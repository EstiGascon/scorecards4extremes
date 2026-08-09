"""
Quaver Compute Backend — Uses native Quaver compute() API
==========================================================

Replaces manual MARS retrieval + custom scoring with Quaver's built-in:
  - compute() for score computation (with orography_correction for 2t)
  - scoredata() for score retrieval
  - normalised_difference() for two-model comparison
  - confintmakers.block_bootstrap for significance testing
  - event() for threshold-based categorical verification

Manual fallbacks:
  - SDFOR-based orography classification (custom_mask GRIBs)
  - Threshold-weighted MAE/RMSE (twMAE, twRMSE)
  - CSV export alongside database storage
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import season_utils


def _load_quaver():
    """Import quaver and vtb modules, return them."""
    try:
        import quaver
        from quaver import (compute, forecast, specifics, scoredata,
                            surfaceobservations, event, DateSequence, StepSequence,
                            ensemble_maker, confintmakers)
        try:
            from quaver import stationclimatology
        except ImportError:
            stationclimatology = None
        import vtb
        return quaver, vtb
    except ImportError as e:
        print(f"ERROR: Could not import quaver/vtb: {e}")
        print("Make sure the ecmwf-toolbox module is loaded.")
        sys.exit(1)


def _get_n_members(config, model_key):
    """Return number of ensemble members for a model, or None if deterministic."""
    mode = config.get('mode', 'deterministic')
    if mode != 'ensemble':
        return None
    q = config['read_data'][model_key].get('quaver', {})
    number = q.get('number', None)
    if number is None:
        return None
    if isinstance(number, list):
        return len(number)
    if isinstance(number, str) and '/' in number:
        parts = number.split('/')
        if 'to' in parts:
            idx = parts.index('to')
            return int(parts[idx + 1]) - int(parts[0]) + 1
        return len(parts)
    return None


def _build_vstream(config, model_key, orog_type=None):
    """Build a vstream label (max 20 chars) for a model.

    Including orog_type as a 2-char suffix ensures each orography class
    (flat/hilly/complex) gets its own vstream in the ScoreDB, avoiding
    overwrite/collision issues when the same model is computed multiple times
    with different SDFOR masks.
    """
    variable = config['variable']
    cfg = config['read_data'][model_key]
    name = cfg.get('name', 'unknown')
    user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
    label = f"{user}_{variable}_{name}"
    if orog_type:
        # Append 2-char orog abbreviation: flat→fl, hilly→hi, complex→co
        label = f"{label}_{orog_type[:2]}"
    return label[:20]


def _get_steps(config):
    """Get forecast steps in hours from config.
    
    Configs may specify either 'steps' (hours) or 'forecast_days' (days).
    """
    steps = config.get('steps')
    if steps:
        return steps
    
    forecast_days = config.get('forecast_days', [1, 3, 5])
    return [d * 24 for d in forecast_days]


def _get_dates(config):
    """Get date range from config as list of datetime objects."""
    start = pd.to_datetime(config['start_date'])
    end = pd.to_datetime(config['end_date'])
    # Generate daily dates (or 12-hourly if needed)
    dates = pd.date_range(start, end, freq='D')
    return dates


def _get_grid(config, model_key):
    """Get the native Gaussian grid for a model.
    
    For obs verification, Quaver needs the native model grid (e.g. O1280).
    This is NOT the same as the 0.1/0.1 lat-lon grid used for interpolation.
    """
    cfg = config['read_data'][model_key]
    q = cfg.get('quaver', {})
    
    # Check for native grid specification
    native_grid = q.get('native_grid')
    if native_grid:
        return native_grid
    
    # Default Gaussian grids based on known model configurations
    name = cfg.get('name', '')
    expver = q.get('expver', '')
    
    if name in ('ifs_oper', 'ifs_ens') or expver == '1':
        return 'O1280'  # tco1279
    elif name in ('ifs_4p4km', 'iekm') or expver == 'iekm':
        return 'O2560'  # tco2559
    elif name in ('hybrid_ens', 'hybrid_det') or expver == 'iy2u':
        return 'O1280'  # hybrid uses tco1279
    else:
        return 'O1280'  # safe default


def _build_events(config):
    """Build Quaver event objects from threshold config.
    
    IMPORTANT: Quaver's ClassFinder.in_globals() walks the call stack looking
    for 'event' (the class) in frame globals.  When we import quaver symbols
    inside a function, they land in f_locals, not f_globals, so ClassFinder
    can't find the 'event' class and raises KeyError('event').
    
    Fix: inject the Quaver classes into THIS module's globals so
    ClassFinder.in_globals() can discover them — f_globals of every frame
    defined in quaver_compute_backend.py is this module's __dict__.
    Also inject into __main__ for extra safety.
    """
    import sys
    from quaver import event

    # Inject into THIS module's globals: these are the f_globals that
    # ClassFinder.in_globals() sees when walking frames from this module.
    globals()['event'] = event

    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        setattr(main_mod, 'event', event)
    
    thresh_cfg = config['threshold']
    method = thresh_cfg['method']
    
    events = []
    
    if method == 'fixed':
        fixed = thresh_cfg['fixed']
        value = fixed['value']
        event_type = fixed['event_type']
        operator = '<' if event_type == 'below' else '>'
        
        ev = event(
            threshold_type='abs',
            threshold_operator=operator,
            threshold_value=value,
            anomaly='no',
        )
        events.append(ev)
    
    elif method in ('dataset_climatology', 'station_climatology'):
        pcfg = thresh_cfg[method]
        percentile = pcfg.get('percentile', pcfg.get('value', None))
        # event_type can be in the sub-dict or at the threshold level
        event_type = pcfg.get('event_type', thresh_cfg.get('event_type', 'above'))
        operator = '<' if event_type == 'below' else '>'
        
        if percentile is not None:
            # Percentile events for quants (quantile score) and ct (Brier score).
            # threshold_type='percentile' works for quants; ct also works.
            ev = event(
                threshold_type='percentile',
                threshold_operator=operator,
                threshold_value=float(percentile),
                anomaly='no',
            )
            events.append(ev)
            print(f"  Percentile event: {operator} {percentile}th percentile "
                  f"→ quants + ct (Brier) via Quaver compute().")
        else:
            print(f"  Percentile threshold (method='{method}'): no percentile value found.")
        
        # NOTE: deterministic CT scores (ETS, PSS, POD, FAR) still use VTB fallback
        # because they need per-station percentile thresholds from STVL climatology.
        print(f"  Continuous scores (CRPS, spread, etc.) computed via Quaver compute().")
    
    return events


def _build_orography_correction(config, model_key):
    """Build interpolation_postprocessor string for orography correction.
    
    For 2t, always use orography correction with the model's own orography.
    For other variables, no correction needed.
    """
    if config['variable'] != '2t':
        return None
    
    # Check if model has a custom orography file
    cfg_aux = config.get('auxiliary_fields', {})
    
    if model_key == 'forecast_model1':
        orog_path = cfg_aux.get('model1', {}).get('orog_path')
    else:
        orog_path = cfg_aux.get('model2', {}).get('orog_path')
    
    if orog_path and os.path.exists(orog_path):
        return f'orography_correction:file={orog_path}'
    else:
        # Default: Quaver retrieves orography from the experiment itself
        return 'orography_correction'


def _create_sdfor_masks(config, model_key=None, n_members=None):
    """Create SDFOR mask GRIB files for flat/hilly/complex classification.
    
    If model_key is given, uses model-specific sdfor_path from
    auxiliary_fields.model1/model2.sdfor_path, falling back to the
    top-level auxiliary_fields.sdfor_path.

    Supports pre-generated masks via auxiliary_fields.model1/2.sdfor_masks dict:
      sdfor_masks:
        flat:    /path/to/sdfor_o2560_flat.grib
        hilly:   /path/to/sdfor_o2560_hilly.grib
        complex: /path/to/sdfor_o2560_complex.grib
    When present, these are used directly without any mask creation.

    n_members: if set (ensemble mode), the mask GRIB is written with N fields,
    one per member (number=1..N), so VTB's xarray 'fieldset' dimension aligns
    with the ensemble and the (50, 1) broadcast error is avoided.
    """
    cfg_aux = config.get('auxiliary_fields', {})
    if model_key == 'forecast_model1':
        model_cfg = cfg_aux.get('model1', {})
    elif model_key == 'forecast_model2':
        model_cfg = cfg_aux.get('model2', {})
    else:
        model_cfg = {}

    # If pre-generated masks are specified, use them directly
    precomputed = model_cfg.get('sdfor_masks', {})
    if precomputed:
        masks = {}
        all_ok = True
        for orog_type, path in precomputed.items():
            if os.path.exists(path):
                masks[orog_type] = path
                print(f"  Using pre-generated SDFOR mask: {path}")
            else:
                print(f"  WARNING: pre-generated SDFOR mask not found: {path}")
                all_ok = False
        if all_ok and masks:
            return masks
        print("  Falling back to mask creation from sdfor_path...")

    try:
        import metview as mv
    except ImportError:
        print("  WARNING: metview not available for SDFOR mask creation")
        return {}
    
    # Model-specific sdfor_path takes priority over top-level sdfor_path
    if model_key == 'forecast_model1':
        sdfor_path = model_cfg.get('sdfor_path') or cfg_aux.get('sdfor_path')
        orog_template_path = model_cfg.get('orog_path')
    elif model_key == 'forecast_model2':
        sdfor_path = model_cfg.get('sdfor_path') or cfg_aux.get('sdfor_path')
        orog_template_path = model_cfg.get('orog_path')
    else:
        sdfor_path = cfg_aux.get('sdfor_path')
        orog_template_path = None
    if not sdfor_path or not os.path.exists(sdfor_path):
        print("  WARNING: SDFOR field not available, cannot create orography masks")
        return {}
    
    orog_ranges = config['filter'].get('orography_ranges', {})
    if not orog_ranges:
        return {}
    
    # Output directory for masks — per-model subdir to avoid grid collisions
    output_dir = config.get('save', {}).get('output_directory', './results')
    model_suffix = f'_{model_key}' if model_key else ''
    mask_dir = os.path.join(output_dir, f'sdfor_masks{model_suffix}')
    os.makedirs(mask_dir, exist_ok=True)
    
    sdfor = mv.read(sdfor_path)

    # Regrid SDFOR to the model's native orography grid if they differ.
    # This handles cases where the SDFOR file resolution doesn't match the
    # forecast field resolution (e.g. sdfor_tco2559 is at N~1280 but
    # tp24 DestinE fields are at true O2560 with 26M points).
    if orog_template_path and os.path.exists(orog_template_path):
        import eccodes
        try:
            sdfor_npts = int(mv.grib_get_long(sdfor[0], 'numberOfDataPoints'))
        except Exception:
            sdfor_npts = None
        try:
            with open(orog_template_path, 'rb') as _f:
                _gh = eccodes.codes_grib_new_from_file(_f)
                orog_npts = eccodes.codes_get(_gh, 'numberOfDataPoints')
                eccodes.codes_release(_gh)
        except Exception:
            orog_npts = None
        if sdfor_npts is not None and orog_npts is not None and sdfor_npts != orog_npts:
            print(f"  Regridding SDFOR from {sdfor_npts} → {orog_npts} points "
                  f"to match model native grid")
            orog_template = mv.read(orog_template_path)
            sdfor = mv.regrid(
                data=sdfor,
                grid_definition_mode='template',
                template_data=orog_template[0],
            )

    masks = {}
    
    for orog_type, (lo, hi) in orog_ranges.items():
        # For ensemble, include member count in filename so det and ens masks
        # coexist in the same directory without colliding.
        if n_members and n_members > 1:
            mask_path = os.path.join(mask_dir, f'sdfor_mask_{orog_type}_ens{n_members}.grib')
        else:
            mask_path = os.path.join(mask_dir, f'sdfor_mask_{orog_type}.grib')
        
        if not os.path.exists(mask_path):
            # Create binary mask: 1 where sdfor is in range, 0 elsewhere
            mask_mv = (sdfor >= lo) * (sdfor < hi)

            if n_members and n_members > 1:
                # Expand single-field mask to N fields, one per ensemble member.
                # VTB reads each GRIB message as one 'fieldset' element keyed by
                # 'number'. The ensemble fieldset is also keyed by 'number' (1..N).
                # Matching coordinates lets xarray broadcast (N, pts) + (N, pts)
                # instead of failing with (N, 1) mismatch.
                import tempfile as _tempfile
                import eccodes as _eccodes
                _fd, _tmp = _tempfile.mkstemp(suffix='.grib')
                os.close(_fd)
                mv.write(_tmp, mask_mv)  # write single-field copy first
                with open(_tmp, 'rb') as _fin:
                    _gh_src = _eccodes.codes_grib_new_from_file(_fin)
                with open(mask_path, 'wb') as _fout:
                    for mbr in range(1, n_members + 1):
                        _gh = _eccodes.codes_clone(_gh_src)
                        _eccodes.codes_set(_gh, 'number', mbr)
                        _eccodes.codes_write(_gh, _fout)
                        _eccodes.codes_release(_gh)
                _eccodes.codes_release(_gh_src)
                os.unlink(_tmp)
                print(f"  Created ensemble SDFOR mask: {mask_path} "
                      f"({n_members} members, sdfor {lo}-{hi})")
            else:
                mv.write(mask_path, mask_mv)
                print(f"  Created SDFOR mask: {mask_path} (sdfor {lo}-{hi})")
        else:
            print(f"  SDFOR mask exists: {mask_path}")
        
        masks[orog_type] = mask_path
    
    return masks


def compute_scores_for_model(config, model_key, dates_batch, events, sdfor_mask=None, orog_type=None):
    """Run Quaver compute() for one model.

    Follows the operational pattern:
      1. Continuous scores (rmsef, mef, maef) — one compute() call
      2. CT scores with events — separate compute() call (both det and ens)

    Args:
        config: Full config dict
        model_key: 'forecast_model1' or 'forecast_model2'
        dates_batch: List of dates to process
        events: List of Quaver event objects
        sdfor_mask: Path to SDFOR mask GRIB (for orography filtering)
        orog_type: Orography class label (for vstream uniqueness)

    Returns:
        vstream label used
    """
    quaver, vtb = _load_quaver()
    from quaver import (compute, forecast, specifics, surfaceobservations,
                        StepSequence, DateSequence)
    try:
        from quaver import stationclimatology
    except ImportError:
        stationclimatology = None

    # Inject Quaver classes into THIS module's globals AND __main__.
    # Required because Quaver's ClassFinder.in_globals() walks the call stack.
    import sys
    from quaver import event as _event_cls
    _quaver_syms = {'event': _event_cls, 'compute': compute,
                    'forecast': forecast, 'specifics': specifics,
                    'surfaceobservations': surfaceobservations,
                    'StepSequence': StepSequence}
    if stationclimatology is not None:
        _quaver_syms['stationclimatology'] = stationclimatology
    globals().update(_quaver_syms)
    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        for _name, _obj in _quaver_syms.items():
            setattr(main_mod, _name, _obj)

    cfg = config['read_data'][model_key]
    q = cfg.get('quaver', {})
    # Pad numeric expver to 4 chars so it matches GRIB header encoding.
    _raw_expver = str(q.get('expver', '0001'))
    if _raw_expver.isdigit():
        q = dict(q, expver=f'{int(_raw_expver):04d}')
    variable = config['variable']
    mode = config.get('mode', 'deterministic')
    steps = _get_steps(config)
    vstream = _build_vstream(config, model_key, orog_type=orog_type)

    # Build orography correction string (only for 2t)
    orog_correction = _build_orography_correction(config, model_key)

    # Build preprocessors list
    preprocessors = []
    if sdfor_mask and os.path.exists(sdfor_mask):
        mask_label = os.path.splitext(os.path.basename(sdfor_mask))[0]
        preprocessors.append(f'custom_mask:file={sdfor_mask},label={mask_label}')
    if config['filter'].get('remove_coastal_stations', False):
        preprocessors.append('mask_over_sea')

    # --- Build reference (same for all calls) ---
    ref_kwargs = {}
    if stationclimatology is not None:
        ref_kwargs['climatology'] = stationclimatology()
    obs_filter = ['toss'] if variable == 'tp24' else []
    if obs_filter:
        ref_kwargs['observation_filter'] = obs_filter
    ref = surfaceobservations(**ref_kwargs)

    # --- Parameter and domain config ---
    param_map = {'2t': '2t', '10ff': '10ff', 'tp24': 'tp'}
    quaver_param = param_map.get(variable, variable)
    area = config.get('extract_points', {}).get('area', 'europe')
    domains = [area] if isinstance(area, str) else area
    period = 24 if variable == 'tp24' else None

    # --- Database routing ---
    # od-class → marsod; rd-class → no database key (let MARS auto-route to marsrd).
    # Forcing database=fdb for rd-class breaks retrieval: iekm tp is in marsrd not fdb.
    _cls = q.get('class', 'od').lower()
    if 'database' in q:
        db = q['database']
    elif _cls == 'rd':
        db = None   # MARS auto-routes to marsrd
    else:
        db = 'marsod'

    # --- Build scores lists ---
    if mode == 'deterministic':
        continuous_scores = ['rmsef', 'mef', 'maef']
    else:
        continuous_scores = ['crps', 'fcrps', 'spread', 'diags', 'rmsef']

    # --- Build step sequence ---
    max_step = max(steps)
    min_step = min(steps)
    step_freq = steps[1] - steps[0] if len(steps) > 1 else 24
    step_seq = StepSequence(min_step, max_step, step_freq)

    # --- Process dates using DateSequence pattern (12-hourly, matching examples) ---
    # Convert dates to integer format YYYYMMDDHH for DateSequence
    start_int = int(dates_batch[0].strftime('%Y%m%d00'))
    end_int = int(dates_batch[-1].strftime('%Y%m%d00'))

    print(f"\n  Model: {cfg.get('name', model_key)} | vstream: {vstream}")
    print(f"  Dates: {dates_batch[0].strftime('%Y%m%d')} - {dates_batch[-1].strftime('%Y%m%d')} "
          f"({len(dates_batch)} dates)")
    print(f"  Steps: {steps}")

    for datein in DateSequence(start_int, end_int, 24):
        # --- Build forecast ---
        if mode == 'ensemble':
            members = q.get('number', '1/to/50')
            if isinstance(members, str):
                parts = members.split('/')
                if 'to' in parts:
                    start_m = int(parts[0])
                    end_m = int(parts[parts.index('to') + 1])
                    member_list = list(range(start_m, end_m + 1))
                else:
                    member_list = [int(x) for x in parts]
            else:
                member_list = members

            fc_kw = dict(
                date=datein, step=step_seq,
                Class=q['class'], expver=q['expver'],
                stream=q.get('stream', 'enfo'), type=q.get('type', 'pf'),
                number=member_list,
            )
            if db is not None:
                fc_kw['database'] = db
            fc = forecast(**fc_kw)
        else:
            fc_kw = dict(
                date=datein, step=step_seq,
                Class=q['class'], expver=q['expver'],
                stream=q.get('stream', 'oper'), type=q.get('type', 'fc'),
            )
            if db is not None:
                fc_kw['database'] = db
            fc = forecast(**fc_kw)

        # --- Common compute kwargs ---
        base_compute_kwargs = dict(
            forecast=fc,
            reference=ref,
            vstream=vstream,
            overwrite='yes',
            spatial_mean_weights='station_density',
            ignore_missing='no',
        )
        if orog_correction:
            base_compute_kwargs['interpolation_postprocessor'] = orog_correction
        if preprocessors:
            base_compute_kwargs['preprocess'] = preprocessors

        # --- Call 1: Continuous scores ---
        native_grid = _get_grid(config, model_key)
        spec_cont_kw = dict(
            levtype='sfc',
            parameter=[quaver_param],
            grid=native_grid,
            score=continuous_scores,
            domain=domains,
        )
        if period:
            spec_cont_kw['period'] = period
        # For ensemble, include events for ct and quants in the continuous call
        if mode == 'ensemble' and events:
            spec_cont_kw['score'] = continuous_scores + ['ct']
            thresh_method = config['threshold'].get('method', '')
            if thresh_method in ('station_climatology', 'dataset_climatology'):
                spec_cont_kw['score'].append('quants')
            spec_cont_kw['events'] = events

        try:
            compute(
                **base_compute_kwargs,
                specifics=specifics(**spec_cont_kw),
            )
            print(f"    ✓ [{datein}] Continuous scores computed")
        except Exception as e:
            import traceback as _tb
            print(f"    ⚠ [{datein}] Continuous compute() failed: {e}")
            print(_tb.format_exc())

        # --- Call 2: CT scores (deterministic only — ensemble CT is in call 1) ---
        if mode == 'deterministic' and events:
            spec_ct_kw = dict(
                levtype='sfc',
                parameter=[quaver_param],
                grid=native_grid,
                score=['ct'],
                domain=domains,
                events=events,
            )
            if period:
                spec_ct_kw['period'] = period

            try:
                compute(
                    **base_compute_kwargs,
                    specifics=specifics(**spec_ct_kw),
                )
                print(f"    ✓ [{datein}] CT scores computed")
            except Exception as e:
                import traceback as _tb
                print(f"    ⚠ [{datein}] CT compute() failed: {e}")
                print(_tb.format_exc())

    return vstream


def retrieve_scores(config, vstream, model_key, score_name, domain='europe', step=None):
    """Retrieve scores from database using scoredata().
    
    Returns pandas DataFrame with columns: date, step, value, etc.
    """
    quaver, _ = _load_quaver()
    from quaver import scoredata
    
    cfg = config['read_data'][model_key]
    q = cfg.get('quaver', {})
    variable = config['variable']
    
    param_map = {'2t': '2t', '10ff': '10ff', 'tp24': 'tp'}
    quaver_param = param_map.get(variable, variable)
    
    dates = _get_dates(config)
    
    sd_kwargs = dict(
        parameter=quaver_param,
        levtype='sfc',
        score=score_name,
        domain_name=domain,
        date=dates,
        vstream=vstream,
        Class=q['class'],
        expver=q['expver'],
        stream=q.get('stream', 'oper'),
        type=q.get('type', 'fc'),
    )
    
    if step is not None:
        sd_kwargs['step'] = pd.to_timedelta(f"{step}h")
    
    sd = scoredata(**sd_kwargs)
    return sd


def compute_and_compare(config, model_names):
    """Full workflow: compute scores for both models and compare.
    
    This is the main entry point for the quaver_compute backend.
    
    Steps:
    1. Create SDFOR masks if needed
    2. compute() for model1 and model2 (all dates, steps, events)
    3. scoredata() to retrieve scores
    4. normalised_difference() for comparison
    5. block_bootstrap for significance
    6. Compute twMAE/twRMSE manually (not in Quaver)
    7. Export to CSV + plots
    
    Returns:
        dict with all results for downstream plotting/saving
    """
    quaver, vtb = _load_quaver()
    from quaver import scoredata, confintmakers
    
    variable = config['variable']
    mode = config.get('mode', 'deterministic')
    dates = _get_dates(config)
    steps = _get_steps(config)
    
    # Build events
    events = _build_events(config)
    
    # Get orography types
    orography_types = config['filter'].get('orography_type', [None])
    if isinstance(orography_types, str):
        orography_types = [orography_types]
    if not orography_types:
        orography_types = [None]
    
    # Create SDFOR masks — per model so grids match (model1=O1280, model2=O2560).
    # For ensemble mode, masks are expanded to N fields (one per member) so that
    # VTB's xarray 'fieldset' dimension aligns and (N, pts) + (N, pts) broadcasts.
    sdfor_masks1 = {}
    sdfor_masks2 = {}
    if any(o is not None for o in orography_types):
        n_members1 = _get_n_members(config, 'forecast_model1')
        n_members2 = _get_n_members(config, 'forecast_model2')
        sdfor_masks1 = _create_sdfor_masks(config, model_key='forecast_model1',
                                            n_members=n_members1)
        sdfor_masks2 = _create_sdfor_masks(config, model_key='forecast_model2',
                                            n_members=n_members2)
    
    # Season handling
    seasons = config['filter'].get('season', [None])
    seasons_to_process = season_utils.parse_seasons_config(seasons)
    
    all_results = []
    
    for season in seasons_to_process:
        # Filter dates by season
        season_label, season_months_list = season_utils.resolve_season(season)
        if season_months_list:
            season_dates = [d for d in dates if d.month in season_months_list]
        else:
            season_dates = list(dates)
        
        if not season_dates:
            print(f"  No dates for season {season_label}, skipping")
            continue
        
        for orog_type in orography_types:
            print(f"\n{'='*60}")
            print(f"Computing: season={season_label}, orography={orog_type}")
            print(f"{'='*60}")
            
            sdfor_mask1 = sdfor_masks1.get(orog_type) if orog_type else None
            sdfor_mask2 = sdfor_masks2.get(orog_type) if orog_type else None
            
            # Split season_dates into manageable chunks so that MARS retrieval
            # inside compute() stays within the 48-hour wall-time limit.
            # Default: 90 dates per chunk (~3 months). Configurable via
            # 'chunk_size_days' in the config (set to 0 or null to disable).
            chunk_size = config.get('chunk_size_days', 90) or len(season_dates)
            date_chunks = [season_dates[i:i + chunk_size]
                           for i in range(0, len(season_dates), chunk_size)]
            n_chunks = len(date_chunks)
            if n_chunks > 1:
                print(f"\n  Auto-chunking: {len(season_dates)} dates → "
                      f"{n_chunks} chunks of up to {chunk_size} days")

            # Compute scores chunk by chunk — results accumulate in VTB database.
            # vstream is deterministic (config + orog_type), same across all chunks.
            vstream1 = vstream2 = None
            for chunk_idx, chunk in enumerate(date_chunks, 1):
                if n_chunks > 1:
                    print(f"\n  --- Chunk {chunk_idx}/{n_chunks}: "
                          f"{chunk[0].strftime('%Y-%m-%d')} → "
                          f"{chunk[-1].strftime('%Y-%m-%d')} "
                          f"({len(chunk)} dates) ---")
                vstream1 = compute_scores_for_model(
                    config, 'forecast_model1', chunk, events, sdfor_mask1,
                    orog_type=orog_type)
                vstream2 = compute_scores_for_model(
                    config, 'forecast_model2', chunk, events, sdfor_mask2,
                    orog_type=orog_type)

            # Retrieve across all season_dates (DB has accumulated all chunks)
            result = _retrieve_and_compare(
                config, vstream1, vstream2, season_dates, steps, events,
                season_label, orog_type)
            
            all_results.append(result)
    
    return all_results


def _get_retrieval_postprocessing_name(config, model_key, orog_type):
    """Reconstruct the postprocessing_name stored by compute() for this model/orog combo.

    The postprocessing_name is built from the preprocessors applied in compute():
    1. custom_mask: stores 'masked:<label>' where label = basename(mask_path, no ext)
    2. mask_over_sea: stores 'over_sea' (if remove_coastal_stations is True)
    Joined with commas in application order.

    Returns 'na' when no preprocessing was applied (default ScoreDB value).
    """
    parts = []

    if orog_type is not None:
        # Determine the mask file that was used for this model/orog combo
        cfg_aux = config.get('auxiliary_fields', {})
        if model_key == 'forecast_model1':
            model_cfg = cfg_aux.get('model1', {})
        elif model_key == 'forecast_model2':
            model_cfg = cfg_aux.get('model2', {})
        else:
            model_cfg = {}

        precomputed = model_cfg.get('sdfor_masks', {})
        if precomputed and orog_type in precomputed:
            # Pre-generated mask (e.g. sdfor_o2560_flat.grib)
            mask_path = precomputed[orog_type]
        else:
            # Mask created by _create_sdfor_masks in the output directory
            output_dir = config.get('save', {}).get('output_directory', './results')
            mask_path = os.path.join(
                output_dir, f'sdfor_masks_{model_key}', f'sdfor_mask_{orog_type}.grib')

        mask_label = os.path.splitext(os.path.basename(mask_path))[0]
        parts.append(f'masked:{mask_label}')

    if config['filter'].get('remove_coastal_stations', False):
        parts.append('over_sea')

    return ','.join(parts) if parts else 'na'


def _retrieve_and_compare(config, vstream1, vstream2, dates, steps, events,
                           season, orog_type):
    """Retrieve scores from database and compute differences.

    Uses native Quaver scoredata() to retrieve:
      - Continuous scores (rmse, bias, mae / crps, spread, etc.)
      - CT-derived scores (ets(ct), pss(ct) for deterministic; bs(ct) for ensemble)
      - Raw CT for POD/FAR derivation

    For non-binary threshold-weighted scores (twMAE, twCRPS), delegates to
    _compute_tw_scores_via_vtb() which extracts station-level data.

    Returns dict compatible with the existing pipeline's result format.
    """
    quaver, _ = _load_quaver()
    from quaver import scoredata, confintmakers

    variable = config['variable']
    mode = config.get('mode', 'deterministic')
    param_map = {'2t': '2t', '10ff': '10ff', 'tp24': 'tp'}
    quaver_param = param_map.get(variable, variable)

    cfg1 = config['read_data']['forecast_model1']
    cfg2 = config['read_data']['forecast_model2']
    q1 = cfg1.get('quaver', {})
    q2 = cfg2.get('quaver', {})

    fc1_name = cfg1.get('name', 'model1')
    fc2_name = cfg2.get('name', 'model2')

    area = config.get('extract_points', {}).get('area', 'europe')

    # Map between Quaver score names and our pipeline names
    if mode == 'deterministic':
        score_map = {
            'rmsef': 'rmse',
            'mef': 'bias',
            'maef': 'mae',
        }
    else:
        score_map = {
            'crps': 'CRPS',
            'fcrps': 'fCRPS',
            'spread': 'spread',
            'diags': 'diagonal_score',
            'rmsef': 'ens_mean_rmse',
        }

    # Event-based scores (need event parameter in scoredata)
    event_scores = {}
    if events:
        if mode == 'ensemble':
            event_scores['bs(ct)'] = 'Brier'
            thresh_method = config['threshold'].get('method', '')
            if thresh_method in ('station_climatology', 'dataset_climatology'):
                event_scores['quants'] = 'quantile_score'
        else:
            # Deterministic: ETS and PSS from ct
            event_scores['ets(ct)'] = 'ETS'
            event_scores['pss(ct)'] = 'PSS'

    # Derive postprocessing_name to match what compute() stored in ScoreDB.
    pname1 = _get_retrieval_postprocessing_name(config, 'forecast_model1', orog_type)
    pname2 = _get_retrieval_postprocessing_name(config, 'forecast_model2', orog_type)
    print(f"  [INFO] scoredata postprocessing_name: model1='{pname1}', model2='{pname2}'")

    # When stationclimatology is used, compute() stores CT with climat='rodw'.
    # scoredata defaults to climat='era5' for percentile events, causing 0 results.
    # Pass climat='rodw' explicitly to override the default.
    thresh_method = config['threshold'].get('method', '')
    sd_event_extra = {}
    if thresh_method == 'station_climatology':
        sd_event_extra['climat'] = 'rodw'

    # Retrieve per-step scores
    results_by_leadtime = []
    overall_scores = {
        'model1_name': fc1_name,
        'model2_name': fc2_name,
    }

    for step in steps:
        step_td = pd.to_timedelta(f"{step}h")
        day = step // 24

        row = {
            'lead_time': day,
            'step': step,
        }

        # --- Continuous scores (no event parameter) ---
        for q_score, our_score in score_map.items():
            row = _retrieve_score_pair(
                row, q_score, our_score,
                quaver_param, area, dates, step_td,
                vstream1, vstream2, q1, q2,
                pname1, pname2,
                event_obj=None,
            )

        # --- Event-based scores ---
        if event_scores and events:
            ev = events[0]  # primary event for scoredata retrieval
            for q_score, our_score in event_scores.items():
                row = _retrieve_score_pair(
                    row, q_score, our_score,
                    quaver_param, area, dates, step_td,
                    vstream1, vstream2, q1, q2,
                    pname1, pname2,
                    event_obj=ev,
                    sd_extra=sd_event_extra,
                )

        # --- POD/FAR from raw CT (deterministic) ---
        if events and mode == 'deterministic':
            ev = events[0]
            row = _retrieve_pod_far_from_ct(
                row, quaver_param, area, dates, step_td,
                vstream1, vstream2, q1, q2,
                pname1, pname2, ev,
                sd_extra=sd_event_extra,
            )

        # --- Derived ensemble scores ---
        if mode == 'ensemble':
            row = _compute_derived_ensemble_scores(
                row, events, quaver_param, area, dates, step_td,
                vstream1, q1, pname1,
                sd_extra=sd_event_extra,
            )

        results_by_leadtime.append(row)

    # --- VTB direct scores (both deterministic and ensemble modes) ---
    # det: RMSE/Bias/MAE + twRMSE/twBias/twMAE + ETS/PSS/POD/FAR
    # ens: CRPS/fCRPS/spread/Brier + twCRPS/twMAE/tw_spread_skill + ETS/PSS/POD/FAR
    # Run in a subprocess so that Metview/Quaver C++ memory is released before
    # VTB loads the iekm O2560 fields (otherwise both sets coexist and OOM).
    vtb_results = _run_vtb_direct_in_subprocess(config, dates, steps)
    if vtb_results:
        for i, step in enumerate(steps):
            if step in vtb_results:
                step_vtb = vtb_results[step]
                for score_name in step_vtb.get('fc1', {}):
                    v1 = step_vtb['fc1'].get(score_name, np.nan)
                    v2 = step_vtb['fc2'].get(score_name, np.nan)
                    results_by_leadtime[i][f'{score_name}_fc1'] = v1
                    results_by_leadtime[i][f'{score_name}_fc2'] = v2
                    results_by_leadtime[i][f'{score_name}_diff'] = (
                        v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)
                    # Significance for VTB direct scores uses the existing Quaver-based
                    # normalised_difference result when the score name matches, otherwise
                    # conservatively set to False (no significance claim).
                    existing_sig_key = f'{score_name}_is_significant'
                    if existing_sig_key not in results_by_leadtime[i]:
                        results_by_leadtime[i][existing_sig_key] = False

    # Build overall scores (mean across lead times)
    by_lt_df = pd.DataFrame(results_by_leadtime)

    all_score_names = list(score_map.values()) + list(event_scores.values())
    if events and mode == 'deterministic':
        all_score_names += ['POD', 'FAR']
    if mode == 'ensemble':
        all_score_names += ['spread_error_ratio']
        all_score_names += ['tw_spread_error_ratio']
    if vtb_results:
        # Gather all VTB direct score names from the first non-empty step
        for step_vtb in vtb_results.values():
            all_score_names += list(step_vtb.get('fc1', {}).keys())
            break

    for score_name in set(all_score_names):
        for suffix in ['_fc1', '_fc2', '_diff']:
            col = f'{score_name}{suffix}'
            if col in by_lt_df.columns:
                overall_scores[col] = by_lt_df[col].mean()
        sig_col = f'{score_name}_is_significant'
        if sig_col in by_lt_df.columns:
            overall_scores[sig_col] = bool(by_lt_df[sig_col].all())
        for ci_suffix in ['_diff_ci_low', '_diff_ci_high']:
            col = f'{score_name}{ci_suffix}'
            if col in by_lt_df.columns:
                overall_scores[col] = by_lt_df[col].mean()

    # Determine threshold_value for downstream plotting
    thresh_cfg = config['threshold']
    method = thresh_cfg['method']
    if method == 'fixed':
        threshold_value = thresh_cfg['fixed']['value']
    elif method == 'dataset_climatology':
        threshold_value = thresh_cfg['dataset_climatology']['percentile']
    elif method == 'station_climatology':
        threshold_value = thresh_cfg['station_climatology']['percentile']
    else:
        threshold_value = 0

    results = {
        'by_leadtime': by_lt_df,
        'season': season,
        'orog_type': orog_type,
    }

    return {
        'orog_type': orog_type,
        'season': season,
        'data': None,
        'results': results,
        'overall_scores': overall_scores,
        'threshold_value': threshold_value,
        'vstream1': vstream1,
        'vstream2': vstream2,
    }


def _retrieve_score_pair(row, q_score, our_score,
                         quaver_param, area, dates, step_td,
                         vstream1, vstream2, q1, q2,
                         pname1, pname2, event_obj=None,
                         sd_extra=None):
    """Retrieve a score for both models, compute diff and significance."""
    from quaver import scoredata, confintmakers

    sd_common = dict(
        parameter=quaver_param, levtype='sfc',
        score=q_score, domain_name=area,
        date=dates, step=step_td,
    )
    if event_obj is not None:
        sd_common['event'] = event_obj
        if sd_extra:
            sd_common.update(sd_extra)

    try:
        sd1 = scoredata(
            **sd_common,
            vstream=vstream1,
            Class=q1['class'], expver=q1['expver'],
            stream=q1.get('stream', 'oper'),
            type=q1.get('type', 'fc'),
            postprocessing_name=pname1,
        )
        sd2 = scoredata(
            **sd_common,
            vstream=vstream2,
            Class=q2['class'], expver=q2['expver'],
            stream=q2.get('stream', 'oper'),
            type=q2.get('type', 'fc'),
            postprocessing_name=pname2,
        )

        mean1 = sd1.mean()
        mean2 = sd2.mean()

        v1 = float(mean1['value'].iloc[0]) if len(mean1) > 0 else np.nan
        v2 = float(mean2['value'].iloc[0]) if len(mean2) > 0 else np.nan

        row[f'{our_score}_fc1'] = v1
        row[f'{our_score}_fc2'] = v2
        row[f'{our_score}_diff'] = v2 - v1

        # Significance via normalised difference + block bootstrap
        try:
            ndiff = sd2.normalised_difference(sd1, normalisation_method='control')
            ndiff_mean = ndiff.mean(
                confint_maker=confintmakers.block_bootstrap(confidence=95))
            if 'value_lower' in ndiff_mean.columns and 'value_upper' in ndiff_mean.columns:
                ci_low = float(ndiff_mean['value_lower'].iloc[0])
                ci_high = float(ndiff_mean['value_upper'].iloc[0])
                row[f'{our_score}_is_significant'] = (ci_low > 0) or (ci_high < 0)
                row[f'{our_score}_diff_ci_low'] = ci_low
                row[f'{our_score}_diff_ci_high'] = ci_high
            else:
                row[f'{our_score}_is_significant'] = False
        except Exception as e:
            print(f"    [WARN] Significance failed for {our_score}: {e}")
            row[f'{our_score}_is_significant'] = False

    except Exception as e:
        print(f"    [WARN] Could not retrieve {q_score} for step {step_td}: {e}")
        row[f'{our_score}_fc1'] = np.nan
        row[f'{our_score}_fc2'] = np.nan
        row[f'{our_score}_diff'] = np.nan
        row[f'{our_score}_is_significant'] = False

    return row


def _retrieve_pod_far_from_ct(row, quaver_param, area, dates, step_td,
                              vstream1, vstream2, q1, q2,
                              pname1, pname2, event_obj,
                              sd_extra=None):
    """Retrieve raw deterministic CT and derive POD/FAR.

    Quaver det CT order: value_0=Misses, value_1=Hits, value_2=Correct, value_3=FalseAlarms
    """
    from quaver import scoredata

    sd_common = dict(
        parameter=quaver_param, levtype='sfc',
        score='ct', domain_name=area,
        date=dates, step=step_td,
        event=event_obj,
    )
    if sd_extra:
        sd_common.update(sd_extra)

    try:
        for model_label, vstream_id, q_cfg, pname in [
            ('fc1', vstream1, q1, pname1), ('fc2', vstream2, q2, pname2)]:
            sd_raw = scoredata(
                **sd_common,
                vstream=vstream_id,
                Class=q_cfg['class'], expver=q_cfg['expver'],
                stream=q_cfg.get('stream', 'oper'),
                type=q_cfg.get('type', 'fc'),
                postprocessing_name=pname,
            )
            ct_data = sd_raw.data()
            if ct_data is not None and len(ct_data) > 0:
                M = ct_data['value_0'].sum()
                H = ct_data['value_1'].sum()
                F = ct_data['value_3'].sum()

                pod = H / (H + M) if (H + M) > 0 else np.nan
                far = F / (H + F) if (H + F) > 0 else np.nan
                row[f'POD_{model_label}'] = pod
                row[f'FAR_{model_label}'] = far
            else:
                row[f'POD_{model_label}'] = np.nan
                row[f'FAR_{model_label}'] = np.nan

        row['POD_diff'] = row.get('POD_fc2', np.nan) - row.get('POD_fc1', np.nan)
        row['FAR_diff'] = row.get('FAR_fc2', np.nan) - row.get('FAR_fc1', np.nan)
        row['POD_is_significant'] = False
        row['FAR_is_significant'] = False
    except Exception as e:
        print(f"    [WARN] Could not compute POD/FAR from raw CT: {e}")
        row['POD_fc1'] = row['POD_fc2'] = row['POD_diff'] = np.nan
        row['FAR_fc1'] = row['FAR_fc2'] = row['FAR_diff'] = np.nan
        row['POD_is_significant'] = False
        row['FAR_is_significant'] = False

    return row


def _compute_derived_ensemble_scores(row, events, quaver_param, area, dates, step_td,
                                     vstream1, q1, pname1,
                                     sd_extra=None):
    """Compute spread/error ratio for ensemble mode."""

    # --- Spread/error ratio ---
    spread1 = row.get('spread_fc1', np.nan)
    spread2 = row.get('spread_fc2', np.nan)
    rmse1 = row.get('ens_mean_rmse_fc1', np.nan)
    rmse2 = row.get('ens_mean_rmse_fc2', np.nan)

    row['spread_error_ratio_fc1'] = (
        spread1 / rmse1 if not np.isnan(spread1) and not np.isnan(rmse1) and rmse1 > 1e-10
        else np.nan)
    row['spread_error_ratio_fc2'] = (
        spread2 / rmse2 if not np.isnan(spread2) and not np.isnan(rmse2) and rmse2 > 1e-10
        else np.nan)
    row['spread_error_ratio_diff'] = (
        row['spread_error_ratio_fc2'] - row['spread_error_ratio_fc1'])
    row['spread_error_ratio_is_significant'] = False

    return row


def _vtb_direct_subprocess_worker(config, dates, steps, result_path):
    """Worker function executed in a child process for VTB direct scoring.

    Dispatches to deterministic or ensemble implementation based on config mode.
    Runs the appropriate function and pickles the result to result_path.
    The child process exits after pickling, freeing all Metview/VTB C++ memory.
    """
    import pickle as _pickle
    try:
        mode = config.get('mode', 'deterministic')
        if mode == 'ensemble':
            result = _compute_vtb_direct_ens_scores(config, dates, steps)
        else:
            result = _compute_vtb_direct_det_scores(config, dates, steps)
    except Exception as e:
        import traceback as _tb
        print(f"  [VTB subprocess] ERROR: {e}")
        print(_tb.format_exc())
        result = {}
    with open(result_path, 'wb') as fh:
        _pickle.dump(result, fh, protocol=4)


def _run_vtb_direct_in_subprocess(config, dates, steps):
    """Run _compute_vtb_direct_det_scores() in a subprocess.

    Quaver/Metview holds C++ memory that cannot be released via Python gc.
    Running VTB direct in a child process ensures the child starts with a
    fresh address space (fork-on-start), and process exit reclaims all memory
    before the parent continues.
    """
    import multiprocessing as _mp
    import pickle as _pickle
    import tempfile as _tmpfile

    # Use a temp file to pass results from child → parent
    with _tmpfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tf:
        result_path = tf.name

    print(f"\n  [VTB direct] Launching subprocess (PID will appear below) ...")
    ctx = _mp.get_context('fork')
    proc = ctx.Process(
        target=_vtb_direct_subprocess_worker,
        args=(config, dates, steps, result_path),
    )
    proc.start()
    print(f"  [VTB direct] Subprocess PID: {proc.pid}")
    proc.join()

    if proc.exitcode != 0:
        print(f"  [VTB direct] Subprocess exited with code {proc.exitcode} — no VTB scores")
        return {}

    try:
        with open(result_path, 'rb') as fh:
            vtb_results = _pickle.load(fh)
    except Exception as e:
        print(f"  [VTB direct] Failed to read subprocess results: {e}")
        vtb_results = {}
    finally:
        import os as _os
        try:
            _os.unlink(result_path)
        except OSError:
            pass

    return vtb_results


def _compute_vtb_direct_det_scores(config, dates, steps):
    """Compute all deterministic scores via VTB direct method (MARS + STVL).

    Implements the validated VTB direct pattern from test_vtb_direct_iekm.py
    and test_vtb_direct_10ff_2t.py for integration into the qcompute backend.

    Computes per step and model:
      - RMSE, Bias, MAE  (unweighted, all observations)
      - twRMSE, twBias, twMAE  (threshold-weighted, obs-exceedance mask)
      - ETS, PSS, POD, FAR  (contingency table from VTB event masks)

    Variable-specific preprocessing:
      - 10ff:  wind speed from u/v components via Fieldset arithmetic
      - 2t:    lapse-rate correction using model orography (config auxiliary_fields)
      - tp24:  24h de-accumulation (subtract previous step)

    Threshold configuration:
      - threshold.station_climatology.percentile: int or list of ints
      - Single percentile → scores named 'ETS', 'PSS', 'twRMSE', etc.
      - Multiple percentiles → 'ETS_p95', 'ETS_p98', ... plus primary without suffix

    Returns:
        dict: {step_hours: {'fc1': {score: value}, 'fc2': {score: value}}}
        Empty dict when not applicable (ensemble mode, fixed threshold, VTB import fails).
    """
    mode = config.get('mode', 'deterministic')
    if mode != 'deterministic':
        return {}

    variable = config['variable']

    # Only station/dataset climatology thresholds are supported (need STVL percentiles)
    thresh_cfg = config.get('threshold', {})
    method = thresh_cfg.get('method', '')
    if method not in ('station_climatology', 'dataset_climatology'):
        return {}

    pcfg = thresh_cfg[method]
    percentile_cfg = pcfg.get('percentile', pcfg.get('value'))
    if percentile_cfg is None:
        return {}
    event_type = pcfg.get('event_type', thresh_cfg.get('event_type', 'above'))

    # Normalise to list; support both single int and list of ints
    if isinstance(percentile_cfg, (int, float)):
        percentiles = [int(percentile_cfg)]
    else:
        percentiles = [int(p) for p in percentile_cfg]

    operator = '>' if event_type == 'above' else '<'
    percentiles_and_ops = [(operator, p) for p in percentiles]
    multi_threshold = len(percentiles) > 1

    # VTB imports
    try:
        import vtb
        import xarray as xr
        from vtb.metricss import xmetrics as vtb_xmetrics
        from vtb.metricss._score_array_interface import complete_quantiles
        from vtb.metricss.event import BinaryEvent
    except ImportError as e:
        print(f"  [VTB direct] VTB not available: {e}")
        return {}

    G = 9.80665  # standard gravity [m s⁻²], for geopotential → metres

    obs_sources = config.get('read_data', {}).get('quaver_obs', {}).get('sources', ['synop'])

    # Build pandas-format steps and valid datetimes for STVL obs retrieval
    pandas_steps = [pd.to_timedelta(f"{s}h") for s in steps]
    vdates = sorted(set(d + s for d in dates for s in pandas_steps))

    # Extract model MARS kwargs
    def _mars_cfg(mk):
        cfg_m = config['read_data'].get(mk, {})
        q = cfg_m.get('quaver', {})
        raw_expver = str(q.get('expver', '0001'))
        expver = f'{int(raw_expver):04d}' if raw_expver.isdigit() else raw_expver
        return {
            'name': cfg_m.get('name', mk),
            'class_': q.get('class', 'od'),
            'expver': expver,
            'stream': q.get('stream', 'oper'),
            'type': q.get('type', 'fc'),
        }

    model_cfgs = {
        'fc1': _mars_cfg('forecast_model1'),
        'fc2': _mars_cfg('forecast_model2'),
    }
    model_mk = {'fc1': 'forecast_model1', 'fc2': 'forecast_model2'}

    # Orography paths for 2t lapse-rate correction
    orog_paths = {}
    if variable == '2t':
        cfg_aux = config.get('auxiliary_fields', {})
        orog_paths['fc1'] = cfg_aux.get('model1', {}).get('orog_path', '')
        orog_paths['fc2'] = cfg_aux.get('model2', {}).get('orog_path', '')

    # STVL parameter name — tp24 maps to 'tp' in STVL (not 'tp24')
    stvl_param = 'tp' if variable == 'tp24' else variable

    # --- Fix STVL climatology headers for VTB FieldMetrics compatibility ---
    # STVL returns number 0-based; VTB FieldMetrics expects type=pb,
    # numberOfForecastsInEnsemble=100, number 1-based.
    def _fix_clim_headers(station_clim):
        fixed = []
        for i in range(len(station_clim)):
            f = station_clim[i]
            v = f.header_get("number")
            if isinstance(v, list): v = v[0]
            if isinstance(v, list): v = v[0]
            f = f.header_set(type="pb", numberOfForecastsInEnsemble=100, number=int(v) + 1)
            fixed.append(f)
        return vtb.Fieldset(*fixed)

    # --- CT + threshold-weighted scores from a FieldMetrics object ---
    def _ct_tw_scores(fm, obs_arr, fc_vals, pcts_ops, multi):
        """Return dict of twRMSE/twBias/twMAE/ETS/PSS/POD/FAR for each threshold."""
        cl_obs = complete_quantiles(fm._arrays["observation_climatology"], qtype="percentile")
        scores = {}
        for (op, pctl) in pcts_ops:
            bev = BinaryEvent(operator=op, value=pctl, type="percentile", is_anomaly=False)
            ev_obs_da = vtb_xmetrics.event_occurrences(bev, forecasts=obs_arr,
                                                        forecast_climatology=cl_obs)
            ev_fc_da  = vtb_xmetrics.event_occurrences(bev, forecasts=fc_vals,
                                                        forecast_climatology=cl_obs)
            ev_obs_b, ev_fc_b = xr.broadcast(ev_obs_da, ev_fc_da)

            # Threshold-weighted continuous scores (weight = obs exceedance)
            w_obs = ev_obs_da.fillna(0.0)
            n_extreme = int(w_obs.sum().values)

            if n_extreme > 0:
                tw_rmse = float(
                    vtb_xmetrics.root_mean_squared_error(obs_arr, fc_vals, weights=w_obs).values)
                tw_bias = float(
                    vtb_xmetrics.mean_error(obs_arr, fc_vals, weights=w_obs).values)
                err_da  = vtb_xmetrics.error(obs_arr, fc_vals)
                tw_mae  = float(abs(err_da).weighted(w_obs).mean().values)
            else:
                tw_rmse = tw_bias = tw_mae = float('nan')

            # Contingency table counts (manual — VTB CT has _stack_arr coordinate bug)
            ov = ev_obs_b.values
            fv = ev_fc_b.values
            valid    = ~np.isnan(ov) & ~np.isnan(fv)
            obs_ev   = (ov == 1.0) & valid
            fc_ev    = (fv == 1.0) & valid
            n_h = int((obs_ev  &  fc_ev).sum())
            n_m = int((obs_ev  & ~fc_ev).sum())
            n_f = int((~obs_ev &  fc_ev & valid).sum())
            n_c = int((~obs_ev & ~fc_ev & valid).sum())
            n_tot = n_h + n_m + n_f + n_c

            if n_tot > 0:
                hr    = (n_h + n_m) * (n_h + n_f) / n_tot
                ets_d = n_h + n_m + n_f - hr
                ets   = (n_h - hr) / ets_d if ets_d != 0 else float('nan')
            else:
                ets = float('nan')
            pod  = n_h / (n_h + n_m) if (n_h + n_m) > 0 else float('nan')
            pofd = n_f / (n_f + n_c) if (n_f + n_c) > 0 else float('nan')
            pss  = (pod - pofd) if not (np.isnan(pod) or np.isnan(pofd)) else float('nan')
            far  = n_f / (n_h + n_f) if (n_h + n_f) > 0 else float('nan')

            suf = f"_p{pctl}" if multi else ""
            scores[f"twRMSE{suf}"] = tw_rmse
            scores[f"twBias{suf}"] = tw_bias
            scores[f"twMAE{suf}"]  = tw_mae
            scores[f"ETS{suf}"]    = ets
            scores[f"PSS{suf}"]    = pss
            scores[f"POD{suf}"]    = pod
            scores[f"FAR{suf}"]    = far

        # For multiple thresholds: also expose primary (last configured) without suffix
        # so downstream code can always access 'ETS', 'PSS', etc. regardless of mode.
        if multi:
            pp = pcts_ops[-1][1]
            for base in ('twRMSE', 'twBias', 'twMAE', 'ETS', 'PSS', 'POD', 'FAR'):
                scores[base] = scores.get(f"{base}_p{pp}", float('nan'))

        return scores

    # ===========================================================================
    # Retrieve obs + climatology from STVL (shared across both models)
    # ===========================================================================
    print(f"\n  [VTB direct] {variable}: retrieving STVL obs + climatology ...")
    # For tp24, STVL parameter is 'tp' with period=24h to select 24h accumulation obs
    stvl_period = pd.to_timedelta('24h') if variable == 'tp24' else None

    try:
        obs_kw = dict(table="observation", parameter=stvl_param,
                      date=vdates, sources=obs_sources)
        if stvl_period is not None:
            obs_kw['period'] = stvl_period
        obs_fs = vtb.media.stvl_retrieve(**obs_kw)
        print(f"    obs: {len(obs_fs)} fields")

        clim_kw = dict(table="climatology", parameter=stvl_param,
                       reference_datetimes=vdates,
                       climatology={"category": "percentiles", "climate_period": "1980-2009"})
        if stvl_period is not None:
            clim_kw['period'] = stvl_period
        clim_raw = vtb.media.stvl_retrieve(**clim_kw)
        clim_fs = _fix_clim_headers(clim_raw)
        print(f"    clim: {len(clim_fs)} fields (headers fixed)")
    except Exception as e:
        import traceback as _tb
        print(f"  [VTB direct] STVL retrieval failed: {e}")
        print(_tb.format_exc())
        return {}

    # ===========================================================================
    # Initialise results structure
    # ===========================================================================
    results = {s: {'fc1': {}, 'fc2': {}} for s in steps}

    # ===========================================================================
    # Per-model loop
    # ===========================================================================
    for label in ('fc1', 'fc2'):
        mcfg = model_cfgs[label]
        mk   = model_mk[label]
        print(f"\n  [VTB direct] Model: {mcfg['name']} (label={label})")

        mars_kw = dict(
            class_=mcfg['class_'],
            expver=mcfg['expver'],
            stream=mcfg['stream'],
            type=mcfg['type'],
        )

        # ------------------------------------------------------------------
        # Retrieve and preprocess forecast fields
        # ------------------------------------------------------------------
        try:
            if variable == '10ff':
                # Retrieve 10u and 10v, compute wind speed at obs locations
                fc_u = vtb.media.mars_retrieve(
                    parameter="10u", levtype="sfc",
                    date=dates, step=pandas_steps, **mars_kw,
                )
                fc_v = vtb.media.mars_retrieve(
                    parameter="10v", levtype="sfc",
                    date=dates, step=pandas_steps, **mars_kw,
                )
                u_at = fc_u.aligned(obs_fs)
                v_at = fc_v.aligned(obs_fs)
                # Fieldset arithmetic: element-wise sqrt(u² + v²)
                fc_processed = (u_at * u_at + v_at * v_at) ** 0.5
                print(f"    10ff: {len(fc_processed)} wind-speed fields at obs stations")

            elif variable == '2t':
                fc_raw = vtb.media.mars_retrieve(
                    parameter="2t", levtype="sfc",
                    date=dates, step=pandas_steps, **mars_kw,
                )
                fc_at = fc_raw.aligned(obs_fs)

                # Lapse-rate correction (VTB pattern — applied to full Fieldset
                # before FieldMetrics so every date/step gets the same per-station
                # correction).
                # Sign convention: model_height > obs_height → model T too cold
                #   → add correction (+0.0065 * orog_diff K)
                # Stations with |height difference| > 500 m are masked (NaN).
                orog_path = orog_paths.get(label, '')
                if orog_path and os.path.exists(orog_path):
                    orog_grib = vtb.Fieldset(orog_path)
                    orog_at   = orog_grib.aligned(obs_fs)
                    model_h   = orog_at.array[0] / G      # geopotential → metres
                    try:
                        obs_elev = obs_fs.elevations       # VTB point-data property
                    except AttributeError:
                        _df  = obs_fs[0].to_dataframe()
                        _col = 'elevation' if 'elevation' in _df.columns else 'altitude'
                        obs_elev = _df[_col].values
                    orog_diff = model_h - obs_elev         # [m]; + when model higher
                    corr      = orog_diff * 0.0065         # [K]; + warms the forecast
                    corr[np.abs(orog_diff) > 500] = np.nan
                    fc_processed = fc_at + corr
                    n_valid = int(np.isfinite(corr).sum())
                    print(f"    Lapse-rate: model_h={np.nanmean(model_h):.0f}m, "
                          f"obs_h={np.nanmean(obs_elev):.0f}m, "
                          f"mean corr={np.nanmean(corr):+.3f} K "
                          f"({n_valid}/{len(corr)} stations within ±500 m)")
                else:
                    fc_processed = fc_at
                    if orog_path:
                        print(f"    Lapse-rate skipped — orog file not found: {orog_path}")
                    else:
                        print(f"    Lapse-rate skipped — no orog_path in config")

            elif variable == 'tp24':
                # For de-accumulation, also retrieve the step-24h predecessor fields.
                # tp(step) - tp(step-24h) gives the 24h window accumulation.
                # step=24h is the special case: equals tp(24) directly (no subtraction).
                extra_tds = [pd.to_timedelta(f"{s - 24}h")
                             for s in steps if s > 24]
                all_steps_td = sorted(set(pandas_steps + extra_tds))
                fc_tp = vtb.media.mars_retrieve(
                    parameter="tp", levtype="sfc",
                    date=dates, step=all_steps_td, **mars_kw,
                )
                fc_processed = fc_tp  # de-accumulation happens in the per-step loop
                print(f"    tp: {len(fc_tp)} fields retrieved (steps={[str(s) for s in all_steps_td]})")

            else:
                print(f"    Unsupported variable '{variable}' — skipping")
                continue

        except Exception as e:
            import traceback as _tb
            print(f"    Forecast retrieval/preprocessing failed: {e}")
            print(_tb.format_exc())
            continue

        # ------------------------------------------------------------------
        # Per-step scoring
        # ------------------------------------------------------------------
        for step_h in steps:
            step_td = pd.to_timedelta(f"{step_h}h")
            try:
                if variable == 'tp24':
                    fc_step = fc_processed.header_filter(step=step_td)
                    if step_h > 24:
                        prev_td  = pd.to_timedelta(f"{step_h - 24}h")
                        fc_prev  = fc_processed.header_filter(step=prev_td)
                        fc_step  = fc_step - fc_prev   # de-accumulate
                    # FieldMetrics handles obs alignment internally for gridded tp
                    fm = vtb.FieldMetrics(
                        forecasts=fc_step,
                        observations=obs_fs,
                        observation_climatology=clim_fs,
                    )
                else:
                    fc_step = fc_processed.header_filter(step=step_td)
                    fm = vtb.FieldMetrics(
                        forecasts=fc_step,
                        observations=obs_fs,
                        observation_climatology=clim_fs,
                    )

                obs_arr = fm._arrays["observations"]
                fc_arr  = fm._arrays["forecasts"]
                fc_vals = fc_arr.isel(number=0) if "number" in fc_arr.dims else fc_arr

                # Log station count on the first step only
                if step_h == steps[0]:
                    _st_dim = [d for d in obs_arr.dims if 'station' in d or 'location' in d]
                    if _st_dim:
                        _n_st = obs_arr.sizes[_st_dim[0]]
                        print(f"    Stations after obs∩clim alignment: {_n_st}")

                # Unweighted continuous scores
                rmse = float(vtb_xmetrics.root_mean_squared_error(obs_arr, fc_vals).values)
                bias = float(vtb_xmetrics.mean_error(obs_arr, fc_vals).values)
                err_da = vtb_xmetrics.error(obs_arr, fc_vals)
                mae  = float(abs(err_da).mean().values)

                step_scores = {'rmse': rmse, 'bias': bias, 'mae': mae}

                # Threshold-weighted + CT scores
                tw_ct = _ct_tw_scores(fm, obs_arr, fc_vals, percentiles_and_ops,
                                      multi_threshold)
                step_scores.update(tw_ct)

                results[step_h][label] = step_scores

                thresh_str = "/".join(f"p{p}" for p in percentiles)
                ets_val = step_scores.get('ETS', float('nan'))
                print(f"    Step {step_h}h: RMSE={rmse:.4f} Bias={bias:+.4f} "
                      f"MAE={mae:.4f}  ETS({thresh_str})={ets_val:.4f}")

            except Exception as e:
                import traceback as _tb
                print(f"    Step {step_h}h failed: {e}")
                print(_tb.format_exc())

        # Free forecast Fieldsets now to avoid peak-memory when loading the
        # next model (fc1 O1280 + fc2 O2560 simultaneously would OOM).
        import gc as _gc
        del fc_processed
        try: del fc_raw   # noqa: E701
        except NameError: pass
        try: del fc_at    # noqa: E701
        except NameError: pass
        try: del fc_u, fc_v, u_at, v_at  # noqa: E701
        except NameError: pass
        try: del fc_tp    # noqa: E701
        except NameError: pass
        _gc.collect()

    return results


def _compute_vtb_direct_ens_scores(config, dates, steps):
    """Compute ensemble scores via VTB direct method (MARS + STVL).

    Computes per step and model:
      - CRPS, fCRPS         (unconditional, all obs)
      - spread              (mean ensemble std of members)
      - spread_skill_ratio  (spread / RMSE of ensemble mean)
      - ens_mean_rmse       (RMSE of ensemble mean)
      - Brier               (probability score for threshold exceedance)
      - twCRPS, twfCRPS     (threshold-weighted CRPS / fair CRPS)
      - twMAE               (ensemble-mean MAE on extreme obs)
      - tw_spread_skill     (spread / RMSE on extreme obs cases)
      - ETS, PSS, POD, FAR  (majority-vote binary from ensemble probability)

    Threshold-weighted scores use the binary obs-exceedance mask as weights
    (weight=1 for extreme obs, 0 otherwise), identical to the det convention.
    """
    mode = config.get('mode', 'deterministic')
    if mode != 'ensemble':
        return {}

    variable = config['variable']
    thresh_cfg = config.get('threshold', {})
    method = thresh_cfg.get('method', '')
    if method not in ('station_climatology', 'dataset_climatology'):
        return {}

    pcfg = thresh_cfg[method]
    percentile_cfg = pcfg.get('percentile', pcfg.get('value'))
    if percentile_cfg is None:
        return {}
    event_type = pcfg.get('event_type', thresh_cfg.get('event_type', 'above'))

    if isinstance(percentile_cfg, (int, float)):
        percentiles = [int(percentile_cfg)]
    else:
        percentiles = [int(p) for p in percentile_cfg]

    operator = '>' if event_type == 'above' else '<'
    percentiles_and_ops = [(operator, p) for p in percentiles]
    multi_threshold = len(percentiles) > 1

    try:
        import vtb
        import xarray as xr
        from vtb.metricss import xmetrics as vtb_xmetrics
        from vtb.metricss._score_array_interface import complete_quantiles
        from vtb.metricss.event import BinaryEvent
    except ImportError as e:
        print(f"  [VTB direct ens] VTB not available: {e}")
        return {}

    G = 9.80665
    obs_sources = config.get('read_data', {}).get('quaver_obs', {}).get('sources', ['synop'])
    pandas_steps = [pd.to_timedelta(f"{s}h") for s in steps]
    vdates = sorted(set(d + s for d in dates for s in pandas_steps))

    # Parse ensemble member list from config
    ens_cfg = config.get('ensemble', {})
    n_members_default = ens_cfg.get('n_members', 50)

    def _parse_numbers(q):
        num_str = str(q.get('number', f'1/to/{n_members_default}'))
        if '/to/' in num_str:
            parts = num_str.split('/to/')
            return list(range(int(parts[0]), int(parts[1]) + 1))
        return [int(x) for x in num_str.split('/') if x.strip().isdigit()]

    def _mars_cfg_ens(mk):
        cfg_m = config['read_data'].get(mk, {})
        q = cfg_m.get('quaver', {})
        raw_expver = str(q.get('expver', '0001'))
        expver = f'{int(raw_expver):04d}' if raw_expver.isdigit() else raw_expver
        return {
            'name':    cfg_m.get('name', mk),
            'class_':  q.get('class', 'od'),
            'expver':  expver,
            'stream':  q.get('stream', 'enfo'),
            'type':    q.get('type', 'pf'),
            'numbers': _parse_numbers(q),
        }

    model_cfgs = {
        'fc1': _mars_cfg_ens('forecast_model1'),
        'fc2': _mars_cfg_ens('forecast_model2'),
    }

    orog_paths = {}
    if variable == '2t':
        cfg_aux = config.get('auxiliary_fields', {})
        orog_paths['fc1'] = cfg_aux.get('model1', {}).get('orog_path', '')
        orog_paths['fc2'] = cfg_aux.get('model2', {}).get('orog_path', '')

    stvl_param  = 'tp' if variable == 'tp24' else variable
    stvl_period = pd.to_timedelta('24h') if variable == 'tp24' else None

    def _fix_clim_headers(station_clim):
        fixed = []
        for i in range(len(station_clim)):
            f = station_clim[i]
            v = f.header_get("number")
            if isinstance(v, list): v = v[0]
            if isinstance(v, list): v = v[0]
            f = f.header_set(type="pb", numberOfForecastsInEnsemble=100, number=int(v) + 1)
            fixed.append(f)
        return vtb.Fieldset(*fixed)

    # ------------------------------------------------------------------
    # STVL obs + climatology
    # ------------------------------------------------------------------
    print(f"\n  [VTB direct ens] {variable}: retrieving STVL obs + climatology ...")
    try:
        obs_kw = dict(table="observation", parameter=stvl_param,
                      date=vdates, sources=obs_sources)
        if stvl_period is not None:
            obs_kw['period'] = stvl_period
        obs_fs = vtb.media.stvl_retrieve(**obs_kw)
        print(f"    obs: {len(obs_fs)} fields")

        clim_kw = dict(table="climatology", parameter=stvl_param,
                       reference_datetimes=vdates,
                       climatology={"category": "percentiles", "climate_period": "1980-2009"})
        if stvl_period is not None:
            clim_kw['period'] = stvl_period
        clim_raw = vtb.media.stvl_retrieve(**clim_kw)
        clim_fs = _fix_clim_headers(clim_raw)
        print(f"    clim: {len(clim_fs)} fields (headers fixed)")
    except Exception as e:
        import traceback as _tb
        print(f"  [VTB direct ens] STVL retrieval failed: {e}")
        print(_tb.format_exc())
        return {}

    results = {s: {'fc1': {}, 'fc2': {}} for s in steps}

    # ------------------------------------------------------------------
    # Per-model loop
    # ------------------------------------------------------------------
    for label in ('fc1', 'fc2'):
        mcfg = model_cfgs[label]
        print(f"\n  [VTB direct ens] Model: {mcfg['name']} (label={label}, "
              f"n_members={len(mcfg['numbers'])})")

        mars_kw = dict(
            class_=mcfg['class_'],
            expver=mcfg['expver'],
            stream=mcfg['stream'],
            type=mcfg['type'],
            number=mcfg['numbers'],
        )

        try:
            if variable == '10ff':
                fc_u = vtb.media.mars_retrieve(
                    parameter="10u", levtype="sfc",
                    date=dates, step=pandas_steps, **mars_kw,
                )
                fc_v = vtb.media.mars_retrieve(
                    parameter="10v", levtype="sfc",
                    date=dates, step=pandas_steps, **mars_kw,
                )
                u_at = fc_u.aligned(obs_fs)
                v_at = fc_v.aligned(obs_fs)
                fc_processed = (u_at * u_at + v_at * v_at) ** 0.5
                print(f"    10ff ens: {len(fc_processed)} wind-speed fields at obs stations")

            elif variable == '2t':
                fc_raw = vtb.media.mars_retrieve(
                    parameter="2t", levtype="sfc",
                    date=dates, step=pandas_steps, **mars_kw,
                )
                fc_at = fc_raw.aligned(obs_fs)
                orog_path = orog_paths.get(label, '')
                if orog_path and os.path.exists(orog_path):
                    orog_grib = vtb.Fieldset(orog_path)
                    orog_at   = orog_grib.aligned(obs_fs)
                    model_h   = orog_at.array[0] / G
                    try:
                        obs_elev = obs_fs.elevations
                    except AttributeError:
                        _df  = obs_fs[0].to_dataframe()
                        _col = 'elevation' if 'elevation' in _df.columns else 'altitude'
                        obs_elev = _df[_col].values
                    orog_diff = model_h - obs_elev
                    corr      = orog_diff * 0.0065
                    corr[np.abs(orog_diff) > 500] = np.nan
                    fc_processed = fc_at + corr
                    n_valid = int(np.isfinite(corr).sum())
                    print(f"    Lapse-rate: mean corr={np.nanmean(corr):+.3f} K "
                          f"({n_valid}/{len(corr)} stations within ±500 m)")
                else:
                    fc_processed = fc_at

            elif variable == 'tp24':
                extra_tds = [pd.to_timedelta(f"{s - 24}h") for s in steps if s > 24]
                all_steps_td = sorted(set(pandas_steps + extra_tds))
                fc_tp = vtb.media.mars_retrieve(
                    parameter="tp", levtype="sfc",
                    date=dates, step=all_steps_td, **mars_kw,
                )
                fc_processed = fc_tp
                print(f"    tp ens: {len(fc_tp)} fields retrieved")

            else:
                print(f"    Unsupported variable '{variable}' — skipping")
                continue

        except Exception as e:
            import traceback as _tb
            print(f"    Forecast retrieval failed: {e}")
            print(_tb.format_exc())
            continue

        # ------------------------------------------------------------------
        # Per-step scoring
        # ------------------------------------------------------------------
        for step_h in steps:
            step_td = pd.to_timedelta(f"{step_h}h")
            try:
                if variable == 'tp24':
                    fc_step = fc_processed.header_filter(step=step_td)
                    if step_h > 24:
                        prev_td = pd.to_timedelta(f"{step_h - 24}h")
                        fc_prev = fc_processed.header_filter(step=prev_td)
                        fc_step = fc_step - fc_prev
                    fm = vtb.FieldMetrics(
                        forecasts=fc_step,
                        observations=obs_fs,
                        observation_climatology=clim_fs,
                    )
                else:
                    fc_step = fc_processed.header_filter(step=step_td)
                    fm = vtb.FieldMetrics(
                        forecasts=fc_step,
                        observations=obs_fs,
                        observation_climatology=clim_fs,
                    )

                obs_arr = fm._arrays["observations"]
                fc_arr  = fm._arrays["forecasts"]

                if "number" not in fc_arr.dims:
                    print(f"    Step {step_h}h: WARNING — no 'number' dim in fc_arr, "
                          f"dims={list(fc_arr.dims)}")
                    continue

                if step_h == steps[0]:
                    _st_dim = [d for d in obs_arr.dims if 'station' in d or 'location' in d]
                    if _st_dim:
                        print(f"    Stations after obs∩clim alignment: "
                              f"{obs_arr.sizes[_st_dim[0]]}")

                ens_mean = fc_arr.mean('number')

                # CRPS and fCRPS
                crps_alpha, crps_beta, crps_da, fcrps_da = vtb_xmetrics.crps(
                    obs_arr, fc_arr, dim='number', returns='all')
                crps_mean  = float(crps_da.mean().values)
                fcrps_mean = float(fcrps_da.mean().values)

                # Spread (mean std of members)
                spread_da   = vtb_xmetrics.spread(fc_arr, dim='number')
                spread_mean = float(spread_da.mean().values)

                # RMSE of ensemble mean
                rmse_ens = float(
                    vtb_xmetrics.root_mean_squared_error(obs_arr, ens_mean).values)

                step_scores = {
                    'CRPS':             crps_mean,
                    'fCRPS':            fcrps_mean,
                    'spread':           spread_mean,
                    'spread_skill_ratio': spread_mean / rmse_ens if rmse_ens > 1e-10 else float('nan'),
                    'ens_mean_rmse':    rmse_ens,
                }

                # Threshold-based scores
                cl_obs = complete_quantiles(
                    fm._arrays["observation_climatology"], qtype="percentile")

                for (op, pctl) in percentiles_and_ops:
                    bev = BinaryEvent(operator=op, value=pctl, type="percentile",
                                      is_anomaly=False)
                    ev_obs_da = vtb_xmetrics.event_occurrences(
                        bev, forecasts=obs_arr, forecast_climatology=cl_obs)
                    ev_fc_da  = vtb_xmetrics.event_occurrences(
                        bev, forecasts=fc_arr, forecast_climatology=cl_obs)

                    w_obs     = ev_obs_da.fillna(0.0)
                    n_extreme = int(w_obs.sum().values)

                    # Brier score: BS = mean((P_f - o)^2)
                    # P_f = fraction of members forecasting the event
                    if "number" in ev_fc_da.dims:
                        prob_fc = ev_fc_da.mean('number')
                    else:
                        prob_fc = ev_fc_da
                    ev_obs_bin = ev_obs_da.fillna(0.0)
                    brier = float(((prob_fc - ev_obs_bin) ** 2).mean().values)

                    # Proper twCRPS/twfCRPS via chaining function (Taggart 2022):
                    #   g(x) = max(x, T_i) for upper tail (min for lower)
                    #   twCRPS = CRPS(g(fc_arr), g(obs_arr))
                    # Per-station T_i extracted from cl_obs (quantile_ dim, values 0.01-0.99).
                    # Falls back to conditional weighted mean if extraction fails.
                    try:
                        t_per_station = cl_obs.sel(quantile_=pctl / 100.0, method='nearest')
                        if op == '>':
                            g_obs_arr = xr.where(obs_arr >= t_per_station, obs_arr, t_per_station)
                            g_fc_arr  = xr.where(fc_arr  >= t_per_station, fc_arr,  t_per_station)
                        else:
                            g_obs_arr = xr.where(obs_arr <= t_per_station, obs_arr, t_per_station)
                            g_fc_arr  = xr.where(fc_arr  <= t_per_station, fc_arr,  t_per_station)
                        _, _, tw_crps_da_chn, tw_fcrps_da_chn = vtb_xmetrics.crps(
                            g_obs_arr, g_fc_arr, dim='number', returns='all')
                        tw_crps  = float(tw_crps_da_chn.mean().values)
                        tw_fcrps = float(tw_fcrps_da_chn.mean().values)
                    except Exception as _echn:
                        print(f"        twCRPS chaining failed ({_echn}); using conditional weighted mean")
                        if n_extreme > 0:
                            tw_crps  = float(crps_da.weighted(w_obs).mean().values)
                            tw_fcrps = float(fcrps_da.weighted(w_obs).mean().values)
                        else:
                            tw_crps = tw_fcrps = float('nan')

                    # tw_mae, tw_rmse, tw_spread_skill: conditional on extreme obs (diagnostic only)
                    if n_extreme > 0:
                        err_da    = vtb_xmetrics.error(obs_arr, ens_mean)
                        tw_mae    = float(abs(err_da).weighted(w_obs).mean().values)
                        tw_rmse   = float(vtb_xmetrics.root_mean_squared_error(
                            obs_arr, ens_mean, weights=w_obs).values)
                        tw_spread = float(spread_da.weighted(w_obs).mean().values)
                        tw_spread_skill = (tw_spread / tw_rmse
                                           if tw_rmse > 1e-10 else float('nan'))
                    else:
                        tw_mae = tw_spread_skill = float('nan')
                        tw_rmse = float('nan')

                    suf = f"_p{pctl}" if multi_threshold else ""
                    step_scores[f"Brier{suf}"]           = brier
                    step_scores[f"twCRPS{suf}"]          = tw_crps
                    step_scores[f"twfCRPS{suf}"]         = tw_fcrps
                    step_scores[f"twMAE{suf}"]           = tw_mae
                    step_scores[f"tw_spread_skill{suf}"] = tw_spread_skill

                if multi_threshold:
                    pp = percentiles_and_ops[-1][1]
                    for base in ('Brier', 'twCRPS', 'twfCRPS', 'twMAE', 'tw_spread_skill'):
                        step_scores[base] = step_scores.get(f"{base}_p{pp}", float('nan'))

                results[step_h][label] = step_scores

                thresh_str = "/".join(f"p{p}" for p in percentiles)
                print(f"    Step {step_h}h: CRPS={crps_mean:.4f} fCRPS={fcrps_mean:.4f} "
                      f"spread={spread_mean:.4f}  "
                      f"Brier({thresh_str})={step_scores.get('Brier', float('nan')):.4f}  "
                      f"twCRPS={step_scores.get('twCRPS', float('nan')):.4f}")

            except Exception as e:
                import traceback as _tb
                print(f"    Step {step_h}h failed: {e}")
                print(_tb.format_exc())

        import gc as _gc
        del fc_processed
        try: del fc_raw   # noqa: E701
        except NameError: pass
        try: del fc_at    # noqa: E701
        except NameError: pass
        try: del fc_u, fc_v, u_at, v_at  # noqa: E701
        except NameError: pass
        try: del fc_tp    # noqa: E701
        except NameError: pass
        _gc.collect()

    return results


# ---------------------------------------------------------------------------
# LEGACY SHIM — retained for backward compatibility with external callers
# ---------------------------------------------------------------------------
def _compute_tw_scores_via_vtb(config, dates, steps):
    """Compute threshold-weighted scores (twMAE, twCRPS) via VTB data extraction.

    These scores are NOT available in Quaver's native compute() API.
    We must:
      1. Extract obs + forecast point data at station locations via VTB
      2. Retrieve STVL station climatology for per-station thresholds
      3. Select cases where obs exceeds the station's threshold
      4. Compute twMAE (deterministic) or twCRPS (ensemble) on those cases

    Returns:
        dict: {step: {'fc1': {score: value}, 'fc2': {score: value}}}
        Empty dict if no tw scores are requested or extraction fails.
    """
    mode = config.get('mode', 'deterministic')
    requested_scores = config.get('scores', {}).get(
        'deterministic' if mode == 'deterministic' else 'ensemble', [])

    # Check if any tw scores are requested
    tw_det_scores = {'twMAE', 'twRMSE'}
    tw_ens_scores = {'twCRPS', 'tw_quantile_score', 'tw_spread_error_ratio'}
    needed_tw = set()
    for s in requested_scores:
        if s in tw_det_scores or s in tw_ens_scores:
            needed_tw.add(s)
    if not needed_tw:
        return {}

    print(f"\n  Computing threshold-weighted scores via VTB: {needed_tw}")

    try:
        import quaver_backend
        from filter import load_extracted_data
    except ImportError as e:
        print(f"    WARNING: Cannot compute tw scores via VTB: {e}")
        return {}

    # Get threshold config
    thresh_cfg = config['threshold']
    method = thresh_cfg['method']
    if method == 'fixed':
        fixed_threshold = thresh_cfg['fixed']['value']
        event_type = thresh_cfg['fixed'].get('event_type', thresh_cfg.get('event_type', 'above'))
    elif method in ('station_climatology', 'dataset_climatology'):
        fixed_threshold = None
        pcfg = thresh_cfg[method]
        event_type = pcfg.get('event_type', thresh_cfg.get('event_type', 'above'))
    else:
        return {}

    # --- Extract point data via quaver_backend ---
    preprocess_cfg = config.get('preprocess', {})
    preprocess_settings = {
        'wind_speed_from_components': preprocess_cfg.get('wind_speed_from_components', False),
        'lapse_rate_correction': preprocess_cfg.get('lapse_rate_correction', False),
        'lapse_rate': preprocess_cfg.get('lapse_rate', -0.0065),
        'precipitation_accumulation_hours': preprocess_cfg.get('precipitation_accumulation_hours', None),
    }
    paths = {
        'fc1_name': config['read_data']['forecast_model1']['name'],
        'fc2_name': config['read_data']['forecast_model2']['name'],
    }

    try:
        extraction_info = quaver_backend.extract_points_quaver(config, paths, preprocess_settings)
    except Exception as e:
        print(f"    WARNING: VTB point extraction failed: {e}")
        return {}

    variable = config['variable']
    fc1_name = config['read_data']['forecast_model1']['name']
    fc2_name = config['read_data']['forecast_model2']['name']

    data = load_extracted_data(
        variable=variable,
        point_data_path=extraction_info['output_path'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        steps=steps,
        fc1_name=fc1_name,
        fc2_name=fc2_name,
        save_format='pandas',
        config=config,
    )

    # --- Get per-station thresholds ---
    if fixed_threshold is not None:
        threshold_series = pd.Series(fixed_threshold, index=data.index)
    else:
        threshold_series, event_type = quaver_backend._threshold_station_climatology(
            config, data)
        print(f"    Per-station thresholds: {threshold_series.notna().sum()} stations matched")

    # --- Compute tw scores per step ---
    results = {}
    for step in steps:
        step_data = data[data['step'] == step]
        if len(step_data) == 0:
            continue

        step_thresholds = threshold_series.loc[step_data.index]
        obs_vals = step_data['obs_value']

        # Select above-threshold cases (or below for cold extremes)
        valid = obs_vals.notna() & step_thresholds.notna()
        if event_type == 'above':
            above_mask = valid & (obs_vals > step_thresholds)
        else:
            above_mask = valid & (obs_vals < step_thresholds)

        results[step] = {'fc1': {}, 'fc2': {}}

        for model_label, fc_prefix in [('fc1', 'fc1'), ('fc2', 'fc2')]:
            fc_col = f'{fc_prefix}_value'
            if fc_col not in step_data.columns:
                continue

            fc_vals = step_data[fc_col]
            case_mask = above_mask & fc_vals.notna()
            obs_above = obs_vals[case_mask].values
            fc_above = fc_vals[case_mask].values

            if len(obs_above) == 0:
                continue

            if 'twMAE' in needed_tw:
                results[step][model_label]['twMAE'] = float(np.mean(np.abs(fc_above - obs_above)))
            if 'twRMSE' in needed_tw:
                results[step][model_label]['twRMSE'] = float(
                    np.sqrt(np.mean((fc_above - obs_above) ** 2)))

            # Ensemble tw scores
            if mode == 'ensemble':
                # Look for member columns (fc1_member_1, fc1_member_2, ...)
                member_cols = [c for c in step_data.columns
                               if c.startswith(f'{fc_prefix}_member_')]
                if member_cols and ('twCRPS' in needed_tw or 'tw_quantile_score' in needed_tw):
                    ens_vals = step_data.loc[case_mask, member_cols].values
                    if 'twCRPS' in needed_tw:
                        try:
                            from scores import crps_ensemble
                            crps_val = crps_ensemble(obs_above, ens_vals)
                            results[step][model_label]['twCRPS'] = float(np.mean(crps_val))
                        except ImportError:
                            # Manual CRPS computation
                            crps_val = _manual_crps(obs_above, ens_vals)
                            results[step][model_label]['twCRPS'] = float(np.mean(crps_val))

                if member_cols and 'tw_spread_error_ratio' in needed_tw:
                    ens_vals = step_data.loc[case_mask, member_cols].values
                    if len(ens_vals) > 0:
                        ens_mean = ens_vals.mean(axis=1)
                        tw_spread = float(np.mean(ens_vals.std(axis=1, ddof=1)))
                        tw_rmse = float(np.sqrt(np.mean((ens_mean - obs_above) ** 2)))
                        if tw_rmse > 1e-10:
                            results[step][model_label]['tw_spread_error_ratio'] = tw_spread / tw_rmse
                        else:
                            results[step][model_label]['tw_spread_error_ratio'] = np.nan

    n_ok = sum(1 for s in results if results[s].get('fc1') or results[s].get('fc2'))
    print(f"    tw scores computed for {n_ok}/{len(steps)} steps")
    return results


def _manual_crps(obs, ensemble):
    """Compute CRPS for each sample given obs (n,) and ensemble (n, m)."""
    n, m = ensemble.shape
    # CRPS = E|X-y| - 0.5 * E|X-X'|
    abs_diff_obs = np.abs(ensemble - obs[:, None]).mean(axis=1)
    # E|X-X'| approximation
    sorted_ens = np.sort(ensemble, axis=1)
    abs_diff_ens = np.zeros(n)
    for i in range(m):
        for j in range(i + 1, m):
            abs_diff_ens += np.abs(sorted_ens[:, i] - sorted_ens[:, j])
    abs_diff_ens = abs_diff_ens * 2.0 / (m * (m - 1))
    return abs_diff_obs - 0.5 * abs_diff_ens


def run_quaver_compute_workflow(config, model_names):
    """Main entry point — run full workflow using Quaver compute().
    
    Called from run.py when backend='quaver_compute'.
    
    Returns:
        list of result dicts, one per season/orog combination
    """
    print("\n" + "=" * 80)
    print("QUAVER COMPUTE BACKEND")
    print("=" * 80)
    print("\nUsing native Quaver compute() API for score computation")
    print("  - Orography correction: automatic for 2t")
    print("  - Scores stored to database + exported to CSV")
    print("  - Significance via block bootstrap")
    
    results = compute_and_compare(config, model_names)
    
    # Export results to CSV
    output_dir = config.get('save', {}).get('output_directory', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    for r in results:
        season = r.get('season', '')
        orog = r.get('orog_type', '')
        suffix = ''
        if season:
            suffix += f'_{season}'
        if orog:
            suffix += f'_{orog}'
        
        variable = config['variable']
        thresh_cfg = config['threshold']
        method = thresh_cfg['method']
        if method == 'fixed':
            thresh_label = str(thresh_cfg['fixed']['value'])
        elif method == 'dataset_climatology':
            thresh_label = f"{thresh_cfg['dataset_climatology']['percentile']}th"
        elif method == 'station_climatology':
            thresh_label = f"p{thresh_cfg['station_climatology']['percentile']}"
        else:
            thresh_label = method
        
        # Save by-leadtime scores
        by_lt_csv = os.path.join(output_dir, f"scores_by_leadtime_{variable}_{thresh_label}{suffix}.csv")
        r['results']['by_leadtime'].to_csv(by_lt_csv, index=False)
        print(f"  ✓ Saved: {by_lt_csv}")
        
        # Save overall scores
        overall_csv = os.path.join(output_dir, f"overall_scores_{variable}_{thresh_label}{suffix}.csv")
        pd.DataFrame([r['overall_scores']]).to_csv(overall_csv, index=False)
        print(f"  ✓ Saved: {overall_csv}")
    
    return results
