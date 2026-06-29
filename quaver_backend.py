"""
QUAVER/VTB BACKEND FOR SCORECARDS4EXTREMES
==========================================
When backend='quaver' is selected in config, this module handles:
  - Data retrieval (MARS for forecasts, STVL/VINO for observations)
  - Point extraction (grid → station alignment via VTB)
  - Threshold computation (station/area climatology via STVL percentiles)
  - Score computation (VTB FieldMetrics where available, custom otherwise)

Requires:
  module load python3   (or: module load quaver/3.6.4)
  → gives access to vtb and quaver packages

When backend='local' (default), the existing pipeline is used unchanged.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Lazy imports — only loaded when backend='quaver' is actually used
_vtb = None
_metview = None


def _load_vtb():
    """Lazy-load vtb and metview to avoid import errors when not available."""
    global _vtb, _metview
    if _vtb is None:
        import vtb as _vtb_mod
        _vtb = _vtb_mod
    if _metview is None:
        import metview as _mv_mod
        _metview = _mv_mod
    return _vtb, _metview


# =============================================================================
# STEP 1: DATA RETRIEVAL
# =============================================================================

def retrieve_forecasts(config, model_key='forecast_model1'):
    """
    Retrieve forecast data from MARS via VTB.

    Config keys used (under read_data.<model_key>.quaver):
      class_, expver, stream, type, grid (optional)

    Returns: vtb.Fieldset of forecast fields
    """
    vtb, _ = _load_vtb()

    cfg = config['read_data'][model_key]
    q = cfg['quaver']

    dates = pd.date_range(config['start_date'], config['end_date'], freq='24h')
    steps = _build_steps(config)

    mars_kw = dict(
        parameter=_variable_to_param(config['variable']),
        levtype='sfc',
        date=dates,
        step=steps,
        stream=q['stream'],
        type=q['type'],
        class_=q['class'],
        expver=q['expver'],
    )
    if 'grid' in q and q['grid']:
        mars_kw['grid'] = q['grid']
    if q.get('number'):
        mars_kw['number'] = q['number']

    print(f"    MARS retrieve: class={q['class']}, expver={q['expver']}, "
          f"stream={q['stream']}, type={q['type']}")
    print(f"    dates={dates[0].strftime('%Y%m%d')}..{dates[-1].strftime('%Y%m%d')}, "
          f"steps={steps}")

    forecasts = vtb.media.mars_retrieve(**mars_kw)
    print(f"    → received {len(forecasts)} fields")
    return forecasts


def retrieve_observations(config, forecast_dates=None, steps=None):
    """
    Retrieve surface observations from STVL/VINO via VTB.

    Returns: vtb.Fieldset of observation fields
    """
    vtb, _ = _load_vtb()

    if forecast_dates is None:
        forecast_dates = pd.date_range(config['start_date'], config['end_date'], freq='24h')
    if steps is None:
        steps = _build_steps(config)

    # Compute verification (valid) dates: fc_date + step
    vdates = sorted(set(d + s for d in forecast_dates for s in steps))

    param = _variable_to_param(config['variable'])
    q_obs = config['read_data'].get('quaver_obs', {})
    sources = q_obs.get('sources', ['synop'])

    kw = dict(
        table='observation',
        parameter=param,
        date=vdates,
        sources=sources,
    )

    # For precipitation, specify accumulation period
    if config['variable'] == 'tp24':
        kw['period'] = pd.to_timedelta('24h')

    print(f"    STVL retrieve: {param}, {len(vdates)} valid dates, sources={sources}")
    observations = vtb.media.stvl_retrieve(**kw)
    print(f"    → received {len(observations)} observation fields")
    return observations


def retrieve_climatology(config, forecast_dates=None, steps=None):
    """
    Retrieve station climatology (mean + percentiles) from STVL.

    Returns: vtb.Fieldset of climatology fields
    """
    vtb, _ = _load_vtb()

    if forecast_dates is None:
        forecast_dates = pd.date_range(config['start_date'], config['end_date'], freq='24h')
    if steps is None:
        steps = _build_steps(config)

    vdates = sorted(set(d + s for d in forecast_dates for s in steps))
    param = _variable_to_param(config['variable'])

    # Retrieve both mean (em) and percentiles (pb)
    clim_mean = vtb.media.stvl_retrieve(
        table='climatology',
        parameter=param,
        date=vdates,
        climatology={'climate_period': '1980-2009', 'category': 'mean'},
    )

    clim_pct = vtb.media.stvl_retrieve(
        table='climatology',
        parameter=param,
        date=vdates,
        climatology={'climate_period': '1980-2009', 'category': 'percentiles'},
    )

    print(f"    Climatology: {len(clim_mean)} mean + {len(clim_pct)} percentile fields")
    return clim_mean, clim_pct


def _format_threshold_suffix(config):
    """Format threshold value for filename to match filter.py naming convention."""
    if 'threshold' not in config:
        return ""
    threshold_cfg = config['threshold']
    method = threshold_cfg.get('method', '')
    variable = config.get('variable', '')
    if method == 'fixed':
        value = threshold_cfg.get('fixed', {}).get('value', 0)
        if variable == 'tp24':
            return f"_{value:.0f}mm"
        elif variable == '2t':
            return f"_{value:.1f}C"
        elif variable == '10ff':
            return f"_{value:.1f}ms"
        else:
            return f"_{value:.1f}"
    elif method in ('dataset_climatology', 'station_climatology'):
        pct = threshold_cfg.get(method, {}).get('percentile', 99)
        if pct == 1: return "_1st"
        elif pct == 2: return "_2nd"
        elif pct == 3: return "_3rd"
        else: return f"_{pct}th"
    return ""


# =============================================================================
# STEP 3: POINT EXTRACTION (alignment)
# =============================================================================

def extract_points_quaver(config, paths_or_fieldsets, preprocess_settings):
    """
    Extract forecast at observation stations using VTB alignment.

    When backend='quaver', this replaces the metview-based extract_points.py.
    It retrieves forecasts + obs via VTB, aligns forecasts to obs locations,
    and returns a DataFrame in the same format as the local pipeline.

    Output: one parquet file per forecast day (e.g. _day1.parquet, _day3.parquet),
            each containing ALL dates for that day. Matches local pipeline format.

    Returns: dict with 'output_path', 'save_format', 'fc1_name', 'fc2_name'
    """
    vtb, _ = _load_vtb()

    variable = config['variable']
    output_path = Path(config['extract_points'].get(
        'output_path', f'/perm/{os.environ.get("USER","user")}/scorecards4extremes/extracted_points/{variable}_quaver'))
    output_path.mkdir(parents=True, exist_ok=True)

    fc1_name = config['read_data']['forecast_model1']['name']
    fc2_name = config['read_data']['forecast_model2']['name']

    dates = pd.date_range(config['start_date'], config['end_date'], freq='24h')
    steps = _build_steps(config)
    is_ensemble = config.get('mode', 'deterministic') == 'ensemble'
    n_members = config.get('ensemble', {}).get('n_members', 50) if is_ensemble else 0

    # Build filename base (include threshold suffix to match filter.py expectations)
    ens_tag = "_ens" if is_ensemble else ""
    threshold_suffix = _format_threshold_suffix(config)
    filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}{ens_tag}{threshold_suffix}"

    # Map each step to its forecast day number
    step_to_day = {}
    for s in steps:
        step_h = int(s.total_seconds() / 3600)
        day = max(1, (step_h + 23) // 24)  # step 24→day1, 72→day3, 120→day5
        step_to_day[step_h] = day
    forecast_days = sorted(set(step_to_day.values()))

    # Check if all output files already exist
    if config.get('skip_extraction_if_exists', False):
        expected_files = [output_path / f"{filename_base}_day{d}.parquet" for d in forecast_days]
        if all(f.exists() for f in expected_files):
            print(f"\n  ✓ Skipping extraction — {len(expected_files)} files already exist")
            return {
                'output_path': output_path,
                'save_format': 'pandas',
                'fc1_name': fc1_name,
                'fc2_name': fc2_name,
                'ensemble': is_ensemble,
            }

    # Area filtering via VTB domain
    area = config['extract_points'].get('area', None)
    domain = _get_vtb_domain(area) if area else None

    # Clean up any old parquet files from previous runs to avoid stale appends
    for d in forecast_days:
        old_file = output_path / f"{filename_base}_day{d}.parquet"
        if old_file.exists():
            old_file.unlink()

    print(f"\n  Retrieving & aligning data via VTB...")
    print(f"  Steps: {[int(s.total_seconds()/3600) for s in steps]} → forecast days: {forecast_days}")
    print(f"  Dates: {dates[0].strftime('%Y%m%d')} to {dates[-1].strftime('%Y%m%d')} ({len(dates)} days)")

    # Retrieve static auxiliary fields (sdfor + lsm + orog) once
    aux_fields = _retrieve_auxiliary_fields(config)

    # Collect rows grouped by forecast day
    rows_by_day = {d: [] for d in forecast_days}
    # For ensemble, flush rows to disk periodically to limit memory
    flush_interval = 5 if is_ensemble else 999999

    # Process each date
    for date_idx, date in enumerate(dates):
        date_str = date.strftime('%Y%m%d')

        if (date_idx + 1) % 10 == 0 or date_idx == 0:
            print(f"    [{date_str}] date {date_idx+1}/{len(dates)}...")

        try:
            # For precipitation, retrieve extra steps for 24h accumulation differencing
            # tp at step S is accumulated from T+0, so tp24 = tp[S] - tp[S-24]
            accum_hours = preprocess_settings.get('precipitation_accumulation_hours', 24)
            if variable == 'tp24':
                extra_steps = set()
                for s in steps:
                    s_h = int(s.total_seconds() / 3600)
                    prev_h = s_h - accum_hours
                    if prev_h > 0:
                        extra_steps.add(pd.to_timedelta(f"{prev_h}h"))
                all_fc_steps = sorted(set(steps) | extra_steps)
            else:
                all_fc_steps = steps

            # Retrieve forecasts for this date
            fc1_fs = _retrieve_one_day(config, 'forecast_model1', date, all_fc_steps)
            fc2_fs = _retrieve_one_day(config, 'forecast_model2', date, all_fc_steps)

            # Apply regridding if configured
            regrid_cfg = config.get('regrid', {})
            if regrid_cfg.get('enabled', False):
                fc1_fs = _regrid_fieldset(fc1_fs, regrid_cfg, f"fc1_{date_str}")
                fc2_fs = _regrid_fieldset(fc2_fs, regrid_cfg, f"fc2_{date_str}")

            # Retrieve observations for this day's valid times
            vdates = sorted(set(date + s for s in steps))
            param = _variable_to_param(variable)
            q_obs = config['read_data'].get('quaver_obs', {})
            sources = q_obs.get('sources', ['synop'])

            obs_kw = dict(table='observation', parameter=param, date=vdates, sources=sources)
            if variable == 'tp24':
                obs_kw['period'] = pd.to_timedelta('24h')
            obs_fs = vtb.media.stvl_retrieve(**obs_kw)

            if len(obs_fs) == 0:
                continue

            # Align forecasts to observation station locations
            fc1_aligned = fc1_fs.aligned(obs_fs)
            fc2_aligned = fc2_fs.aligned(obs_fs)

            # Align auxiliary fields to observation station locations
            aux_at_stations = _align_auxiliary_to_obs(aux_fields, obs_fs)

            # Convert to DataFrame rows
            fc1_factor = config['read_data']['forecast_model1'].get('unit_conversion_factor', 1.0)
            fc2_factor = config['read_data']['forecast_model2'].get('unit_conversion_factor', 1.0)
            rows = _fieldsets_to_dataframe(
                fc1_aligned, fc2_aligned, obs_fs, date, steps,
                fc1_name, fc2_name, variable, is_ensemble, n_members,
                preprocess_settings, domain, aux_at_stations,
                fc1_unit_factor=fc1_factor, fc2_unit_factor=fc2_factor,
                accum_hours=accum_hours if variable == 'tp24' else 0,
            )

            # Distribute rows into per-day buckets
            for row in rows:
                day = step_to_day.get(row.get('step', 24), 1)
                rows_by_day[day].append(row)

            # Free VTB fieldsets to limit memory growth (critical for ensemble)
            del fc1_fs, fc2_fs, obs_fs, fc1_aligned, fc2_aligned, rows
            import gc; gc.collect()

            # Periodically flush rows to disk for ensemble to limit memory
            if is_ensemble and (date_idx + 1) % flush_interval == 0:
                for day in forecast_days:
                    if rows_by_day[day]:
                        out_file = output_path / f"{filename_base}_day{day}.parquet"
                        chunk_df = pd.DataFrame(rows_by_day[day])
                        if out_file.exists():
                            existing = pd.read_parquet(out_file)
                            chunk_df = pd.concat([existing, chunk_df], ignore_index=True)
                        chunk_df.to_parquet(out_file, index=False)
                rows_by_day = {d: [] for d in forecast_days}
                gc.collect()

        except Exception as e:
            print(f"    [{date_str}] ERROR: {e}")
            continue

    # Save one parquet file per forecast day (append to any flushed data)
    for day in forecast_days:
        out_file = output_path / f"{filename_base}_day{day}.parquet"
        if rows_by_day[day]:
            df = pd.DataFrame(rows_by_day[day])
            if out_file.exists():
                existing = pd.read_parquet(out_file)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(out_file, index=False)
            print(f"  Day {day}: {len(df):,} rows → {out_file.name}")
        elif out_file.exists():
            df = pd.read_parquet(out_file)
            print(f"  Day {day}: {len(df):,} rows → {out_file.name} (from flush)")
        else:
            print(f"  Day {day}: no data")

    total_rows = sum(len(v) for v in rows_by_day.values())
    print(f"\n  ✓ VTB extraction complete: {total_rows:,} total rows → {output_path}")

    return {
        'output_path': output_path,
        'save_format': 'pandas',
        'fc1_name': fc1_name,
        'fc2_name': fc2_name,
        'ensemble': is_ensemble,
    }


# =============================================================================
# STEP 5: THRESHOLD COMPUTATION (Quaver-aware)
# =============================================================================

def compute_threshold_quaver(config, data):
    """
    Compute extreme-event threshold using Quaver/VTB methods.

    Supported methods (config threshold.method):
      - 'fixed'                  : user-defined value (no VTB needed)
      - 'station_climatology'    : per-station percentile from STVL climatology
      - 'area_mean_climatology'  : area-mean of station climatological percentile
      - 'model_percentile'       : percentile of one model's forecast distribution
      - 'dataset_climatology'    : percentile from the extracted obs data (existing)

    Returns: (threshold, event_type) or (per_station_thresholds, event_type)
    """
    cfg = config['threshold']
    method = cfg['method']

    if method == 'fixed':
        event_type = cfg['fixed']['event_type']
        threshold = cfg['fixed']['value']
        print(f"  Fixed threshold: {threshold}")
        return threshold, event_type

    event_type = cfg.get('event_type', 'above')

    if method == 'station_climatology':
        return _threshold_station_climatology(config, data)

    elif method == 'area_mean_climatology':
        return _threshold_area_mean_climatology(config, data)

    elif method == 'model_percentile':
        return _threshold_model_percentile(config, data)

    elif method == 'dataset_climatology':
        # Original behaviour — compute percentile from extracted obs
        percentile = cfg['dataset_climatology']['percentile']
        threshold = np.nanpercentile(data['obs_value'], percentile)
        print(f"  Dataset climatology: {percentile}th pct = {threshold:.3f}")
        return threshold, event_type

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def _threshold_station_climatology(config, data):
    """
    Per-station threshold from STVL observation climatology.
    Each station gets its own percentile-based threshold.

    Uses nearest-neighbour spatial matching (tolerance 0.1°) between
    Quaver-extracted stations and STVL climatology stations, following
    the same approach as vtb.tools.aligned_geodfs(max_cluster_size=0.1).

    Config keys:
      threshold.station_climatology.percentile: int (e.g. 99)

    Returns: (Series with per-row threshold, event_type)
    """
    from scipy.spatial import cKDTree

    vtb, _ = _load_vtb()
    cfg = config['threshold']['station_climatology']
    percentile = cfg['percentile']
    event_type = config['threshold'].get('event_type', 'above')

    print(f"  Station climatology threshold: p{percentile} per station via STVL")

    # Retrieve climatology percentiles from STVL
    # Use a single representative date — climatology is the same for all dates
    if 'valid_time' in data.columns:
        vdates = pd.to_datetime(data['valid_time'].unique(), format='%Y%m%d')
    else:
        vdates = pd.to_datetime(data['date'].unique(), format='%Y%m%d')

    param = _variable_to_param(config['variable'])

    # Build STVL retrieval kwargs matching the reference VTB approach
    stvl_kwargs = dict(
        table='climatology',
        parameter=param,
        reference_datetimes=sorted(vdates)[0],
        climatology={'climate_period': '1980-2009', 'category': 'percentiles'},
    )
    preprocess = config.get('preprocess', {})
    accum_hours = preprocess.get('precipitation_accumulation_hours', None)
    if accum_hours:
        stvl_kwargs['period'] = pd.to_timedelta(f'{accum_hours}h')
        print(f"    STVL period: {accum_hours}h (precipitation accumulation)")

    clim_pct = vtb.media.stvl_retrieve(**stvl_kwargs)

    # Handle empty STVL result
    if clim_pct.is_void or len(clim_pct) == 0:
        print(f"  WARNING: STVL climatology returned no data for parameter={param}")
        print(f"  Falling back to dataset_climatology (pooled p{percentile})")
        threshold_val = np.nanpercentile(data['obs_value'].dropna(), percentile)
        print(f"  → Using uniform threshold = {threshold_val:.3f} for all stations")
        threshold_series = pd.Series(threshold_val, index=data.index)
        return threshold_series, event_type

    try:
        clim_df = clim_pct.to_dataframe()
    except (AttributeError, Exception) as e:
        print(f"  WARNING: STVL climatology to_dataframe failed: {e}")
        print(f"  Falling back to dataset_climatology (pooled p{percentile})")
        threshold_val = np.nanpercentile(data['obs_value'].dropna(), percentile)
        print(f"  → Using uniform threshold = {threshold_val:.3f} for all stations")
        threshold_series = pd.Series(threshold_val, index=data.index)
        return threshold_series, event_type

    # STVL returns columns value_0..value_98 = 1st..99th percentile
    pct_col = f'value_{percentile - 1}'
    if pct_col not in clim_df.columns:
        print(f"  WARNING: column {pct_col} not in STVL climatology (available: value_0..value_{len([c for c in clim_df.columns if c.startswith('value_')])-1})")
        print(f"  Falling back to dataset_climatology (pooled p{percentile})")
        threshold_val = np.nanpercentile(data['obs_value'].dropna(), percentile)
        print(f"  → Using uniform threshold = {threshold_val:.3f} for all stations")
        threshold_series = pd.Series(threshold_val, index=data.index)
        return threshold_series, event_type

    print(f"    STVL climatology: {len(clim_df)} entries, {clim_df['stnid'].nunique()} stations")
    print(f"    Using column {pct_col} for p{percentile}")

    # --- Nearest-neighbour spatial matching ---
    # Quaver station_ids are sequential indices, not STVL station codes,
    # so we match by coordinates with 0.1° tolerance (same as
    # vtb.tools.aligned_geodfs max_cluster_size=0.1).
    MAX_DIST = 0.1  # degrees

    # If multiple STVL entries exist per station, average the threshold
    clim_grouped = clim_df.groupby(['latitude', 'longitude'])[pct_col].mean().reset_index()
    clim_coords = clim_grouped[['latitude', 'longitude']].values
    clim_thresholds = clim_grouped[pct_col].values

    # Get unique data stations and build KDTree lookup
    data_unique = data[['lat', 'lon']].drop_duplicates().dropna()
    data_coords = data_unique[['lat', 'lon']].values

    tree = cKDTree(clim_coords)
    distances, indices = tree.query(data_coords)

    # Build mapping: (lat, lon) -> threshold for matched stations
    station_threshold_map = {}
    for i, (_, row) in enumerate(data_unique.iterrows()):
        if distances[i] <= MAX_DIST:
            station_threshold_map[(row['lat'], row['lon'])] = clim_thresholds[indices[i]]

    n_matched_stations = len(station_threshold_map)
    n_total_stations = len(data_unique)
    print(f"    Nearest-neighbour matching (tolerance {MAX_DIST}°): "
          f"{n_matched_stations}/{n_total_stations} unique stations matched")

    # Map thresholds to every row in data via merge on (lat, lon)
    mapping_df = pd.DataFrame(
        list(station_threshold_map.items()),
        columns=['_key', '_thr']
    )
    mapping_df[['lat', 'lon']] = pd.DataFrame(mapping_df['_key'].tolist(), index=mapping_df.index)
    mapping_df = mapping_df.drop(columns='_key')

    data_with_thr = data[['lat', 'lon']].merge(mapping_df, on=['lat', 'lon'], how='left')
    threshold_series = pd.Series(data_with_thr['_thr'].values, index=data.index)

    n_valid = threshold_series.notna().sum()
    n_total = len(threshold_series)
    n_missing = n_total - n_valid
    mean_thr = np.nanmean(threshold_series)
    print(f"  → {n_valid}/{n_total} rows matched ({n_missing} missing), mean threshold = {mean_thr:.3f}")

    if n_valid == 0:
        print(f"  WARNING: No stations matched! Falling back to dataset_climatology")
        threshold_val = np.nanpercentile(data['obs_value'].dropna(), percentile)
        print(f"  → Using uniform threshold = {threshold_val:.3f} for all stations")
        threshold_series = pd.Series(threshold_val, index=data.index)

    return threshold_series, event_type


def _threshold_area_mean_climatology(config, data):
    """
    Single threshold = area-mean of station climatological percentile.

    Config keys:
      threshold.area_mean_climatology.percentile: int

    Returns: (float, event_type)
    """
    cfg = config['threshold']['area_mean_climatology']
    percentile = cfg['percentile']
    event_type = config['threshold'].get('event_type', 'above')

    print(f"  Area-mean climatology: mean of p{percentile} across all stations")

    # Get per-station thresholds first
    station_thresholds, _ = _threshold_station_climatology(config, data)
    threshold = np.nanmean(station_thresholds)

    print(f"  → area-mean threshold = {threshold:.3f}")
    return threshold, event_type


def _threshold_model_percentile(config, data):
    """
    Per-station percentile computed from one model's forecast distribution.

    Config keys:
      threshold.model_percentile.percentile: int
      threshold.model_percentile.model: 'fc1' or 'fc2'

    Returns: (per-station Series, event_type)
    """
    cfg = config['threshold']['model_percentile']
    percentile = cfg['percentile']
    model = cfg.get('model', 'fc1')
    event_type = config['threshold'].get('event_type', 'above')

    col = f'{model}_value' if f'{model}_value' in data.columns else 'fc1_value'
    print(f"  Model percentile: p{percentile} of {model} at each station")

    threshold_series = data.groupby('station_id')[col].transform(
        lambda x: np.nanpercentile(x, percentile)
    )

    mean_thr = threshold_series.mean()
    print(f"  → mean threshold across stations = {mean_thr:.3f}")
    return threshold_series, event_type


# =============================================================================
# STEP 6: SCORE COMPUTATION (VTB where possible, custom otherwise)
# =============================================================================

def compute_scores_quaver(config, data, threshold, event_type, model_names, is_ensemble=True):
    """
    Compute verification scores using VTB where available,
    falling back to custom implementations otherwise.

    VTB-available scores:
      - CRPS, fair_CRPS (FieldMetrics.crps)
      - Contingency table → ETS, Brier, ROCA, PSS (FieldMetrics.contingency_table)
      - RMSE, bias, mean_error (FieldMetrics.error, DomainMetrics)
      - Spread (FieldMetrics.spread)

    Custom scores (not in VTB):
      - twCRPS (threshold-weighted CRPS)
      - tw_quantile_score
      - BSS (Brier Skill Score)
      - extreme_spread_skill_ratio

    Returns: (overall_scores, results) in same format as ens_scores/det_scores
    """
    vtb, _ = _load_vtb()

    requested_scores = _get_requested_scores(config, is_ensemble)
    fc1_name = model_names['fc1_name']
    fc2_name = model_names['fc2_name']

    # Scores we can compute via VTB
    vtb_scores = {'CRPS', 'fair_CRPS', 'spread', 'ens_spread',
                  'rmse', 'ens_mean_rmse', 'bias', 'ens_mean_bias',
                  'mae', 'ens_mean_mae',
                  'ETS', 'PSS', 'POD', 'FAR', 'Brier', 'ROCA'}

    # Scores we compute with custom code
    custom_scores = {'twCRPS', 'BSS', 'quantile_score', 'tw_quantile_score',
                     'diagonal_score', 'extreme_spread_skill_ratio',
                     'twMAE', 'twRMSE', 'correlation'}

    # Determine which path each score takes
    use_vtb = set(requested_scores) & vtb_scores
    use_custom = set(requested_scores) & custom_scores
    # Anything not in either set falls back to custom
    use_custom |= set(requested_scores) - vtb_scores - custom_scores

    print(f"\n  Scores via VTB:    {sorted(use_vtb) if use_vtb else 'none'}")
    print(f"  Scores via custom: {sorted(use_custom) if use_custom else 'none'}")

    # Group data by forecast day / step
    if 'forecast_day' in data.columns:
        group_col = 'forecast_day'
    elif 'step' in data.columns:
        group_col = 'step'
    else:
        group_col = None

    rows_by_lt = []
    groups = data.groupby(group_col) if group_col else [(None, data)]

    for group_val, gdata in groups:
        row = {'forecast_day': group_val, 'n_samples': len(gdata)}
        # Add lead_time (hours) for compatibility with plot.py
        if group_col == 'forecast_day' and group_val is not None:
            row['lead_time'] = int(group_val) * 24
        elif group_col == 'step' and group_val is not None:
            row['lead_time'] = int(group_val)

        if isinstance(threshold, pd.Series):
            thr = threshold.loc[gdata.index]
        else:
            thr = threshold
        row['threshold'] = float(np.nanmean(thr)) if isinstance(thr, (pd.Series, np.ndarray)) else float(thr)

        if is_ensemble:
            # Build member arrays
            fc1_members = _extract_member_columns(gdata, 'fc1')
            fc2_members = _extract_member_columns(gdata, 'fc2')
            obs = gdata['obs_value'].values

            # ---- VTB scores ----
            if 'CRPS' in use_vtb or 'fair_CRPS' in use_vtb:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    crps_val, fair_crps_val = _vtb_crps(members, obs)
                    if 'CRPS' in use_vtb:
                        row[f'CRPS_{prefix}'] = crps_val
                    if 'fair_CRPS' in use_vtb:
                        row[f'fair_CRPS_{prefix}'] = fair_crps_val

            if 'ens_spread' in use_vtb or 'spread' in use_vtb:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    row[f'ens_spread_{prefix}'] = np.nanmean(np.nanstd(members, axis=1))

            if any(s in use_vtb for s in ('ens_mean_rmse', 'rmse')):
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    ens_mean = np.nanmean(members, axis=1)
                    row[f'ens_mean_rmse_{prefix}'] = np.sqrt(np.nanmean((ens_mean - obs) ** 2))

            if any(s in use_vtb for s in ('ens_mean_bias', 'bias')):
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    ens_mean = np.nanmean(members, axis=1)
                    row[f'ens_mean_bias_{prefix}'] = np.nanmean(ens_mean - obs)

            if any(s in use_vtb for s in ('ens_mean_mae', 'mae')):
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    ens_mean = np.nanmean(members, axis=1)
                    row[f'ens_mean_mae_{prefix}'] = np.nanmean(np.abs(ens_mean - obs))

            if 'Brier' in use_vtb:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    row[f'Brier_{prefix}'] = _compute_brier(members, obs, thr, event_type)

            if 'ETS' in use_vtb:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    ens_mean = np.nanmean(members, axis=1)
                    row[f'ETS_{prefix}'] = _compute_ets_from_arrays(ens_mean, obs, thr, event_type)

            # ---- Custom scores ----
            if 'twCRPS' in use_custom:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    row[f'twCRPS_{prefix}'] = _compute_twcrps(members, obs, thr, event_type)

            if 'BSS' in use_custom:
                # Extract threshold percentile for definitional p_c (= ERA5 convention)
                _thr_cfg = config.get('threshold', {})
                _pctl = (
                    _thr_cfg.get('station_climatology', {}).get('percentile') or
                    _thr_cfg.get('local_obs_climatology', {}).get('percentile') or
                    _thr_cfg.get('dataset_climatology', {}).get('percentile')
                )
                # Brier Skill Score: 1 - BS/BS_clim
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    bs = row.get(f'Brier_{prefix}', _compute_brier(members, obs, thr, event_type))
                    bs_clim = _compute_brier_climatology(obs, thr, event_type)
                    row[f'BSS_{prefix}'] = 1.0 - bs / bs_clim if bs_clim > 0 else 0.0

            if 'quantile_score' in use_custom:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    row[f'quantile_score_{prefix}'] = _compute_quantile_score(members, obs, event_type)

            if 'tw_quantile_score' in use_custom:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    row[f'tw_quantile_score_{prefix}'] = _compute_tw_quantile_score(
                        members, obs, thr, event_type)

            if 'diagonal_score' in use_custom:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    row[f'diagonal_score_{prefix}'] = _compute_diagonal_score(members, obs)

            if 'extreme_spread_skill_ratio' in use_custom:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    row[f'extreme_spread_skill_ratio_{prefix}'] = _compute_tw_spread_skill(
                        members, obs, thr, event_type)

            if 'twMAE' in use_custom or 'twMAE' in requested_scores:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    ens_mean = np.nanmean(members, axis=1)
                    row[f'twMAE_{prefix}'] = _compute_tw_mae(ens_mean, obs, thr, event_type)

            if 'twRMSE' in use_custom or 'twRMSE' in requested_scores:
                for prefix, members in [('fc1', fc1_members), ('fc2', fc2_members)]:
                    ens_mean = np.nanmean(members, axis=1)
                    row[f'twRMSE_{prefix}'] = _compute_tw_rmse(ens_mean, obs, thr, event_type)

        else:
            # Deterministic scores
            fc1 = gdata['fc1_value'].values
            fc2 = gdata['fc2_value'].values
            obs = gdata['obs_value'].values

            for prefix, fc in [('fc1', fc1), ('fc2', fc2)]:
                if 'rmse' in use_vtb or 'rmse' in requested_scores:
                    row[f'rmse_{prefix}'] = np.sqrt(np.nanmean((fc - obs) ** 2))
                if 'bias' in use_vtb or 'bias' in requested_scores:
                    row[f'bias_{prefix}'] = np.nanmean(fc - obs)
                if 'mae' in use_vtb or 'mae' in requested_scores:
                    row[f'mae_{prefix}'] = np.nanmean(np.abs(fc - obs))
                if 'correlation' in requested_scores:
                    valid = ~(np.isnan(fc) | np.isnan(obs))
                    if valid.sum() > 2:
                        row[f'correlation_{prefix}'] = np.corrcoef(fc[valid], obs[valid])[0, 1]
                    else:
                        row[f'correlation_{prefix}'] = np.nan
                if 'ETS' in requested_scores:
                    row[f'ETS_{prefix}'] = _compute_ets_from_arrays(fc, obs, thr, event_type)
                if 'PSS' in requested_scores:
                    row[f'PSS_{prefix}'] = _compute_pss_from_arrays(fc, obs, thr, event_type)
                if 'POD' in requested_scores:
                    h, m, f, cn = _contingency(fc, obs, thr, event_type)
                    row[f'POD_{prefix}'] = h / (h + m) if (h + m) > 0 else np.nan
                if 'FAR' in requested_scores:
                    h, m, f, cn = _contingency(fc, obs, thr, event_type)
                    row[f'FAR_{prefix}'] = f / (h + f) if (h + f) > 0 else np.nan
                if 'twMAE' in requested_scores:
                    row[f'twMAE_{prefix}'] = _compute_tw_mae(fc, obs, thr, event_type)
                if 'twRMSE' in requested_scores:
                    row[f'twRMSE_{prefix}'] = _compute_tw_rmse(fc, obs, thr, event_type)

        # Compute differences for all scores
        for score in requested_scores:
            k1 = f'{score}_fc1'
            k2 = f'{score}_fc2'
            if k1 in row and k2 in row:
                row[f'{score}_diff'] = row[k2] - row[k1]

        # Bootstrap significance testing
        boot_cfg = config.get('bootstrap', {})
        if boot_cfg.get('enabled', False):
            n_boot = boot_cfg.get('n_samples', 100)
            confidence = boot_cfg.get('confidence_level', 0.95)
            # Use block bootstrap by date to account for spatial correlation
            boot_dates = gdata['date'].values if 'date' in gdata.columns else None
            for score in requested_scores:
                if f'{score}_fc1' not in row or f'{score}_fc2' not in row:
                    continue
                if is_ensemble:
                    is_sig, ci_lo, ci_hi = _bootstrap_paired_significance_ens(
                        fc1_members, fc2_members, obs, thr, event_type,
                        score, n_boot, confidence, dates=boot_dates)
                else:
                    is_sig, ci_lo, ci_hi = _bootstrap_paired_significance_det(
                        fc1, fc2, obs, thr, event_type,
                        score, n_boot, confidence, dates=boot_dates)
                row[f'{score}_is_significant'] = is_sig
                row[f'{score}_diff_ci_low'] = ci_lo
                row[f'{score}_diff_ci_high'] = ci_hi

        rows_by_lt.append(row)

    by_leadtime = pd.DataFrame(rows_by_lt)

    # Build overall_scores (period-mean)
    overall_scores = {
        'model1_name': fc1_name,
        'model2_name': fc2_name,
    }
    for col in by_leadtime.columns:
        if col not in ('forecast_day', 'n_samples', 'threshold'):
            overall_scores[col] = by_leadtime[col].mean()

    results = {'by_leadtime': by_leadtime}
    return overall_scores, results


# =============================================================================
# SCORE COMPUTATION HELPERS
# =============================================================================

def _vtb_crps(members, obs):
    """Compute CRPS and fair CRPS using the VTB xmetrics formula."""
    import xarray as xr
    vtb, _ = _load_vtb()

    n_samples, n_members = members.shape
    fc_xa = xr.DataArray(members, dims=['sample', 'number'])
    obs_xa = xr.DataArray(obs, dims=['sample'])

    # VTB xmetrics CRPS: crps(forecasts, observations, member_dim)
    try:
        crps_arr = vtb.xmetrics.crps(forecasts=fc_xa, observations=obs_xa, member_dim='number')
        crps_mean = float(np.nanmean(crps_arr.values))
    except Exception:
        # Fallback to manual CRPS computation
        crps_mean = _compute_crps_manual(members, obs)

    # Fair CRPS adjustment: CRPS_fair = CRPS - spread_term / (2 * M * (M-1))
    spread_term = 0.0
    for i in range(n_members):
        for j in range(i + 1, n_members):
            spread_term += np.nanmean(np.abs(members[:, i] - members[:, j]))
    fair_crps = crps_mean - spread_term / (n_members * (n_members - 1)) if n_members > 1 else crps_mean

    return crps_mean, fair_crps


def _compute_crps_manual(members, obs):
    """Manual CRPS computation when VTB xmetrics is not available."""
    n_samples, n_members = members.shape
    crps_per_sample = np.zeros(n_samples)

    for i in range(n_samples):
        sorted_m = np.sort(members[i, :])
        o = obs[i]
        if np.isnan(o):
            crps_per_sample[i] = np.nan
            continue
        # CRPS = E|X-y| - 0.5 * E|X-X'|
        term1 = np.nanmean(np.abs(sorted_m - o))
        diffs = np.abs(sorted_m[:, None] - sorted_m[None, :])
        term2 = np.nanmean(diffs) / 2.0
        crps_per_sample[i] = term1 - term2

    return float(np.nanmean(crps_per_sample))


def _compute_twcrps(members, obs, threshold, event_type):
    """
    Threshold-weighted CRPS: weights the integrand of CRPS by a chaining
    function that emphasises the tail beyond the threshold.

    w(y) = |y - t| for y beyond threshold, 0 otherwise
    """
    n_samples, n_members = members.shape
    tw_crps = np.zeros(n_samples)

    for i in range(n_samples):
        o = obs[i]
        if np.isnan(o):
            tw_crps[i] = np.nan
            continue

        if isinstance(threshold, pd.Series):
            t = threshold.iloc[i]
        elif isinstance(threshold, np.ndarray):
            t = threshold[i]
        else:
            t = float(threshold)
        sorted_m = np.sort(members[i, :])

        # Use chain function weight: w(z) = (z - t)^2 for z >= t (above), or (t - z)^2 for z <= t (below)
        # Simplified: compute CRPS only over the extreme portion
        if event_type == 'above':
            # Filter to values above threshold
            extreme_mask = (sorted_m >= t) | (o >= t)
            if not extreme_mask.any():
                tw_crps[i] = 0.0
                continue
            # Weight function: max(0, z - t)
            w_obs = max(0.0, o - t)
            w_members = np.maximum(0.0, sorted_m - t)
        else:
            extreme_mask = (sorted_m <= t) | (o <= t)
            if not extreme_mask.any():
                tw_crps[i] = 0.0
                continue
            w_obs = max(0.0, t - o)
            w_members = np.maximum(0.0, t - sorted_m)

        term1 = np.nanmean(np.abs(w_members - w_obs))
        diffs = np.abs(w_members[:, None] - w_members[None, :])
        term2 = np.nanmean(diffs) / 2.0
        tw_crps[i] = term1 - term2

    return float(np.nanmean(tw_crps[~np.isnan(tw_crps)]))


def _compute_brier(members, obs, threshold, event_type):
    """Brier Score: mean squared error of ensemble probability vs binary obs."""
    n_samples, n_members = members.shape
    bs_sum = 0.0
    count = 0

    for i in range(n_samples):
        o = obs[i]
        if np.isnan(o):
            continue
        if isinstance(threshold, pd.Series):
            t = threshold.iloc[i]
        elif isinstance(threshold, np.ndarray):
            t = threshold[i]
        else:
            t = float(threshold)

        if event_type == 'above':
            obs_event = 1.0 if o >= t else 0.0
            fc_prob = np.sum(members[i, :] >= t) / n_members
        else:
            obs_event = 1.0 if o <= t else 0.0
            fc_prob = np.sum(members[i, :] <= t) / n_members

        bs_sum += (fc_prob - obs_event) ** 2
        count += 1

    return bs_sum / count if count > 0 else np.nan


def _compute_brier_climatology(obs, threshold, event_type):
    """Climatological Brier Score for BSS reference.

    Uses the sample event frequency (Murphy 1973 definition). Per-station
    thresholds are supported via array-valued threshold.
    """
    if isinstance(threshold, pd.Series):
        thr_arr = threshold.values
    elif isinstance(threshold, np.ndarray):
        thr_arr = threshold
    else:
        thr_arr = None

    if thr_arr is not None:
        valid_mask = ~np.isnan(obs) & ~np.isnan(thr_arr)
        if not valid_mask.any():
            return np.nan
        obs_v = obs[valid_mask]
        thr_v = thr_arr[valid_mask]
        clim_freq = (np.mean(obs_v >= thr_v) if event_type == 'above'
                     else np.mean(obs_v <= thr_v))
    else:
        t = float(threshold)
        valid = obs[~np.isnan(obs)]
        if len(valid) == 0:
            return np.nan
        clim_freq = (np.mean(valid >= t) if event_type == 'above'
                     else np.mean(valid <= t))

    return clim_freq * (1 - clim_freq)


def _compute_quantile_score(members, obs, event_type):
    """Quantile score at the extreme quantile (p99 for above, p01 for below).

    Extracts the ensemble's alpha-quantile as a point forecast and evaluates
    it with the pinball loss. Matches the ens_scores.py implementation.
    """
    n_samples, n_members = members.shape
    alpha = 0.99 if event_type == 'above' else 0.01
    sorted_members = np.sort(members, axis=1)
    q_idx = min(int(np.ceil(alpha * n_members)) - 1, n_members - 1)

    qs_total = 0.0
    count = 0
    for i in range(n_samples):
        if np.isnan(obs[i]):
            continue
        q_hat = sorted_members[i, q_idx]
        diff = obs[i] - q_hat
        qs_total += (alpha - (1.0 if diff < 0 else 0.0)) * diff
        count += 1

    return qs_total / count if count > 0 else np.nan


def _compute_tw_quantile_score(members, obs, threshold, event_type):
    """Threshold-weighted quantile score per Taggart (2022).

    Uses the chaining function g(x) = max(x, T) for upper tail (min for lower):
      twQS_alpha(q, y) = qs_alpha(g(q), g(y))
    where q = ensemble alpha-quantile (p99 for above, p01 for below).

    This is a proper score — unlike the conditional approach (selecting only
    extreme obs), the chaining function scores ALL cases while focusing the
    discrimination on the tail region. Per-station thresholds are applied
    row-by-row so each station uses its own climatological T_i.
    """
    n_samples, n_members = members.shape
    alpha = 0.99 if event_type == 'above' else 0.01
    sorted_members = np.sort(members, axis=1)
    q_idx = min(int(np.ceil(alpha * n_members)) - 1, n_members - 1)

    qs_total = 0.0
    count = 0
    for i in range(n_samples):
        if np.isnan(obs[i]):
            continue
        if isinstance(threshold, pd.Series):
            t = threshold.iloc[i]
        elif isinstance(threshold, np.ndarray):
            t = threshold[i]
        else:
            t = float(threshold)

        q_hat = sorted_members[i, q_idx]  # alpha-quantile of ensemble
        y = obs[i]

        # Apply chaining function: focus score on the tail
        if event_type == 'above':
            g_q = max(q_hat, t)
            g_y = max(y, t)
        else:
            g_q = min(q_hat, t)
            g_y = min(y, t)

        # Standard quantile score on transformed values
        diff = g_y - g_q
        qs_total += (alpha - (1.0 if diff < 0 else 0.0)) * diff
        count += 1

    return qs_total / count if count > 0 else np.nan


def _compute_diagonal_score(members, obs, ncat=20):
    """Diagonal score using observation climatology percentiles.
    
    Follows VTB xmetrics.diagonal() methodology:
    - Derives observation climatology percentiles from pooled obs
    - For each tau, checks if obs exceeds climatological threshold
    - Computes ensemble probability of exceeding the same threshold
    - Scores mismatches between observation events and forecast probability
    
    Args:
        members: (n_samples, n_members) forecast ensemble
        obs: (n_samples,) observations
        ncat: number of categories (VTB default=20 → 19 tau levels)
    
    Returns:
        float: mean diagonal score
    """
    n_samples, n_members = members.shape
    
    # Filter valid samples
    valid = ~np.isnan(obs)
    for j in range(n_members):
        valid &= ~np.isnan(members[:, j])
    
    if valid.sum() < 10:
        return np.nan
    
    obs_v = obs[valid]
    members_v = members[valid]
    n_valid = valid.sum()
    
    # Observation climatology percentiles
    taus = np.arange(1, ncat, dtype=float) / ncat  # [0.05, 0.10, ..., 0.95]
    obs_clim_pctls = np.percentile(obs_v, taus * 100)
    
    ds_total = 0.0
    n_valid_taus = 0
    
    for k, tau in enumerate(taus):
        threshold_k = obs_clim_pctls[k]
        
        # Skip non-distinct thresholds
        if np.all(obs_v == threshold_k):
            continue
        
        # Observation event: did obs exceed climatological percentile?
        obs_ev = (obs_v > threshold_k).astype(float)
        
        # Ensemble probability of exceeding the same percentile
        p = (members_v > threshold_k).sum(axis=1).astype(float) / n_members
        
        # VTB formula
        dst = obs_ev * (p <= (1.0 - tau)) * tau + (1.0 - obs_ev) * (p > (1.0 - tau)) * (1.0 - tau)
        
        ds_total += dst.sum()
        n_valid_taus += 1
    
    if n_valid_taus == 0:
        return np.nan
    
    return ds_total / (n_valid_taus * n_valid)


def _compute_ensemble_score(score_name, members, obs, threshold, event_type):
    """Dispatch to the appropriate score function for a single (members, obs) sample.
    Used by bootstrap resampling. members: (n_samples, n_members), obs: (n_samples,).
    """
    if score_name == 'twCRPS':
        return _compute_twcrps(members, obs, threshold, event_type)
    elif score_name == 'Brier':
        return _compute_brier(members, obs, threshold, event_type)
    elif score_name == 'BSS':
        bs = _compute_brier(members, obs, threshold, event_type)
        bs_clim = _compute_brier_climatology(obs, threshold, event_type)
        return 1.0 - bs / bs_clim if bs_clim > 0 else 0.0
    elif score_name == 'quantile_score':
        return _compute_quantile_score(members, obs, event_type)
    elif score_name == 'tw_quantile_score':
        return _compute_tw_quantile_score(members, obs, threshold, event_type)
    elif score_name == 'extreme_spread_skill_ratio':
        return _compute_tw_spread_skill(members, obs, threshold, event_type)
    elif score_name == 'twMAE':
        ens_mean = np.nanmean(members, axis=1)
        return _compute_tw_mae(ens_mean, obs, threshold, event_type)
    elif score_name == 'twRMSE':
        ens_mean = np.nanmean(members, axis=1)
        return _compute_tw_rmse(ens_mean, obs, threshold, event_type)
    elif score_name == 'CRPS':
        crps_val, _ = _vtb_crps(members, obs)
        return crps_val
    elif score_name == 'fair_CRPS':
        _, fair_val = _vtb_crps(members, obs)
        return fair_val
    elif score_name in ('ens_spread', 'spread'):
        return np.nanmean(np.nanstd(members, axis=1))
    elif score_name in ('ens_mean_rmse', 'rmse'):
        ens_mean = np.nanmean(members, axis=1)
        return np.sqrt(np.nanmean((ens_mean - obs) ** 2))
    elif score_name in ('ens_mean_bias', 'bias'):
        ens_mean = np.nanmean(members, axis=1)
        return np.nanmean(ens_mean - obs)
    elif score_name in ('ens_mean_mae', 'mae'):
        ens_mean = np.nanmean(members, axis=1)
        return np.nanmean(np.abs(ens_mean - obs))
    elif score_name == 'ETS':
        ens_mean = np.nanmean(members, axis=1)
        return _compute_ets_from_arrays(ens_mean, obs, threshold, event_type)
    elif score_name == 'diagonal_score':
        return _compute_diagonal_score(members, obs)
    else:
        return np.nan


def _compute_det_score(score_name, fc, obs, threshold, event_type):
    """Dispatch to the appropriate score function for deterministic data.
    Used by bootstrap resampling. fc: (n_samples,), obs: (n_samples,).
    """
    if score_name == 'rmse':
        return np.sqrt(np.nanmean((fc - obs) ** 2))
    elif score_name == 'bias':
        return np.nanmean(fc - obs)
    elif score_name == 'mae':
        return np.nanmean(np.abs(fc - obs))
    elif score_name == 'ETS':
        return _compute_ets_from_arrays(fc, obs, threshold, event_type)
    elif score_name == 'PSS':
        return _compute_pss_from_arrays(fc, obs, threshold, event_type)
    elif score_name == 'twMAE':
        return _compute_tw_mae(fc, obs, threshold, event_type)
    elif score_name == 'twRMSE':
        return _compute_tw_rmse(fc, obs, threshold, event_type)
    elif score_name == 'correlation':
        valid = ~(np.isnan(fc) | np.isnan(obs))
        if valid.sum() > 2:
            return np.corrcoef(fc[valid], obs[valid])[0, 1]
        return np.nan
    elif score_name == 'POD':
        h, m, f, cn = _contingency(fc, obs, threshold, event_type)
        return h / (h + m) if (h + m) > 0 else np.nan
    elif score_name == 'FAR':
        h, m, f, cn = _contingency(fc, obs, threshold, event_type)
        return f / (h + f) if (h + f) > 0 else np.nan
    else:
        return np.nan


def _bootstrap_paired_significance_ens(fc1_members, fc2_members, obs, threshold,
                                       event_type, score_name,
                                       n_bootstrap=100, confidence=0.95,
                                       max_samples=200000, dates=None):
    """Test if fc1-fc2 difference is significant via paired bootstrap (ensemble).

    When ``dates`` is provided (array of per-row date values), uses a block
    bootstrap that resamples whole dates at a time.  This correctly accounts for
    the spatial correlation between stations observed on the same date, which
    would otherwise inflate the effective sample size and make every box appear
    significant.  Falls back to row-level bootstrap when fewer than 5 unique
    dates are available.

    Returns (is_significant, ci_low, ci_high).
    """
    n = len(obs)
    if n < 20:
        return False, np.nan, np.nan

    # Convert threshold to numpy to avoid pandas label-based indexing issues
    if isinstance(threshold, pd.Series):
        threshold = threshold.values

    # Decide whether to use block bootstrap (resample by date)
    use_block = False
    if dates is not None:
        unique_dates, date_inverse = np.unique(dates, return_inverse=True)
        n_dates = len(unique_dates)
        if n_dates >= 5:
            use_block = True
            # Pre-build index lists per date for efficiency
            date_indices = [np.where(date_inverse == d_idx)[0]
                            for d_idx in range(n_dates)]

    if not use_block and n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        fc1_members = fc1_members[idx]
        fc2_members = fc2_members[idx]
        obs = obs[idx]
        threshold = threshold[idx] if isinstance(threshold, np.ndarray) else threshold
        n = max_samples

    boot_diffs = []
    for _ in range(n_bootstrap):
        if use_block:
            # Block bootstrap: resample whole dates to preserve spatial correlation
            sampled = np.random.choice(n_dates, n_dates, replace=True)
            idx = np.concatenate([date_indices[d] for d in sampled])
        else:
            idx = np.random.choice(n, n, replace=True)

        obs_b = obs[idx]
        fc1_b = fc1_members[idx]
        fc2_b = fc2_members[idx]
        thr_b = threshold[idx] if isinstance(threshold, np.ndarray) else threshold

        v1 = _compute_ensemble_score(score_name, fc1_b, obs_b, thr_b, event_type)
        v2 = _compute_ensemble_score(score_name, fc2_b, obs_b, thr_b, event_type)

        if not (np.isnan(v1) or np.isnan(v2)):
            boot_diffs.append(v1 - v2)

    if len(boot_diffs) < n_bootstrap * 0.3:
        return False, np.nan, np.nan

    alpha = 1 - confidence
    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    is_significant = (ci_low > 0) or (ci_high < 0)
    return is_significant, ci_low, ci_high


def _bootstrap_paired_significance_det(fc1, fc2, obs, threshold,
                                       event_type, score_name,
                                       n_bootstrap=100, confidence=0.95,
                                       max_samples=200000, dates=None):
    """Test if fc1-fc2 difference is significant via paired bootstrap (deterministic).

    See ``_bootstrap_paired_significance_ens`` for block-bootstrap details.

    Returns (is_significant, ci_low, ci_high).
    """
    n = len(obs)
    if n < 20:
        return False, np.nan, np.nan

    # Convert threshold to numpy to avoid pandas label-based indexing issues
    if isinstance(threshold, pd.Series):
        threshold = threshold.values

    # Decide whether to use block bootstrap (resample by date)
    use_block = False
    if dates is not None:
        unique_dates, date_inverse = np.unique(dates, return_inverse=True)
        n_dates = len(unique_dates)
        if n_dates >= 5:
            use_block = True
            date_indices = [np.where(date_inverse == d_idx)[0]
                            for d_idx in range(n_dates)]

    if not use_block and n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        fc1 = fc1[idx]
        fc2 = fc2[idx]
        obs = obs[idx]
        threshold = threshold[idx] if isinstance(threshold, np.ndarray) else threshold
        n = max_samples

    boot_diffs = []
    for _ in range(n_bootstrap):
        if use_block:
            sampled = np.random.choice(n_dates, n_dates, replace=True)
            idx = np.concatenate([date_indices[d] for d in sampled])
        else:
            idx = np.random.choice(n, n, replace=True)

        obs_b = obs[idx]
        fc1_b = fc1[idx]
        fc2_b = fc2[idx]
        thr_b = threshold[idx] if isinstance(threshold, np.ndarray) else threshold

        v1 = _compute_det_score(score_name, fc1_b, obs_b, thr_b, event_type)
        v2 = _compute_det_score(score_name, fc2_b, obs_b, thr_b, event_type)

        if not (np.isnan(v1) or np.isnan(v2)):
            boot_diffs.append(v1 - v2)

    if len(boot_diffs) < n_bootstrap * 0.3:
        return False, np.nan, np.nan

    alpha = 1 - confidence
    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    is_significant = (ci_low > 0) or (ci_high < 0)
    return is_significant, ci_low, ci_high


def _compute_tw_spread_skill(members, obs, threshold, event_type):
    """Threshold-weighted spread-skill ratio for extreme cases."""
    n_samples, n_members = members.shape
    spreads = []
    errors = []

    for i in range(n_samples):
        if np.isnan(obs[i]):
            continue
        if isinstance(threshold, pd.Series):
            t = threshold.iloc[i]
        elif isinstance(threshold, np.ndarray):
            t = threshold[i]
        else:
            t = float(threshold)
        is_extreme = (obs[i] > t) if event_type == 'above' else (obs[i] < t)
        if not is_extreme:
            continue
        spreads.append(np.nanstd(members[i, :]))
        errors.append(np.abs(np.nanmean(members[i, :]) - obs[i]))

    if len(spreads) == 0:
        return np.nan
    mean_spread = np.mean(spreads)
    mean_error = np.mean(errors)
    return mean_spread / mean_error if mean_error > 0 else np.nan


# =============================================================================
# DETERMINISTIC SCORE HELPERS
# =============================================================================

def _contingency(fc, obs, threshold, event_type):
    """Return (hits, misses, false_alarms, correct_negatives)."""
    if isinstance(threshold, pd.Series):
        t = threshold.values
    elif isinstance(threshold, np.ndarray):
        t = threshold
    else:
        t = float(threshold)

    if event_type == 'above':
        fc_yes = fc >= t
        obs_yes = obs >= t
    else:
        fc_yes = fc <= t
        obs_yes = obs <= t

    valid = ~(np.isnan(fc) | np.isnan(obs))
    h = np.sum(fc_yes[valid] & obs_yes[valid])
    m = np.sum(~fc_yes[valid] & obs_yes[valid])
    f = np.sum(fc_yes[valid] & ~obs_yes[valid])
    cn = np.sum(~fc_yes[valid] & ~obs_yes[valid])
    return int(h), int(m), int(f), int(cn)


def _compute_ets_from_arrays(fc, obs, threshold, event_type):
    """Equitable Threat Score."""
    h, m, f, cn = _contingency(fc, obs, threshold, event_type)
    n = h + m + f + cn
    if n == 0:
        return np.nan
    h_ref = (h + m) * (h + f) / n
    denom = h + m + f - h_ref
    return (h - h_ref) / denom if denom > 0 else np.nan


def _compute_pss_from_arrays(fc, obs, threshold, event_type):
    """Peirce Skill Score (Hanssen-Kuipers)."""
    h, m, f, cn = _contingency(fc, obs, threshold, event_type)
    pod = h / (h + m) if (h + m) > 0 else 0.0
    pofd = f / (f + cn) if (f + cn) > 0 else 0.0
    return pod - pofd


def _compute_tw_mae(fc, obs, threshold, event_type):
    """Threshold-weighted MAE — only extreme observed cases."""
    if isinstance(threshold, pd.Series):
        t = threshold.values
    elif isinstance(threshold, np.ndarray):
        t = threshold
    else:
        t = float(threshold)
    if event_type == 'above':
        mask = obs >= t
    else:
        mask = obs <= t
    valid = mask & ~np.isnan(fc) & ~np.isnan(obs)
    if valid.sum() == 0:
        return np.nan
    return float(np.nanmean(np.abs(fc[valid] - obs[valid])))


def _compute_tw_rmse(fc, obs, threshold, event_type):
    """Threshold-weighted RMSE — only extreme observed cases."""
    if isinstance(threshold, pd.Series):
        t = threshold.values
    elif isinstance(threshold, np.ndarray):
        t = threshold
    else:
        t = float(threshold)
    if event_type == 'above':
        mask = obs >= t
    else:
        mask = obs <= t
    valid = mask & ~np.isnan(fc) & ~np.isnan(obs)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((fc[valid] - obs[valid]) ** 2)))


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _variable_to_param(variable):
    """Map config variable name to MARS/STVL parameter name."""
    mapping = {
        '2t': '2t',
        '10ff': '10si',  # 10m wind speed (instantaneous)
        'tp24': 'tp',
    }
    return mapping.get(variable, variable)


def _build_steps(config):
    """Build pandas TimedeltaIndex of forecast steps from config."""
    if 'steps' in config and config['steps']:
        return pd.to_timedelta([f"{s}h" for s in config['steps']])
    elif 'forecast_days' in config and config['forecast_days']:
        freq = config.get('lead_time_frequency', 24)
        all_steps = []
        for day in config['forecast_days']:
            start_h = (day - 1) * 24
            end_h = day * 24
            all_steps.extend(range(start_h, end_h + 1, freq))
        return pd.to_timedelta([f"{s}h" for s in sorted(set(all_steps))])
    else:
        return pd.to_timedelta(['24h', '48h', '72h', '96h', '120h'])


def _get_vtb_domain(area):
    """Map config area name to a bbox tuple (north, west, south, east).

    Uses the same bboxes as extract_points.py / filter.py so that quaver
    and local pipelines operate on identical geographic regions.
    Falls back to VTB Domain definitions for names not in the table.
    """
    # Hardcoded bboxes — must match extract_points.py get_area_bbox()
    _AREAS = {
        'europe':           (68, -15, 27, 50),
        'nh_extratropics':  (90, -180, 20, 180),
        'tropics':          (20, -180, -20, 180),
    }
    if area in _AREAS:
        north, west, south, east = _AREAS[area]
        print(f"    Area bbox '{area}': N={north}, W={west}, S={south}, E={east}")
        return (north, west, south, east)

    # Fallback: try VTB Domain
    vtb, _ = _load_vtb()
    vtb_names = {
        'n.hem': 'n.hem', 'nhem': 'n.hem',
        's.hem': 's.hem', 'shem': 's.hem',
        'tropics30': 'tropics30',
        'extrop30': 'extrop30',
    }
    name = vtb_names.get(area, area)
    try:
        dom = vtb.Domain.from_name(name)
        tiles = dom.to_lists()  # list of [north, west, south, east]
        if tiles:
            north, west, south, east = tiles[0]
            print(f"    Area domain '{name}': N={north}, W={west}, S={south}, E={east}")
            return (north, west, south, east)
        print(f"    [WARN] Domain '{name}' has no tiles, no spatial filtering applied")
        return None
    except Exception:
        print(f"    [WARN] Unknown VTB domain '{area}', no spatial filtering applied")
        return None


def _date_to_forecast_day(date, steps):
    """Get the forecast day number from step list (day 1 = 0-24h, etc)."""
    max_h = max(s.total_seconds() / 3600 for s in steps)
    return int(max_h / 24) or 1


def _retrieve_auxiliary_fields(config):
    """
    Retrieve static auxiliary fields (sdfor, lsm) for orography and
    coastal filtering.

    Tries MARS retrieval first. If auxiliary_fields paths are configured
    and MARS fails, falls back to reading local GRIB files.

    Returns: dict with 'sdfor' and 'lsm' VTB Fieldsets (or None per field).
    """
    vtb, metview = _load_vtb()

    result = {'sdfor': None, 'lsm': None, 'orog': None, 'orog_model2': None}
    cfg_aux = config.get('auxiliary_fields', {})

    # All three are static analysis fields from the operational IFS
    _static_mars_kw = dict(
        levtype='sfc',
        class_='od',
        stream='oper',
        type='an',
        expver='1',
        date='latest',
        step=0,
    )

    # --- sdfor (sub-grid orography standard deviation, param 160228) ---
    try:
        print("  Retrieving sdfor from MARS...")
        result['sdfor'] = vtb.media.mars_retrieve(parameter='sdfor', **_static_mars_kw)
        print(f"    → sdfor retrieved ({len(result['sdfor'])} field(s))")
    except Exception as e:
        print(f"    MARS sdfor failed ({e}), trying local file...")
        sdfor_path = cfg_aux.get('sdfor_path')
        if sdfor_path and os.path.exists(sdfor_path):
            result['sdfor'] = vtb.Fieldset(sdfor_path)
            print(f"    → sdfor loaded from {sdfor_path}")
        else:
            print(f"    ⚠ sdfor not available — orography filtering will be skipped")

    # --- lsm (land-sea mask, param 172) ---
    try:
        print("  Retrieving lsm from MARS...")
        result['lsm'] = vtb.media.mars_retrieve(parameter='lsm', **_static_mars_kw)
        print(f"    → lsm retrieved ({len(result['lsm'])} field(s))")
    except Exception as e:
        print(f"    MARS lsm failed ({e}), trying local file...")
        lsm_path = cfg_aux.get('model1', {}).get('lsm_path') or cfg_aux.get('lsm_path')
        if lsm_path and os.path.exists(lsm_path):
            result['lsm'] = vtb.Fieldset(lsm_path)
            print(f"    → lsm loaded from {lsm_path}")
        else:
            print(f"    ⚠ lsm not available — coastal filtering will be skipped")

    # --- orog (geopotential at surface) for model1 ---
    # Use per-model GRIB files when configured (ensures correct resolution)
    orog_path_m1 = cfg_aux.get('model1', {}).get('orog_path') or cfg_aux.get('orog_path')
    if orog_path_m1 and os.path.exists(orog_path_m1):
        result['orog'] = vtb.Fieldset(orog_path_m1)
        print(f"    → orog model1 loaded from {orog_path_m1}")
    else:
        try:
            print("  Retrieving orog (surface geopotential) from MARS...")
            result['orog'] = vtb.media.mars_retrieve(parameter='z', **_static_mars_kw)
            print(f"    → orog retrieved ({len(result['orog'])} field(s))")
        except Exception as e:
            print(f"    ⚠ orog not available ({e}) — lapse-rate correction will not use model height")

    # --- orog for model2 (if different from model1) ---
    orog_path_m2 = cfg_aux.get('model2', {}).get('orog_path')
    orog_path_m1 = cfg_aux.get('model1', {}).get('orog_path') or cfg_aux.get('orog_path')
    if orog_path_m2 and orog_path_m2 != orog_path_m1 and os.path.exists(orog_path_m2):
        result['orog_model2'] = vtb.Fieldset(orog_path_m2)
        print(f"    → orog model2 loaded from {orog_path_m2}")
    else:
        result['orog_model2'] = result['orog']  # same orog for both models

    return result


def _align_auxiliary_to_obs(aux_fields, obs_fs):
    """
    Interpolate static auxiliary fields (sdfor, lsm, orog) to observation
    station locations using VTB Fieldset.aligned().

    orog is surface geopotential → converted to height in metres (z / g).

    Returns: dict with 'sdfor_values', 'lsm_values', 'height_values'
             as numpy arrays indexed by observation order (or None).
    """
    result = {'sdfor_values': None, 'lsm_values': None, 'height_values': None, 'height_values_model2': None}

    if aux_fields.get('sdfor') is not None:
        try:
            sdfor_at_obs = aux_fields['sdfor'].aligned(obs_fs)
            sdfor_df = sdfor_at_obs.to_dataframe()
            result['sdfor_values'] = sdfor_df['value_0'].values if 'value_0' in sdfor_df.columns else None
        except Exception as e:
            print(f"    [WARN] Could not align sdfor to stations: {e}")

    if aux_fields.get('lsm') is not None:
        try:
            lsm_at_obs = aux_fields['lsm'].aligned(obs_fs)
            lsm_df = lsm_at_obs.to_dataframe()
            result['lsm_values'] = lsm_df['value_0'].values if 'value_0' in lsm_df.columns else None
        except Exception as e:
            print(f"    [WARN] Could not align lsm to stations: {e}")

    if aux_fields.get('orog') is not None:
        try:
            orog_at_obs = aux_fields['orog'].aligned(obs_fs)
            orog_df = orog_at_obs.to_dataframe()
            if 'value_0' in orog_df.columns:
                # Convert surface geopotential (m²/s²) to height (m): h = z / g
                result['height_values'] = orog_df['value_0'].values / 9.80665
        except Exception as e:
            print(f"    [WARN] Could not align orog to stations: {e}")

    # Model2 orography (may differ from model1 if different grid)
    orog_m2 = aux_fields.get('orog_model2')
    if orog_m2 is not None and orog_m2 is not aux_fields.get('orog'):
        try:
            orog_m2_at_obs = orog_m2.aligned(obs_fs)
            orog_m2_df = orog_m2_at_obs.to_dataframe()
            if 'value_0' in orog_m2_df.columns:
                result['height_values_model2'] = orog_m2_df['value_0'].values / 9.80665
        except Exception as e:
            print(f"    [WARN] Could not align orog_model2 to stations: {e}")
    else:
        result['height_values_model2'] = result['height_values']

    return result


def _retrieve_one_day(config, model_key, date, steps):
    """Retrieve one day of forecast fields from MARS.
    
    When 'interpolation' is configured, uses direct MARS CLI to avoid
    VTB/Metview quoting issue with the interpolation keyword.
    """
    vtb, _ = _load_vtb()
    cfg = config['read_data'][model_key]
    q = cfg['quaver']

    interpolation = q.get('interpolation', '')

    if interpolation:
        # Use direct MARS CLI to avoid VTB quoting the interpolation value
        return _retrieve_via_mars_cli(config, q, date, steps)

    mars_kw = dict(
        parameter=_variable_to_param(config['variable']),
        levtype='sfc',
        date=date,
        step=steps,
        stream=q['stream'],
        type=q['type'],
        class_=q['class'],
        expver=q['expver'],
    )
    if 'grid' in q and q['grid']:
        mars_kw['grid'] = q['grid']
    if q.get('number'):
        mars_kw['number'] = q['number']

    return vtb.media.mars_retrieve(**mars_kw)


def _retrieve_via_mars_cli(config, q, date, steps):
    """Retrieve fields via direct MARS CLI call with conservative interpolation.
    
    Uses the 'mars' command directly to avoid VTB/Metview quoting issues
    with the 'interpolation' keyword value.
    """
    import subprocess

    vtb, _ = _load_vtb()

    param = _variable_to_param(config['variable'])
    # Unique label for this model (class+expver) to avoid file collisions
    model_label = f"{q['class']}_{q['expver']}"
    # Ensure steps is formatted as MARS slash-separated string (handle Timedelta, int, etc.)
    def _step_to_str(s):
        if hasattr(s, 'total_seconds'):
            return str(int(s.total_seconds() / 3600))
        return str(int(s))
    if hasattr(steps, '__iter__') and not isinstance(steps, str):
        step_str = '/'.join(_step_to_str(s) for s in steps)
    else:
        step_str = _step_to_str(steps)

    # Format date as YYYYMMDD string for MARS
    if hasattr(date, 'strftime'):
        date_str = date.strftime('%Y%m%d')
    else:
        date_str = str(date).replace('-', '')[:8]

    interpolation = q.get('interpolation', 'grid-box-average')

    # Build MARS request text line by line
    lines = [
        "retrieve,",
        f"    class      = {q['class']},",
        f"    type       = {q['type']},",
        f"    stream     = {q['stream']},",
        f"    expver     = {q['expver']},",
        f"    levtype    = sfc,",
        f"    param      = {param},",
        f"    date       = {date_str},",
        f"    time       = 0000,",
        f"    step       = {step_str},",
    ]
    if 'grid' in q and q['grid']:
        lines.append(f"    grid       = {q['grid']},")
    lines.append(f"    interpolation = {interpolation},")
    if q.get('number'):
        lines.append(f"    number     = {q['number']},")

    tmpdir = os.environ.get('TMPDIR', '/tmp')
    target_path = os.path.join(tmpdir, f"mars_{model_label}_{os.getpid()}_{date_str}.grib")
    lines.append(f'    target     = "{target_path}"')

    mars_request = "\n".join(lines)

    req_path = os.path.join(tmpdir, f"mars_req_{model_label}_{os.getpid()}_{date_str}.req")
    with open(req_path, 'w') as f:
        f.write(mars_request + "\n")

    print(f"      MARS CLI ({date_str}): interpolation={interpolation}, grid={q.get('grid', 'native')}")

    try:
        result = subprocess.run(
            ['mars', req_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            all_output = result.stdout + '\n' + result.stderr
            err_lines = [l for l in all_output.splitlines() if 'ERROR' in l]
            for l in err_lines[:5]:
                print(f"        {l}")
            raise RuntimeError(f"MARS CLI failed for {date_str}: exit code {result.returncode}")

        # Read the resulting GRIB into a VTB Fieldset
        fs = vtb.Fieldset(target_path)
        print(f"        → {len(fs)} fields retrieved")
        return fs

    finally:
        # Clean up request file (keep target GRIB — VTB may memory-map it)
        if os.path.exists(req_path):
            try:
                os.remove(req_path)
            except OSError:
                pass


def _regrid_fieldset(fieldset, regrid_cfg, label=""):
    """
    Regrid a VTB Fieldset.

    Two methods (chosen by regrid.method config, defaults to 'mir'):
      1. 'mir' : Write to GRIB, run mir CLI, read back (supports grid-box-average)
      2. 'metview' : Use metview.regrid() in-process (supports grid-box-average via
                     INTERPOLATION='grid_box_average')

    Config keys (under 'regrid'):
      target_grid:    e.g. "0.1/0.1"
      interpolation:  e.g. "grid-box-average" (conservative)
      method:         "mir" or "metview" (default: "mir")
    """
    import subprocess

    vtb, metview = _load_vtb()

    target_grid = regrid_cfg.get('target_grid', '0.1/0.1')
    interpolation = regrid_cfg.get('interpolation', 'grid-box-average')
    method = regrid_cfg.get('method', 'mir')

    print(f"      Regridding {label}: grid={target_grid}, interp={interpolation}, method={method}")

    if method == 'metview':
        # Use metview's built-in regridding (no subprocess)
        try:
            mv_fs = fieldset.to_metview()
            # metview.regrid() uses INTERPOLATION keyword
            mv_interp = interpolation.replace('-', '_')  # grid-box-average → grid_box_average
            regridded_mv = metview.regrid(
                data=mv_fs,
                grid=target_grid.split('/'),
                interpolation=mv_interp,
            )
            regridded = vtb.Fieldset.from_metview_fieldset(regridded_mv)
            print(f"        → done (metview)")
            return regridded
        except Exception as e:
            print(f"        [WARN] metview regrid failed for {label}: {e}")
            return fieldset

    # method == 'mir': use MIR CLI subprocess
    tmpdir = os.environ.get('TMPDIR', '/tmp')
    in_path = os.path.join(tmpdir, f"mir_in_{label}_{os.getpid()}.grib")
    out_path = os.path.join(tmpdir, f"mir_out_{label}_{os.getpid()}.grib")

    try:
        # Write fieldset to temp GRIB
        fieldset.write(in_path)

        in_size_mb = os.path.getsize(in_path) / 1e6
        print(f"        Input: {in_size_mb:.1f} MB")

        # Run MIR
        cmd = [
            'mir',
            f'--grid={target_grid}',
            f'--interpolation={interpolation}',
            in_path,
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"        [WARN] MIR failed for {label}: {result.stderr.strip()[:200]}")
            return fieldset

        out_size_mb = os.path.getsize(out_path) / 1e6
        print(f"        Output: {out_size_mb:.1f} MB → done")

        regridded = vtb.Fieldset(out_path)
        return regridded

    except subprocess.TimeoutExpired:
        print(f"        [WARN] MIR timed out for {label} (>10 min), using original grid")
        return fieldset
    except Exception as e:
        print(f"        [WARN] Regrid failed for {label}: {e}")
        return fieldset
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def _fieldsets_to_dataframe(fc1, fc2, obs, date, steps,
                            fc1_name, fc2_name, variable,
                            is_ensemble, n_members,
                            preprocess_settings, domain,
                            aux_at_stations=None,
                            fc1_unit_factor=1.0, fc2_unit_factor=1.0,
                            accum_hours=0):
    """
    Convert VTB-aligned Fieldsets into list of row dicts matching
    the existing pipeline's DataFrame format.

    aux_at_stations: dict with 'sdfor_values' and 'lsm_values' arrays
                     aligned to obs station order (from _align_auxiliary_to_obs).
    fc1_unit_factor/fc2_unit_factor: multiply forecast values by this factor
                                     (e.g. 1000.0 for m→mm).
    """
    rows = []

    fc1_df = fc1.to_dataframe()
    fc2_df = fc2.to_dataframe()
    obs_df = obs.to_dataframe()

    # Use obs as the reference for station locations
    if len(obs_df) == 0:
        return rows

    # For each step, build matched rows
    # Pre-extract previous-step DataFrames for accumulation differencing (tp24)
    fc1_prev_dfs = {}
    fc2_prev_dfs = {}
    if accum_hours > 0:
        for step in steps:
            step_h = int(step.total_seconds() / 3600)
            prev_h = step_h - accum_hours
            if prev_h > 0:
                prev_td = pd.to_timedelta(f"{prev_h}h")
                try:
                    fc1_prev_fs = fc1.header_filter(step=prev_td)
                    fc2_prev_fs = fc2.header_filter(step=prev_td)
                    fc1_prev_dfs[step_h] = fc1_prev_fs.to_dataframe()
                    fc2_prev_dfs[step_h] = fc2_prev_fs.to_dataframe()
                except Exception:
                    pass

    for step in steps:
        step_h = int(step.total_seconds() / 3600)
        vtime = date + step
        vtime_str = vtime.strftime('%Y%m%d')
        date_str = date.strftime('%Y%m%d')
        forecast_day = (step_h // 24) + 1 if step_h > 0 else 1

        # Filter dataframes by step
        # VTB fieldsets have 'step' in their header
        try:
            fc1_step = fc1.header_filter(step=step)
            fc2_step = fc2.header_filter(step=step)
            obs_step = obs.header_filter(date=vtime)
        except Exception:
            continue

        if len(obs_step) == 0:
            continue

        obs_step_df = obs_step.to_dataframe()

        for idx, obs_row in obs_step_df.iterrows():
            row = {
                'date': date_str,
                'step': step_h,
                'valid_time': vtime_str,
                'station_id': obs_row.get('station_id', idx),
                'lat': obs_row.get('latitude', np.nan),
                'lon': obs_row.get('longitude', np.nan),
                'obs_value': obs_row.get('value_0', np.nan),
                'obs_height': obs_row.get('elevation', obs_row.get('altitude', 0.0)),
                'forecast_day': forecast_day,
            }

            # Add auxiliary fields (sdfor, lsm, height) for orography/coastal filtering + lapse-rate
            if aux_at_stations is not None:
                sdfor_vals = aux_at_stations.get('sdfor_values')
                lsm_vals = aux_at_stations.get('lsm_values')
                height_vals = aux_at_stations.get('height_values')
                row['sdfor'] = float(sdfor_vals[idx]) if sdfor_vals is not None and idx < len(sdfor_vals) else 0.0
                row['lsm'] = float(lsm_vals[idx]) if lsm_vals is not None and idx < len(lsm_vals) else 1.0
                row['fc1_height'] = float(height_vals[idx]) if height_vals is not None and idx < len(height_vals) else 0.0
                height_vals_m2 = aux_at_stations.get('height_values_model2')
                row['fc2_height'] = float(height_vals_m2[idx]) if height_vals_m2 is not None and idx < len(height_vals_m2) else row['fc1_height']

            # Apply area filtering (bbox check)
            if domain is not None:
                north, west, south, east = domain
                lat, lon = row['lat'], row['lon']
                if lat < south or lat > north or lon < west or lon > east:
                    continue

            if is_ensemble:
                # Extract member values from aligned fieldsets
                fc1_step_df = fc1_step.to_dataframe()
                fc2_step_df = fc2_step.to_dataframe()
                fc1_prev_df = fc1_prev_dfs.get(step_h)
                fc2_prev_df = fc2_prev_dfs.get(step_h)
                for m in range(n_members):
                    col = f'value_{m}' if f'value_{m}' in fc1_step_df.columns else f'value_0'
                    fc1_val = (fc1_step_df.iloc[idx].get(col, np.nan) if idx < len(fc1_step_df) else np.nan)
                    fc2_val = (fc2_step_df.iloc[idx].get(col, np.nan) if idx < len(fc2_step_df) else np.nan)
                    # Subtract previous step for accumulation differencing
                    if accum_hours > 0 and step_h > accum_hours and fc1_prev_df is not None:
                        fc1_prev_val = fc1_prev_df.iloc[idx].get(col, 0.0) if idx < len(fc1_prev_df) else 0.0
                        fc2_prev_val = fc2_prev_df.iloc[idx].get(col, 0.0) if idx < len(fc2_prev_df) else 0.0
                        fc1_val = fc1_val - fc1_prev_val
                        fc2_val = fc2_val - fc2_prev_val
                    row[f'fc1_member_{m}'] = fc1_val * fc1_unit_factor
                    row[f'fc2_member_{m}'] = fc2_val * fc2_unit_factor
            else:
                fc1_step_df = fc1_step.to_dataframe()
                fc2_step_df = fc2_step.to_dataframe()
                fc1_val = (fc1_step_df.iloc[idx].get('value_0', np.nan) if idx < len(fc1_step_df) else np.nan)
                fc2_val = (fc2_step_df.iloc[idx].get('value_0', np.nan) if idx < len(fc2_step_df) else np.nan)
                # Subtract previous step for accumulation differencing (tp24)
                if accum_hours > 0 and step_h > accum_hours:
                    fc1_prev_df = fc1_prev_dfs.get(step_h)
                    fc2_prev_df = fc2_prev_dfs.get(step_h)
                    if fc1_prev_df is not None and idx < len(fc1_prev_df):
                        fc1_val = fc1_val - fc1_prev_df.iloc[idx].get('value_0', 0.0)
                    if fc2_prev_df is not None and idx < len(fc2_prev_df):
                        fc2_val = fc2_val - fc2_prev_df.iloc[idx].get('value_0', 0.0)
                row['fc1_value'] = fc1_val * fc1_unit_factor
                row['fc2_value'] = fc2_val * fc2_unit_factor

            # Apply preprocessing for 2t
            if variable == '2t':
                # Temperature conversion K → °C (VTB/MARS always returns Kelvin)
                if row.get('obs_value', 0) > 100:
                    row['obs_value'] -= 273.15
                for key in ('fc1_value', 'fc2_value'):
                    if key in row and row[key] > 100:
                        row[key] -= 273.15
                # Ensemble members
                for m in range(n_members):
                    for prefix in ('fc1', 'fc2'):
                        mk = f'{prefix}_member_{m}'
                        if mk in row and row[mk] > 100:
                            row[mk] -= 273.15
                # Note: lapse rate correction is applied later in filter.py
                # using obs_height, fc1_height, fc2_height columns

            rows.append(row)

    return rows


def _extract_member_columns(data, prefix):
    """Extract member columns from DataFrame as (n_samples, n_members) array."""
    member_cols = sorted([c for c in data.columns if c.startswith(f'{prefix}_member_')],
                         key=lambda c: int(c.split('_')[-1]))
    if not member_cols:
        return np.empty((len(data), 0))
    return data[member_cols].values


def _get_requested_scores(config, is_ensemble):
    """Get list of requested score names from config."""
    scores_cfg = config.get('scores', {})
    if is_ensemble:
        return scores_cfg.get('ensemble', ['CRPS', 'twCRPS', 'Brier', 'BSS', 'ens_spread'])
    else:
        return scores_cfg.get('deterministic', ['ETS', 'PSS', 'bias', 'rmse'])
