#!/usr/bin/env python3
"""
Scorecards4Extremes - Main Workflow Runner
===========================================

Simple workflow-based tool for extreme weather verification

USAGE:
  1. Edit config.yaml with your options for each step
  2. Run: python run.py
  3. Done!

WORKFLOW STEPS:
  Step 1: Read Data (Quaver/local GRIB, VINO/local .gpt)
  Step 2: Pre-process Data (wind speed, lapse-rate, etc.)
  Step 3: Extract Point Data (nearest gridpoint)
  Step 4: Filter Data (lead time, season, orography, quality)
  Step 5: Calculate Threshold (station/fixed/dataset climatology)
  Step 6: Calculate Verification Scores (ETS, PSS, twMAE, etc.)
  Step 7: Bootstrap Significance Testing
  Step 8: Save Results (CSV files)
  Step 9: Plot Results (heatmaps, summary)
"""

import os
import sys
import subprocess
from pathlib import Path

# Set up TMPDIR automatically if not already set
if 'TMPDIR' not in os.environ:
    # Try ECMWF location first, then fall back to standard tmp
    user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
    ecmwf_tmp = f'/ec/res4/scratch/{user}/tmp'
    if os.path.exists('/ec/res4/scratch'):
        tmpdir = ecmwf_tmp
    else:
        tmpdir = f'/tmp/{user}'
    
    # Create directory if it doesn't exist
    os.makedirs(tmpdir, exist_ok=True)
    os.environ['TMPDIR'] = tmpdir
    print(f"✓ TMPDIR automatically set to: {tmpdir}")

# Set up environment for metview before any imports
if 'METVIEW_PYTHON_START_TIMEOUT' not in os.environ:
    os.environ['METVIEW_PYTHON_START_TIMEOUT'] = '30'

# Try to load modules using modulecmd
try:
    # Load ecmwf-toolbox/new module
    subprocess.run(['modulecmd', 'python', 'load', 'ecmwf-toolbox/new'], 
                   capture_output=True, check=False)
    # Load python3 module
    subprocess.run(['modulecmd', 'python', 'load', 'python3'], 
                   capture_output=True, check=False)
except Exception:
    # Modules might already be loaded or not needed
    pass

import yaml
import pandas as pd
from pathlib import Path

# Import workflow steps
import read_data
import preprocess
import filter
import threshold as threshold_module
import det_scores
import bootstrap
import save
import plot
import season_utils

# extract_points imports Metview at module level. Defer the import so that
# skip-extraction runs don't require Metview to be available.
extract_points = None
def _get_extract_points():
    global extract_points
    if extract_points is None:
        import extract_points as _ep
        extract_points = _ep
    return extract_points

# Ensemble modules (optional - only needed for ensemble mode)
try:
    import extract_points_ensemble
    ENSEMBLE_EXTRACT_AVAILABLE = True
except ImportError:
    ENSEMBLE_EXTRACT_AVAILABLE = False

try:
    import ens_scores
    ENS_SCORES_AVAILABLE = True
except ImportError:
    ENS_SCORES_AVAILABLE = False

ENSEMBLE_AVAILABLE = ENSEMBLE_EXTRACT_AVAILABLE  # extraction is the hard requirement

# Quaver/VTB backend (optional - only needed when backend='quaver')
try:
    import quaver_backend
    QUAVER_AVAILABLE = True
except ImportError:
    QUAVER_AVAILABLE = False

# Quaver compute backend (optional - uses native compute() API)
try:
    import quaver_compute_backend
    QUAVER_COMPUTE_AVAILABLE = True
except ImportError:
    QUAVER_COMPUTE_AVAILABLE = False

# Quaver/VTB extraction-only backend (optional - backend='quaver_extract')
# Implements ONLY step 3 (point extraction -> parquet); scoring stays local.
try:
    import quaver_extract
    QUAVER_EXTRACT_AVAILABLE = True
except ImportError:
    QUAVER_EXTRACT_AVAILABLE = False


def load_config(config_file='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"\nERROR: {config_file} not found!")
        print("Make sure config.yaml is in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR loading config: {e}")
        sys.exit(1)


def print_header():
    """Print tool header"""
    print("\n" + "="*80)
    print("SCORECARDS FOR EXTREMES - WORKFLOW RUNNER")
    print("="*80)
    print("\nThis tool follows a 9-step workflow for extreme weather verification")
    print("Each step has options you can configure in config.yaml")
    print("="*80)


def print_config_summary(config):
    """Print configuration summary"""
    print("\nConfiguration Summary:")
    print(f"  Variable: {config['variable']}")
    print(f"  Backend: {config.get('backend', 'local')}")
    print(f"  Dates: {config['start_date']} to {config['end_date']}")
    
    # Handle both steps and forecast_days
    if 'forecast_days' in config and config['forecast_days']:
        freq = config.get('lead_time_frequency', 24)
        print(f"  Forecast days: {config['forecast_days']} (every {freq}h)")
    elif 'steps' in config:
        print(f"  Steps: {config['steps']}")
    
    print(f"  Forecast Model 1: {config['read_data']['forecast_model1']['name']}")
    print(f"  Forecast Model 2: {config['read_data']['forecast_model2']['name']}")
    # quaver_extract backend uses read_data.quaver_obs instead of observation_source
    obs_src = config['read_data'].get('observation_source')
    if obs_src is None and 'quaver_obs' in config['read_data']:
        obs_src = "quaver (" + ", ".join(config['read_data']['quaver_obs'].get('sources', [])) + ")"
    print(f"  Observation source: {obs_src}")
    print(f"  Threshold method: {config['threshold']['method']}")
    print(f"  Output: {config['save']['output_directory']}")


def _aggregate_to_daily_mean(data, threshold_value, model_names, config=None):
    """
    Aggregate sub-daily forecast/obs data to daily means.

    Groups by (lat, lon, date, forecast_day) where
      forecast_day = ((step - 1) // 24) + 1

    This is the correct grouping because det_scores.py stratifies by forecast_day.
    Returns (agg_df, new_threshold_series).
    """
    import pandas as pd

    data = data.copy()
    data['forecast_day'] = ((data['step'] - 1) // 24).astype(int) + 1

    group_cols = ['lat', 'lon', 'date', 'forecast_day']

    agg_cols = ['obs_value', 'fc1_value', 'fc2_value',
                'fc1_value_uncorrected', 'fc2_value_uncorrected']
    agg_cols = [c for c in agg_cols if c in data.columns]

    # Include ensemble member columns (fc1_member_*, fc2_member_*) in the mean
    # aggregation so that daily-mean member values are used when the threshold is
    # a daily-mean climatological percentile (local_obs_climatology).
    member_cols = [c for c in data.columns
                   if c.startswith('fc1_member_') or c.startswith('fc2_member_')]
    agg_cols = agg_cols + member_cols

    # step gets the mean; all other non-group, non-agg columns keep first value
    meta_cols = [c for c in data.columns
                 if c not in agg_cols and c not in group_cols and c != 'step']

    agg_dict = {col: (col, 'mean') for col in agg_cols}
    agg_dict['step'] = ('step', 'mean')
    for col in meta_cols:
        agg_dict[col] = (col, 'first')

    agg = data.groupby(group_cols, sort=False).agg(**agg_dict).reset_index()

    # Use the canonical midpoint step for each forecast_day instead of the
    # actual mean, which varies between orography types when some groups have
    # fewer sub-daily obs (e.g. complex terrain missing step 24 gives mean 14
    # instead of 15). Formula: day * 24 - (24 - freq) / 2  where freq is the
    # sub-daily step interval.  For freq=6: day*24 - 9 → 15, 39, 63, 87, 111.
    freq = int(config.get('lead_time_frequency', 6)) if config else 6
    agg['step'] = (agg['forecast_day'] * 24 - (24 - freq) // 2).astype(int)

    # Re-align the per-station threshold Series to the aggregated index.
    if isinstance(threshold_value, pd.Series):
        thr_df = data[group_cols].copy()
        thr_df['_thr'] = threshold_value.values
        thr_map = thr_df.groupby(group_cols)['_thr'].first().reset_index()
        agg = agg.merge(thr_map, on=group_cols, how='left')
        new_threshold = pd.Series(agg['_thr'].values, index=agg.index)
        agg = agg.drop(columns=['_thr'])
    else:
        new_threshold = threshold_value

    agg = agg.reset_index(drop=True)
    if isinstance(new_threshold, pd.Series):
        new_threshold = pd.Series(new_threshold.values, index=agg.index)

    print(f"    Daily aggregation: {len(data):,} sub-daily rows → {len(agg):,} daily rows")

    return agg, new_threshold


def main():
    """Main workflow execution"""
    print_header()
    
    # Load configuration from command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        print("\nERROR: No config file specified.")
        print("Usage: python run.py <config_file.yaml>")
        print("Example: python run.py config_tp24_local_p99obsclim.yaml")
        sys.exit(1)
    
    print(f"\nLoading configuration from {config_file}...")
    config = load_config(config_file)
    
    print_config_summary(config)
    
    # input("\nPress Enter to start workflow...")
    
    # ====================================================================
    # MULTI-REGION SUPPORT
    # ====================================================================
    # Check if user wants to process multiple regions
    extract_cfg = config.get('extract_points', {})
    area_config = extract_cfg.get('area', None)
    
    # Determine which areas to process
    if isinstance(area_config, list):
        # Multiple areas specified - process each one
        areas_to_process = area_config
        print(f"\n🌍 Multiple regions detected: {areas_to_process}")
        print(f"   Will process each region separately and save results independently\n")
    elif area_config is not None:
        # Single area specified
        areas_to_process = [area_config]
    else:
        # No area specified - global
        areas_to_process = [None]
    
    # Store original config values to restore between iterations
    original_output_path = extract_cfg.get('output_path', './extracted_points')
    original_result_dir = config['save']['output_directory']
    
    # Loop through each area
    for area_idx, current_area in enumerate(areas_to_process):
        if len(areas_to_process) > 1:
            print("\n" + "="*80)
            print(f"🌍 PROCESSING REGION {area_idx + 1}/{len(areas_to_process)}: {current_area or 'GLOBAL'}")
            print("="*80)
        
        # Update config with current area
        config['extract_points']['area'] = current_area
        
        # Update output paths to include area suffix
        if current_area and len(areas_to_process) > 1:
            area_suffix = f"_{current_area}" if isinstance(current_area, str) else "_custom"
            config['extract_points']['output_path'] = f"{original_output_path}{area_suffix}"
            config['save']['output_directory'] = f"{original_result_dir}{area_suffix}"
        else:
            # Single area or None - use original paths
            config['extract_points']['output_path'] = original_output_path
            config['save']['output_directory'] = original_result_dir
    
        try:
            # ====================================================================
            # STEP 1: READ DATA
            # ====================================================================
            paths = read_data.run_step1(config)
            
            # ====================================================================
            # STEP 2: PRE-PROCESS DATA
            # ====================================================================
            preprocess_settings = preprocess.run_step2(config, paths)
            
            # ====================================================================
            # CHECK MODE: ENSEMBLE or DETERMINISTIC
            # ====================================================================
            run_mode = config.get('mode', 'deterministic')
            
            if run_mode == 'ensemble':
                # ============================================================
                # ENSEMBLE WORKFLOW (Steps 3-9)
                # ============================================================
                backend = config.get('backend', 'local')
                if backend == 'quaver_extract' and not QUAVER_EXTRACT_AVAILABLE:
                    print("\nERROR: backend='quaver_extract' but quaver_extract.py not importable "
                          "(check vtb/metview modules)")
                    sys.exit(1)
                if not ENSEMBLE_AVAILABLE and backend not in ('quaver', 'quaver_compute', 'quaver_extract'):
                    print("\nERROR: Ensemble extraction module not found (extract_points_ensemble.py)")
                    sys.exit(1)
                if backend == 'quaver' and not QUAVER_AVAILABLE:
                    print("\nERROR: backend='quaver' but quaver_backend.py not importable (check vtb/quaver modules)")
                    sys.exit(1)
                if backend == 'quaver_compute' and not QUAVER_COMPUTE_AVAILABLE:
                    print("\nERROR: backend='quaver_compute' but quaver_compute_backend.py not importable")
                    sys.exit(1)
                
                # --- Quaver Compute backend: use native compute() API ---
                if backend == 'quaver_compute':
                    model_names = {
                        'fc1_name': paths['fc1_name'],
                        'fc2_name': paths['fc2_name']
                    }
                    results = quaver_compute_backend.run_quaver_compute_workflow(config, model_names)
                    
                    # Plot
                    output_dir = Path(config.get('save', {}).get('output_directory', './results'))
                    threshold_value = results[0].get('threshold_value', 0) if results else 0
                    if len(results) > 1:
                        plot.run_step9(config, results, threshold_value, output_dir, model_names, season=None)
                    elif len(results) == 1:
                        plot.run_step9(config, results[0], threshold_value, output_dir, model_names, results[0].get('season'))
                    
                    print("\n" + "=" * 80)
                    print("✓ ENSEMBLE WORKFLOW COMPLETE! (quaver_compute)")
                    print("=" * 80)
                    continue
                
                print("\n" + "="*80)
                print(f"ENSEMBLE VERIFICATION MODE (backend={backend})")
                print("="*80)
                
                model_names = {
                    'fc1_name': paths['fc1_name'],
                    'fc2_name': paths['fc2_name']
                }
                
                # STEP 3 (ensemble): Extract ensemble members
                skip_extraction = config.get('skip_extraction_if_exists', False)
                variable = config['variable']
                fc1_name = paths['fc1_name']
                fc2_name = paths['fc2_name']
                ens_output_path = Path(config['extract_points'].get('output_path', f'./extracted_points/{variable}_ens'))
                ens_filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}_ens"
                
                ens_files_exist = False
                if skip_extraction:
                    day_files = sorted(ens_output_path.glob(f"{ens_filename_base}_day*.parquet"))
                    ens_files_exist = len(day_files) > 0
                
                if ens_files_exist:
                    print(f"\n✓ Skipping ensemble extraction - {len(day_files)} files found")
                    point_data_path = {
                        'output_path': ens_output_path,
                        'save_format': 'pandas',
                        'fc1_name': fc1_name,
                        'fc2_name': fc2_name,
                        'ensemble': True,
                    }
                elif backend == 'quaver':
                    point_data_path = quaver_backend.extract_points_quaver(config, paths, preprocess_settings)
                elif backend == 'quaver_extract':
                    point_data_path = quaver_extract.run_step3_ensemble(config, paths, preprocess_settings)
                else:
                    point_data_path = extract_points_ensemble.run_step3_ensemble(config, paths, preprocess_settings)
                
                # Parse seasons and orography config (needed before skip check)
                seasons_config = config['filter'].get('season', None)
                seasons_to_process = season_utils.parse_seasons_config(seasons_config)

                orog_config = config['filter'].get('orography_type', None)
                if isinstance(orog_config, list):
                    orography_types = orog_config
                elif orog_config == 'all' or orog_config is None:
                    orography_types = [None]
                else:
                    orography_types = [orog_config]

                # ---- Skip scoring if existing CSVs are present ----
                skip_scoring = config.get('skip_scoring_if_exists', False)
                all_season_orog_results = []
                skip_to_plotting = False
                output_dir = Path(config['save']['output_directory'])

                if skip_scoring:
                    def _find_ens_csv(d, pattern):
                        matches = sorted(d.glob(pattern))
                        return matches[0] if matches else None

                    all_files_ok = True
                    for _s in seasons_to_process:
                        _sl, _ = season_utils.resolve_season(_s)
                        _ss = f"_{_sl}" if _sl else ""
                        for _o in orography_types:
                            _os = f"_{_o}" if _o else ""
                            _tail = f"*{_ss}{_os}.csv"
                            if not (_find_ens_csv(output_dir, f"scores_by_leadtime_{variable}{_tail}") and
                                    _find_ens_csv(output_dir, f"overall_scores_{variable}{_tail}")):
                                all_files_ok = False
                                break
                        if not all_files_ok:
                            break

                    if all_files_ok:
                        print("\n" + "="*80)
                        print("STEPS 4-8: LOADING EXISTING ENSEMBLE RESULTS")
                        print("="*80)
                        print(f"\n✓ Skipped - result files already exist in: {output_dir}")
                        print("  Loading existing scores for plotting...")
                        print("  To recalculate, set 'skip_scoring_if_exists: false' in config")

                        for _s in seasons_to_process:
                            _sl, _ = season_utils.resolve_season(_s)
                            _ss = f"_{_sl}" if _sl else ""
                            for _o in orography_types:
                                _os = f"_{_o}" if _o else ""
                                _tail = f"*{_ss}{_os}.csv"
                                _by_lt_f = _find_ens_csv(output_dir, f"scores_by_leadtime_{variable}{_tail}")
                                _overall_f = _find_ens_csv(output_dir, f"overall_scores_{variable}{_tail}")
                                _by_lt = pd.read_csv(_by_lt_f)
                                _overall = pd.read_csv(_overall_f)
                                _thr = (float(_by_lt['threshold'].iloc[0])
                                        if 'threshold' in _by_lt.columns
                                        else config['threshold'].get('fixed', {}).get('value', 0.0))
                                all_season_orog_results.append({
                                    'orog_type': _o,
                                    'data': None,
                                    'results': {'by_leadtime': _by_lt},
                                    'overall_scores': _overall.to_dict('records')[0] if len(_overall) > 0 else {},
                                    'season': _sl,
                                    'threshold_value': _thr,
                                })
                        threshold_value = all_season_orog_results[0]['threshold_value']
                        skip_to_plotting = True

                if not skip_to_plotting:
                    import gc as _gc
                    import re as _re
                    from collections import defaultdict as _defaultdict
                    _gc.collect()

                    # STEP 4 (ensemble): Discover per-day parquet files
                    ens_out = point_data_path['output_path']
                    ens_pattern = f"{variable}_{fc1_name}_vs_{fc2_name}_ens_*day*.parquet"
                    ens_day_files = sorted(Path(ens_out).glob(ens_pattern))

                    if not ens_day_files:
                        print(f"\nERROR: No ensemble data files found: {ens_out / ens_pattern}")
                        raise FileNotFoundError(f"No ensemble parquet files in {ens_out}")

                    print(f"\n  Found {len(ens_day_files)} ensemble day files")
                    print("  Streaming one file at a time to minimise peak memory usage")

                    score_backend = config.get('scores', {}).get('backend', config.get('backend', 'local'))
                    threshold_method_ens = config.get('threshold', {}).get('method', 'fixed')

                    # --- Pre-compute global threshold for dataset_climatology ---
                    # For local_obs_climatology/fixed/station_climatology the threshold
                    # can be computed per-file since it doesn't depend on the full obs
                    # distribution of the current forecast period.
                    _global_threshold_by_so = {}
                    if threshold_method_ens == 'dataset_climatology':
                        print("\n  Pre-loading obs for dataset_climatology threshold ...")
                        _obs_meta = ['date', 'step', 'obs_value', 'lat', 'lon',
                                     'sdfor', 'lsm', 'forecast_day']
                        _obs_dfs = []
                        for _f in ens_day_files:
                            _avail = pd.read_parquet(_f, columns=[]).columns.tolist()
                            _cols = [c for c in _obs_meta if c in _avail]
                            _obs_dfs.append(pd.read_parquet(_f, columns=_cols))
                        _all_obs = pd.concat(_obs_dfs, ignore_index=True)
                        del _obs_dfs; _gc.collect()
                        for _s in seasons_to_process:
                            _sl, _sml = season_utils.resolve_season(_s)
                            if _sml and 'date' in _all_obs.columns:
                                _sm = pd.to_datetime(_all_obs['date'], format='%Y%m%d').dt.month
                                _sd = _all_obs[_sm.isin(_sml)]
                            else:
                                _sd = _all_obs
                            for _ot in orography_types:
                                if _ot:
                                    _or = config['filter'].get('orography_ranges', {})
                                    if _ot in _or and 'sdfor' in _sd.columns:
                                        _lo, _hi = _or[_ot]
                                        _od = _sd[(_sd['sdfor'] >= _lo) & (_sd['sdfor'] < _hi)].reset_index(drop=True)
                                    else:
                                        _od = _sd.reset_index(drop=True)
                                else:
                                    _od = _sd.reset_index(drop=True)
                                if len(_od) > 0:
                                    _t, _et = threshold_module.run_step5(config, _od)
                                    _global_threshold_by_so[(_sl, _ot)] = (_t, _et)
                        del _all_obs; _gc.collect()
                        print("  ✓ Threshold pre-computed for all season/orography combinations")

                    # --- Per-variable QC parameters ---
                    _do_2t_qc = (config.get('variable') == '2t')
                    if _do_2t_qc:
                        _min_temp = config['filter'].get('min_valid_temperature', -60.0)
                        _max_temp = config['filter'].get('max_valid_temperature', 60.0)
                    _do_outlier = config['filter'].get('remove_outliers', False)
                    _outlier_std = config['filter'].get('outlier_threshold_std', 5.0)

                    # Accumulator: (season_label, orog_type) → list of day_scores dicts
                    _day_results_accum = _defaultdict(list)
                    # Default event_type from config (overwritten by run_step5 calls below)
                    _evt = config.get('threshold', {}).get('event_type', 'above')

                    # --- Stream one day file at a time ---
                    print("\n" + "="*70)
                    print("STEPS 4-6 (ensemble): per-day streaming")
                    print("="*70)

                    for _fi, _f in enumerate(ens_day_files):
                        _m = _re.search(r'_day(\d+)\.parquet$', _f.name)
                        if not _m:
                            print(f"  WARNING: cannot parse day from {_f.name}, skipping")
                            continue
                        _day_num = int(_m.group(1))
                        print(f"\n  [{_fi+1}/{len(ens_day_files)}] {_f.name} ...", flush=True)

                        _df = pd.read_parquet(_f)
                        for _col in _df.select_dtypes(include='float64').columns:
                            _df[_col] = _df[_col].astype('float32')
                        print(f"    {len(_df):,} rows", flush=True)

                        # 2t physical bounds QC
                        if _do_2t_qc:
                            _nb = len(_df)
                            _vm = ((_df['obs_value'] >= _min_temp) &
                                   (_df['obs_value'] <= _max_temp))
                            for _mc in _df.columns:
                                if _mc.startswith('fc1_member_') or _mc.startswith('fc2_member_'):
                                    _vm &= (_df[_mc] >= _min_temp) & (_df[_mc] <= _max_temp)
                            _df = _df[_vm].reset_index(drop=True)
                            if len(_df) < _nb:
                                print(f"    2t QC: {_nb:,} → {len(_df):,}")

                        # Outlier filter
                        if _do_outlier and len(_df) > 0:
                            _obs_m = _df['obs_value'].mean()
                            _obs_s = _df['obs_value'].std()
                            _nb = len(_df)
                            _df = _df[
                                (_df['obs_value'] - _obs_m).abs() <= _outlier_std * _obs_s
                            ].reset_index(drop=True)
                            if len(_df) < _nb:
                                print(f"    Outlier filter: {_nb:,} → {len(_df):,}")

                        for _season in seasons_to_process:
                            _season_label, _season_months_list = season_utils.resolve_season(_season)
                            if _season_months_list and 'date' in _df.columns:
                                _dm = pd.to_datetime(_df['date'], format='%Y%m%d').dt.month
                                _sd = _df[_dm.isin(_season_months_list)].reset_index(drop=True)
                            else:
                                _sd = _df.copy()
                            if len(_sd) == 0:
                                continue

                            for _orog_type in orography_types:
                                if _orog_type:
                                    _orog_ranges = config['filter'].get('orography_ranges', {})
                                    if _orog_type in _orog_ranges and 'sdfor' in _sd.columns:
                                        _lo, _hi = _orog_ranges[_orog_type]
                                        _od = _sd[(_sd['sdfor'] >= _lo) &
                                                  (_sd['sdfor'] < _hi)].reset_index(drop=True)
                                    else:
                                        _od = _sd.copy()
                                else:
                                    _od = _sd.copy()
                                if len(_od) == 0:
                                    continue

                                # STEP 5b (dataset_climatology before threshold)
                                if (threshold_method_ens == 'dataset_climatology' and
                                        config.get('lead_time_frequency', 24) < 24):
                                    _od, _ = _aggregate_to_daily_mean(_od, None, model_names, config)

                                # STEP 5: Threshold
                                if threshold_method_ens == 'dataset_climatology':
                                    _so_key = (_season_label, _orog_type)
                                    if _so_key not in _global_threshold_by_so:
                                        continue
                                    _thr_val, _evt = _global_threshold_by_so[_so_key]
                                else:
                                    _thr_val, _evt = threshold_module.run_step5(config, _od)

                                # STEP 5b (local_obs_climatology after threshold)
                                if threshold_method_ens == 'local_obs_climatology':
                                    _od, _thr_val = _aggregate_to_daily_mean(
                                        _od, _thr_val, model_names, config)

                                # STEP 6: Score this single day
                                if score_backend == 'quaver' or not ENS_SCORES_AVAILABLE:
                                    print("  ERROR: quaver backend does not support per-day "
                                          "streaming; set backend to 'local'")
                                    sys.exit(1)

                                _day_score = ens_scores.score_single_day_file(
                                    config, _day_num, _od,
                                    _thr_val, _evt, model_names)
                                if _day_score is not None:
                                    _day_results_accum[(_season_label, _orog_type)].append(
                                        _day_score)
                                    print(f"    {_season_label or 'all'} / "
                                          f"{_orog_type or 'all'}: "
                                          f"n={_day_score['n_samples']:,} ✓",
                                          flush=True)

                        del _df; _gc.collect()

                    # --- STEP 8: Aggregate per-day results and save ---
                    print("\n" + "="*70)
                    print("STEP 8: Aggregating and saving ensemble results")
                    print("="*70)

                    for _season in seasons_to_process:
                        _season_label, _ = season_utils.resolve_season(_season)
                        if _season_label:
                            print(f"\n{'='*70}")
                            print(f"SEASON: {_season_label}")
                            print(f"{'='*70}")

                        for _orog_type in orography_types:
                            if _orog_type:
                                print(f"\n  Orography: {_orog_type.upper()}")

                            _so_key = (_season_label, _orog_type)
                            _day_results = _day_results_accum.get(_so_key, [])
                            if not _day_results:
                                print(f"    No results for season={_season_label}, "
                                      f"orog={_orog_type}")
                                continue

                            overall_scores, results = ens_scores.aggregate_day_results(
                                _day_results, model_names)
                            threshold_value = _day_results[0]['threshold']
                            event_type = _evt  # from last iteration; same for all

                            output_dir = save.run_step8(
                                config, overall_scores, results,
                                _orog_type, _season_label, threshold_value)

                            all_season_orog_results.append({
                                'orog_type': _orog_type,
                                'data': None,
                                'results': results,
                                'overall_scores': overall_scores,
                                'season': _season_label,
                                'threshold_value': threshold_value,
                            })
                
                # Save observation counts summary
                save.save_observation_counts(all_season_orog_results, output_dir, config)

                # Reorganize results to group by orography type (all seasons per orog),
                # matching the deterministic scorecard layout: flat DJF, flat MAM, …,
                # hilly DJF, hilly MAM, …, complex DJF, …
                if len(all_season_orog_results) > 1 and len(orography_types) > 1 and len(seasons_to_process) > 1:
                    reorganized_results = []
                    for orog_type in orography_types:
                        for result in all_season_orog_results:
                            if result['orog_type'] == orog_type:
                                reorganized_results.append(result)
                    all_season_orog_results = reorganized_results
                    print(f"✓ Reorganized: Grouped by orography type (all seasons per type)\n")

                # STEP 9: Plot
                if len(all_season_orog_results) > 1:
                    plot.run_step9(config, all_season_orog_results, threshold_value, output_dir, model_names, season=None)
                elif len(all_season_orog_results) == 1:
                    plot.run_step9(config, all_season_orog_results[0], threshold_value, output_dir, model_names, all_season_orog_results[0].get('season'))
                
                print("\n" + "="*80)
                print("✓ ENSEMBLE WORKFLOW COMPLETE!")
                print("="*80)
                print(f"\nResults saved to: {output_dir}")
                continue  # Skip deterministic workflow below
            
            # ====================================================================
            # DETERMINISTIC WORKFLOW (Steps 3-9) - original code below
            # ====================================================================
            backend = config.get('backend', 'local')

            if backend == 'quaver_extract' and not QUAVER_EXTRACT_AVAILABLE:
                print("\nERROR: backend='quaver_extract' but quaver_extract.py not importable "
                      "(check vtb/quaver modules, e.g. `module load quaver/3.6.4`)")
                sys.exit(1)

            # --- Quaver Compute backend: use native compute() API ---
            if backend == 'quaver_compute':
                if not QUAVER_COMPUTE_AVAILABLE:
                    print("\nERROR: backend='quaver_compute' but quaver_compute_backend.py not importable")
                    sys.exit(1)
                model_names = {
                    'fc1_name': paths['fc1_name'],
                    'fc2_name': paths['fc2_name']
                }
                results = quaver_compute_backend.run_quaver_compute_workflow(config, model_names)
                
                # Plot
                output_dir = Path(config.get('save', {}).get('output_directory', './results'))
                threshold_value = results[0].get('threshold_value', 0) if results else 0
                if len(results) > 1:
                    plot.run_step9(config, results, threshold_value, output_dir, model_names, season=None)
                elif len(results) == 1:
                    plot.run_step9(config, results[0], threshold_value, output_dir, model_names, results[0].get('season'))
                
                print("\n" + "=" * 80)
                print("✓ WORKFLOW COMPLETE! (quaver_compute)")
                print("=" * 80)
                continue
            
            # ====================================================================
            # STEP 3: EXTRACT POINT DATA
            # ====================================================================
            skip_extraction = config.get('skip_extraction_if_exists', False)
            variable = config['variable']
            fc1_name = paths['fc1_name']
            fc2_name = paths['fc2_name']

            if skip_extraction:
                # Only check for existing files if skip is enabled
                output_path = Path(config['extract_points']['output_path'])

                # Extraction output has no threshold suffix — threshold is applied at scoring time
                filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}"
                
                # Check for day-specific parquet files (the actual output format)
                forecast_days = config.get('forecast_days', [])
                if forecast_days:
                    existing_days = [d for d in forecast_days
                                     if (output_path / f"{filename_base}_day{d}.parquet").exists()]
                    missing_days  = [d for d in forecast_days if d not in existing_days]
                    all_files_exist = len(missing_days) == 0
                else:
                    # Fallback: check for consolidated file
                    parquet_file = output_path / f"{filename_base}.parquet"
                    all_files_exist = parquet_file.exists()
                    missing_days = []

                print("\n" + "="*80)
                print("STEP 3: EXTRACT POINT DATA")
                print("="*80)
                if all_files_exist:
                    print(f"\n✓ Skipped - extracted data already exists for forecast days: {forecast_days}")
                    print("  To re-extract, set 'skip_extraction_if_exists: false' in config")
                    point_data_path = {
                        'output_path': output_path,
                        'save_format': config['extract_points'].get('save_format', 'pandas'),
                        'fc1_name': fc1_name,
                        'fc2_name': fc2_name
                    }
                elif existing_days:
                    # Some days exist — only extract the missing ones
                    print(f"\n  Already extracted: days {existing_days}")
                    print(f"  Missing — will extract: days {missing_days}")
                    config_partial = dict(config)
                    config_partial['forecast_days'] = missing_days
                    if backend == 'quaver':
                        point_data_path = quaver_backend.extract_points_quaver(config_partial, paths, preprocess_settings)
                    elif backend == 'quaver_extract':
                        point_data_path = quaver_extract.run_step3(config_partial, paths, preprocess_settings)
                    else:
                        point_data_path = _get_extract_points().run_step3(config_partial, paths, preprocess_settings)
                    # Restore full forecast_days so scoring uses all days
                    point_data_path['forecast_days'] = forecast_days
                else:
                    # No files exist at all — full extraction
                    if backend == 'quaver':
                        point_data_path = quaver_backend.extract_points_quaver(config, paths, preprocess_settings)
                    elif backend == 'quaver_extract':
                        point_data_path = quaver_extract.run_step3(config, paths, preprocess_settings)
                    else:
                        point_data_path = _get_extract_points().run_step3(config, paths, preprocess_settings)
            else:
                # Skip is disabled, always extract from scratch
                if backend == 'quaver':
                    point_data_path = quaver_backend.extract_points_quaver(config, paths, preprocess_settings)
                elif backend == 'quaver_extract':
                    point_data_path = quaver_extract.run_step3(config, paths, preprocess_settings)
                else:
                    point_data_path = _get_extract_points().run_step3(config, paths, preprocess_settings)
            
            # ====================================================================
            # STEP 4: FILTER DATA
            # ====================================================================
            model_names = {
                'fc1_name': paths['fc1_name'],
                'fc2_name': paths['fc2_name']
            }

            # Check if we should evaluate multiple orography types
            orog_config = config['filter'].get('orography_type', None)
            if isinstance(orog_config, list):
                # List of orography types - evaluate each separately (will create multi-column heatmap)
                orography_types = orog_config
            elif orog_config == 'all' or orog_config is None:
                # No orography filtering - use all data together (single column)
                orography_types = [None]
            else:
                # Single orography type specified
                orography_types = [orog_config]
            
            # Get seasons from config - support processing multiple seasons
            seasons_config = config['filter'].get('season', None)
            # Handle both list and string format for season, including month groups
            seasons_to_process = season_utils.parse_seasons_config(seasons_config)
            
            print(f"\n{'='*80}")
            print(f"PROCESSING {len(seasons_to_process)} SEASON(S): {seasons_to_process}")
            print(f"{'='*80}")
            
            # Master list to collect ALL results (all seasons × all orography types)
            all_season_orog_results = []
            
            # Loop through each season
            for season_idx, season in enumerate(seasons_to_process):
                season_label, season_months_list = season_utils.resolve_season(season)
                if season_label:
                    print(f"\n{'='*80}")
                    print(f"SEASON {season_idx + 1}/{len(seasons_to_process)}: {season_label}")
                    print(f"{'='*80}\n")
                
                # Temporarily set season in config for this iteration
                original_season = config['filter'].get('season')
                config['filter']['season'] = season
                
                season_suffix = f"_{season_label}" if season_label else ""
                
                # Check if we should skip scoring and load existing results
                skip_scoring = config.get('skip_scoring_if_exists', False)
                skip_to_plotting = False
                
                if skip_scoring:
                    # Only check for existing files if skip is enabled
                    output_dir = Path(config['save']['output_directory'])
                    
                    # Files are saved with a threshold suffix (e.g. _8.0ms) that is not
                    # known yet at this point. Use glob to find matching files regardless
                    # of the threshold part of the filename.
                    def _find_csv(output_dir, pattern):
                        """Return the first file matching glob pattern, or None."""
                        matches = sorted(output_dir.glob(pattern))
                        return matches[0] if matches else None

                    # Check if all required result files exist
                    all_files_exist = True
                    for orog_type in orography_types:
                        orog_suffix = f"_{orog_type}" if orog_type else ""
                        tail = f"*{season_suffix}{orog_suffix}.csv"
                        if not (_find_csv(output_dir, f"scores_by_leadtime_{variable}{tail}") and
                                _find_csv(output_dir, f"overall_scores_{variable}{tail}")):
                            all_files_exist = False
                            break
                    
                    if all_files_exist:
                        print("\n" + "="*80)
                        print("STEPS 4-7: LOADING EXISTING RESULTS")
                        print("="*80)
                        print(f"\n✓ Skipped - result files already exist in: {output_dir}")
                        print("  Loading existing scores for plotting...")
                        print("  To recalculate, set 'skip_scoring_if_exists: false' in config")
                        
                        # Load existing results
                        all_results = []
                        for orog_type in orography_types:
                            orog_suffix = f"_{orog_type}" if orog_type else ""
                            tail = f"*{season_suffix}{orog_suffix}.csv"
                            
                            by_leadtime = pd.read_csv(_find_csv(output_dir, f"scores_by_leadtime_{variable}{tail}"))
                            overall = pd.read_csv(_find_csv(output_dir, f"overall_scores_{variable}{tail}"))
                            
                            all_results.append({
                                'orog_type': orog_type,
                                'data': None,  # Not needed for plotting
                                'results': {'by_leadtime': by_leadtime},
                                'overall_scores': overall.to_dict('records')[0] if len(overall) > 0 else {}
                            })
                        
                        # Get threshold value from the first result (for plotting)
                        if 'threshold' in all_results[0]['results']['by_leadtime'].columns:
                            threshold_value = all_results[0]['results']['by_leadtime']['threshold'].iloc[0]
                        else:
                            threshold_value = config['threshold'].get('fixed', {}).get('value', 30.0)
                        
                        # Add season label to loaded results for comprehensive plotting
                        for result in all_results:
                            result['season'] = season_label
                        
                        # Add all loaded results from this season to master list
                        all_season_orog_results.extend(all_results)
                        
                        skip_to_plotting = True
                # If skip_scoring is False or files don't exist, skip_to_plotting remains False
                
                if not skip_to_plotting:
                    # Store results for all orography types (for multi-column plotting)
                    all_results = []
                    
                    # Loop through orography types.
                    # Each call to run_step4 applies the orog filter DURING loading
                    # so only ~1/3 of rows are loaded at a time — avoids OOM.
                    for orog_type in orography_types:
                        if orog_type:
                            print(f"\n{'='*70}")
                            print(f"EVALUATING OROGRAPHY TYPE: {orog_type.upper()}")
                            print(f"{'='*70}\n")
                            original_orog = config['filter'].get('orography_type')
                            config['filter']['orography_type'] = orog_type
                        
                        data = filter.run_step4(config, point_data_path, preprocess_settings, model_names)
                        
                        # ====================================================================
                        # STEP 5b (dataset_climatology): DAILY AVERAGING BEFORE THRESHOLD
                        # When data is sub-daily (freq < 24h) and the threshold is a
                        # pooled dataset percentile, aggregate to daily means FIRST so
                        # the pooled percentile is derived from the same distribution
                        # that will be scored (daily means, not 6-hourly peaks).
                        # ====================================================================
                        if (config.get('threshold', {}).get('method') == 'dataset_climatology' and
                                config.get('lead_time_frequency', 24) < 24):
                            data, _ = _aggregate_to_daily_mean(data, None, model_names, config)
                            print("  ✓ Step 5b: aggregated sub-daily data to daily means (dataset_climatology)")

                        # ====================================================================
                        # STEP 5: CALCULATE THRESHOLD
                        # ====================================================================
                        threshold_value, event_type = threshold_module.run_step5(config, data)
                        
                        # ====================================================================
                        # STEP 5b: DAILY AVERAGING (local_obs_climatology only)
                        # The local obs climatology is built from daily means of all
                        # sub-daily obs.  When the parquet has sub-daily steps we must
                        # average obs and forecasts to daily means before scoring so
                        # the distribution matches the climatological percentiles.
                        # ====================================================================
                        if config.get('threshold', {}).get('method') == 'local_obs_climatology':
                            data, threshold_value = _aggregate_to_daily_mean(data, threshold_value, model_names, config)
                            print("  ✓ Step 5b: aggregated sub-daily data to daily means (local_obs_climatology)")

                        # ====================================================================
                        # STEP 5b (fixed threshold): DAILY AVERAGING
                        # Fixed thresholds (e.g. 8 m/s for 10ff) are typically derived
                        # from daily-mean climatologies.  When the parquet is sub-daily
                        # we must average to daily means before scoring so the
                        # distribution matches what the threshold represents.
                        # ====================================================================
                        if (config.get('threshold', {}).get('method') == 'fixed' and
                                config.get('lead_time_frequency', 24) < 24 and
                                not config.get('skip_daily_aggregation', False)):
                            data, threshold_value = _aggregate_to_daily_mean(data, threshold_value, model_names, config)
                            print("  ✓ Step 5b: aggregated sub-daily data to daily means (fixed threshold)")

                        # ====================================================================
                        # STEP 6: CALCULATE VERIFICATION SCORES
                        # ====================================================================
                        score_backend = config.get('scores', {}).get('backend', config.get('backend', 'local'))
                        if score_backend == 'quaver':
                            overall_scores, results = quaver_backend.compute_scores_quaver(
                                config, data, threshold_value, event_type, model_names, is_ensemble=False)
                        else:
                            overall_scores, results = det_scores.run_step6(config, data, threshold_value, event_type, model_names)
                        
                        # ====================================================================
                        # STEP 7: BOOTSTRAP SIGNIFICANCE TESTING
                        # ====================================================================
                        bootstrap_results = bootstrap.run_step7(config, data, threshold_value, event_type)
                        
                        # Free data before next orog type load
                        del data
                        import gc as _gc
                        _gc.collect()
                        
                        # Store for multi-column plotting
                        all_results.append({
                            'orog_type': orog_type,
                            'data': None,
                            'results': results,
                            'overall_scores': overall_scores
                        })
                        
                        if orog_type:
                            config['filter']['orography_type'] = original_orog
                    
                        # ====================================================================
                        # STEP 8: SAVE RESULTS
                        # ====================================================================
                        output_dir = save.run_step8(config, overall_scores, results, orog_type, season_label, threshold_value)
                
                    # Add season label to results for comprehensive plotting
                    for result in all_results:
                        result['season'] = season_label
                        result['threshold_value'] = threshold_value
                    
                    # Add all results from this season to master list
                    all_season_orog_results.extend(all_results)
                
                # Restore original season config
                config['filter']['season'] = original_season
                
                if season_label:
                    print(f"\n{'='*80}")
                    print(f"✓ SEASON COMPLETE: {season_label}")
                    print(f"{'='*80}\n")
            
            # ====================================================================
            # STEP 9: PLOT RESULTS (ALL SEASONS × ALL OROGRAPHIES)
            # ====================================================================
            print(f"\n{'='*80}")
            print(f"CREATING COMPREHENSIVE HEATMAPS")
            print(f"Conditions: {len(all_season_orog_results)} ({len(seasons_to_process)} seasons × {len(orography_types)} orographies)")
            print(f"{'='*80}\n")
            
            # Reorganize results to group by orography type (all seasons for LOW, then MID, then HIGH)
            # Instead of by season (all orographies for DJF, then MAM, etc.)
            if len(all_season_orog_results) > 1 and len(orography_types) > 1 and len(seasons_to_process) > 1:
                reorganized_results = []
                for orog_type in orography_types:
                    for result in all_season_orog_results:
                        if result['orog_type'] == orog_type:
                            reorganized_results.append(result)
                all_season_orog_results = reorganized_results
                print(f"✓ Reorganized: Grouped by orography type (all seasons per type)\n")
            
            # Save observation counts summary
            save.save_observation_counts(all_season_orog_results, output_dir, config)

            # Create comprehensive multi-column heatmap with all conditions
            if len(all_season_orog_results) > 1:
                plot.run_step9(config, all_season_orog_results, threshold_value, output_dir, model_names, season=None)
            elif len(all_season_orog_results) == 1:
                # Single result - use original single-column plot
                plot.run_step9(config, all_season_orog_results[0], threshold_value, output_dir, model_names, all_season_orog_results[0].get('season'))
            
            # ====================================================================
            # DONE WITH THIS REGION!
            # ====================================================================
            if len(areas_to_process) > 1:
                print("\n" + "="*80)
                print(f"✓ REGION COMPLETE: {current_area or 'GLOBAL'}")
                print("="*80)
                print(f"Results saved to: {output_dir}")
                print("="*80 + "\n")
            else:
                # Only one region - print full completion message
                print("\n" + "="*80)
                print("✓ WORKFLOW COMPLETE!")
                print("="*80)
                print(f"\nResults saved to: {output_dir}")
                print("\nWhat was done:")
                print("  ✓ Read forecast and observation data")
                print("  ✓ Pre-processed data (variable-specific)")
                print("  ✓ Extracted point forecasts at observation locations")
                print("  ✓ Filtered data by your criteria")
                print("  ✓ Calculated extreme threshold")
                print("  ✓ Computed verification scores")
                print("  ✓ Tested statistical significance")
                print("  ✓ Saved results to CSV")
                print("  ✓ Created plots")
                print("\n" + "="*80 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nWorkflow interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nERROR in region '{current_area}': {e}")
            import traceback
            traceback.print_exc()
            # Continue to next region if there are multiple, otherwise exit
            if len(areas_to_process) == 1:
                sys.exit(1)
            else:
                print(f"\n⚠️  Skipping region '{current_area}' due to error, continuing with next region...\n")
                continue
    
    # ====================================================================
    # ALL REGIONS COMPLETE - FINAL SUMMARY
    # ====================================================================
    if len(areas_to_process) > 1:
        print("\n" + "="*80)
        print("🎉 ALL REGIONS COMPLETE!")
        print("="*80)
        print(f"\nProcessed {len(areas_to_process)} regions:")
        for area in areas_to_process:
            area_suffix = f"_{area}" if area and isinstance(area, str) else "_custom" if area else ""
            result_dir = f"{original_result_dir}{area_suffix}"
            print(f"  ✓ {area or 'GLOBAL':20s} → {result_dir}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
