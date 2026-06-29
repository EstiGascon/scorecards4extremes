"""
STEP 5: CALCULATE THRESHOLD FOR EXTREMES
=========================================
Methods:
  1. fixed              : User-specified value (e.g., -5°C)
  2. dataset_climatology: Percentile from the extracted observation data
  3. station_climatology: Per-station percentile from STVL observation climatology (quaver backend)
  4. area_mean_climatology: Area-mean of station clim percentile (quaver backend)
  5. model_percentile   : Percentile of one model's forecast distribution per station
  6. local_obs_climatology: Per-station per-month percentile from locally computed obs climatology files
                            (GEO NCOLS files produced by obs_clim_local/obsclim.py)
"""

import os
import numpy as np
import pandas as pd


def run_step5(config, data):
    """
    Execute Step 5: Calculate Threshold
    Returns threshold value (scalar or per-station Series) and event type
    """
    print("\n" + "="*80)
    print("STEP 5: CALCULATE THRESHOLD FOR EXTREMES")
    print("="*80)
    
    cfg = config['threshold']
    method = cfg['method']

    # event_type lives at the top level of the threshold block for all methods.
    # For backwards compatibility also check inside the fixed sub-block.
    if 'event_type' in cfg:
        event_type = cfg['event_type']
    elif method == 'fixed' and 'event_type' in cfg.get('fixed', {}):
        event_type = cfg['fixed']['event_type']
    else:
        event_type = cfg['event_type']  # will raise a clear KeyError if missing

    print(f"\nMethod: {method}")
    print(f"Event type: {event_type}")
    
    if method == 'station_climatology':
        backend = config.get('backend', 'local')
        if backend == 'quaver':
            import quaver_backend
            threshold, event_type = quaver_backend.compute_threshold_quaver(config, data)
        else:
            print("\n  [Station climatology not yet implemented for local backend]")
            print("  Falling back to dataset climatology...")
            percentile = cfg['station_climatology']['percentile']
            threshold = np.percentile(data['obs_value'], percentile)
            print(f"  {percentile}th percentile: {threshold:.2f}")
    
    elif method == 'fixed':
        threshold = cfg['fixed']['value']
        print(f"  Fixed threshold: {threshold:.2f}")
        if event_type == 'above':
            n_events = np.sum(data['obs_value'] > threshold)
        else:
            n_events = np.sum(data['obs_value'] < threshold)
        pct_events = 100 * n_events / len(data)
        print(f"  Events in dataset: {n_events} ({pct_events:.1f}%)")
    
    elif method == 'dataset_climatology':
        percentile = cfg['dataset_climatology']['percentile']
        use_filtered = cfg['dataset_climatology']['use_filtered_data']

        if use_filtered:
            print(f"  Using filtered data for climatology")

        if 'forecast_day' in data.columns:
            # Per-forecast-day threshold: pool obs for each forecast day separately
            # so each heatmap box (condition) gets its own climatological threshold.
            print(f"  Computing per-forecast-day p{percentile} from dataset observations")
            threshold = data.groupby('forecast_day')['obs_value'].transform(
                lambda x: np.nanpercentile(x, percentile)
            )
            for day, grp in data.groupby('forecast_day'):
                thr_day = float(np.nanpercentile(grp['obs_value'], percentile))
                if event_type == 'above':
                    n_ev = int(np.sum(grp['obs_value'] > thr_day))
                else:
                    n_ev = int(np.sum(grp['obs_value'] < thr_day))
                pct_ev = 100 * n_ev / len(grp)
                print(f"    Day {int(day)}: p{percentile} = {thr_day:.2f}, events = {n_ev} ({pct_ev:.1f}%)")
        else:
            # Fallback: single global threshold (no forecast_day column)
            threshold = np.percentile(data['obs_value'], percentile)
            print(f"  {percentile}th percentile threshold (global): {threshold:.2f}")
            if event_type == 'above':
                n_events = np.sum(data['obs_value'] > threshold)
            else:
                n_events = np.sum(data['obs_value'] < threshold)
            pct_events = 100 * n_events / len(data)
            print(f"  Events in dataset: {n_events} ({pct_events:.1f}%)")

    elif method == 'area_mean_climatology':
        import quaver_backend
        threshold, event_type = quaver_backend.compute_threshold_quaver(config, data)

    elif method == 'model_percentile':
        cfg_mp = cfg['model_percentile']
        percentile = cfg_mp['percentile']
        model = cfg_mp.get('model', 'fc1')
        col = f'{model}_value' if f'{model}_value' in data.columns else 'fc1_value'
        print(f"\n  Model percentile: p{percentile} of {model} at each station")

        if 'station_id' in data.columns:
            threshold = data.groupby('station_id')[col].transform(
                lambda x: np.nanpercentile(x, percentile)
            )
        else:
            threshold = np.nanpercentile(data[col], percentile)
            print(f"  (no station_id column — using global percentile)")

        mean_thr = float(np.nanmean(threshold))
        print(f"  Mean threshold across stations: {mean_thr:.2f}")

    elif method == 'local_obs_climatology':
        threshold = _compute_local_obs_climatology_threshold(config, data)

    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    print("\n✓ Step 5 complete")
    
    return threshold, event_type


# =============================================================================
# LOCAL OBS CLIMATOLOGY — helper functions
# =============================================================================

def _parse_geo_ncols(filepath):
    """
    Parse a Metview GEO NCOLS file (as written by obsclim.py) into a DataFrame.

    The file format is:
      #GEO
      #FORMAT NCOLS
      #COLUMNS
      stnid  latitude  longitude  level  date  time  height  q0 q1 ... q100
      # Missing values represented by 3e+38
      #METADATA
      ...
      #DATA
      <data rows>

    Returns a DataFrame indexed by stnid (int) with all quantile columns as floats.
    Missing values (3e38) are replaced with NaN.
    """
    rows = []
    header = None
    in_data = False
    with open(filepath, 'r') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('#DATA'):
                in_data = True
                continue
            if not in_data:
                if line.startswith('#COLUMNS'):
                    continue
                # The column header line immediately follows #COLUMNS (no leading #)
                if not line.startswith('#') and header is None and 'stnid' in line:
                    header = line.split()
                continue
            if line.startswith('#') or not line.strip():
                continue
            rows.append(line.split())

    if header is None or not rows:
        raise ValueError(f"Could not parse GEO NCOLS file: {filepath}")

    df = pd.DataFrame(rows, columns=header[:len(rows[0])])
    df['stnid'] = df['stnid'].astype(int)
    df = df.set_index('stnid')

    # Convert all quantile columns to float, replace missing (3e38)
    # Handles integer percentiles (q0..q100) and fractional ones (q99p5, q99p9)
    import re as _re
    quant_cols = [c for c in df.columns
                  if c.startswith('q') and (_re.match(r'^\d+$', c[1:]) or _re.match(r'^\d+p\d+$', c[1:]))]
    df[quant_cols] = df[quant_cols].astype(float).replace(3e38, np.nan)
    # Also replace values very close to 3e38 (floating point representation)
    for col in quant_cols:
        df.loc[df[col] > 1e37, col] = np.nan

    return df, quant_cols


def _load_clim_file_for_month(clim_dir, param, window_days, month, n_years, first_year, last_year, min_avail_pct):
    """
    Build the climatology filename for a given month and load it.

    Filename pattern: clim_{param}_{window}_{mm}_{nyears}years_{fyear}_{lyear}_{pct}
    e.g.: clim_tp_1_01_20years_2005_2024_65
    """
    mm = f"{month:02d}"
    fname = (f"clim_{param}_{window_days}_{mm}_{n_years}years"
             f"_{first_year}_{last_year}_{min_avail_pct}")
    fpath = os.path.join(clim_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Local obs climatology file not found: {fpath}\n"
            f"  Expected pattern: clim_{param}_<window>_<MM>_<N>years_<Y1>_<Y2>_<pct>"
        )
    return _parse_geo_ncols(fpath)


def _compute_local_obs_climatology_threshold(config, data):
    """
    Compute a per-row threshold from locally produced obs climatology files.

    Config keys (under threshold.local_obs_climatology):
      path            : directory containing the clim_* files
      parameter       : e.g. 'tp'
      window_days     : aggregation window in days (1 for tp24)
      n_years         : number of years in climatology (e.g. 20)
      first_year      : first year of climatology period (e.g. 2005)
      last_year       : last year of climatology period (e.g. 2024)
      min_availability_pct : minimum data availability % used when generating files (e.g. 65)
      percentile      : which percentile to use as threshold (e.g. 99)
      max_match_dist  : max lat/lon distance (degrees) for station matching (default 0.1)

    Station matching is done by nearest lat/lon (within max_match_dist) because the
    parquet pipeline uses internal sequential station IDs (e.g. 'S168') while the
    climatology files use WMO station IDs. Stations with no nearby clim entry get NaN.

    The data DataFrame must have:
      - 'station_id' column
      - 'lat' and 'lon' columns
      - 'date' column (YYYYMMDD string)

    Returns a pd.Series aligned to data.index with per-row thresholds.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("scipy is required for local_obs_climatology (lat/lon matching)")

    cfg = config['threshold']['local_obs_climatology']
    clim_dir     = cfg['path']
    param        = cfg.get('parameter', 'tp')
    window_days  = cfg.get('window_days', 1)
    n_years      = cfg.get('n_years', 20)
    first_year   = str(cfg.get('first_year', 2005))
    last_year    = str(cfg.get('last_year', 2024))
    min_avail    = int(cfg.get('min_availability_pct', 65))
    _pct_raw     = cfg.get('percentile', 99)
    percentile   = float(_pct_raw)
    max_dist     = float(cfg.get('max_match_dist', 0.1))
    # Build column key: integer percentiles → 'q99', fractional → 'q99p5'
    if percentile == int(percentile):
        quant_col = f'q{int(percentile)}'
    else:
        quant_col = f'q{percentile}'.replace('.', 'p')

    # Derive month from the VALID time (init date + step), not the init date.
    # Using the init-date month is wrong at long lead times where the valid date
    # crosses a month boundary (e.g. init Nov-28 + 120h = valid Dec-3 → need Dec clim).
    if 'step' in data.columns:
        date_col = data['date']
        if pd.api.types.is_datetime64_any_dtype(date_col):
            init_dates = date_col
        else:
            init_dates = pd.to_datetime(date_col.astype(str), format='%Y%m%d')
        valid_dates = init_dates + pd.to_timedelta(data['step'], unit='h')
        months = valid_dates.dt.month
    else:
        # Fallback: use init date month (no step available)
        date_col = data['date']
        if pd.api.types.is_datetime64_any_dtype(date_col):
            months = date_col.dt.month
        else:
            months = date_col.astype(str).str[4:6].astype(int)

    threshold_vals = np.full(len(data), np.nan)
    clim_cache = {}
    missing_months = set()

    for month in sorted(months.unique()):
        row_mask = (months == month).values
        if month not in clim_cache:
            try:
                df_clim, _ = _load_clim_file_for_month(
                    clim_dir, param, window_days, month,
                    n_years, first_year, last_year, min_avail
                )
                # Build a KD-tree over clim stations (lat/lon)
                clim_lats = df_clim['latitude'].astype(float).values
                clim_lons = df_clim['longitude'].astype(float).values
                clim_tree = cKDTree(np.column_stack([clim_lats, clim_lons]))
                clim_cache[month] = (df_clim, clim_tree)
            except FileNotFoundError as e:
                missing_months.add(month)
                print(f"  WARNING: {e}")
                continue

        df_clim, clim_tree = clim_cache[month]

        if quant_col not in df_clim.columns:
            print(f"  WARNING: column '{quant_col}' not in climatology file (month {month:02d})")
            continue

        # For each row in this month, find the threshold via nearest clim station.
        # NOTE: station_id in the parquet is a per-date row index, NOT a persistent
        # station identifier, so it cannot be used to deduplicate lat/lon lookups.
        # We use the actual per-row lat/lon directly.
        row_indices = np.where(row_mask)[0]
        row_lats    = data['lat'].iloc[row_indices].values
        row_lons    = data['lon'].iloc[row_indices].values

        dists, nn_idxs = clim_tree.query(np.column_stack([row_lats, row_lons]))

        # Map distance → threshold; set NaN where too far
        row_thresholds = np.where(
            dists <= max_dist,
            df_clim[quant_col].values[nn_idxs],
            np.nan
        )

        threshold_vals[row_indices] = row_thresholds

    result = pd.Series(threshold_vals, index=data.index)

    n_total   = len(data)
    n_matched = result.notna().sum()
    unique_months = sorted(months.unique())
    print(f"\n  Local obs climatology: p{percentile} from {param} clim files")
    print(f"  Clim dir:   {clim_dir}")
    print(f"  Window:     {window_days} day(s),  period: {first_year}–{last_year}")
    print(f"  Matching:   lat/lon nearest-neighbour, max dist {max_dist}°")
    print(f"  Months in data: {[f'{m:02d}' for m in unique_months]}")
    if missing_months:
        print(f"  WARNING: no clim files for months: {sorted(missing_months)}")
    print(f"  Rows matched: {n_matched}/{n_total}  ({100*n_matched/n_total:.1f}%)")
    n_unmatched_stns = result.isna().groupby(data['station_id']).all().sum()
    print(f"  Stations with no clim match: {n_unmatched_stns}")

    valid = threshold_vals[~np.isnan(threshold_vals)]
    if len(valid):
        print(f"  Threshold stats — mean: {np.mean(valid):.2f},  median: {np.median(valid):.2f}")

    return result
