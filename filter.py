"""
STEP 4: FILTER DATA
===================
Filter extracted point data by:
  - Lead time
  - Season (DJF, MAM, JJA, SON)
  - Orography (low/mid/high terrain)
  - Quality control (outliers)
"""

import gc
import pandas as pd
import numpy as np
import re
import metview as mv
from pathlib import Path
from datetime import datetime, timedelta
import season_utils
from utils import format_threshold_string as _format_threshold_string


def load_extracted_data(variable, point_data_path, start_date, end_date, steps, fc1_name, fc2_name, save_format='pandas', config=None):
    """Load extracted point data - loads by forecast day for memory efficiency"""
    print("  Loading extracted point data for both forecast models...")
    
    point_data_path = Path(point_data_path)
    
    if save_format == 'pandas':
        # Check if we have forecast-day organized files (new memory-efficient format)
        day_pattern = f"{variable}_{fc1_name}_vs_{fc2_name}_day*.parquet"
        day_files = sorted(point_data_path.glob(day_pattern))
        
        if day_files:
            # Load day-by-day for memory efficiency, using iterative concat.
            # Key: apply orog, coastal, and dtype optimisations PER FILE so we
            # never hold more than (accumulated_so_far + one new file) in memory,
            # and each load is already filtered to the target subset.
            print(f"    Found {len(day_files)} forecast day files (memory-efficient mode)")

            # Determine filters to apply during loading
            cfg_f = config.get('filter', {}) if config else {}
            orog_type_load = cfg_f.get('orography_type', None)
            # orography_type may be a list when processing all types in sequence;
            # at load time we only apply filter when a single type is specified.
            if isinstance(orog_type_load, list):
                orog_type_load = None  # handled by caller iterating orog types
            orog_ranges_load = cfg_f.get('orography_ranges',
                {'low': [0, 40], 'mid': [40, 120], 'high': [120, 3000]})
            apply_orog = (orog_type_load and orog_type_load not in ('all',)
                          and orog_type_load in orog_ranges_load)
            apply_coastal = cfg_f.get('remove_coastal_stations', False)
            coastal_thresh = cfg_f.get('coastal_lsm_threshold', 0.9)

            df = None
            n_loaded = 0
            for day_file in day_files:
                df_day = pd.read_parquet(day_file)

                # Extract forecast day from filename (e.g., "day1.parquet" -> 1)
                day_match = re.search(r'day(\d+)\.parquet', day_file.name)
                if day_match:
                    forecast_day = int(day_match.group(1))
                    df_day['forecast_day'] = forecast_day

                # Filter by date and step if needed
                if start_date and end_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
                    df_day = df_day[(df_day['date'] >= start_dt) & (df_day['date'] <= end_dt)]

                if steps:
                    df_day = df_day[df_day['step'].isin(steps)]

                # ---- EARLY FILTERS (reduces rows before concat) ----
                # Apply orography filter during load (key memory saving: ~1/3 rows per type)
                if apply_orog and 'sdfor' in df_day.columns:
                    lo, hi = orog_ranges_load[orog_type_load]
                    df_day = df_day[(df_day['sdfor'] >= lo) & (df_day['sdfor'] < hi)]

                # Apply coastal filter during load
                if apply_coastal and 'lsm' in df_day.columns:
                    df_day = df_day[df_day['lsm'] >= coastal_thresh]

                # ---- DTYPE OPTIMISATION (halves numeric memory) ----
                float_cols = ['lat', 'lon', 'obs_height', 'fc1_height', 'fc2_height',
                              'sdfor', 'lsm', 'obs_value', 'fc1_value', 'fc2_value',
                              'fc1_value_uncorrected', 'fc2_value_uncorrected']
                for col in float_cols:
                    if col in df_day.columns and df_day[col].dtype == np.float64:
                        df_day[col] = df_day[col].astype(np.float32)
                if 'step' in df_day.columns:
                    df_day['step'] = df_day['step'].astype(np.int32)
                # Drop valid_time — date column is used for season filtering
                if 'valid_time' in df_day.columns:
                    df_day = df_day.drop(columns=['valid_time'])
                # ---- END OPTIMISATION ----

                if len(df_day) > 0:
                    print(f"      Loaded {day_file.name}: {len(df_day):,} rows")
                    if df is None:
                        df = df_day
                    else:
                        df = pd.concat([df, df_day], ignore_index=True)
                    n_loaded += 1

                del df_day
                gc.collect()

            if df is None:
                raise RuntimeError("No data matched the filter criteria")
            
        else:
            # Fallback: Load from single combined file (old format)
            filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}"
            parquet_file = point_data_path / f"{filename_base}.parquet"
            pickle_file = point_data_path / f"{filename_base}.pkl"
            csv_file = point_data_path / f"{filename_base}.csv"
            
            if parquet_file.exists():
                print(f"    Reading from: {parquet_file.name}")
                df = pd.read_parquet(parquet_file)
            elif pickle_file.exists():
                print(f"    Reading from: {pickle_file.name}")
                df = pd.read_pickle(pickle_file)
            elif csv_file.exists():
                print(f"    Reading from: {csv_file.name}")
                df = pd.read_csv(csv_file)
            else:
                raise FileNotFoundError(f"No pandas data file found. Expected forecast day files or {parquet_file}, {pickle_file}, or {csv_file}")
            
            # Filter by date and step if needed
            if start_date and end_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            if steps:
                df = df[df['step'].isin(steps)]
        
        print(f"  Loaded {len(df):,} valid forecast-observation pairs")
        return df
    
    else:
        # Legacy mode: load from individual .geo files
        print("  [Legacy mode: loading individual .geo files]")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            
            for step in steps:
                fc1_file = point_data_path / f"{variable}_{fc1_name}_points_{date_str}_{step}.geo"
                fc2_file = point_data_path / f"{variable}_{fc2_name}_points_{date_str}_{step}.geo"
                obs_file = point_data_path / f"{variable}_obs_points_{date_str}_{step}.geo"
                
                if not (fc1_file.exists() and fc2_file.exists() and obs_file.exists()):
                    continue
                
                try:
                    fc1_gpt = mv.read(str(fc1_file))
                    fc2_gpt = mv.read(str(fc2_file))
                    obs_gpt = mv.read(str(obs_file))
                    
                    fc1_values = mv.values(fc1_gpt)
                    fc2_values = mv.values(fc2_gpt)
                    obs_values = mv.values(obs_gpt)
                    fc1_lats = mv.latitudes(fc1_gpt)
                    fc1_lons = mv.longitudes(fc1_gpt)
                    
                    # Get heights if available
                    try:
                        fc1_heights = mv.levels(fc1_gpt)
                        obs_heights = mv.levels(obs_gpt)
                    except:
                        fc1_heights = [0] * len(fc1_values)
                        obs_heights = [0] * len(obs_values)
                    
                    for i in range(len(fc1_values)):
                        all_data.append({
                            'date': date_str,
                            'step': step,
                            'lat': fc1_lats[i],
                            'lon': fc1_lons[i],
                            'fc1_value': fc1_values[i],
                            'fc2_value': fc2_values[i],
                            'obs_value': obs_values[i],
                            'fc_height': fc1_heights[i] if i < len(fc1_heights) else 0,
                            'obs_height': obs_heights[i] if i < len(obs_heights) else 0
                        })
                except Exception as e:
                    pass
            
            current_dt += timedelta(days=1)
        
        df = pd.DataFrame(all_data)
        
        # Remove NaN values
        df = df[~(np.isnan(df['fc1_value']) | np.isnan(df['fc2_value']) | np.isnan(df['obs_value']))]
        
        print(f"  Loaded {len(df)} valid forecast-observation pairs")
        return df


def apply_lapse_rate_correction(data, lapse_rate=-0.0065):
    """Apply lapse-rate correction to temperature data for BOTH models.
    
    Formula: T_corrected = T_model + lapse_rate * (obs_height - model_height)
    With lapse_rate = -0.0065 K/m: if station is higher → forecast cools down.
    """
    # Remove rows with missing/invalid station elevations
    if 'obs_height' in data.columns:
        valid_h = (data['obs_height'] > -500) & (data['obs_height'] < 9000)
        n_invalid = (~valid_h).sum()
        if n_invalid > 0:
            data = data[valid_h].copy()
            print(f"  Removed {n_invalid} rows with missing/invalid station elevations")

    # Each model has its own height field
    height_diff_fc1 = data['obs_height'] - data['fc1_height']
    height_diff_fc2 = data['obs_height'] - data['fc2_height']
    data['fc1_value_uncorrected'] = data['fc1_value'].copy()
    data['fc2_value_uncorrected'] = data['fc2_value'].copy()
    data['fc1_value'] = data['fc1_value'] + lapse_rate * height_diff_fc1
    data['fc2_value'] = data['fc2_value'] + lapse_rate * height_diff_fc2

    # Correct ensemble members if present
    fc1_member_cols = sorted([c for c in data.columns if c.startswith('fc1_member_')],
                              key=lambda c: int(c.split('_')[-1]))
    fc2_member_cols = sorted([c for c in data.columns if c.startswith('fc2_member_')],
                              key=lambda c: int(c.split('_')[-1]))
    for col in fc1_member_cols:
        data[col] = data[col] + lapse_rate * height_diff_fc1
    for col in fc2_member_cols:
        data[col] = data[col] + lapse_rate * height_diff_fc2

    return data


def run_step4(config, extraction_info, preprocess_settings, model_names):
    """
    Execute Step 4: Filter Data
    Returns filtered DataFrame with both models
    """
    print("\n" + "="*80)
    print("STEP 4: FILTER DATA")
    print("="*80)
    
    # Extract info from step 3
    if isinstance(extraction_info, dict):
        point_data_path = extraction_info['output_path']
        save_format = extraction_info.get('save_format', 'pandas')
    else:
        # Legacy: just a path
        point_data_path = extraction_info
        save_format = 'pandas'
    
    # Load data for BOTH models
    data = load_extracted_data(
        variable=config['variable'],
        point_data_path=point_data_path,
        start_date=config['start_date'],
        end_date=config['end_date'],
        steps=config.get('steps'),  # May be None if using forecast_days
        fc1_name=model_names['fc1_name'],
        fc2_name=model_names['fc2_name'],
        save_format=save_format,
        config=config
    )
    
    original_len = len(data)
    
    # Check if lapse-rate correction was already applied during extraction
    has_uncorrected_columns = 'fc1_value_uncorrected' in data.columns
    
    # Always apply lapse-rate correction for 2t
    if config['variable'] == '2t':
        if has_uncorrected_columns:
            print(f"  ⚠️  Note: Lapse-rate correction already applied during extraction (Step 3)")
            print(f"     Skipping to avoid double-correction. Heights stored: fc1_height, fc2_height.")
        elif 'fc1_height' in data.columns and 'obs_height' in data.columns:
            lapse_rate = preprocess_settings.get('lapse_rate', -0.0065)
            data = apply_lapse_rate_correction(data, lapse_rate)
            print(f"  Applied lapse-rate correction ({lapse_rate} K/m) to both models")
        else:
            print(f"  ⚠️  Lapse-rate correction skipped: missing height columns (fc1_height/obs_height)")
    
    print(f"\n  Initial dataset: {len(data)} rows")
    print(f"    Unique stations: {data['station_id'].nunique()}")
    print(f"    Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"    Steps: {sorted(data['step'].unique())}")
    
    cfg = config['filter']
    
    # ========================================================================
    # GEOGRAPHIC AREA FILTERING
    # ========================================================================
    # Check if area filtering was already applied during extraction
    cfg_extract = config.get('extract_points', {})
    area_in_extraction = cfg_extract.get('area', None)
    
    area_name = cfg.get('area', None)
    if area_name:
        if area_in_extraction:
            print(f"\n  ⚡ Area filtering already applied during extraction: {area_in_extraction}")
            print(f"     Skipping redundant filter step (data already filtered)")
            # If user specified different area in filter than extraction, warn them
            if area_name != area_in_extraction:
                print(f"  ⚠️  WARNING: You configured area='{area_name}' in filter step,")
                print(f"     but data was extracted with area='{area_in_extraction}'.")
                print(f"     Cannot change area after extraction. Ignoring filter setting.")
        else:
            # No area filter during extraction, apply it now
            # Define named areas [North, West, South, East]
            areas = {
                'europe': [68, -15, 27, 50],
                'nh_extratropics': [90, -180, 20, 180],
                'tropics': [20, -180, -20, 180]
            }
            
            if area_name in areas:
                area_coords = areas[area_name]
                lat_north, lon_west, lat_south, lon_east = area_coords
                before = len(data)
                data = data[(data['lat'] >= lat_south) & (data['lat'] <= lat_north) &
                           (data['lon'] >= lon_west) & (data['lon'] <= lon_east)]
                print(f"\n  Area filtering ({area_name}): {len(data)} rows (removed {before - len(data)})")
                print(f"    Area: lat [{lat_south}, {lat_north}], lon [{lon_west}, {lon_east}]")
            else:
                print(f"\n  Warning: Unknown area '{area_name}', skipping area filter")
    elif area_in_extraction:
        print(f"\n  ⚡ Area pre-filtered during extraction: {area_in_extraction}")
    
    # ========================================================================
    # LEAD TIME FILTERING
    # ========================================================================
    # Filter by lead time
    if cfg.get('lead_times'):
        data = data[data['step'].isin(cfg['lead_times'])]
        print(f"  Lead time filter: {len(data)} rows (from {original_len})")
    
    # ========================================================================
    # SEASON FILTERING
    # ========================================================================
    # Filter by season (supports standard codes like DJF and custom month groups like [8,9,10,11])
    season = cfg.get('season')
    if season:
        # Handle list format: take the first entry for this filter pass
        if isinstance(season, list):
            season = season[0] if season else None
        
        season_label, months = season_utils.resolve_season(season)
        
        if months:
            # Extract month from date string (YYYYMMDD)
            data['month'] = data['date'].astype(str).str[4:6].astype(int)
            before = len(data)
            data = data[data['month'].isin(months)]
            print(f"\n  Season filtering ({season_label}): {len(data)} rows (removed {before - len(data)})")
            print(f"    Months: {months}")
            # Drop temporary month column
            data = data.drop(columns=['month'])
        elif season_label:
            print(f"\n  Warning: Could not resolve season '{season}', skipping season filter")
    
    # ========================================================================
    # OROGRAPHY FILTERING
    # ========================================================================
    if cfg.get('orography_type'):
        orography_type = cfg['orography_type']
        orography_ranges = cfg.get('orography_ranges', {
            'low': [0, 40],
            'mid': [40, 120],
            'high': [120, 3000]
        })
        
        if 'sdfor' not in data.columns:
            print(f"  ⚠ Warning: sdfor column not found in data. Skipping orography filter.")
        else:
            if orography_type in orography_ranges:
                min_sdfor, max_sdfor = orography_ranges[orography_type]
                before = len(data)
                data = data[(data['sdfor'] >= min_sdfor) & (data['sdfor'] < max_sdfor)]
                print(f"  Orography filter ({orography_type}): {len(data)} rows (sdfor: {min_sdfor}-{max_sdfor}, removed {before - len(data)})")
            else:
                print(f"  ⚠ Unknown orography type: {orography_type}")
    
    # ========================================================================
    # COASTAL STATION FILTERING (LSM threshold)
    # ========================================================================
    # Remove coastal stations based on land-sea mask
    # LSM (land-sea mask) ranges from 0 (water) to 1 (land)
    # Default threshold 0.9 keeps only stations on solid land, far from coasts
    # Future option: set to "coastal" to keep ONLY coastal stations (lsm <= 0.9)
    remove_coastal = cfg.get('remove_coastal_stations', True)  # Default: remove coastal
    lsm_threshold = cfg.get('coastal_lsm_threshold', 0.9)  # Default: 0.9
    
    if remove_coastal:
        if 'lsm' in data.columns:
            before = len(data)
            data = data[data['lsm'] > lsm_threshold]
            removed = before - len(data)
            if removed > 0:
                print(f"  Coastal filter (lsm > {lsm_threshold}): {len(data)} rows (removed {removed} coastal stations, {100*removed/before:.1f}%)")
            else:
                print(f"  Coastal filter (lsm > {lsm_threshold}): {len(data)} rows (no coastal stations removed)")
        else:
            print(f"  ⚠ Warning: lsm column not found. Skipping coastal filtering.")
    else:
        print(f"  Coastal filter: disabled (keeping all stations including coastal)")
    
    # Quality control - remove outliers (both models)
    # NOTE: For precipitation (tp24), forecast outlier removal is SKIPPED because
    # precipitation is heavily right-skewed and 5*std cutoff removes genuine
    # extreme events (the very events we want to verify). The max_valid_precipitation
    # QC below already handles sensor errors.
    if cfg.get('remove_outliers', False) and config['variable'] != 'tp24':
        threshold_std = cfg.get('outlier_threshold_std', 5.0)
        fc1_mean = data['fc1_value'].mean()
        fc1_std = data['fc1_value'].std()
        fc2_mean = data['fc2_value'].mean()
        fc2_std = data['fc2_value'].std()
        before = len(data)
        # Remove outliers from either model
        data = data[(np.abs(data['fc1_value'] - fc1_mean) < threshold_std * fc1_std) & 
                    (np.abs(data['fc2_value'] - fc2_mean) < threshold_std * fc2_std)]
        print(f"  Outlier removal: {len(data)} rows (removed {before - len(data)})")
    
    # Variable-specific quality control
    if config['variable'] == 'tp24':
        # Remove anomalously high precipitation observations (likely sensor errors or outliers)
        max_precip = cfg.get('max_valid_precipitation', None)
        if max_precip is not None:
            before = len(data)
            data = data[data['obs_value'] <= max_precip]
            print(f"  Precipitation QC (max={max_precip}mm): {len(data)} rows (removed {before - len(data)})")
    
    if config['variable'] == '2t':
        # Remove extreme temperature outliers (likely missing data flags or sensor errors)
        min_temp = cfg.get('min_valid_temperature', -60.0)  # Default: -60°C
        max_temp = cfg.get('max_valid_temperature', 60.0)   # Default: +60°C
        before = len(data)
        valid_mask = (data['obs_value'] >= min_temp) & (data['obs_value'] <= max_temp)
        # Also filter rows where any forecast value (det or ensemble member) is out of range
        for col in data.columns:
            if col in ('fc1_value', 'fc2_value') or col.startswith('fc1_member_') or col.startswith('fc2_member_'):
                valid_mask &= (data[col] >= min_temp) & (data[col] <= max_temp)
        data = data[valid_mask]
        removed = before - len(data)
        if removed > 0:
            print(f"  Temperature QC ({min_temp}°C to {max_temp}°C): {len(data)} rows (removed {removed})")

    if config['variable'] == '10ff':
        # Remove bad wind speed obs (missing-value flags often appear as very large numbers)
        min_wind = cfg.get('min_valid_wind_speed', 0.0)
        max_wind = cfg.get('max_valid_wind_speed', 100.0)
        before = len(data)
        valid_mask = (data['obs_value'] >= min_wind) & (data['obs_value'] <= max_wind)
        for col in data.columns:
            if col in ('fc1_value', 'fc2_value') or col.startswith('fc1_member_') or col.startswith('fc2_member_'):
                valid_mask &= (data[col] >= min_wind) & (data[col] <= max_wind)
        data = data[valid_mask]
        removed = before - len(data)
        if removed > 0:
            print(f"  Wind speed QC ({min_wind}–{max_wind} m/s): {len(data)} rows (removed {removed})")
    
    print(f"\nFinal dataset: {len(data)} rows")
    print("\n✓ Step 4 complete")
    
    return data
