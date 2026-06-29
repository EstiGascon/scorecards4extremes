"""
STEP 3: EXTRACT POINT DATA
===========================
Extract forecast at observation locations using nearest gridpoint
"""

import metview as mv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from utils import format_threshold_string as _format_threshold_string


def get_area_bbox(area_config):
    """
    Get bounding box for area filtering
    
    Args:
        area_config: str (area name) or list [N, W, S, E] or None
    
    Returns:
        tuple: (lat_north, lon_west, lat_south, lon_east) or None
    """
    if area_config is None:
        return None
    
    # Define predefined areas [North, West, South, East]
    areas = {
        'europe': [68, -15, 27, 50],
        'nh_extratropics': [90, -180, 20, 180],
        'tropics': [20, -180, -20, 180],
        'north_america': [72, -170, 20, -50],
        'asia': [75, 50, 10, 150]
    }
    
    # Get area coordinates
    if isinstance(area_config, str):
        if area_config in areas:
            return tuple(areas[area_config])
        else:
            print(f"  Warning: Unknown area '{area_config}', no area filtering applied")
            return None
    elif isinstance(area_config, list) and len(area_config) == 4:
        return tuple(area_config)
    else:
        print(f"  Warning: Invalid area format '{area_config}', expected string or [N,W,S,E]")
        return None


def filter_extracted_data_by_area(data_dict, area_bbox):
    """
    Filter extracted data by geographic area
    This reduces memory usage by filtering after extraction but before saving
    
    Args:
        data_dict: dict with 'lat', 'lon', etc. arrays
        area_bbox: tuple (lat_north, lon_west, lat_south, lon_east) or None
    
    Returns:
        Filtered data_dict
    """
    if area_bbox is None:
        return data_dict
    
    lat_north, lon_west, lat_south, lon_east = area_bbox
    
    # Filter all arrays by area
    lats = data_dict['lat']
    lons = data_dict['lon']
    
    # Find indices of points in area
    mask = [(lat >= lat_south and lat <= lat_north and 
             lon >= lon_west and lon <= lon_east)
            for lat, lon in zip(lats, lons)]
    
    if not any(mask):
        return None  # No points in area
    
    # Filter all fields
    filtered_data = {}
    for key, values in data_dict.items():
        if isinstance(values, list):
            filtered_data[key] = [v for v, m in zip(values, mask) if m]
        else:
            filtered_data[key] = values  # Keep non-list items as-is
    
    return filtered_data


def extract_points(config, variable, fc1_path, fc2_path, fc1_name, fc2_name, obs_path, output_path, 
                  start_date, end_date, steps, preprocess_settings, save_format='pandas'):
    """
    Extract point forecasts from GRIB files at observation locations
    Processes TWO forecast models for comparison
    Saves as pandas DataFrame (recommended) or individual .geo files
    NEW: Supports early area filtering to reduce file sizes and memory usage
    """
    fc1_path = Path(fc1_path)
    fc2_path = Path(fc2_path)
    obs_path = Path(obs_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Load auxiliary fields from config
    cfg_aux = config.get('auxiliary_fields', {})
    
    # Get area filtering configuration from extract_points section
    cfg_extract = config.get('extract_points', {})
    area_config = cfg_extract.get('area', None)
    if area_config:
        print(f"  ⚡ EARLY AREA FILTERING ENABLED: {area_config}")
        print(f"     This will extract ONLY stations in the specified region")
        print(f"     Expected memory reduction: 50-90% depending on region size")
    else:
        print(f"  ℹ️  No area filtering - extracting ALL global stations")
        print(f"     Tip: Set 'area' in extract_points config to reduce memory usage")
    
    # Load height data if needed for temperature lapse-rate correction
    # Each model can have its own orography/lsm
    height_field_fc1 = None
    height_field_fc2 = None
    if variable == '2t' and preprocess_settings.get('lapse_rate_correction', False):
        try:
            # Model 1 height field
            lsm_path_fc1 = cfg_aux.get('model1', {}).get('lsm_path', cfg_aux.get('lsm_path', '/ec/vol/destine/continuous_evaluation/lsm_tco1279.grib'))
            orog_path_fc1 = cfg_aux.get('model1', {}).get('orog_path', cfg_aux.get('orog_path', '/ec/vol/destine/continuous_evaluation/hres_orog.grib'))
            
            lsm_fc1 = mv.read(lsm_path_fc1)
            orog_fc1 = mv.read(orog_path_fc1)
            height_field_fc1 = orog_fc1 / 9.80665  # Convert geopotential (m²/s²) to height in metres
            print(f"  Loaded height data for {fc1_name} lapse-rate correction")
            print(f"    LSM: {lsm_path_fc1}")
            print(f"    Orography: {orog_path_fc1}")
            
            # Model 2 height field
            lsm_path_fc2 = cfg_aux.get('model2', {}).get('lsm_path', cfg_aux.get('lsm_path', '/ec/vol/destine/continuous_evaluation/lsm_tco1279.grib'))
            orog_path_fc2 = cfg_aux.get('model2', {}).get('orog_path', cfg_aux.get('orog_path', '/ec/vol/destine/continuous_evaluation/hres_orog.grib'))
            
            lsm_fc2 = mv.read(lsm_path_fc2)
            orog_fc2 = mv.read(orog_path_fc2)
            height_field_fc2 = orog_fc2 / 9.80665  # Convert geopotential (m²/s²) to height in metres
            print(f"  Loaded height data for {fc2_name} lapse-rate correction")
            print(f"    LSM: {lsm_path_fc2}")
            print(f"    Orography: {orog_path_fc2}")
        except Exception as e:
            print(f"  Warning: Could not load height data: {e}")
    
    # Load standard deviation of orography (sdfor) for terrain filtering
    sdfor_field = None
    try:
        sdfor_path = cfg_aux.get('sdfor_path', '/ec/vol/destine/continuous_evaluation/sdfor_tco1279.grib')
        sdfor_field = mv.read(sdfor_path)
        print(f"  Loaded sdfor for terrain classification: {sdfor_path}")
    except Exception as e:
        print(f"  Warning: Could not load sdfor data: {e}")
        print(f"  Terrain filtering will not be available")
    
    # Load land-sea mask (lsm) for coastal station filtering
    lsm_field = None
    try:
        lsm_path = cfg_aux.get('lsm_path', cfg_aux.get('model1', {}).get('lsm_path', '/ec/vol/destine/continuous_evaluation/lsm_tco1279.grib'))
        lsm_field = mv.read(lsm_path)
        print(f"  Loaded LSM for coastal filtering: {lsm_path}")
    except Exception as e:
        print(f"  Warning: Could not load LSM data: {e}")
        print(f"  Coastal filtering will not be available")
    
    print(f"  Extracting points for TWO forecast models...")
    print(f"  Model 1: {fc1_name}")
    print(f"  Model 2: {fc2_name}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Steps: {steps}")
    print(f"  Save format: {save_format}")
    
    # Calculate total iterations for progress
    total_days = (end_dt - start_dt).days + 1
    total_iterations = total_days * len(steps)
    print(f"  Total files to process: {total_iterations}")
    
    # Data collection for pandas mode
    # Organize by forecast day for memory efficiency
    data_by_forecast_day = {}  # {day: [data_rows]}
    forecast_days_config = config.get('forecast_days', None)
    
    current_dt = start_dt
    processed_pairs = 0
    errors = 0
    iteration = 0
    first_extraction_done = False  # Track if we've done first successful extraction for diagnostics
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y%m%d')
        day_number = (current_dt - start_dt).days + 1
        
        # Progress: Processing day X of Y
        print(f"\n  Processing day {day_number}/{total_days}: {date_str}")
        
        # Read GRIB files ONCE per day for both models
        fc1_grib = None
        fc2_grib = None
        
        try:
            if variable == '2t':
                # Model 1
                fc1_grib_file = fc1_path / f"2t_{date_str}.grib"
                if fc1_grib_file.exists():
                    fc1_grib = mv.read(str(fc1_grib_file))
                    print(f"    ✓ Loaded {fc1_name} GRIB")
                else:
                    print(f"    ✗ Missing {fc1_name} GRIB")
                
                # Model 2
                fc2_grib_file = fc2_path / f"2t_{date_str}.grib"
                if fc2_grib_file.exists():
                    fc2_grib = mv.read(str(fc2_grib_file))
                    print(f"    ✓ Loaded {fc2_name} GRIB")
                else:
                    print(f"    ✗ Missing {fc2_name} GRIB")
            
            elif variable == '10ff':
                # Model 1 - wind components
                fc1_u_file = fc1_path / f"10u_{date_str}.grib"
                fc1_v_file = fc1_path / f"10v_{date_str}.grib"
                if fc1_u_file.exists() and fc1_v_file.exists():
                    fc1_u_grib = mv.read(str(fc1_u_file))
                    fc1_v_grib = mv.read(str(fc1_v_file))
                    print(f"    ✓ Loaded {fc1_name} wind components")
                else:
                    print(f"    ✗ Missing {fc1_name} wind components")
                
                # Model 2 - wind components
                fc2_u_file = fc2_path / f"10u_{date_str}.grib"
                fc2_v_file = fc2_path / f"10v_{date_str}.grib"
                if fc2_u_file.exists() and fc2_v_file.exists():
                    fc2_u_grib = mv.read(str(fc2_u_file))
                    fc2_v_grib = mv.read(str(fc2_v_file))
                    print(f"    ✓ Loaded {fc2_name} wind components")
                else:
                    print(f"    ✗ Missing {fc2_name} wind components")
            
            elif variable == 'tp24':
                # Model 1 - try multiple naming patterns
                fc1_patterns = [
                    fc1_path / f"tp24_{date_str}.grib",
                    fc1_path / f"tp_{date_str}.grib",
                    fc1_path / f"tp24_{date_str}_0.25degree.grib",
                    fc1_path / f"tp_{date_str}_0.25degree.grib",
                    fc1_path / f"tp24_{date_str}_0.1degree.grib",
                    fc1_path / f"tp_{date_str}_0.1degree.grib"
                ]
                fc1_grib = None
                for fc1_file in fc1_patterns:
                    if fc1_file.exists():
                        fc1_grib = mv.read(str(fc1_file))
                        print(f"    ✓ Loaded {fc1_name} precipitation GRIB")
                        break
                if fc1_grib is None:
                    print(f"    ✗ Missing {fc1_name} precipitation GRIB")
                
                # Model 2 - try multiple naming patterns
                fc2_patterns = [
                    fc2_path / f"tp24_{date_str}.grib",
                    fc2_path / f"tp_{date_str}.grib",
                    fc2_path / f"tp24_{date_str}_0.25degree.grib",
                    fc2_path / f"tp_{date_str}_0.25degree.grib",
                    fc2_path / f"tp24_{date_str}_0.1degree.grib",
                    fc2_path / f"tp_{date_str}_0.1degree.grib"
                ]
                fc2_grib = None
                for fc2_file in fc2_patterns:
                    if fc2_file.exists():
                        fc2_grib = mv.read(str(fc2_file))
                        print(f"    ✓ Loaded {fc2_name} precipitation GRIB")
                        break
                if fc2_grib is None:
                    print(f"    ✗ Missing {fc2_name} precipitation GRIB")
        
        except Exception as e:
            if current_dt == start_dt:
                print(f"    [ERROR] Failed to read GRIB files for {date_str}: {e}")
            fc1_grib = None
            fc2_grib = None
        
        # Now process each step for this day
        print(f"    Extracting {len(steps)} lead times: {steps}")
        day_start_pairs = processed_pairs
        
        for step in steps:
            iteration += 1
            
            try:
                # Extract specific step from pre-loaded GRIB files
                fc1_step = None
                fc2_step = None
                
                if variable == '2t':
                    if fc1_grib is not None:
                        fc1_step = mv.read(data=fc1_grib, step=step) - 273.15  # Convert to Celsius
                    if fc2_grib is not None:
                        fc2_step = mv.read(data=fc2_grib, step=step) - 273.15  # Convert to Celsius
                
                elif variable == '10ff':
                    # Use pre-loaded wind component GRIBs
                    if 'fc1_u_grib' in locals() and 'fc1_v_grib' in locals():
                        u1_step = mv.read(data=fc1_u_grib, step=step)
                        v1_step = mv.read(data=fc1_v_grib, step=step)
                        fc1_step = mv.sqrt(u1_step * u1_step + v1_step * v1_step)
                    
                    if 'fc2_u_grib' in locals() and 'fc2_v_grib' in locals():
                        u2_step = mv.read(data=fc2_u_grib, step=step)
                        v2_step = mv.read(data=fc2_v_grib, step=step)
                        fc2_step = mv.sqrt(u2_step * u2_step + v2_step * v2_step)
                
                elif variable == 'tp24':
                    # Use pre-loaded precipitation GRIBs
                    # For 24h accumulation: tp[step] - tp[step-24]
                    # Special case: If step-24=0, just use tp[step] (for models without T+0)
                    cfg_rd = config.get('read_data', {})
                    fc1_unit_factor = cfg_rd.get('forecast_model1', {}).get('unit_conversion_factor', 1000.0)
                    fc2_unit_factor = cfg_rd.get('forecast_model2', {}).get('unit_conversion_factor', 1000.0)
                    if fc1_grib is not None and step >= 24:
                        try:
                            tp1_step = mv.read(data=fc1_grib, step=step)
                            if step == 24:
                                # For step 24, no T+0 available in some models, use T+24 directly
                                fc1_step = tp1_step * fc1_unit_factor
                            else:
                                tp1_prev = mv.read(data=fc1_grib, step=step-24)
                                fc1_step = (tp1_step - tp1_prev) * fc1_unit_factor
                        except Exception as e:
                            fc1_step = None
                            if iteration == 1:
                                print(f"      Warning: Could not extract {fc1_name} tp24 at step {step}: {e}")
                    
                    if fc2_grib is not None and step >= 24:
                        try:
                            tp2_step = mv.read(data=fc2_grib, step=step)
                            if step == 24:
                                # For step 24, no T+0 available in some models, use T+24 directly
                                fc2_step = tp2_step * fc2_unit_factor
                            else:
                                tp2_prev = mv.read(data=fc2_grib, step=step-24)
                                fc2_step = (tp2_step - tp2_prev) * fc2_unit_factor
                        except Exception as e:
                            fc2_step = None
                            if iteration == 1:
                                print(f"      Warning: Could not extract {fc2_name} tp24 at step {step}: {e}")
                
                # Skip if either forecast is missing
                if fc1_step is None or fc2_step is None:
                    errors += 1
                    continue
                
                if iteration == 1:
                    print(f"    Both models have data, proceeding to observations...")
                
                # Calculate observation valid time from base date + step
                # Base date is at 00:00 UTC
                base_dt = datetime.strptime(date_str, '%Y%m%d')
                valid_dt = base_dt + timedelta(hours=step)
                vdate_str = valid_dt.strftime('%Y%m%d')
                vtime_str = valid_dt.strftime('%H')
                
                # Observation file format: 2t_obs_YYYYMMDDHH.geo
                obs_file = obs_path / f"{variable}_obs_{vdate_str}{vtime_str}.geo"
                if iteration == 1:
                    print(f"    Looking for obs: {obs_file}")
                    print(f"    Obs exists: {obs_file.exists()}")
                
                if not obs_file.exists():
                    if iteration == 1:
                        print(f"    [DEBUG] No observation file found, skipping")
                    errors += 1
                    continue
                
                if iteration == 1:
                    print(f"    Reading observations...")
                obs = mv.read(str(obs_file))
                obs = mv.remove_duplicates(obs)
                
                # Convert observations from Kelvin to Celsius for 2t
                if variable == '2t':
                    obs = obs - 273.15
                
                if iteration == 1:
                    print(f"    Observations loaded: {mv.count(obs)} stations")
                    if area_config:
                        print(f"    Area filtering will be applied AFTER extraction")
                    print(f"    Extracting nearest grid points for both models...")
                
                # Extract nearest grid points for BOTH models
                fc1_points = mv.nearest_gridpoint(fc1_step, obs)
                fc2_points = mv.nearest_gridpoint(fc2_step, obs)
                
                if iteration == 1:
                    print(f"    Extraction complete, collecting data...")
                
                # Collect data based on save format
                if save_format == 'pandas':
                    # Extract values to lists and build DataFrame rows
                    lats = mv.latitudes(obs)
                    lons = mv.longitudes(obs)
                    obs_vals = mv.values(obs)
                    fc1_vals = mv.values(fc1_points)
                    fc2_vals = mv.values(fc2_points)
                    
                    # Show sample values before preprocessing (first time only)
                    if iteration == 1 and len(obs_vals) > 0:
                        print(f"    📊 Sample values BEFORE preprocessing (first 3 stations):")
                        for i in range(min(3, len(obs_vals))):
                            print(f"      Station {i+1}: Obs={obs_vals[i]:.2f}, {fc1_name}={fc1_vals[i]:.2f}, {fc2_name}={fc2_vals[i]:.2f}")
                    
                    # Get station IDs if available
                    try:
                        station_ids = mv.geopoints_id(obs)
                    except:
                        station_ids = [f"S{i}" for i in range(len(lats))]
                    
                    # Get station elevations directly from Metview geopoints object
                    # IMPORTANT: mv.elevations() returns values in Metview's internal order,
                    # which matches mv.latitudes()/mv.longitudes()/mv.values() ordering.
                    # Do NOT parse the text file for elevations — mv.read() reorders stations
                    # internally, so text-file order != Metview object order.
                    obs_heights = [0.0] * len(lats)
                    if variable == '2t' and preprocess_settings.get('lapse_rate_correction', False):
                        try:
                            obs_heights_raw = mv.elevations(obs)
                            obs_heights = list(obs_heights_raw)
                            
                            if not first_extraction_done:
                                import numpy as _np
                                _h = _np.array(obs_heights)
                                # Count valid vs missing (99999 or >9000 are missing indicators)
                                valid_mask = (_h > -500) & (_h < 9000)
                                n_valid = int(valid_mask.sum())
                                n_missing = len(_h) - n_valid
                                if n_valid > 0:
                                    print(f"    ✓ Extracted {n_valid}/{len(obs_heights)} valid station elevations (via mv.elevations)")
                                    print(f"       Height range: {_h[valid_mask].min():.0f}m to {_h[valid_mask].max():.0f}m")
                                    if n_missing > 0:
                                        print(f"       ({n_missing} stations with missing heights will be removed)")
                                else:
                                    print(f"    ⚠️  All station elevations are missing or invalid")
                                del _h, valid_mask
                        
                        except Exception as e:
                            if not first_extraction_done:
                                print(f"    ⚠️  Could not extract station elevations: {e}")
                                print(f"       Using 0.0 for all stations")
                            obs_heights = [0.0] * len(lats)
                    
                    # Extract standard deviation of orography at observation points (before filtering)
                    sdfor_vals = [0.0] * len(lats)
                    if sdfor_field is not None:
                        sdfor_at_obs = mv.nearest_gridpoint(sdfor_field, obs)
                        sdfor_vals = mv.values(sdfor_at_obs)
                    
                    # Extract land-sea mask at observation points (for coastal filtering)
                    lsm_vals = [1.0] * len(lats)  # Default to land if not available
                    if lsm_field is not None:
                        lsm_at_obs = mv.nearest_gridpoint(lsm_field, obs)
                        lsm_vals = mv.values(lsm_at_obs)
                    
                    # Apply lapse-rate correction for 2t if enabled
                    fc1_heights = [0.0] * len(lats)
                    fc2_heights = [0.0] * len(lats)
                    fc1_vals_uncorrected = list(fc1_vals)
                    fc2_vals_uncorrected = list(fc2_vals)
                    if variable == '2t' and (height_field_fc1 is not None or height_field_fc2 is not None):
                        lapse_rate = preprocess_settings.get('lapse_rate', -0.0065)
                        
                        # Get model heights at observation points
                        if height_field_fc1 is not None:
                            height_at_obs_fc1 = mv.nearest_gridpoint(height_field_fc1, obs)
                            fc1_heights = mv.values(height_at_obs_fc1)
                        
                        if height_field_fc2 is not None:
                            height_at_obs_fc2 = mv.nearest_gridpoint(height_field_fc2, obs)
                            fc2_heights = mv.values(height_at_obs_fc2)
                        
                        # Apply corrections
                        corrections_fc1 = []
                        corrections_fc2 = []
                        
                        for i in range(len(lats)):
                            # Model 1 correction
                            if height_field_fc1 is not None:
                                height_diff = obs_heights[i] - fc1_heights[i]
                                correction = lapse_rate * height_diff
                                fc1_vals[i] = fc1_vals[i] + correction
                                corrections_fc1.append(correction)
                            else:
                                corrections_fc1.append(0.0)
                            
                            # Model 2 correction
                            if height_field_fc2 is not None:
                                height_diff = obs_heights[i] - fc2_heights[i]
                                correction = lapse_rate * height_diff
                                fc2_vals[i] = fc2_vals[i] + correction
                                corrections_fc2.append(correction)
                            else:
                                corrections_fc2.append(0.0)
                        
                        # Quality filtering: Remove extreme corrections (matching original script)
                        # This catches unrealistic corrections from bad height data
                        import numpy as np
                        temp_min = -100.0
                        temp_max = 60.0
                        max_correction = 50.0  # °C
                        max_height_diff = 10000.0  # meters
                        
                        valid_indices = []
                        filtered_count = 0
                        extreme_temp_count = 0
                        extreme_correction_count = 0
                        extreme_height_count = 0
                        missing_height_count = 0
                        
                        for i in range(len(lats)):
                            # Check for missing observation height
                            # Only flag 9999/99999 as missing (old tool convention)
                            # Do NOT remove stations at sea level (elevation=0)
                            height_valid = not (obs_heights[i] >= 9999)
                            
                            # Check temperature bounds
                            temp_ok = (temp_min <= fc1_vals[i] <= temp_max and 
                                      temp_min <= fc2_vals[i] <= temp_max and
                                      temp_min <= obs_vals[i] <= temp_max)
                            
                            # Check correction magnitude
                            corr_ok = (abs(corrections_fc1[i]) <= max_correction and 
                                      abs(corrections_fc2[i]) <= max_correction)
                            
                            # Check height difference
                            height_ok = (abs(obs_heights[i] - fc1_heights[i]) <= max_height_diff and
                                        abs(obs_heights[i] - fc2_heights[i]) <= max_height_diff)
                            
                            if height_valid and temp_ok and corr_ok and height_ok:
                                valid_indices.append(i)
                            else:
                                filtered_count += 1
                                if not height_valid:
                                    missing_height_count += 1
                                if not temp_ok:
                                    extreme_temp_count += 1
                                if not corr_ok:
                                    extreme_correction_count += 1
                                if not height_ok:
                                    extreme_height_count += 1
                        
                        # Apply filtering
                        if filtered_count > 0:
                            original_count = len(lats)
                            lats = [lats[i] for i in valid_indices]
                            lons = [lons[i] for i in valid_indices]
                            obs_vals = [obs_vals[i] for i in valid_indices]
                            fc1_vals = [fc1_vals[i] for i in valid_indices]
                            fc2_vals = [fc2_vals[i] for i in valid_indices]
                            sdfor_vals = [sdfor_vals[i] for i in valid_indices]
                            lsm_vals = [lsm_vals[i] for i in valid_indices]
                            obs_heights = [obs_heights[i] for i in valid_indices]
                            fc1_heights = [fc1_heights[i] for i in valid_indices]
                            fc2_heights = [fc2_heights[i] for i in valid_indices]
                            station_ids = [station_ids[i] for i in valid_indices]
                            
                            if not first_extraction_done:
                                pct = 100 * filtered_count / original_count
                                print(f"    🔍 Post-correction quality filtering: removed {filtered_count}/{original_count} points ({pct:.1f}%)")
                                if missing_height_count > 0:
                                    print(f"       - Missing observation heights (>=9999): {missing_height_count}")
                                if extreme_temp_count > 0:
                                    print(f"       - Extreme temperatures (outside {temp_min}°C to {temp_max}°C): {extreme_temp_count}")
                                if extreme_correction_count > 0:
                                    print(f"       - Extreme corrections (>{max_correction}°C): {extreme_correction_count}")
                                if extreme_height_count > 0:
                                    print(f"       - Extreme height differences (>{max_height_diff}m): {extreme_height_count}")
                        
                        if not first_extraction_done:
                            print(f"    Applied lapse-rate correction (rate: {lapse_rate} K/m)")
                            if len(obs_vals) > 0:
                                print(f"    📊 Sample values AFTER lapse-rate correction (first 3 stations):")
                                for i in range(min(3, len(obs_vals))):
                                    corr1 = obs_heights[i] - fc1_heights[i]
                                    corr2 = obs_heights[i] - fc2_heights[i]
                                    print(f"      Station {i+1}: Obs={obs_vals[i]:.2f}°C (h={obs_heights[i]:.0f}m)")
                                    print(f"                 {fc1_name}={fc1_vals[i]:.2f}°C (h={fc1_heights[i]:.0f}m, Δh={corr1:.0f}m)")
                                    print(f"                 {fc2_name}={fc2_vals[i]:.2f}°C (h={fc2_heights[i]:.0f}m, Δh={corr2:.0f}m)")
                            else:
                                print(f"    ⚠️  No valid data points remaining after quality filtering!")
                            first_extraction_done = True
                    
                    # Determine which forecast day this step belongs to
                    # Use the mapping created during step computation
                    if '_step_to_forecast_day' in config:
                        forecast_day = config['_step_to_forecast_day'].get(step, ((step - 1) // 24) + 1 if step > 0 else 1)
                    else:
                        # Fallback: Day 1 = 1-24h, Day 2 = 25-48h, etc. (matches: day = ((lt-1)//24)+1)
                        forecast_day = ((step - 1) // 24) + 1 if step > 0 else 1
                    
                    if forecast_day not in data_by_forecast_day:
                        data_by_forecast_day[forecast_day] = []
                    
                    # Add each station's data
                    for i in range(len(lats)):
                        data_by_forecast_day[forecast_day].append({
                            'date': date_str,
                            'step': step,
                            'valid_time': f"{vdate_str}{vtime_str}",
                            'station_id': station_ids[i],
                            'lat': lats[i],
                            'lon': lons[i],
                            'obs_height': obs_heights[i],
                            'fc1_height': fc1_heights[i],
                            'fc2_height': fc2_heights[i],
                            'sdfor': sdfor_vals[i],
                            'lsm': lsm_vals[i],
                            'obs_value': obs_vals[i],
                            'fc1_value': fc1_vals[i],
                            'fc2_value': fc2_vals[i],
                            'fc1_value_uncorrected': fc1_vals_uncorrected[i],
                            'fc2_value_uncorrected': fc2_vals_uncorrected[i]
                        })
                    
                    processed_pairs += len(lats)
                    
                else:  # Legacy mode: save individual .geo files
                    fc1_out = output_path / f"{variable}_{fc1_name}_points_{date_str}_{step}.geo"
                    fc2_out = output_path / f"{variable}_{fc2_name}_points_{date_str}_{step}.geo"
                    obs_out = output_path / f"{variable}_obs_points_{date_str}_{step}.geo"
                    
                    mv.write(str(fc1_out), fc1_points)
                    mv.write(str(fc2_out), fc2_points)
                    mv.write(str(obs_out), obs)
                    processed_pairs += 1
                    
                    if iteration == 1:
                        print(f"    Files saved:")
                        print(f"      {fc1_out}")
                        print(f"      {fc2_out}")
                        print(f"      {obs_out}")
                
            except Exception as e:
                errors += 1
                if day_number <= 2:  # Show errors for first 2 days
                    print(f"      ✗ Step {step}h: {str(e)[:80]}")
        
        # Day summary
        day_pairs = processed_pairs - day_start_pairs
        print(f"    → Extracted {day_pairs} station-pairs for this day")
        print(f"    → Total so far: {processed_pairs} pairs, {errors} errors")
        
        # Save data by forecast day periodically to avoid memory buildup (every 10 calendar days)
        if save_format == 'pandas' and day_number % 10 == 0:
            for fday in list(data_by_forecast_day.keys()):
                if len(data_by_forecast_day[fday]) > 0:
                    # Apply area filtering if configured
                    data_to_save = data_by_forecast_day[fday]
                    if area_config:
                        area_bbox = get_area_bbox(area_config)
                        if area_bbox:
                            original_count = len(data_to_save)
                            # Filter by lat/lon
                            lat_north, lon_west, lat_south, lon_east = area_bbox
                            data_to_save = [
                                row for row in data_to_save
                                if (row['lat'] >= lat_south and row['lat'] <= lat_north and
                                    row['lon'] >= lon_west and row['lon'] <= lon_east)
                            ]
                            if day_number == 10:  # Report once
                                print(f"    🔍 Area filtering: {original_count} → {len(data_to_save)} stations ({100*len(data_to_save)/original_count:.1f}%)")
                    
                    df_batch = pd.DataFrame(data_to_save)
                    filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}_day{fday}"
                    # Write a uniquely-named batch file (avoids growing read-append-write)
                    batch_file = output_path / f"{filename_base}_batch_{day_number:04d}.parquet"
                    df_batch.to_parquet(batch_file, index=False, compression='snappy')
                    
                    del df_batch
                    data_by_forecast_day[fday] = []  # Clear memory for this day
            
            # Force garbage collection
            import gc
            gc.collect()
            print(f"  💾 Saved forecast day batches at calendar day {day_number}")
        
        current_dt += timedelta(days=1)
    
    print(f"\n  Extraction complete:")
    
    if save_format == 'pandas':
        import gc
        
        # Save any remaining data by forecast day
        print(f"\n  Finalizing forecast day files...")
        saved_files = []
        total_rows_by_day = {}
        
        for fday in sorted(data_by_forecast_day.keys()):
            if len(data_by_forecast_day[fday]) > 0:
                # Apply area filtering if configured
                data_to_save = data_by_forecast_day[fday]
                if area_config:
                    area_bbox = get_area_bbox(area_config)
                    if area_bbox:
                        original_count = len(data_to_save)
                        # Filter by lat/lon
                        lat_north, lon_west, lat_south, lon_east = area_bbox
                        data_to_save = [
                            row for row in data_to_save
                            if (row['lat'] >= lat_south and row['lat'] <= lat_north and
                                row['lon'] >= lon_west and row['lon'] <= lon_east)
                        ]
                        print(f"    🔍 Area filtering day {fday}: {original_count} → {len(data_to_save)} stations ({100*len(data_to_save)/original_count:.1f}%)")
                
                df_final = pd.DataFrame(data_to_save)
                filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}_day{fday}"
                final_file = output_path / f"{filename_base}.parquet"
                
                # Collect all numbered batch files for this forecast day
                batch_files = sorted(output_path.glob(f"{filename_base}_batch_*.parquet"))
                if batch_files:
                    dfs = [pd.read_parquet(f) for f in batch_files]
                    if len(df_final) > 0:
                        dfs.append(df_final)
                    df_final = pd.concat(dfs, ignore_index=True)
                    del dfs
                    for f in batch_files:
                        f.unlink()
                
                # Save final file for this forecast day
                df_final.to_parquet(final_file, index=False, compression='snappy')
                total_rows_by_day[fday] = len(df_final)
                saved_files.append(final_file)
                print(f"    ✓ Day {fday}: {len(df_final):,} rows → {final_file.name}")
                del df_final
                gc.collect()
        
        # Also finalize any batch files for forecast days with no remaining in-memory data
        import re
        batch_pattern = f"{variable}_{fc1_name}_vs_{fc2_name}_day*_batch_*.parquet"
        # Group batch files by forecast day
        fday_batches = {}
        for batch_file in sorted(output_path.glob(batch_pattern)):
            match = re.search(r'_day(\d+)_batch_', batch_file.name)
            if match:
                fday = int(match.group(1))
                fday_batches.setdefault(fday, []).append(batch_file)
        for fday, batch_files in fday_batches.items():
            if fday not in total_rows_by_day:
                try:
                    dfs = [pd.read_parquet(f) for f in sorted(batch_files)]
                    df = pd.concat(dfs, ignore_index=True)
                    del dfs
                    final_file = output_path / f"{variable}_{fc1_name}_vs_{fc2_name}_day{fday}.parquet"
                    df.to_parquet(final_file, index=False, compression='snappy')
                    total_rows_by_day[fday] = len(df)
                    saved_files.append(final_file)
                    print(f"    ✓ Day {fday}: {len(df):,} rows → {final_file.name}")
                    del df
                    for f in batch_files:
                        f.unlink()
                    gc.collect()
                except Exception as e:
                    print(f"    ⚠️  Warning: Could not finalize day {fday} batches: {e}")
                    for f in batch_files:
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    gc.collect()
        
        # Print summary
        total_rows = sum(total_rows_by_day.values())
        print(f"\n  Extraction Summary:")
        print(f"    {'='*60}")
        print(f"    Total extracted pairs: {total_rows:,}")
        print(f"    Saved {len(saved_files)} forecast day files")
        for fday in sorted(total_rows_by_day.keys()):
            print(f"      Day {fday}: {total_rows_by_day[fday]:,} rows")
        print(f"    {'='*60}")
        print(f"    ℹ Files organized by forecast day for memory efficiency")
        print(f"    ℹ Filter step will process one day at a time")
        print(f"    {'='*60}")
        
        if errors > 0:
            print(f"    ⚠ Skipped: {errors} steps (no data or errors)")
    
    else:
        # Legacy mode: individual .geo files
        print(f"    ✓ Processed {processed_pairs} forecast-observation sets")
        if errors > 0:
            print(f"    ⚠ Errors/skipped: {errors} files")
        if processed_pairs == 0:
            raise RuntimeError("No data was extracted! Check paths and date range.")


def run_step3(config, paths, preprocess_settings):
    """
    Execute Step 3: Extract Point Data
    Returns path to extracted points and model names
    """
    print("\n" + "="*80)
    print("STEP 3: EXTRACT POINT DATA")
    print("="*80)
    
    cfg = config['extract_points']
    output_path = Path(cfg['output_path'])
    
    print(f"\nOutput directory: {output_path}")
    
    # Determine lead times from config
    if 'forecast_days' in config and config['forecast_days'] is not None:
        # Use forecast days mode
        forecast_days = config['forecast_days']
        frequency = config.get('lead_time_frequency', 6)  # Default 6h
        
        steps = []
        step_to_forecast_day = {}  # Map each step to its requested forecast day
        
        for day in forecast_days:
            # Day 1 = 1-24h, Day 2 = 25-48h, etc. (matches: day = ((lt-1)//24)+1)
            # Generate steps within each day
            day_start = (day - 1) * 24 + 1  # 1, 25, 49, 73, 97, ...
            day_end = day * 24 + 1          # 25, 49, 73, 97, 121, ... (exclusive)
            
            # Generate steps at specified frequency within the day
            # Example: Day 1, freq=6 → [1, 7, 13, 19] or with offset [6, 12, 18, 24]
            # Adjust start to align with frequency (e.g., start at 6 for freq=6)
            if frequency > 1:
                offset = frequency - (day_start % frequency) if (day_start % frequency) != 0 else 0
                day_start += offset
            day_steps = list(range(day_start, day_end, frequency))
            
            # Add steps and map to forecast day
            for step in day_steps:
                if step not in steps:
                    steps.append(step)
                    step_to_forecast_day[step] = day  # Map step to its forecast day
        
        # Remove step 0 for precipitation variables (need 24h accumulation)
        if config['variable'] in ['tp24', 'tp'] and 0 in steps:
            steps.remove(0)
            if 0 in step_to_forecast_day:
                del step_to_forecast_day[0]
        
        steps = sorted(steps)
        print(f"Lead time mode: Forecast days {forecast_days} with {frequency}h frequency")
        print(f"Computed steps: {steps}")
        
        # Store mapping for later use
        config['_step_to_forecast_day'] = step_to_forecast_day
    else:
        # Use explicit steps
        steps = config['steps']
        print(f"Lead time mode: Explicit steps {steps}")
    
    # Note: area filtering happens in step 4, not during extraction
    # We extract ALL global data here for maximum reusability
    print(f"Extraction: Global (no area/orography filtering)")
    
    save_format = cfg.get('save_format', 'pandas')
    
    # Get model names from paths
    fc1_name = paths['fc1_name']
    fc2_name = paths['fc2_name']
    
    # Extract points for BOTH models
    extract_points(
        config=config,
        variable=config['variable'],
        fc1_path=paths['fc1_path'],
        fc2_path=paths['fc2_path'],
        fc1_name=fc1_name,
        fc2_name=fc2_name,
        obs_path=paths['obs_path'],
        output_path=str(output_path),
        start_date=config['start_date'],
        end_date=config['end_date'],
        steps=steps,
        preprocess_settings=preprocess_settings,
        save_format=save_format
    )
    
    print("\n✓ Step 3 complete")
    
    return {
        'output_path': output_path,
        'save_format': save_format,
        'fc1_name': fc1_name,
        'fc2_name': fc2_name
    }
