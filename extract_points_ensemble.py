"""
STEP 3 (ENSEMBLE): EXTRACT ENSEMBLE POINT DATA
================================================
Extract ensemble forecast members at observation locations using nearest gridpoint.
Produces a parquet file per forecast day with columns:
  date, step, valid_time, station_id, lat, lon, obs_value, 
  fc1_member_0..fc1_member_50, fc2_member_0..fc2_member_50, forecast_day
  (member 0 = control forecast, members 1-50 = perturbed)

Performance: uses batch mv.read(data=, step=) to get all members at once per step,
then batch mv.nearest_gridpoint on the entire fieldset. This avoids the per-member
select bottleneck (~2s per mv.read call × 51 members × 11 steps × 2 models).
"""

import metview as mv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import gc
import shutil


def _extract_raw_members_at_step(grib_data, step, obs_lats, obs_lons,
                                 include_control, n_members, model_prefix,
                                 unit_factor=1000.0, grib_param='tp'):
    """Extract raw (cumulative) values for all ensemble members at one step.

    Returns:
        (result_dict, member_map) where
          result_dict: {'{model_prefix}_member_{m}': np.array}
          member_map:  {member_number: np.array} raw values for later subtraction
        Both empty dicts on failure.
    """
    try:
        step_fields = mv.read(data=grib_data, param=grib_param, step=step)
    except Exception:
        return {}, {}

    n_fields = len(step_fields)
    if n_fields == 0:
        return {}, {}

    # Batch interpolation: shape (n_fields, n_stations)
    try:
        vals_2d = np.array(mv.nearest_gridpoint(step_fields, obs_lats, obs_lons))
    except Exception:
        return {}, {}

    if vals_2d.ndim == 1:
        vals_2d = vals_2d.reshape(1, -1)

    # Build member→row mapping from GRIB metadata.
    member_map = {}  # member_number -> row index
    for i in range(n_fields):
        try:
            ftype = mv.grib_get_string(step_fields[i], 'type')
            fnum = int(mv.grib_get_long(step_fields[i], 'number'))
        except Exception:
            continue
        if ftype == 'cf':
            member_map[0] = i
        else:
            member_map[fnum] = i

    # Convert units (e.g. m → mm). Factor is 1000 for IFS/HRES, 1.0 for AIFS (already mm).
    vals_2d = vals_2d * unit_factor

    # Build raw cache (per-member arrays for potential subtraction later)
    raw_cache = {}
    for m_num, row_idx in member_map.items():
        raw_cache[m_num] = vals_2d[row_idx].copy()

    # Build result dict with expected member column names
    result = {}
    if include_control and 0 in member_map:
        result[f'{model_prefix}_member_0'] = vals_2d[member_map[0]]
    for m in range(1, n_members + 1):
        if m in member_map:
            result[f'{model_prefix}_member_{m}'] = vals_2d[member_map[m]]

    return result, raw_cache


def extract_ensemble_points(config, variable, fc1_path, fc2_path, fc1_name, fc2_name,
                            obs_path, output_path, start_date, end_date, steps):
    """
    Extract ensemble forecast members at observation station locations.
    
    For each date/step: reads all ensemble members (cf + 50 pf), extracts values
    at observation locations via nearest_gridpoint, saves as wide-format parquet.
    
    Args:
        config: full config dict
        variable: str, e.g. 'tp24'
        fc1_path, fc2_path: Path to ensemble GRIB directories
        fc1_name, fc2_name: model label strings
        obs_path: path to observation .gpt files
        output_path: where to save parquet files
        start_date, end_date: 'YYYY-MM-DD' strings
        steps: list of int forecast hours, e.g. [0,24,48,...,240]
    """
    fc1_path = Path(fc1_path)
    fc2_path = Path(fc2_path)
    obs_path = Path(obs_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Ensemble config
    ens_cfg = config.get('ensemble', {})
    n_members = ens_cfg.get('n_members', 50)  # number of perturbed members
    include_control = ens_cfg.get('include_control', True)

    # Area filtering
    from extract_points import get_area_bbox, filter_extracted_data_by_area
    cfg_extract = config.get('extract_points', {})
    area_config = cfg_extract.get('area', None)
    area_bbox = get_area_bbox(area_config) if area_config else None

    # Load auxiliary fields for filtering (sdfor for orography, lsm for coastal)
    cfg_aux = config.get('auxiliary_fields', {})
    sdfor_field = None
    try:
        sdfor_path = cfg_aux.get('sdfor_path', '/ec/vol/destine/continuous_evaluation/sdfor_tco1279.grib')
        sdfor_field = mv.read(sdfor_path)
        print(f"  Loaded sdfor for terrain classification: {sdfor_path}")
    except Exception as e:
        print(f"  Warning: Could not load sdfor data: {e}")

    lsm_field = None
    try:
        lsm_path = cfg_aux.get('lsm_path', cfg_aux.get('model1', {}).get('lsm_path', '/ec/vol/destine/continuous_evaluation/lsm_tco1279.grib'))
        lsm_field = mv.read(lsm_path)
        print(f"  Loaded LSM for coastal filtering: {lsm_path}")
    except Exception as e:
        print(f"  Warning: Could not load LSM data: {e}")

    print(f"  Ensemble extraction: {n_members} perturbed members + control={include_control}")
    print(f"  Model 1: {fc1_name} ({fc1_path})")
    print(f"  Model 2: {fc2_name} ({fc2_path})")
    print(f"  Dates: {start_date} to {end_date}")
    print(f"  Steps: {steps}")
    if area_bbox:
        print(f"  Area filter: {area_config} -> {area_bbox}")

    total_days = (end_dt - start_dt).days + 1
    data_by_forecast_day = {}

    # Determine forecast_days from config or steps
    forecast_days_config = config.get('forecast_days', None)

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y%m%d')
        day_number = (current_dt - start_dt).days + 1
        print(f"\n  [{day_number}/{total_days}] {date_str}", flush=True)

        # Skip dates that were already successfully extracted in a previous run.
        # A date is considered complete if at least one tmp file exists for it.
        # (All day-files for a date are written atomically in the flush block below.)
        tmp_dir_check = output_path / '_tmp'
        if tmp_dir_check.is_dir():
            existing_tmp = sorted(tmp_dir_check.glob(f"{date_str}_day*.parquet"))
            if existing_tmp:
                print(f"    ✓ Already extracted ({len(existing_tmp)} day files). Skipping.", flush=True)
                current_dt += timedelta(days=1)
                continue

        # Determine file patterns based on variable
        fc1_file = None
        fc2_file = None
        fc1_u_file = None
        fc1_v_file = None
        fc2_u_file = None
        fc2_v_file = None
        wind_mode = False

        if variable == '10ff':
            # Wind speed: need u and v component files
            wind_mode = True
            for patt_u, patt_v in [
                (f"10u_{fc1_name}_{date_str}.grib", f"10v_{fc1_name}_{date_str}.grib"),
                (f"10u_{date_str}.grib", f"10v_{date_str}.grib"),
            ]:
                if (fc1_path / patt_u).exists() and (fc1_path / patt_v).exists():
                    fc1_u_file = fc1_path / patt_u
                    fc1_v_file = fc1_path / patt_v
                    break
            if fc1_u_file is None:
                # Glob fallback: find any 10u_*_{date}.grib in the directory
                u_matches = sorted(fc1_path.glob(f"10u_*_{date_str}.grib"))
                v_matches = sorted(fc1_path.glob(f"10v_*_{date_str}.grib"))
                if u_matches and v_matches:
                    fc1_u_file = u_matches[0]
                    fc1_v_file = v_matches[0]
            for patt_u, patt_v in [
                (f"10u_{fc2_name}_{date_str}.grib", f"10v_{fc2_name}_{date_str}.grib"),
                (f"10u_{date_str}.grib", f"10v_{date_str}.grib"),
            ]:
                if (fc2_path / patt_u).exists() and (fc2_path / patt_v).exists():
                    fc2_u_file = fc2_path / patt_u
                    fc2_v_file = fc2_path / patt_v
                    break
            if fc2_u_file is None:
                # Glob fallback: find any 10u_*_{date}.grib in the directory
                u_matches = sorted(fc2_path.glob(f"10u_*_{date_str}.grib"))
                v_matches = sorted(fc2_path.glob(f"10v_*_{date_str}.grib"))
                if u_matches and v_matches:
                    fc2_u_file = u_matches[0]
                    fc2_v_file = v_matches[0]
            if fc1_u_file is None or fc2_u_file is None:
                missing = []
                if fc1_u_file is None:
                    missing.append(f"{fc1_name}: 10u/10v")
                if fc2_u_file is None:
                    missing.append(f"{fc2_name}: 10u/10v")
                print(f"    SKIP - missing wind components: {', '.join(missing)}")
                current_dt += timedelta(days=1)
                continue
        elif variable == '2t':
            # Temperature
            for patt in [f"2t_{fc1_name}_{date_str}.grib", f"2t_{date_str}.grib"]:
                if (fc1_path / patt).exists():
                    fc1_file = fc1_path / patt
                    break
            for patt in [f"2t_{fc2_name}_{date_str}.grib", f"2t_{date_str}.grib"]:
                if (fc2_path / patt).exists():
                    fc2_file = fc2_path / patt
                    break
        else:
            # Precipitation (tp24) or other
            for patt in [f"tp_{fc1_name}_{date_str}.grib", f"tp24_{fc1_name}_{date_str}.grib", f"tp_{date_str}.grib", f"tp24_{date_str}.grib"]:
                if (fc1_path / patt).exists():
                    fc1_file = fc1_path / patt
                    break
            # Glob fallback: handle names like "ifsens4aifs_0.25degree" where the
            # resolution suffix appears AFTER the date in the filename
            # e.g. tp_ifsens4aifs_20251021_0.25degree.grib
            if fc1_file is None:
                fc1_base = fc1_name.rsplit('_', 1)[0] if '_' in fc1_name and 'degree' in fc1_name.rsplit('_', 1)[-1] else fc1_name
                for patt in [f"tp_{fc1_base}_{date_str}*.grib", f"tp24_{fc1_base}_{date_str}*.grib"]:
                    matches = sorted(fc1_path.glob(patt))
                    if matches:
                        fc1_file = matches[0]
                        break
            for patt in [f"tp_{fc2_name}_{date_str}.grib", f"tp24_{fc2_name}_{date_str}.grib", f"tp_{date_str}.grib", f"tp24_{date_str}.grib"]:
                if (fc2_path / patt).exists():
                    fc2_file = fc2_path / patt
                    break
            if fc2_file is None:
                fc2_base = fc2_name.rsplit('_', 1)[0] if '_' in fc2_name and 'degree' in fc2_name.rsplit('_', 1)[-1] else fc2_name
                for patt in [f"tp_{fc2_base}_{date_str}*.grib", f"tp24_{fc2_base}_{date_str}*.grib"]:
                    matches = sorted(fc2_path.glob(patt))
                    if matches:
                        fc2_file = matches[0]
                        break

        if not wind_mode and (fc1_file is None or fc2_file is None):
            missing = []
            if fc1_file is None:
                missing.append(f"{fc1_name}")
            if fc2_file is None:
                missing.append(f"{fc2_name}")
            print(f"    SKIP - missing: {', '.join(missing)}")
            current_dt += timedelta(days=1)
            continue

        # Read GRIB files once per day
        try:
            if wind_mode:
                fc1_u_grib = mv.read(str(fc1_u_file))
                fc1_v_grib = mv.read(str(fc1_v_file))
                fc2_u_grib = mv.read(str(fc2_u_file))
                fc2_v_grib = mv.read(str(fc2_v_file))
                fc1_grib = None
                fc2_grib = None
                print(f"    Loaded wind GRIBs: {fc1_name}(u+v), {fc2_name}(u+v)", flush=True)
            else:
                fc1_grib = mv.read(str(fc1_file))
                fc2_grib = mv.read(str(fc2_file))
                print(f"    Loaded GRIBs: {fc1_name}({len(fc1_grib)} msgs), {fc2_name}({len(fc2_grib)} msgs)", flush=True)
        except Exception as e:
            print(f"    ERROR reading GRIBs: {e}")
            current_dt += timedelta(days=1)
            continue

        # Read observation file for this date
        obs_file = None
        if variable == 'tp24':
            obs_patterns = [
                obs_path / f"tp24_obs_{date_str}00.geo",
                obs_path / f"tp24_{date_str}.gpt",
                obs_path / f"tp_{date_str}.gpt",
                obs_path / f"synop_{date_str}.gpt",
            ]
        elif variable == '2t':
            obs_patterns = [
                obs_path / f"2t_obs_{date_str}00.geo",
                obs_path / f"2t_{date_str}.gpt",
                obs_path / f"synop_{date_str}.gpt",
            ]
        elif variable == '10ff':
            obs_patterns = [
                obs_path / f"10ff_obs_{date_str}00.geo",
                obs_path / f"10ff_{date_str}.gpt",
                obs_path / f"synop_{date_str}.gpt",
            ]
        else:
            obs_patterns = [obs_path / f"synop_{date_str}.gpt"]
        for obs_pattern in obs_patterns:
            if obs_pattern.exists():
                obs_file = obs_pattern
                break

        # If obs is a single file for all dates, try that approach
        if obs_file is None:
            # Try loading obs from gpt files matching the step
            pass

        # Per-model unit conversion factors
        cfg_rd = config.get('read_data', {})
        fc1_unit_factor = cfg_rd.get('forecast_model1', {}).get('unit_conversion_factor', 1.0)
        fc2_unit_factor = cfg_rd.get('forecast_model2', {}).get('unit_conversion_factor', 1.0)

        # Variable-specific settings
        # Precipitation deaccumulation
        accum_hours = config.get('preprocess', {}).get('precipitation_accumulation_hours', None)
        need_deaccum = (variable.startswith('tp') and accum_hours is not None)
        if need_deaccum:
            print(f"    Deaccumulation: {accum_hours}h periods", flush=True)

        # Temperature: Kelvin→Celsius conversion, lapse-rate correction
        need_kelvin_to_celsius = (variable == '2t')
        preprocess_cfg = config.get('preprocess', {})
        need_lapse_rate = (variable == '2t' and preprocess_cfg.get('lapse_rate_correction', False))
        lapse_rate = preprocess_cfg.get('lapse_rate', -0.0065)

        # Load height fields for lapse-rate correction (once per day)
        height_field_fc1 = None
        height_field_fc2 = None
        if need_lapse_rate:
            try:
                orog_path_fc1 = cfg_aux.get('model1', {}).get('orog_path')
                if orog_path_fc1:
                    height_field_fc1 = mv.read(orog_path_fc1) / 9.80665
                    print(f"    Loaded orography for {fc1_name}")
            except Exception as e:
                print(f"    Warning: Could not load {fc1_name} orography: {e}")
            try:
                orog_path_fc2 = cfg_aux.get('model2', {}).get('orog_path')
                if orog_path_fc2:
                    height_field_fc2 = mv.read(orog_path_fc2) / 9.80665
                    print(f"    Loaded orography for {fc2_name}")
            except Exception as e:
                print(f"    Warning: Could not load {fc2_name} orography: {e}")

        # GRIB parameter to read
        if variable == '2t':
            grib_param = '2t'
        elif variable == '10ff':
            grib_param = None  # handled separately for u/v
        else:
            grib_param = 'tp'

        for step in steps:
            forecast_day = ((step - 1) // 24) + 1 if step > 0 else 1

            # Skip if not in requested forecast days
            if forecast_days_config and forecast_day not in forecast_days_config:
                continue

            valid_dt = current_dt + timedelta(hours=step)
            valid_date_str = valid_dt.strftime('%Y%m%d')

            # Read observation for the valid time
            obs_gpt = None
            if variable == 'tp24':
                obs_valid_patterns = [
                    obs_path / f"tp24_obs_{valid_date_str}00.geo",
                    obs_path / f"tp24_{valid_date_str}.gpt",
                    obs_path / f"tp_{valid_date_str}.gpt",
                    obs_path / f"synop_{valid_date_str}.gpt",
                ]
            elif variable == '2t':
                obs_valid_patterns = [
                    obs_path / f"2t_obs_{valid_date_str}00.geo",
                    obs_path / f"2t_{valid_date_str}.gpt",
                    obs_path / f"synop_{valid_date_str}.gpt",
                ]
            elif variable == '10ff':
                obs_valid_patterns = [
                    obs_path / f"10ff_obs_{valid_date_str}00.geo",
                    obs_path / f"10ff_{valid_date_str}.gpt",
                    obs_path / f"synop_{valid_date_str}.gpt",
                ]
            else:
                obs_valid_patterns = [obs_path / f"synop_{valid_date_str}.gpt"]
            for obs_pattern in obs_valid_patterns:
                if obs_pattern.exists():
                    try:
                        obs_gpt = mv.read(str(obs_pattern))
                        break
                    except Exception:
                        continue

            if obs_gpt is None:
                continue

            # Get observation station info
            try:
                obs_lats = mv.latitudes(obs_gpt)
                obs_lons = mv.longitudes(obs_gpt)
                obs_values = mv.values(obs_gpt)
                obs_ids = list(range(len(obs_lats)))
            except Exception as e:
                print(f"    Step {step}: obs read error: {e}")
                continue

            n_stations = len(obs_lats)
            if n_stations == 0:
                continue

            # Extract members at this step — variable-specific
            if wind_mode:
                # Wind speed: extract u and v separately, compute sqrt(u²+v²) per member
                fc1_u_vals, _ = _extract_raw_members_at_step(
                    fc1_u_grib, step, obs_lats, obs_lons,
                    include_control, n_members, 'fc1', fc1_unit_factor, grib_param='10u')
                fc1_v_vals, _ = _extract_raw_members_at_step(
                    fc1_v_grib, step, obs_lats, obs_lons,
                    include_control, n_members, 'fc1', fc1_unit_factor, grib_param='10v')
                fc2_u_vals, _ = _extract_raw_members_at_step(
                    fc2_u_grib, step, obs_lats, obs_lons,
                    include_control, n_members, 'fc2', fc2_unit_factor, grib_param='10u')
                fc2_v_vals, _ = _extract_raw_members_at_step(
                    fc2_v_grib, step, obs_lats, obs_lons,
                    include_control, n_members, 'fc2', fc2_unit_factor, grib_param='10v')

                if not fc1_u_vals or not fc1_v_vals or not fc2_u_vals or not fc2_v_vals:
                    print(f"    Step {step}: no valid wind members extracted")
                    continue

                # Compute wind speed per member
                fc1_member_values = {}
                for col in fc1_u_vals:
                    if col in fc1_v_vals:
                        fc1_member_values[col] = np.sqrt(fc1_u_vals[col]**2 + fc1_v_vals[col]**2)
                fc2_member_values = {}
                for col in fc2_u_vals:
                    if col in fc2_v_vals:
                        fc2_member_values[col] = np.sqrt(fc2_u_vals[col]**2 + fc2_v_vals[col]**2)
            else:
                # Standard scalar variable (tp, 2t)
                fc1_member_values, _ = _extract_raw_members_at_step(
                    fc1_grib, step, obs_lats, obs_lons,
                    include_control, n_members, 'fc1', fc1_unit_factor, grib_param=grib_param)
                fc2_member_values, _ = _extract_raw_members_at_step(
                    fc2_grib, step, obs_lats, obs_lons,
                    include_control, n_members, 'fc2', fc2_unit_factor, grib_param=grib_param)

            if not fc1_member_values or not fc2_member_values:
                print(f"    Step {step}: no valid members extracted")
                continue

            # Temperature: convert K → °C for all members
            if need_kelvin_to_celsius:
                for col in fc1_member_values:
                    fc1_member_values[col] = fc1_member_values[col] - 273.15
                for col in fc2_member_values:
                    fc2_member_values[col] = fc2_member_values[col] - 273.15

            # Temperature: lapse-rate correction per member
            if need_lapse_rate:
                obs_heights = np.zeros(n_stations)
                try:
                    obs_heights = np.array(mv.elevations(obs_gpt))
                except Exception:
                    pass

                if height_field_fc1 is not None:
                    fc1_model_heights = np.array(mv.nearest_gridpoint(height_field_fc1, obs_lats, obs_lons)).flatten()
                    height_diff_fc1 = obs_heights - fc1_model_heights
                    correction_fc1 = lapse_rate * height_diff_fc1
                    for col in fc1_member_values:
                        fc1_member_values[col] = fc1_member_values[col] + correction_fc1

                if height_field_fc2 is not None:
                    fc2_model_heights = np.array(mv.nearest_gridpoint(height_field_fc2, obs_lats, obs_lons)).flatten()
                    height_diff_fc2 = obs_heights - fc2_model_heights
                    correction_fc2 = lapse_rate * height_diff_fc2
                    for col in fc2_member_values:
                        fc2_member_values[col] = fc2_member_values[col] + correction_fc2

            # Deaccumulate: tp_Nh = tp[step] - tp[step - accum_hours]
            if need_deaccum and step > accum_hours:
                prev_step = step - accum_hours
                _, fc1_prev_raw = _extract_raw_members_at_step(
                    fc1_grib, prev_step, obs_lats, obs_lons,
                    include_control, n_members, 'fc1', fc1_unit_factor, grib_param=grib_param)
                _, fc2_prev_raw = _extract_raw_members_at_step(
                    fc2_grib, prev_step, obs_lats, obs_lons,
                    include_control, n_members, 'fc2', fc2_unit_factor, grib_param=grib_param)

                for col in list(fc1_member_values.keys()):
                    m_num = int(col.split('_')[-1])
                    if m_num in fc1_prev_raw:
                        fc1_member_values[col] = fc1_member_values[col] - fc1_prev_raw[m_num]
                for col in list(fc2_member_values.keys()):
                    m_num = int(col.split('_')[-1])
                    if m_num in fc2_prev_raw:
                        fc2_member_values[col] = fc2_member_values[col] - fc2_prev_raw[m_num]

            # Convert obs values
            obs_vals = np.array(obs_values)
            if need_kelvin_to_celsius:
                obs_vals = obs_vals - 273.15

            # Extract sdfor and lsm at observation locations
            sdfor_vals = [0.0] * n_stations
            if sdfor_field is not None:
                try:
                    sdfor_at_obs = mv.nearest_gridpoint(sdfor_field, obs_lats, obs_lons)
                    sdfor_vals = list(np.array(sdfor_at_obs).flatten())
                except Exception:
                    pass

            lsm_vals = [1.0] * n_stations
            if lsm_field is not None:
                try:
                    lsm_at_obs = mv.nearest_gridpoint(lsm_field, obs_lats, obs_lons)
                    lsm_vals = list(np.array(lsm_at_obs).flatten())
                except Exception:
                    pass

            # Build row data for this step
            row_data = {
                'date': [date_str] * n_stations,
                'step': [step] * n_stations,
                'valid_time': [valid_date_str] * n_stations,
                'station_id': list(obs_ids),
                'lat': list(obs_lats),
                'lon': list(obs_lons),
                'obs_value': list(obs_vals),
                'forecast_day': [forecast_day] * n_stations,
                'sdfor': sdfor_vals,
                'lsm': lsm_vals,
            }

            # Add member columns
            for col, vals in fc1_member_values.items():
                row_data[col] = list(vals)
            for col, vals in fc2_member_values.items():
                row_data[col] = list(vals)

            # Apply area filter
            if area_bbox:
                row_data = filter_extracted_data_by_area(row_data, area_bbox)
                if row_data is None:
                    continue

            # Accumulate by forecast day
            if forecast_day not in data_by_forecast_day:
                data_by_forecast_day[forecast_day] = []
            data_by_forecast_day[forecast_day].append(row_data)

            n_pts = len(row_data['lat'])
            n_fc1 = len(fc1_member_values)
            n_fc2 = len(fc2_member_values)
            print(f"    Step {step} (day{forecast_day}): {n_pts} stations, {n_fc1} fc1 members, {n_fc2} fc2 members", flush=True)

        # Free GRIB memory before next date
        if wind_mode:
            del fc1_u_grib, fc1_v_grib, fc2_u_grib, fc2_v_grib
        else:
            del fc1_grib, fc2_grib
        gc.collect()

        # Flush this date's data to temporary parquets immediately so that
        # we never hold all 160 days in RAM at once.
        tmp_dir = output_path / '_tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for day, chunks in data_by_forecast_day.items():
            if chunks:
                chunk_dfs = [pd.DataFrame(chunk) for chunk in chunks]
                df_tmp = pd.concat(chunk_dfs, ignore_index=True)
                # Downcast float64 → float32 to halve storage and speed up later reads
                for _col in df_tmp.select_dtypes(include='float64').columns:
                    df_tmp[_col] = df_tmp[_col].astype('float32')
                tmp_file = tmp_dir / f"{date_str}_day{day}.parquet"
                df_tmp.to_parquet(tmp_file, index=False)
                del chunk_dfs, df_tmp
        data_by_forecast_day.clear()
        gc.collect()

        current_dt += timedelta(days=1)

    # Merge temporary per-date parquets into final per-forecast-day files
    filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}_ens"
    saved_files = []
    tmp_dir = output_path / '_tmp'

    tmp_files = sorted(tmp_dir.glob('*_day*.parquet'))
    days_found = sorted(set(int(f.stem.split('_day')[1]) for f in tmp_files))

    for day in days_found:
        day_files = sorted(tmp_dir.glob(f'*_day{day}.parquet'))
        chunk_dfs = [pd.read_parquet(f) for f in day_files]
        df_day = pd.concat(chunk_dfs, ignore_index=True)
        out_file = output_path / f"{filename_base}_day{day}.parquet"
        df_day.to_parquet(out_file, index=False)
        saved_files.append(out_file)
        print(f"  Saved {out_file.name}: {len(df_day):,} rows, {len(df_day.columns)} columns")
        del chunk_dfs, df_day
        gc.collect()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  Total: {len(saved_files)} forecast day files saved to {output_path}")
    return saved_files


def run_step3_ensemble(config, paths, preprocess_settings):
    """
    Execute ensemble extraction (Step 3 for ensemble mode).
    Returns point_data_path dict compatible with filter step.
    """
    print("\n" + "="*80)
    print("STEP 3: EXTRACT ENSEMBLE POINT DATA")
    print("="*80)

    variable = config['variable']
    cfg_extract = config.get('extract_points', {})
    output_path = cfg_extract.get('output_path', f'./extracted_points/{variable}_ens')

    # Get steps
    steps = config.get('steps', [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240])
    if config.get('forecast_days'):
        freq = config.get('lead_time_frequency', 24)
        steps = []
        for day in config['forecast_days']:
            base = (day - 1) * 24
            steps.extend(range(base, base + 24, freq))

    saved_files = extract_ensemble_points(
        config=config,
        variable=variable,
        fc1_path=paths['fc1_path'],
        fc2_path=paths['fc2_path'],
        fc1_name=paths['fc1_name'],
        fc2_name=paths['fc2_name'],
        obs_path=paths.get('obs_path', ''),
        output_path=output_path,
        start_date=config['start_date'],
        end_date=config['end_date'],
        steps=steps,
    )

    print("\n✓ Step 3 (ensemble) complete")
    return {
        'output_path': Path(output_path),
        'save_format': 'pandas',
        'fc1_name': paths['fc1_name'],
        'fc2_name': paths['fc2_name'],
        'ensemble': True,
    }
