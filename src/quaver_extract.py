"""
QUAVER / VTB POINT-EXTRACTION BACKEND  (STEP 1 — DETERMINISTIC & ENSEMBLE)
=========================================================================
Selected via `backend: quaver_extract` in the config.

This module implements ONLY the first step of the scorecards-for-extremes
pipeline: extracting forecast values at observation station locations and
writing the per-forecast-day parquet files. Everything downstream
(threshold, filtering, scoring, plotting) continues to use the existing,
proven local pipeline unchanged.

Deterministic mode -> run_step3()          (one fc1_value/fc2_value per row)
Ensemble mode      -> run_step3_ensemble()  (fc{1,2}_member_0..N per row,
                      matching extract_points_ensemble.py's schema so the
                      ensemble scoring path works without modification).

How it works
------------
  1. Forecasts are retrieved fresh from MARS via VTB
     (`vtb.media.mars_retrieve`) — one day at a time.
  2. Observations are retrieved fresh from STVL via VTB
     (`vtb.media.stvl_retrieve`).
  3. Forecasts are interpolated to the observation station locations with
     `Fieldset.aligned()`. The interpolation method is configurable
     (nearest by default — VTB's native default; bilinear / IDW optional).
  4. Rows are written to one parquet file per forecast day, using the SAME
     schema the local extractor produces, so filter.py / scoring work
     without modification.

Resume support
--------------
Each date is flushed to `<output_path>/_tmp/<date>_day<N>.parquet`. On a
re-run, dates whose tmp files already exist are skipped. Once all dates are
processed the tmp files are merged into the final
`<variable>_<fc1>_vs_<fc2>_day<N>.parquet` files and _tmp is removed.

Requires (on ECMWF Atos):
    module load quaver/3.6.4      # provides vtb + metview
NB: metview must be imported AFTER vtb (handled by quaver_backend._load_vtb).
"""

import gc
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the proven, lazy VTB retrieval helpers from quaver_backend.
# (Importing quaver_backend does NOT import vtb — that happens lazily.)
from quaver_backend import (
    _load_vtb,
    _variable_to_param,
    _get_vtb_domain,
    _retrieve_one_day,
    _align_auxiliary_to_obs,
)
# ENS control-member stream/type resolver (cheap import — no vtb).
from mars_retrieve import _ens_control_stream_type
# Single source of truth for forecast steps — shared with the local extractor
# and MARS retrieval, so all three backends look at identical lead times.
from utils import compute_steps


# ---------------------------------------------------------------------------
# Interpolation selection
# ---------------------------------------------------------------------------
# Maps friendly config names to VTB's point_interpolators registry keys.
# `nearest_point` is VTB's own default (returned as None → let VTB pick it).
_INTERP_ALIASES = {
    'nearest': 'nearest_point',
    'nearest_point': 'nearest_point',
    'nn': 'nearest_point',
    'bilinear': 'linear_interpolation',
    'linear': 'linear_interpolation',
    'linear_interpolation': 'linear_interpolation',
    'idw': 'idw_interpolation',
    'idw_interpolation': 'idw_interpolation',
}


def _make_interpolator(method):
    """Return a VTB interpolator callable for `method`, or None for VTB's default.

    None means "use VTB's built-in default" (nearest_point).
    """
    if not method:
        return None
    key = _INTERP_ALIASES.get(str(method).lower().strip())
    if key is None:
        print(f"    [WARN] Unknown interpolation_method '{method}' — "
              f"falling back to VTB default (nearest_point)")
        return None
    if key == 'nearest_point':
        return None  # VTB default; no explicit interpolator needed
    try:
        from vtb.geo.point_interpolation import point_interpolators
        return point_interpolators[key]()
    except Exception as e:
        print(f"    [WARN] Could not build interpolator '{key}' ({e}) — "
              f"using VTB default (nearest_point)")
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _step_to_forecast_day(step_h):
    """Day 1 = 1..24h, Day 2 = 25..48h, ... (matches extract_points.py)."""
    return ((step_h - 1) // 24) + 1 if step_h > 0 else 1


def _first_col(df, names, default=None):
    """Return the values of the first column in `names` present in `df`."""
    for n in names:
        if n in df.columns:
            return df[n].values
    return default


def _k_to_c(arr):
    """Kelvin → Celsius, only where values look like Kelvin (> 100)."""
    arr = np.asarray(arr, dtype=float)
    return np.where(arr > 100.0, arr - 273.15, arr)


def _retrieve_one_day_param(config, model_key, date, steps, param):
    """Retrieve one day of forecast fields from MARS for an EXPLICIT param.

    Like quaver_backend._retrieve_one_day, but takes `param` directly instead
    of deriving it from `_variable_to_param(config['variable'])`. Needed for
    10ff, which must be retrieved as its raw 10u/10v components (NOT the
    derived '10si' diagnostic — that field isn't archived for research
    streams/expvers such as class=rd).
    """
    vtb, _ = _load_vtb()
    q = config['read_data'][model_key]['quaver']
    mars_kw = dict(
        parameter=param,
        levtype='sfc',
        date=date,
        step=steps,
        stream=q['stream'],
        type=q['type'],
        class_=q['class'],
        expver=q['expver'],
    )
    if q.get('grid'):
        mars_kw['grid'] = q['grid']
    if q.get('number'):
        mars_kw['number'] = q['number']
    return vtb.media.mars_retrieve(**mars_kw)


# ---------------------------------------------------------------------------
# Auxiliary fields (sdfor / lsm / orography)
# ---------------------------------------------------------------------------
# Policy (per user request): RETRIEVE from MARS first; READ a local GRIB file
# only as a fallback when retrieval is not possible.
#   * sdfor, lsm  -> single common field from the operational IFS static
#                    analysis (od/oper/an). These define the orography class
#                    and coastal masks applied identically to both models.
#   * orography   -> retrieved PER MODEL using that model's own MARS identity
#                    (class/expver/stream/grid) so the height matches the
#                    model's native resolution; falls back to the IFS static
#                    analysis, then to the configured local file.

def _load_grib(path):
    """Read a local GRIB file into a VTB Fieldset, or return None."""
    if path and os.path.exists(path):
        vtb, _ = _load_vtb()
        return vtb.Fieldset(path)
    return None


def _retrieve_orog_for_model(config, model_key, static_mars_kw, local_path):
    """Retrieve surface geopotential (orography) for one model.

    Order: model-specific MARS → local per-model GRIB file → IFS static analysis.

    The local per-model file (e.g. aifs_orog.grib) is the correct orography for
    this model and is tried BEFORE the generic IFS static analysis. This matters
    for data-driven models such as AIFS (class=ai), which do not archive a surface
    geopotential in MARS: without this ordering the retrieval would silently fall
    back to IFS orography and apply an IFS-height lapse-rate correction to the AIFS
    forecast, biasing results over complex terrain.
    """
    vtb, _ = _load_vtb()
    q = config['read_data'].get(model_key, {}).get('quaver', {})

    # 1) Model-specific retrieval (correct resolution for this model's grid).
    if q:
        try:
            date_str = str(config['start_date']).replace('-', '')[:8]
            kw = dict(
                parameter='z', levtype='sfc', step=0,
                date=date_str, time=0,
                class_=q.get('class', 'od'),
                expver=q.get('expver', '1'),
                stream=q.get('stream', 'oper'),
                type=q.get('type', 'fc'),
            )
            if q.get('model'):
                kw['model'] = q['model']    # e.g. 'aifs-ens' for AIFS (class=ai)
            if q.get('grid'):
                kw['grid'] = q['grid']
            fs = vtb.media.mars_retrieve(**kw)
            if fs is not None and len(fs) > 0:
                print(f"    → orog [{model_key}] retrieved from MARS "
                      f"(class={kw['class_']}, expver={kw['expver']})")
                return fs
        except Exception as e:
            print(f"    orog [{model_key}] model-specific MARS failed ({e})")

    # 2) Local per-model file (correct orography for this model, e.g. aifs_orog.grib).
    fs = _load_grib(local_path)
    if fs is not None:
        print(f"    → orog [{model_key}] loaded from local file {local_path}")
        return fs

    # 3) Operational IFS static analysis (last resort — NOT model-specific).
    try:
        fs = vtb.media.mars_retrieve(parameter='z', **static_mars_kw)
        if fs is not None and len(fs) > 0:
            print(f"    → orog [{model_key}] retrieved from MARS (IFS static "
                  f"analysis — WARNING: not model-specific orography)")
            return fs
    except Exception as e:
        print(f"    orog [{model_key}] IFS-static MARS failed ({e})")

    print(f"    ⚠ orog [{model_key}] unavailable — lapse-rate correction will not "
          f"use this model's height")
    return None


def _get_auxiliary_fields(config):
    """Retrieve sdfor, lsm and per-model orography (retrieve-first, read-fallback).

    Returns dict: {'sdfor', 'lsm', 'orog', 'orog_model2'} of VTB Fieldsets or None.
    """
    vtb, _ = _load_vtb()
    cfg_aux = config.get('auxiliary_fields', {})
    result = {'sdfor': None, 'lsm': None, 'orog': None, 'orog_model2': None}

    # Static operational IFS analysis request (shared by sdfor/lsm/orog fallback).
    # NOTE: vtb's date parser does NOT understand MARS's "latest" keyword (it tries
    # int()/pandas.to_datetime() and raises). sdfor/lsm/orography change extremely
    # rarely, so any valid analysis date works — use the run's own start_date.
    static_kw = dict(levtype='sfc', class_='od', stream='oper', type='an',
                     expver='1', date=config['start_date'], time=0, step=0)

    # --- sdfor (sub-grid orography std dev) ---
    try:
        print("  Retrieving sdfor from MARS...")
        result['sdfor'] = vtb.media.mars_retrieve(parameter='sdfor', **static_kw)
        print(f"    → sdfor retrieved ({len(result['sdfor'])} field(s))")
    except Exception as e:
        print(f"    sdfor MARS failed ({e}) — reading local file")
        result['sdfor'] = _load_grib(cfg_aux.get('sdfor_path'))
        if result['sdfor'] is None:
            print("    ⚠ sdfor unavailable — orography filtering will be skipped")

    # --- lsm (land-sea mask) ---
    try:
        print("  Retrieving lsm from MARS...")
        result['lsm'] = vtb.media.mars_retrieve(parameter='lsm', **static_kw)
        print(f"    → lsm retrieved ({len(result['lsm'])} field(s))")
    except Exception as e:
        print(f"    lsm MARS failed ({e}) — reading local file")
        lsm_path = cfg_aux.get('model1', {}).get('lsm_path') or cfg_aux.get('lsm_path')
        result['lsm'] = _load_grib(lsm_path)
        if result['lsm'] is None:
            print("    ⚠ lsm unavailable — coastal filtering will be skipped")

    # --- orography per model ---
    print("  Retrieving orography (surface geopotential)...")
    orog_path_m1 = cfg_aux.get('model1', {}).get('orog_path') or cfg_aux.get('orog_path')
    orog_path_m2 = cfg_aux.get('model2', {}).get('orog_path') or orog_path_m1
    result['orog'] = _retrieve_orog_for_model(
        config, 'forecast_model1', static_kw, orog_path_m1)
    result['orog_model2'] = _retrieve_orog_for_model(
        config, 'forecast_model2', static_kw, orog_path_m2)

    return result


# ---------------------------------------------------------------------------
# Per-date extraction
# ---------------------------------------------------------------------------
def _extract_one_date(config, date, step_hours, accum_hours,
                      interpolator, domain, aux_fields,
                      fc1_factor, fc2_factor, step_to_day_map=None):
    """Extract all steps for a single forecast date.

    Returns dict {forecast_day: [row, ...]}.
    """
    vtb, _ = _load_vtb()
    variable = config['variable']
    is_wind = (variable == '10ff')
    # 10ff: retrieve raw 10u/10v components (NOT the derived '10si' diagnostic —
    # that field isn't archived for research streams like class=rd/expver=iekm).
    # STVL observations are requested as '10ff' directly (stations report speed,
    # not components) — matches mars_retrieve.py's STVL convention for wind.
    param = '10ff' if is_wind else _variable_to_param(variable)
    q_obs = config['read_data'].get('quaver_obs', {})
    sources = q_obs.get('sources', ['synop'])
    is_temperature = (variable == '2t')
    step_to_day_map = step_to_day_map or {}

    # Forecast steps to actually retrieve (include the (step - accum) steps for
    # precipitation de-accumulation).
    if accum_hours > 0:
        all_h = set(step_hours)
        for h in step_hours:
            if h - accum_hours > 0:
                all_h.add(h - accum_hours)
        all_h = sorted(all_h)
    else:
        all_h = sorted(set(step_hours))
    all_fc_steps = pd.to_timedelta([f"{h}h" for h in all_h])

    # Retrieve both forecasts for this date.
    if is_wind:
        fc1_u_fs = _retrieve_one_day_param(config, 'forecast_model1', date, all_fc_steps, '10u')
        fc1_v_fs = _retrieve_one_day_param(config, 'forecast_model1', date, all_fc_steps, '10v')
        fc2_u_fs = _retrieve_one_day_param(config, 'forecast_model2', date, all_fc_steps, '10u')
        fc2_v_fs = _retrieve_one_day_param(config, 'forecast_model2', date, all_fc_steps, '10v')
    else:
        fc1_fs = _retrieve_one_day(config, 'forecast_model1', date, all_fc_steps)
        fc2_fs = _retrieve_one_day(config, 'forecast_model2', date, all_fc_steps)

    # Retrieve observations ONE VALID TIME PER CALL (not batched) — batching a
    # date list into a single stvl_retrieve() call makes vtb internally build one
    # Fieldset per date and then run them through Fieldset.aligned_fieldsets(),
    # which re-matches stations by [stationID, lat, lon] proximity ACROSS all the
    # batched dates together. Calling once per date avoids that cross-date
    # alignment step entirely (matches mars_retrieve.py's per-cycle behaviour for
    # methods 1/2). Confirmed 2026-08-04 via vtb source (media/stvl.py,
    # fieldset/fieldset.py) as the cause of a systematic obs-level divergence
    # between quaver_extract (method 3) and mars_retrieve (methods 1/2).
    #
    # IMPORTANT: keep each valid time's Fieldset SEPARATE in a dict (rather than
    # combining them with vtb.Fieldset(*list)) — that constructor runs its own
    # alignment/station-ID check across the whole list and intermittently raises
    # "Only aligned Fieldsets can be appended ... Station IDs differ" whenever
    # station reporting differs between valid times (common — confirmed
    # 2026-08-06 as the cause of ~2/3 method3 job failures). Each stvl_retrieve()
    # call already requests a single date=[vdt], so no further per-step filtering
    # is needed — just index the dict by vtime directly.
    vdates = sorted({date + pd.to_timedelta(f"{h}h") for h in step_hours})
    obs_by_vdt = {}
    for vdt in vdates:
        obs_kw = dict(table='observation', parameter=param, date=[vdt], sources=sources)
        if variable == 'tp24':
            obs_kw['period'] = pd.to_timedelta('24h')
        fs = vtb.media.stvl_retrieve(**obs_kw)
        if fs is not None and len(fs) > 0:
            obs_by_vdt[vdt] = fs
    if not obs_by_vdt:
        return {}

    rows_by_day = {}

    for h in step_hours:
        step_td = pd.to_timedelta(f"{h}h")
        vtime = date + step_td
        forecast_day = step_to_day_map.get(h, _step_to_forecast_day(h))

        # Observations valid at this step's validity time.
        obs_step = obs_by_vdt.get(vtime)
        if obs_step is None or len(obs_step) == 0:
            continue

        # Forecast field(s) for this step (de-accumulate precipitation; combine
        # 10u/10v into wind speed AFTER interpolating each component — for
        # nearest-point interpolation this is numerically identical to
        # combining before interpolation, since nearest is a pure selection).
        if is_wind:
            try:
                fc1_u_step = fc1_u_fs.header_filter(step=step_td)
                fc1_v_step = fc1_v_fs.header_filter(step=step_td)
                fc2_u_step = fc2_u_fs.header_filter(step=step_td)
                fc2_v_step = fc2_v_fs.header_filter(step=step_td)
            except Exception:
                continue
            if (len(fc1_u_step) == 0 or len(fc1_v_step) == 0 or
                    len(fc2_u_step) == 0 or len(fc2_v_step) == 0):
                continue
        else:
            try:
                fc1_step = fc1_fs.header_filter(step=step_td)
                fc2_step = fc2_fs.header_filter(step=step_td)
                if accum_hours > 0 and h > accum_hours:
                    prev_td = pd.to_timedelta(f"{h - accum_hours}h")
                    fc1_step = fc1_step - fc1_fs.header_filter(step=prev_td)
                    fc2_step = fc2_step - fc2_fs.header_filter(step=prev_td)
            except Exception:
                continue
            if len(fc1_step) == 0 or len(fc2_step) == 0:
                continue

        # Interpolate forecasts (and static aux fields) to the obs stations.
        # The template is obs_step, so the aligned rows share obs_step's order.
        try:
            if is_wind:
                fc1_u_at = fc1_u_step.aligned(obs_step, interpolator=interpolator)
                fc1_v_at = fc1_v_step.aligned(obs_step, interpolator=interpolator)
                fc2_u_at = fc2_u_step.aligned(obs_step, interpolator=interpolator)
                fc2_v_at = fc2_v_step.aligned(obs_step, interpolator=interpolator)
            else:
                fc1_at = fc1_step.aligned(obs_step, interpolator=interpolator)
                fc2_at = fc2_step.aligned(obs_step, interpolator=interpolator)
        except Exception as e:
            print(f"      [WARN] alignment failed at {vtime:%Y%m%d} step {h}h: {e}")
            continue
        aux = _align_auxiliary_to_obs(aux_fields, obs_step)

        obs_df = obs_step.to_dataframe()
        n = len(obs_df)
        if n == 0:
            continue

        stn = _first_col(obs_df, ['station_id', 'stnid'], default=np.arange(n))
        lats = _first_col(obs_df, ['latitude', 'lat'], default=np.full(n, np.nan))
        lons = _first_col(obs_df, ['longitude', 'lon'], default=np.full(n, np.nan))
        elev = _first_col(obs_df, ['elevation', 'altitude', 'height'],
                          default=np.zeros(n))
        obs_vals = _first_col(obs_df, ['value_0'], default=np.full(n, np.nan))

        if is_wind:
            fc1_u_df = fc1_u_at.to_dataframe()
            fc1_v_df = fc1_v_at.to_dataframe()
            fc2_u_df = fc2_u_at.to_dataframe()
            fc2_v_df = fc2_v_at.to_dataframe()
            fc1_u_vals = _first_col(fc1_u_df, ['value_0'], default=np.full(n, np.nan))
            fc1_v_vals = _first_col(fc1_v_df, ['value_0'], default=np.full(n, np.nan))
            fc2_u_vals = _first_col(fc2_u_df, ['value_0'], default=np.full(n, np.nan))
            fc2_v_vals = _first_col(fc2_v_df, ['value_0'], default=np.full(n, np.nan))
            fc1_vals = np.sqrt(fc1_u_vals ** 2 + fc1_v_vals ** 2) * fc1_factor
            fc2_vals = np.sqrt(fc2_u_vals ** 2 + fc2_v_vals ** 2) * fc2_factor
        else:
            fc1_df = fc1_at.to_dataframe()
            fc2_df = fc2_at.to_dataframe()
            fc1_vals = _first_col(fc1_df, ['value_0'], default=np.full(n, np.nan)) * fc1_factor
            fc2_vals = _first_col(fc2_df, ['value_0'], default=np.full(n, np.nan)) * fc2_factor

        # Unit conversions.
        if is_temperature:
            obs_vals = _k_to_c(obs_vals)
            fc1_vals = _k_to_c(fc1_vals)
            fc2_vals = _k_to_c(fc2_vals)

        sdfor_vals = aux.get('sdfor_values')
        lsm_vals = aux.get('lsm_values')
        fc1_h = aux.get('height_values')
        fc2_h = aux.get('height_values_model2')

        date_str = date.strftime('%Y%m%d')
        vtime_str = vtime.strftime('%Y%m%d')
        day_rows = rows_by_day.setdefault(forecast_day, [])

        if domain is not None:
            north, west, south, east = domain

        for i in range(n):
            obs_v = obs_vals[i]
            # Drop stations without a valid observation (STVL missing → NaN or huge).
            if obs_v is None or not np.isfinite(obs_v) or abs(obs_v) > 1e30:
                continue
            lat_i = float(lats[i])
            lon_i = float(lons[i])
            if domain is not None:
                if lat_i < south or lat_i > north or lon_i < west or lon_i > east:
                    continue
            day_rows.append({
                'date': date_str,
                'step': h,
                'valid_time': vtime_str,
                'station_id': stn[i],
                'lat': lat_i,
                'lon': lon_i,
                'obs_height': float(elev[i]) if elev is not None else 0.0,
                'fc1_height': float(fc1_h[i]) if fc1_h is not None and i < len(fc1_h) else 0.0,
                'fc2_height': float(fc2_h[i]) if fc2_h is not None and i < len(fc2_h) else 0.0,
                'sdfor': float(sdfor_vals[i]) if sdfor_vals is not None and i < len(sdfor_vals) else 0.0,
                'lsm': float(lsm_vals[i]) if lsm_vals is not None and i < len(lsm_vals) else 1.0,
                'obs_value': float(obs_v),
                'fc1_value': float(fc1_vals[i]),
                'fc2_value': float(fc2_vals[i]),
            })

    if is_wind:
        del fc1_u_fs, fc1_v_fs, fc2_u_fs, fc2_v_fs, obs_by_vdt
    else:
        del fc1_fs, fc2_fs, obs_by_vdt
    gc.collect()
    return rows_by_day


# ---------------------------------------------------------------------------
# Public entry point (same signature/return as extract_points.run_step3)
# ---------------------------------------------------------------------------
def run_step3(config, paths, preprocess_settings):
    """Extract point data via VTB/quaver and write per-forecast-day parquet files.

    Returns dict: {'output_path', 'save_format', 'fc1_name', 'fc2_name'}.
    """
    _load_vtb()  # fail fast with a clear error if vtb/metview are unavailable

    variable = config['variable']
    fc1_name = paths['fc1_name']
    fc2_name = paths['fc2_name']

    ep = config['extract_points']
    output_path = Path(ep['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path / '_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    save_format = ep.get('save_format', 'pandas')
    interp_method = ep.get('interpolation_method', 'nearest')
    interpolator = _make_interpolator(interp_method)

    dates = pd.date_range(config['start_date'], config['end_date'], freq='24h')
    # Use the SAME step logic as the local extractor (utils.compute_steps):
    # e.g. tp24 + forecast_days [1,3,5] -> steps [24, 72, 120] (step 0 dropped).
    step_hours, step_to_day_map = compute_steps(config)
    step_hours = sorted(set(step_hours))
    forecast_days = sorted({step_to_day_map.get(h, _step_to_forecast_day(h))
                            for h in step_hours})

    filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}"

    area = ep.get('area')
    domain = _get_vtb_domain(area) if area else None

    fc1_factor = config['read_data']['forecast_model1'].get('unit_conversion_factor', 1.0)
    fc2_factor = config['read_data']['forecast_model2'].get('unit_conversion_factor', 1.0)
    accum_hours = (preprocess_settings.get('precipitation_accumulation_hours', 24)
                   if variable == 'tp24' else 0)

    print("\n" + "=" * 80)
    print("STEP 3: EXTRACT POINT DATA  (backend = quaver_extract)")
    print("=" * 80)
    print(f"  Variable        : {variable}")
    print(f"  Models          : {fc1_name} vs {fc2_name}")
    print(f"  Interpolation   : {interp_method}"
          f"{' (VTB default nearest_point)' if interpolator is None and interp_method not in ('nearest','nearest_point','nn') else ''}")
    print(f"  Steps (h)       : {step_hours}  →  forecast days {forecast_days}")
    print(f"  Dates           : {dates[0]:%Y%m%d}..{dates[-1]:%Y%m%d} ({len(dates)} days)")
    print(f"  Output          : {output_path}")

    # Retrieve static auxiliary fields (sdfor, lsm, orog) once.
    print("\n  Retrieving auxiliary fields (sdfor, lsm, orography)...")
    aux_fields = _get_auxiliary_fields(config)

    # -------- per-date extraction with resume --------
    for date_idx, date in enumerate(dates):
        date_str = date.strftime('%Y%m%d')

        if list(tmp_dir.glob(f"{date_str}_day*.parquet")):
            print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) ✓ already extracted — skipping")
            continue

        print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) extracting...", flush=True)
        try:
            rows_by_day = _extract_one_date(
                config, date, step_hours, accum_hours,
                interpolator, domain, aux_fields, fc1_factor, fc2_factor,
                step_to_day_map=step_to_day_map,
            )
        except Exception as e:
            print(f"      [WARN] extraction failed for {date_str}: {e}")
            continue

        for day, rows in rows_by_day.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(np.float32)
            df.to_parquet(tmp_dir / f"{date_str}_day{day}.parquet",
                          compression='snappy', index=False)
        gc.collect()

    # -------- merge tmp → final per-day files --------
    print("\n  Merging per-date files into final forecast-day parquet files...")
    saved_files = []
    for day in forecast_days:
        parts = sorted(tmp_dir.glob(f"*_day{day}.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        final_file = output_path / f"{filename_base}_day{day}.parquet"
        df.to_parquet(final_file, compression='snappy', index=False)
        saved_files.append(final_file)
        print(f"    → {final_file.name}: {len(df):,} rows")
        del df
        gc.collect()

    if saved_files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print("    [WARN] No data extracted — keeping _tmp for inspection")

    print(f"\n  ✓ Extraction complete: {len(saved_files)} forecast-day file(s) written")

    return {
        'output_path': output_path,
        'save_format': save_format,
        'fc1_name': fc1_name,
        'fc2_name': fc2_name,
    }


# ===========================================================================
# ENSEMBLE SUPPORT
# ===========================================================================
def _retrieve_ensemble_one_day_param(config, model_key, date, steps, param,
                                     kind, n_members):
    """Retrieve one day of ENSEMBLE forecast fields for an EXPLICIT param.

    kind='perturbed' -> perturbed members (type=pf, number=1/to/n_members).
    kind='control'   -> single control member. stream/type come from
                        _ens_control_stream_type (enfo/cf, or oper/fc for the
                        operational IFS from Cycle 50r1 onwards).

    Explicit `param` (not derived from the variable) so 10ff can be retrieved as
    its raw 10u/10v components, exactly like the deterministic path.
    """
    vtb, _ = _load_vtb()
    q = config['read_data'][model_key]['quaver']
    date_str = (date.strftime('%Y%m%d') if hasattr(date, 'strftime')
                else str(date).replace('-', '')[:8])
    mars_kw = dict(
        parameter=param,
        levtype='sfc',
        date=date,
        step=steps,
        class_=q['class'],
        expver=q['expver'],
    )
    if kind == 'control':
        stream, mtype = _ens_control_stream_type(q, date_str)
        mars_kw['stream'] = stream
        mars_kw['type'] = mtype
    else:
        mars_kw['stream'] = q.get('stream', 'enfo')
        mars_kw['type'] = 'pf'
        mars_kw['number'] = q.get('number', f'1/to/{n_members}')
    if q.get('model'):
        mars_kw['model'] = q['model']       # e.g. 'aifs-ens' for AIFS (class=ai)
    if q.get('grid'):
        mars_kw['grid'] = q['grid']
    return vtb.media.mars_retrieve(**mars_kw)


def _safe_step_filter(fs, step_td):
    """header_filter(step=...) that returns None instead of raising / empty."""
    if fs is None:
        return None
    try:
        sub = fs.header_filter(step=step_td)
    except Exception:
        return None
    if sub is None or len(sub) == 0:
        return None
    return sub


def _member_value_arrays(step_fs, obs_step, interpolator):
    """Align a (multi-member, single-step) fieldset to obs stations and return
    the ordered list of per-member value arrays [value_0, value_1, ...].

    Returns [] on failure / empty input.
    """
    if step_fs is None or len(step_fs) == 0:
        return []
    try:
        at = step_fs.aligned(obs_step, interpolator=interpolator)
    except Exception:
        return []
    df = at.to_dataframe()
    cols = sorted((c for c in df.columns if c.startswith('value_')),
                  key=lambda c: int(c.split('_')[1]))
    return [np.asarray(df[c].values, dtype=float) for c in cols]


def _members_for_step(fs_dict, step_td, obs_step, interpolator, is_wind,
                      factor, n_members, include_control):
    """Return {member_number: value_array} for one model at one step.

    member 0 = control (only if include_control and available); 1..N = perturbed.
    Wind speed is computed per member as sqrt(u^2 + v^2) * factor.
    """
    out = {}
    if is_wind:
        u_arrs = _member_value_arrays(
            _safe_step_filter(fs_dict.get('u_pert'), step_td), obs_step, interpolator)
        v_arrs = _member_value_arrays(
            _safe_step_filter(fs_dict.get('v_pert'), step_td), obs_step, interpolator)
        for i in range(min(len(u_arrs), len(v_arrs))):
            out[i + 1] = np.sqrt(u_arrs[i] ** 2 + v_arrs[i] ** 2) * factor
        if include_control:
            uc = _member_value_arrays(
                _safe_step_filter(fs_dict.get('u_ctrl'), step_td), obs_step, interpolator)
            vc = _member_value_arrays(
                _safe_step_filter(fs_dict.get('v_ctrl'), step_td), obs_step, interpolator)
            if uc and vc:
                out[0] = np.sqrt(uc[0] ** 2 + vc[0] ** 2) * factor
    else:
        arrs = _member_value_arrays(
            _safe_step_filter(fs_dict.get('pert'), step_td), obs_step, interpolator)
        for i, a in enumerate(arrs):
            out[i + 1] = a * factor
        if include_control:
            c = _member_value_arrays(
                _safe_step_filter(fs_dict.get('ctrl'), step_td), obs_step, interpolator)
            if c:
                out[0] = c[0] * factor
    return out


def _extract_one_date_ensemble(config, date, step_hours, accum_hours,
                               interpolator, domain, aux_fields,
                               fc1_factor, fc2_factor, n_members,
                               include_control, step_to_day_map=None):
    """Extract all ensemble members for all steps of a single forecast date.

    Returns dict {forecast_day: [row, ...]} where each row carries
    fc1_member_0..N / fc2_member_0..N columns (schema matches
    extract_points_ensemble.py so the ensemble scoring path is unchanged).
    """
    vtb, _ = _load_vtb()
    variable = config['variable']
    is_wind = (variable == '10ff')
    is_temperature = (variable == '2t')
    # 10ff is retrieved as raw 10u/10v components (the derived '10si' diagnostic
    # isn't archived for research streams); STVL obs are requested as '10ff'.
    param = '10ff' if is_wind else _variable_to_param(variable)
    q_obs = config['read_data'].get('quaver_obs', {})
    sources = q_obs.get('sources', ['synop'])
    step_to_day_map = step_to_day_map or {}

    preprocess_cfg = config.get('preprocess', {})
    need_lapse = (is_temperature and preprocess_cfg.get('lapse_rate_correction', False))
    lapse_rate = preprocess_cfg.get('lapse_rate', -0.0065)

    # Steps to retrieve (+ the (step - accum) support steps for de-accumulation).
    if accum_hours > 0:
        all_h = sorted(set(step_hours) |
                       {h - accum_hours for h in step_hours if h - accum_hours > 0})
    else:
        all_h = sorted(set(step_hours))
    all_fc_steps = pd.to_timedelta([f"{h}h" for h in all_h])

    def _retrieve_model(model_key):
        """Retrieve ensemble fieldsets for one model → dict of Fieldsets."""
        d = {}
        if is_wind:
            d['u_pert'] = _retrieve_ensemble_one_day_param(
                config, model_key, date, all_fc_steps, '10u', 'perturbed', n_members)
            d['v_pert'] = _retrieve_ensemble_one_day_param(
                config, model_key, date, all_fc_steps, '10v', 'perturbed', n_members)
            d['u_ctrl'] = d['v_ctrl'] = None
            if include_control:
                try:
                    d['u_ctrl'] = _retrieve_ensemble_one_day_param(
                        config, model_key, date, all_fc_steps, '10u', 'control', n_members)
                    d['v_ctrl'] = _retrieve_ensemble_one_day_param(
                        config, model_key, date, all_fc_steps, '10v', 'control', n_members)
                except Exception as e:
                    print(f"      [WARN] control wind retrieval failed ({model_key}): {e}")
        else:
            d['pert'] = _retrieve_ensemble_one_day_param(
                config, model_key, date, all_fc_steps, param, 'perturbed', n_members)
            d['ctrl'] = None
            if include_control:
                try:
                    d['ctrl'] = _retrieve_ensemble_one_day_param(
                        config, model_key, date, all_fc_steps, param, 'control', n_members)
                except Exception as e:
                    print(f"      [WARN] control retrieval failed ({model_key}): {e}")
        return d

    fc1 = _retrieve_model('forecast_model1')
    fc2 = _retrieve_model('forecast_model2')

    # Retrieve observations ONE VALID TIME PER CALL (not batched) — see the
    # matching comment in the deterministic extraction path above for why:
    # batching a date list into one stvl_retrieve() call triggers vtb's internal
    # Fieldset.aligned_fieldsets() cross-date station-matching/merging, which a
    # per-date loop avoids entirely (matches mars_retrieve.py's per-cycle
    # behaviour for methods 1/2).
    #
    # IMPORTANT: keep each valid time's Fieldset SEPARATE in a dict (rather than
    # combining them with vtb.Fieldset(*list)) — see matching comment in
    # _extract_one_date above (fixed 2026-08-06: this concatenation intermittently
    # raised "Only aligned Fieldsets can be appended ... Station IDs differ").
    vdates = sorted({date + pd.to_timedelta(f"{h}h") for h in step_hours})
    obs_by_vdt = {}
    for vdt in vdates:
        obs_kw = dict(table='observation', parameter=param, date=[vdt], sources=sources)
        if variable == 'tp24':
            obs_kw['period'] = pd.to_timedelta('24h')
        fs = vtb.media.stvl_retrieve(**obs_kw)
        if fs is not None and len(fs) > 0:
            obs_by_vdt[vdt] = fs
    if not obs_by_vdt:
        return {}

    rows_by_day = {}

    for h in step_hours:
        step_td = pd.to_timedelta(f"{h}h")
        vtime = date + step_td
        forecast_day = step_to_day_map.get(h, _step_to_forecast_day(h))

        obs_step = obs_by_vdt.get(vtime)
        if obs_step is None or len(obs_step) == 0:
            continue

        try:
            fc1_members = _members_for_step(
                fc1, step_td, obs_step, interpolator, is_wind, fc1_factor,
                n_members, include_control)
            fc2_members = _members_for_step(
                fc2, step_td, obs_step, interpolator, is_wind, fc2_factor,
                n_members, include_control)
        except Exception as e:
            print(f"      [WARN] member extraction failed at {vtime:%Y%m%d} step {h}h: {e}")
            continue
        if not fc1_members or not fc2_members:
            continue

        # De-accumulation (tp24): subtract the (step - accum) members, per member.
        if accum_hours > 0 and h > accum_hours:
            prev_td = pd.to_timedelta(f"{h - accum_hours}h")
            fc1_prev = _members_for_step(
                fc1, prev_td, obs_step, interpolator, is_wind, fc1_factor,
                n_members, include_control)
            fc2_prev = _members_for_step(
                fc2, prev_td, obs_step, interpolator, is_wind, fc2_factor,
                n_members, include_control)
            for m in list(fc1_members):
                if m in fc1_prev:
                    fc1_members[m] = fc1_members[m] - fc1_prev[m]
            for m in list(fc2_members):
                if m in fc2_prev:
                    fc2_members[m] = fc2_members[m] - fc2_prev[m]

        obs_df = obs_step.to_dataframe()
        n = len(obs_df)
        if n == 0:
            continue
        stn = _first_col(obs_df, ['station_id', 'stnid'], default=np.arange(n))
        lats = _first_col(obs_df, ['latitude', 'lat'], default=np.full(n, np.nan))
        lons = _first_col(obs_df, ['longitude', 'lon'], default=np.full(n, np.nan))
        elev = _first_col(obs_df, ['elevation', 'altitude', 'height'], default=np.zeros(n))
        obs_vals = _first_col(obs_df, ['value_0'], default=np.full(n, np.nan))

        aux = _align_auxiliary_to_obs(aux_fields, obs_step)
        sdfor_vals = aux.get('sdfor_values')
        lsm_vals = aux.get('lsm_values')
        fc1_h = aux.get('height_values')
        fc2_h = aux.get('height_values_model2')

        # Temperature: Kelvin → Celsius (obs + all members), then lapse-rate.
        if is_temperature:
            obs_vals = _k_to_c(obs_vals)
            for m in fc1_members:
                fc1_members[m] = _k_to_c(fc1_members[m])
            for m in fc2_members:
                fc2_members[m] = _k_to_c(fc2_members[m])
            if need_lapse:
                obs_h = np.asarray(elev, dtype=float)
                if fc1_h is not None:
                    corr1 = lapse_rate * (obs_h - np.asarray(fc1_h, dtype=float))
                    for m in fc1_members:
                        fc1_members[m] = fc1_members[m] + corr1
                if fc2_h is not None:
                    corr2 = lapse_rate * (obs_h - np.asarray(fc2_h, dtype=float))
                    for m in fc2_members:
                        fc2_members[m] = fc2_members[m] + corr2

        # Quality cap: flag stations with unrealistic lapse corrections / height
        # diffs so they are skipped below (mirrors extract_points.py so method3
        # agrees with the deterministic and method1/2 extractors).
        cap_ok = np.ones(n, dtype=bool)
        if is_temperature and need_lapse:
            max_correction = 50.0      # °C
            max_height_diff = 10000.0  # m
            obs_h = np.asarray(elev, dtype=float)
            if fc1_h is not None and len(fc1_h) == n:
                hd1 = obs_h - np.asarray(fc1_h, dtype=float)
                cap_ok &= (np.abs(lapse_rate * hd1) <= max_correction) & (np.abs(hd1) <= max_height_diff)
            if fc2_h is not None and len(fc2_h) == n:
                hd2 = obs_h - np.asarray(fc2_h, dtype=float)
                cap_ok &= (np.abs(lapse_rate * hd2) <= max_correction) & (np.abs(hd2) <= max_height_diff)

        date_str = date.strftime('%Y%m%d')
        vtime_str = vtime.strftime('%Y%m%d')
        day_rows = rows_by_day.setdefault(forecast_day, [])
        if domain is not None:
            north, west, south, east = domain

        for i in range(n):
            obs_v = obs_vals[i]
            if obs_v is None or not np.isfinite(obs_v) or abs(obs_v) > 1e30:
                continue
            if not cap_ok[i]:
                continue
            lat_i = float(lats[i])
            lon_i = float(lons[i])
            if domain is not None:
                if lat_i < south or lat_i > north or lon_i < west or lon_i > east:
                    continue
            row = {
                'date': date_str,
                'step': h,
                'valid_time': vtime_str,
                'station_id': stn[i],
                'lat': lat_i,
                'lon': lon_i,
                'obs_value': float(obs_v),
                'forecast_day': forecast_day,
                'sdfor': float(sdfor_vals[i]) if sdfor_vals is not None and i < len(sdfor_vals) else 0.0,
                'lsm': float(lsm_vals[i]) if lsm_vals is not None and i < len(lsm_vals) else 1.0,
            }
            for m, arr in fc1_members.items():
                row[f'fc1_member_{m}'] = float(arr[i]) if i < len(arr) else np.nan
            for m, arr in fc2_members.items():
                row[f'fc2_member_{m}'] = float(arr[i]) if i < len(arr) else np.nan
            day_rows.append(row)

    del fc1, fc2, obs_by_vdt
    gc.collect()
    return rows_by_day


def run_step3_ensemble(config, paths, preprocess_settings):
    """Ensemble point extraction via VTB/quaver → per-forecast-day parquet files.

    Writes the SAME schema as extract_points_ensemble.py (fc{1,2}_member_0..N),
    so the downstream ensemble threshold/filter/scoring path is unchanged.

    Returns dict: {'output_path','save_format','fc1_name','fc2_name','ensemble'}.
    """
    _load_vtb()  # fail fast if vtb/metview are unavailable

    variable = config['variable']
    fc1_name = paths['fc1_name']
    fc2_name = paths['fc2_name']

    ep = config['extract_points']
    output_path = Path(ep['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path / '_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    save_format = ep.get('save_format', 'pandas')
    interp_method = ep.get('interpolation_method', 'nearest')
    interpolator = _make_interpolator(interp_method)

    ens_cfg = config.get('ensemble', {})
    n_members = ens_cfg.get('n_members', 50)
    include_control = ens_cfg.get('include_control', True)

    dates = pd.date_range(config['start_date'], config['end_date'], freq='24h')
    step_hours, step_to_day_map = compute_steps(config)
    step_hours = sorted(set(step_hours))
    forecast_days = sorted({step_to_day_map.get(h, _step_to_forecast_day(h))
                            for h in step_hours})

    filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}_ens"

    area = ep.get('area')
    domain = _get_vtb_domain(area) if area else None

    fc1_factor = config['read_data']['forecast_model1'].get('unit_conversion_factor', 1.0)
    fc2_factor = config['read_data']['forecast_model2'].get('unit_conversion_factor', 1.0)
    accum_hours = (preprocess_settings.get('precipitation_accumulation_hours', 24)
                   if variable == 'tp24' else 0)

    print("\n" + "=" * 80)
    print("STEP 3: EXTRACT ENSEMBLE POINT DATA  (backend = quaver_extract)")
    print("=" * 80)
    print(f"  Variable        : {variable}")
    print(f"  Models          : {fc1_name} vs {fc2_name}")
    print(f"  Members         : {n_members} perturbed + control={include_control}")
    print(f"  Interpolation   : {interp_method}")
    print(f"  Steps (h)       : {step_hours}  →  forecast days {forecast_days}")
    print(f"  Dates           : {dates[0]:%Y%m%d}..{dates[-1]:%Y%m%d} ({len(dates)} days)")
    print(f"  Output          : {output_path}")

    print("\n  Retrieving auxiliary fields (sdfor, lsm, orography)...")
    aux_fields = _get_auxiliary_fields(config)

    for date_idx, date in enumerate(dates):
        date_str = date.strftime('%Y%m%d')

        if list(tmp_dir.glob(f"{date_str}_day*.parquet")):
            print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) ✓ already extracted — skipping")
            continue

        print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) extracting...", flush=True)
        try:
            rows_by_day = _extract_one_date_ensemble(
                config, date, step_hours, accum_hours,
                interpolator, domain, aux_fields, fc1_factor, fc2_factor,
                n_members, include_control, step_to_day_map=step_to_day_map,
            )
        except Exception as e:
            print(f"      [WARN] extraction failed for {date_str}: {e}")
            continue

        for day, rows in rows_by_day.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(np.float32)
            df.to_parquet(tmp_dir / f"{date_str}_day{day}.parquet",
                          compression='snappy', index=False)
        gc.collect()

    print("\n  Merging per-date files into final forecast-day parquet files...")
    saved_files = []
    for day in forecast_days:
        parts = sorted(tmp_dir.glob(f"*_day{day}.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        final_file = output_path / f"{filename_base}_day{day}.parquet"
        df.to_parquet(final_file, compression='snappy', index=False)
        saved_files.append(final_file)
        print(f"    → {final_file.name}: {len(df):,} rows, {len(df.columns)} cols")
        del df
        gc.collect()

    if saved_files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print("    [WARN] No data extracted — keeping _tmp for inspection")

    print(f"\n  ✓ Ensemble extraction complete: {len(saved_files)} forecast-day file(s) written")

    return {
        'output_path': output_path,
        'save_format': save_format,
        'fc1_name': fc1_name,
        'fc2_name': fc2_name,
        'ensemble': True,
    }
