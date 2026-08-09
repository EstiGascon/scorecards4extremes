"""
cams_extract.py — Step 3 extraction backend for CAMS composition variables.

Reads raw per-site CAMS model NetCDF (<expid>/<expid>_YYYYMMDD_00.nc) and raw
observation NetCDF (AERONET for aod500, AirNow for the surface chemistry
species), aligns them per (site, forecast step), and writes the standard
pipeline parquet schema [date, step, forecast_day, valid_time, station_id,
lat, lon, obs_value, fc1_value, fc2_value, n_obs] — so no external
pre-processing / pre-built parquet is required.

Ported from the reference scripts (read_cams_data_aod_aeronet.py /
read_cams_data_airnow.py, /home/ecm7338/scorecards4extremes/cams_sample_data/)
into a single species-parameterised module, dispatched from run.py like the
other Step-3 backends (backend: 'cams_extract').

Data has no orography (sdfor) / land-sea-mask (lsm) columns — orography
stratification and coastal filtering are skipped downstream for these
variables (see filter.py).
"""

import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils import compute_steps

STEP_TOL_H = 1.5   # +/- window (h) for assigning an obs to a 3-hourly model step
M_AIR = 28.9644    # g/mol, dry air — used for gas kg/kg -> ppb/ppm conversion

# Species registry — add a new CAMS variable by adding one entry here.
#   kind='aeronet': AOD vs AERONET (unitless, no conversion).
#   kind='airnow':  surface chemistry vs AirNow (conv='conc' for PM: kg/m3*1e9
#                   -> ug/m3; conv='vmr' for gases: kg/kg*(M_AIR/molar)*scale
#                   -> ppb (scale=1e9) / ppm (scale=1e6)).
SPECIES = {
    "aod500": dict(kind="aeronet", model_var="aod500", obs_var="AOT-500nm"),
    "pm2p5":  dict(kind="airnow", model_var="pm2p5",     obs_var="PM2.5",  conv="conc"),
    "pm10":   dict(kind="airnow", model_var="pm10",      obs_var="PM10",   conv="conc"),
    "go3":    dict(kind="airnow", model_var="go3_ml137", obs_var="OZONE",  conv="vmr", molar=48.00,   scale=1e9),
    "no2":    dict(kind="airnow", model_var="no2_ml137", obs_var="NO2",    conv="vmr", molar=46.0055, scale=1e9),
    "so2":    dict(kind="airnow", model_var="so2_ml137", obs_var="SO2",    conv="vmr", molar=64.066,  scale=1e9),
    "co":     dict(kind="airnow", model_var="co_ml137",  obs_var="CO",     conv="vmr", molar=28.0101, scale=1e6),
    "no":     dict(kind="airnow", model_var="no_ml137",  obs_var="NO",     conv="vmr", molar=30.0061, scale=1e9),
}


# ---------------------------------------------------------------------------
# Low-level NetCDF readers
# ---------------------------------------------------------------------------
def _decode_ids(da):
    """Decode an xarray char/byte-string site-id variable to a stripped str array."""
    vals = da.values
    if vals.dtype.kind == "S":
        vals = np.char.decode(vals.astype("S"))
    return np.char.strip(vals.astype(str))


def _parse_time_units(units):
    """Parse 'hours since YYYY-MM-DD HH:MM:SS' -> (reference datetime, factor_hours)."""
    m = re.match(r"\s*(\w+)\s+since\s+(.+)", units.strip())
    unit, ref = m.group(1).lower(), m.group(2).strip()
    ref_dt = datetime.strptime(ref[:19], "%Y-%m-%d %H:%M:%S")
    factor = {"hours": 1.0, "minutes": 1 / 60.0, "days": 24.0, "seconds": 1 / 3600.0}[unit]
    return ref_dt, factor


def _to_obs_units(arr, spec):
    """Convert a model array from its native units to the obs units for `spec`."""
    if spec["kind"] != "airnow":
        return arr
    if spec["conv"] == "conc":
        return arr * 1e9                                    # kg/m3 -> ug/m3
    return arr * (M_AIR / spec["molar"]) * spec["scale"]    # kg/kg -> ppb/ppm


def read_model_file(spec, model_dir, expid, base_date, run_hour="00"):
    """Read one experiment's 00-run site file -> dict of values(site, step) + coords."""
    model_file = Path(model_dir) / f"{expid}_{base_date}_{run_hour}.nc"
    with xr.open_dataset(model_file, decode_times=False) as ds:
        site_ids = _decode_ids(ds["site_ids"])
        lats = np.asarray(ds["site_lats"].values, dtype=float)
        lons = np.asarray(ds["site_lons"].values, dtype=float)
        steps = np.asarray(ds["steps"].values, dtype=int)
        vals = np.asarray(ds[spec["model_var"]].values, dtype="float64")  # (site, step)

    vals = _to_obs_units(vals, spec)
    base_dt = datetime.strptime(base_date + run_hour, "%Y%m%d%H")
    return {"expid": expid, "site_ids": site_ids, "lats": lats, "lons": lons,
            "steps": steps, "values": vals, "base_dt": base_dt}


def _read_obs_day(spec, obs_dir, day_dt):
    """Read one daily obs file -> tidy frame (NaN obs dropped), or None if absent."""
    if spec["kind"] == "aeronet":
        f = Path(obs_dir) / f"Aeronet_AOT_L1.5_Table_{day_dt:%Y%m%d}.nc"
        engine = None
    else:
        f = Path(obs_dir) / f"airnow_{day_dt:%Y%m%d}.nc"
        engine = "h5netcdf"
    if not f.exists():
        return None

    kwargs = {"decode_times": False}
    if engine:
        kwargs["engine"] = engine
    with xr.open_dataset(f, **kwargs) as ds:
        ref_dt, factor = _parse_time_units(ds["time"].attrs["units"])
        t = np.asarray(ds["time"].values, dtype=float)
        times = ref_dt + pd.to_timedelta(t * factor, unit="h")
        df = pd.DataFrame({
            "site_id": _decode_ids(ds["site_id"]),
            "obs_lat": np.asarray(ds["latitude"].values, dtype=float),
            "obs_lon": np.asarray(ds["longitude"].values, dtype=float),
            "time": times,
            "obs_value": np.asarray(ds[spec["obs_var"]].values, dtype="float64"),
        })
    return df[np.isfinite(df["obs_value"])].reset_index(drop=True)


def _read_obs_window(spec, obs_dir, base_date, n_days=6, cache=None):
    """Concatenate obs for base_date..base_date+n_days-1 (files read once via `cache`)."""
    cache = {} if cache is None else cache
    base_dt = datetime.strptime(base_date, "%Y%m%d")
    frames = []
    for k in range(n_days):
        day = base_dt + timedelta(days=k)
        key = day.strftime("%Y%m%d")
        if key not in cache:
            cache[key] = _read_obs_day(spec, obs_dir, day)
        if cache[key] is not None:
            frames.append(cache[key])
    if not frames:
        return None
    obs = pd.concat(frames, ignore_index=True)
    return obs.drop_duplicates(subset=["site_id", "time", "obs_value"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Site matching + temporal alignment (identical logic for both species kinds)
# ---------------------------------------------------------------------------
def _build_site_row_map(model, obs):
    """Map each common site_id -> single model row index (nearest lat/lon on ties)."""
    obs_ll = obs.groupby("site_id")[["obs_lat", "obs_lon"]].mean()
    common = sorted(set(model["site_ids"]) & set(obs_ll.index))

    rows_by_id = {}
    for i, sid in enumerate(model["site_ids"]):
        rows_by_id.setdefault(sid, []).append(i)

    site_row = {}
    for sid in common:
        rows = rows_by_id[sid]
        if len(rows) == 1:
            site_row[sid] = rows[0]
        else:
            olat, olon = obs_ll.loc[sid, "obs_lat"], obs_ll.loc[sid, "obs_lon"]
            d = [(model["lats"][r] - olat) ** 2 + (model["lons"][r] - olon) ** 2 for r in rows]
            site_row[sid] = rows[int(np.argmin(d))]
    return site_row


def _aggregate_obs_to_steps(obs, base_dt, steps):
    """Average obs per (site_id, step) where step = nearest model step (+/- tol)."""
    steps_set = set(int(s) for s in steps)
    offset_h = (obs["time"] - base_dt).dt.total_seconds() / 3600.0
    step_bin = (offset_h / 3.0).round() * 3
    keep = ((offset_h - step_bin).abs() <= STEP_TOL_H) & step_bin.isin(steps_set)

    binned = obs.loc[keep].assign(step=step_bin[keep].astype(int))
    return (binned.groupby(["site_id", "step"])
                  .agg(obs_value=("obs_value", "mean"),
                       n_obs=("obs_value", "size"),
                       obs_lat=("obs_lat", "mean"),
                       obs_lon=("obs_lon", "mean"))
                  .reset_index())


def _attach_model_values(agg, model, out_col):
    """Add `out_col` = model value at each row's matched (site, step)."""
    site_row = _build_site_row_map(model, agg)
    step_index = {int(s): i for i, s in enumerate(model["steps"])}
    vals = np.full(len(agg), np.nan)
    for i, r in enumerate(agg.itertuples(index=False)):
        mrow = site_row.get(r.site_id)
        si = step_index.get(int(r.step))
        if mrow is not None and si is not None:
            vals[i] = model["values"][mrow, si]
    out = agg.copy()
    out[out_col] = vals
    return out


def _align(obs, model1, model2, spec):
    """Aggregate obs to model1's steps and attach values from both experiments."""
    base_dt, steps = model1["base_dt"], model1["steps"]
    agg = _aggregate_obs_to_steps(obs, base_dt, steps)

    col1 = f"{spec['model_var']}_{model1['expid']}"
    agg = _attach_model_values(agg, model1, col1)

    col2 = f"{spec['model_var']}_{model2['expid']}"
    agg = _attach_model_values(agg, model2, col2)

    aligned = agg[agg[col1].notna()].copy()
    aligned["valid_time"] = pd.to_datetime(
        [base_dt + timedelta(hours=int(s)) for s in aligned["step"]])
    return aligned.sort_values(["site_id", "step"]).reset_index(drop=True), col1, col2


def _to_pipeline_format(aligned, base_dt, col1, col2):
    """Map the aligned model/obs table into the pipeline's fc1/fc2/obs schema."""
    return pd.DataFrame({
        "date": base_dt.strftime("%Y%m%d"),
        "step": aligned["step"].astype(np.int32),
        "forecast_day": np.ceil(aligned["step"] / 24.0).astype(np.int32),
        "valid_time": aligned["valid_time"].dt.strftime("%Y%m%d%H"),
        "station_id": aligned["site_id"].astype(str),
        "lat": aligned["obs_lat"].astype(np.float32),
        "lon": aligned["obs_lon"].astype(np.float32),
        "obs_value": aligned["obs_value"].astype(np.float32),
        "fc1_value": aligned[col1].astype(np.float32),
        "fc2_value": aligned[col2].astype(np.float32),
        "n_obs": aligned["n_obs"].astype(np.int32),
    })


# ---------------------------------------------------------------------------
# Step 3 entry point
# ---------------------------------------------------------------------------
def run_step3(config, paths, preprocess_settings):
    """Read raw CAMS/obs NetCDF and write per-forecast-day parquet files.

    Returns dict: {'output_path', 'save_format', 'fc1_name', 'fc2_name'}.
    """
    variable = config["variable"]
    if variable not in SPECIES:
        raise ValueError(f"cams_extract: no species registered for variable '{variable}' "
                          f"(known: {sorted(SPECIES)})")
    spec = SPECIES[variable]

    fc1_name = paths["fc1_name"]
    fc2_name = paths["fc2_name"]
    fc1_dir = paths["fc1_path"]
    fc2_dir = paths["fc2_path"]
    obs_dir = paths["obs_path"]

    rd = config["read_data"]
    expid1 = rd["forecast_model1"].get("cams_netcdf", {}).get("expid", fc1_name)
    expid2 = rd["forecast_model2"].get("cams_netcdf", {}).get("expid", fc2_name)
    run_hour = rd["forecast_model1"].get("cams_netcdf", {}).get("run_hour", "00")

    ep = config["extract_points"]
    output_path = Path(ep["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    save_format = ep.get("save_format", "pandas")

    step_hours, _ = compute_steps(config)
    step_hours = sorted(set(step_hours))
    forecast_days = sorted({int(np.ceil(h / 24.0)) for h in step_hours})
    filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}"

    dates = pd.date_range(config["start_date"], config["end_date"], freq="24h")

    print("\n" + "=" * 80)
    print("STEP 3: EXTRACT POINT DATA  (backend = cams_extract)")
    print("=" * 80)
    print(f"  Variable        : {variable}  ({spec['kind']})")
    print(f"  Models          : {fc1_name} ({expid1}) vs {fc2_name} ({expid2})")
    print(f"  Steps (h)       : {step_hours}  →  forecast days {forecast_days}")
    print(f"  Dates           : {dates[0]:%Y%m%d}..{dates[-1]:%Y%m%d} ({len(dates)} days)")
    print(f"  Output          : {output_path}")

    obs_cache = {}
    for date_idx, date in enumerate(dates):
        date_str = date.strftime("%Y%m%d")

        if list(tmp_dir.glob(f"{date_str}_day*.parquet")):
            print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) ✓ already extracted — skipping")
            continue

        print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) extracting...", flush=True)
        try:
            model1 = read_model_file(spec, fc1_dir, expid1, date_str, run_hour)
            model2 = read_model_file(spec, fc2_dir, expid2, date_str, run_hour)
            obs = _read_obs_window(spec, obs_dir, date_str, n_days=6, cache=obs_cache)
            if obs is None:
                print(f"      [WARN] no obs window for {date_str} — skipping")
                continue
            aligned, col1, col2 = _align(obs, model1, model2, spec)
            if aligned.empty:
                print(f"      [WARN] no matched rows for {date_str} — skipping")
                continue
            pipe = _to_pipeline_format(aligned, model1["base_dt"], col1, col2)
            pipe = pipe[pipe["step"].isin(step_hours)]
        except FileNotFoundError as e:
            print(f"      [WARN] missing model file for {date_str}: {e} — skipping")
            continue
        except Exception as e:
            print(f"      [WARN] extraction failed for {date_str}: {e}")
            continue

        for day, grp in pipe.groupby("forecast_day"):
            grp.to_parquet(tmp_dir / f"{date_str}_day{int(day)}.parquet",
                            compression="snappy", index=False)

        # evict obs days no longer needed by any future base date
        nxt_key = (date + timedelta(days=1)).strftime("%Y%m%d")
        for key in [k for k in obs_cache if k < nxt_key]:
            del obs_cache[key]

    print("\n  Merging per-date files into final forecast-day parquet files...")
    saved_files = []
    for day in forecast_days:
        parts = sorted(tmp_dir.glob(f"*_day{day}.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        final_file = output_path / f"{filename_base}_day{day}.parquet"
        df.to_parquet(final_file, compression="snappy", index=False)
        saved_files.append(final_file)
        print(f"    → {final_file.name}: {len(df):,} rows")

    if saved_files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print("    [WARN] No data extracted — keeping _tmp for inspection")

    print(f"\n  ✓ Extraction complete: {len(saved_files)} forecast-day file(s) written")

    return {
        "output_path": output_path,
        "save_format": save_format,
        "fc1_name": fc1_name,
        "fc2_name": fc2_name,
    }
