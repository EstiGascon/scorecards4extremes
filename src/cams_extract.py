"""
cams_extract.py — Step 3 extraction backend for CAMS composition variables.

Reads raw per-site CAMS model NetCDF (<expid>/<expid>_YYYYMMDD_00.nc) and raw
observation NetCDF from any of the supported observation networks, aligns them
per (site, forecast step), and writes the standard pipeline parquet schema
[date, step, forecast_day, valid_time, station_id, lat, lon, obs_value,
fc1_value, fc2_value, n_obs] — so no external pre-processing / pre-built
parquet is required.

Two registries drive the module:

  SPECIES  — model-side, network-independent metadata (model variable name and,
             for gases, molar mass). Keyed by pipeline `variable`; run.py
             auto-selects this backend when config['variable'] is a key here.
  NETWORKS — one entry per observation network: where its obs files live, and
             which species it carries with which obs variable name and unit
             conversion. Membership of a network's `species` map IS the
             availability check for that (network, species) pair.

Add a network by adding one NETWORKS entry; add a species to a network by
adding one entry to that network's `species` map.

The same species can have different obs units in different networks — AirNow
reports gases as volume mixing ratios (ppb/ppm) while Airbase and China report
them as mass concentrations (ug/m3) — so the unit conversion is a property of
the (network, species) pair, not of the species. See CONVERSIONS below.

Ported from the reference scripts (read_cams_data_aod_aeronet.py /
read_cams_data_airnow.py, /home/ecm7338/scorecards4extremes/cams_sample_data/)
and generalised to the ver0d archive layout, dispatched from run.py like the
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
M_AIR = 28.9644    # g/mol, dry air — used for both gas conversions
R_UNIV = 8.31446   # J/(mol K) — used for the gas kg/kg -> ug/m3 conversion

# Model-side metadata, network-independent. Keys are pipeline `variable` names:
# run.py auto-selects this backend for any config whose `variable` is a key here
# (see run.py), so do not rename keys without updating the configs.
# `molar` (g/mol) is only needed by the 'vmr' conversion.
SPECIES = {
    "aod500": dict(model_var="aod500"),
    "pm2p5":  dict(model_var="pm2p5"),
    "pm10":   dict(model_var="pm10"),
    "go3":    dict(model_var="go3_ml137", molar=48.00),
    "no2":    dict(model_var="no2_ml137", molar=46.0055),
    "so2":    dict(model_var="so2_ml137", molar=64.066),
    "co":     dict(model_var="co_ml137",  molar=28.0101),
    "no":     dict(model_var="no_ml137",  molar=30.0061),
}

# How a model value is brought into the obs units of a given network:
#   'none'    — already comparable (AOD, unitless).
#   'conc'    — model kg/m3  -> ug/m3      (PM, all networks).
#   'vmr'     — model kg/kg  -> ppb/ppm    (AirNow gases; `scale` 1e9 / 1e6).
#   'density' — model kg/kg  -> ug/m3      (Airbase/China gases) via the air
#               density rho = p * M_AIR / (R_UNIV * T), with p = exp(lnsp),
#               built from the model file's own `lnsp` and `t_ml137`. This
#               mirrors ver0d's own `$mmr2ugm3` factor
#               (exp(<lnsp>)*($M_air/1000)/$R/<t_ml137>*1E9); requires
#               MODEL_AUX_VARS to be present in the model file.
CONVERSIONS = ("none", "conc", "vmr", "density")

# Model variables read only to support conv='density'.
MODEL_AUX_VARS = dict(lnsp="lnsp", temperature="t_ml137")

# Variable names of the obs coordinate/index fields. Every network in the ver0d
# archive uses these; a network entry may override any of them.
OBS_COORD_VARS = dict(site_id="site_id", latitude="latitude",
                      longitude="longitude", time="time")

# Observation network registry.
#   model_suffix  — model dirs/expids are '<expid><model_suffix>'.
#   obs_subdir    — strftime pattern of the dated subdirectory under obs_path,
#                   or None for files sitting flat in obs_path. Both layouts are
#                   always tried (see _obs_file_candidates), so the sample-data
#                   trees keep working against the same entry.
#   obs_filename  — strftime pattern of the daily obs file.
#   engine        — xarray engine, or None to let xarray choose. None works for
#                   the whole archive: AERONET files are NetCDF-3 classic (which
#                   h5netcdf cannot open) while the rest are HDF5-based.
#   obs_pad_days  — extra days read either side of the obs window. AERONET files
#                   sometimes hold data timestamped in the neighbouring day, so
#                   ver0d widens the AOD search by a day (idl/io/file_list_ob.pro:
#                   "Some Aeronet data for a given time is sometimes in the
#                   previous or next day's file"). Obs still land in a step by
#                   timestamp, so a wider read only recovers misfiled records.
#   domain        — [North, West, South, East] box covering the network's
#                   stations (same order as filter.py areas). Not used to filter
#                   data; it tells the case-study tools (case_studies/) which map
#                   area a config covers when filter.area is null. Set from the
#                   2025 station lists; a few outlying sites fall outside it
#                   (AirNow US-embassy sites abroad, Airbase overseas France).
#   species       — {variable: {obs_var, conv, scale, obs_scale}}; presence here
#                   is what makes a (network, species) pair available.
#                   `obs_scale` (default 1) multiplies the obs values as read, for
#                   archive files whose stored values disagree with their units
#                   attribute.
NETWORKS = {
    "aeronet": dict(
        model_suffix="",
        obs_subdir="%Y%m/%d",
        obs_filename="Aeronet_AOT_L1.5_Table_%Y%m%d.nc",
        engine=None,
        obs_pad_days=1,     # as ver0d does for mode_aod
        domain=[90, -180, -90, 180],

        species={
            "aod500": dict(obs_var="AOT-500nm", conv="none"),
        },
    ),
    "airnow": dict(
        model_suffix="_airnow",
        obs_subdir="%Y%m",
        obs_filename="airnow_%Y%m%d.nc",
        engine=None,
        domain=[65, -165, 15, -55],
        species={
            "pm2p5": dict(obs_var="PM2.5", conv="conc"),
            "pm10":  dict(obs_var="PM10",  conv="conc"),
            "go3":   dict(obs_var="OZONE", conv="vmr", scale=1e9),
            "no2":   dict(obs_var="NO2",   conv="vmr", scale=1e9),
            "so2":   dict(obs_var="SO2",   conv="vmr", scale=1e9),
            "co":    dict(obs_var="CO",    conv="vmr", scale=1e6),
            "no":    dict(obs_var="NO",    conv="vmr", scale=1e9),
        },
    ),
    "airbase": dict(
        model_suffix="_airbase",
        obs_subdir="%Y",
        obs_filename="airbase_ineris_%Y%m%d.nc",
        engine=None,
        domain=[72, -32, 26, 45],
        species={
            "pm2p5": dict(obs_var="PM2.5", conv="conc"),
            "pm10":  dict(obs_var="PM10",  conv="conc"),
            "go3":   dict(obs_var="O3",    conv="density"),
            "no2":   dict(obs_var="NO2",   conv="density"),
            "so2":   dict(obs_var="SO2",   conv="density"),
            # Airbase CO is labelled ug/m3 but stored ~1000x too large (ng/m3):
            # 2024-2025 medians ~2e5-4e5 at every station, vs ~200-400 ug/m3
            # expected; ver0d's own sane range (CO 0-2000) would reject it all.
            "co":    dict(obs_var="CO",    conv="density", obs_scale=1e-3),
        },
    ),
    "china": dict(
        model_suffix="_china_aq",
        obs_subdir="%Y",
        obs_filename="china_gsv_%Y%m%d.nc",
        engine=None,
        domain=[55, 72, 15, 136],
        species={
            "pm2p5": dict(obs_var="PM2.5", conv="conc"),
            "pm10":  dict(obs_var="PM10",  conv="conc"),
            "go3":   dict(obs_var="O3",    conv="density"),
            "no2":   dict(obs_var="NO2",   conv="density"),
            "so2":   dict(obs_var="SO2",   conv="density"),
            "co":    dict(obs_var="CO",    conv="density"),
        },
    ),
}


# ---------------------------------------------------------------------------
# Network / species resolution
# ---------------------------------------------------------------------------
def network_domain(config):
    """(network, [N, W, S, E]) of the obs network a CAMS config reads, else (None, None).

    For configs whose `variable` is a CAMS species; the network is resolved as in
    resolve() (read_data.cams_netcdf.network, or the only network carrying it).
    """
    variable = config.get("variable")
    if variable not in SPECIES:
        return None, None
    network = (config.get("read_data", {}).get("cams_netcdf") or {}).get("network")
    try:
        network = resolve(variable, network)["network"]
    except ValueError:
        return None, None
    return network, NETWORKS[network].get("domain")


def networks_for(variable):
    """Names of the networks that carry `variable`."""
    return sorted(n for n, net in NETWORKS.items() if variable in net["species"])


def resolve(variable, network=None, overrides=None):
    """Resolve (variable, network) into one flat spec used by every reader.

    `network` may be omitted when the variable is carried by exactly one network
    (e.g. aod500 -> aeronet). `overrides` is the config's cams_netcdf block; the
    obs path patterns and engine may be overridden from there for one-off layouts.
    """
    if variable not in SPECIES:
        raise ValueError(
            f"cams_extract: no species registered for variable '{variable}' "
            f"(known: {sorted(SPECIES)})")

    if network is None:
        candidates = networks_for(variable)
        if not candidates:
            raise ValueError(
                f"cams_extract: variable '{variable}' is registered in SPECIES but "
                f"no observation network carries it")
        if len(candidates) > 1:
            raise ValueError(
                f"cams_extract: variable '{variable}' is available from several "
                f"networks {candidates} — set read_data.cams_netcdf.network to "
                f"choose one")
        network = candidates[0]

    if network not in NETWORKS:
        raise ValueError(
            f"cams_extract: unknown observation network '{network}' "
            f"(known: {sorted(NETWORKS)})")

    net = NETWORKS[network]
    if variable not in net["species"]:
        raise ValueError(
            f"cams_extract: network '{network}' does not carry variable "
            f"'{variable}' (it carries: {sorted(net['species'])}; "
            f"'{variable}' is available from: {networks_for(variable) or 'no network'})")

    entry = net["species"][variable]
    conv = entry["conv"]
    if conv not in CONVERSIONS:
        raise ValueError(
            f"cams_extract: unknown conversion '{conv}' for {network}/{variable} "
            f"(known: {list(CONVERSIONS)})")

    overrides = overrides or {}
    res = {
        "variable": variable,
        "network": network,
        "model_var": SPECIES[variable]["model_var"],
        "model_suffix": net["model_suffix"],
        "molar": SPECIES[variable].get("molar"),
        "obs_var": entry["obs_var"],
        "conv": conv,
        "scale": entry.get("scale"),
        "obs_scale": float(entry.get("obs_scale", 1.0)),
        "obs_subdir": overrides.get("obs_subdir", net["obs_subdir"]),
        "obs_filename": overrides.get("obs_filename", net["obs_filename"]),
        "engine": overrides.get("engine", net["engine"]),
        "obs_pad_days": int(overrides.get("obs_pad_days", net.get("obs_pad_days", 0))),
        "coords": {**OBS_COORD_VARS, **net.get("coords", {})},
    }
    if conv == "vmr" and (res["molar"] is None or res["scale"] is None):
        raise ValueError(
            f"cams_extract: {network}/{variable} uses conv='vmr' but is missing "
            f"molar mass (SPECIES) or scale (NETWORKS)")
    return res


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


def air_density(lnsp, temperature):
    """Air density (kg/m3) from log surface pressure and temperature (K)."""
    return np.exp(lnsp) * (M_AIR / 1000.0) / (R_UNIV * temperature)


def _to_obs_units(arr, res, aux=None):
    """Convert a model array from its native units to the obs units for `res`.

    `aux` carries the model file's `lnsp`/`t_ml137` arrays, needed only by the
    'density' conversion.
    """
    conv = res["conv"]
    if conv == "none":
        return arr
    if conv == "conc":
        return arr * 1e9                                   # kg/m3 -> ug/m3
    if conv == "vmr":
        return arr * (M_AIR / res["molar"]) * res["scale"]  # kg/kg -> ppb/ppm
    # 'density': kg/kg -> ug/m3, needs the model's own lnsp and temperature.
    rho = air_density(aux["lnsp"], aux["temperature"])       # kg/m3
    return arr * rho * 1e9


def read_model_file(res, model_dir, expid, base_date, run_hour="00"):
    """Read one experiment's site file -> dict of values(site, step) + coords.

    Model files follow '<model_dir>/<expid>_<YYYYMMDD>_<HH>.nc', so `expid` must
    already carry any network suffix (e.g. '0080_airbase'); the suffix is a
    ver0d settings-file basename, not a string derivable from the network name.
    """
    model_file = Path(model_dir) / f"{expid}_{base_date}_{run_hour}.nc"
    needs_aux = res["conv"] == "density"
    with xr.open_dataset(model_file, decode_times=False) as ds:
        site_ids = _decode_ids(ds["site_ids"])
        lats = np.asarray(ds["site_lats"].values, dtype=float)
        lons = np.asarray(ds["site_lons"].values, dtype=float)
        steps = np.asarray(ds["steps"].values, dtype=int)
        vals = np.asarray(ds[res["model_var"]].values, dtype="float64")  # (site, step)

        aux = None
        if needs_aux:
            missing = [v for v in MODEL_AUX_VARS.values() if v not in ds.variables]
            if missing:
                # Fail loudly: a missing density factor would otherwise silently
                # produce plausible-looking but wrong values. The *_airnow model
                # dirs genuinely lack t_ml137 (they verify in ppb, not ug/m3).
                raise KeyError(
                    f"cams_extract: {res['network']}/{res['variable']} needs "
                    f"conv='density', which requires {sorted(MODEL_AUX_VARS.values())} "
                    f"in the model file, but {missing} are absent from {model_file}")
            aux = {k: np.asarray(ds[v].values, dtype="float64")
                   for k, v in MODEL_AUX_VARS.items()}

    vals = _to_obs_units(vals, res, aux)
    base_dt = datetime.strptime(base_date + run_hour, "%Y%m%d%H")
    return {"expid": expid, "site_ids": site_ids, "lats": lats, "lons": lons,
            "steps": steps, "values": vals, "base_dt": base_dt}


def obs_file_candidates(res, obs_dir, day_dt):
    """Candidate paths for one day's obs file, most-specific first.

    The archive keeps obs in a dated subdirectory whose granularity differs per
    network ('%Y%m/%d' for AERONET, '%Y%m' for AirNow, '%Y' for Airbase/China),
    while the sample-data trees keep them flat. Both are offered so one registry
    entry serves either layout.
    """
    name = day_dt.strftime(res["obs_filename"])
    paths = []
    if res["obs_subdir"]:
        paths.append(Path(obs_dir) / day_dt.strftime(res["obs_subdir"]) / name)
    paths.append(Path(obs_dir) / name)
    return paths


def _read_obs_day(res, obs_dir, day_dt):
    """Read one daily obs file -> tidy frame (NaN obs dropped), or None if absent.

    A file that exists but lacks the species variable (e.g. Airbase 20250502 has
    no PM2.5) counts as absent too: a day without observations, not an error.
    """
    f = next((p for p in obs_file_candidates(res, obs_dir, day_dt) if p.exists()), None)
    if f is None:
        return None

    coords = res["coords"]
    kwargs = {"decode_times": False}
    if res["engine"]:
        kwargs["engine"] = res["engine"]
    with xr.open_dataset(f, **kwargs) as ds:
        if res["obs_var"] not in ds.variables:
            print(f"      [WARN] {f.name} has no '{res['obs_var']}' variable — "
                  f"treating {day_dt:%Y%m%d} as a day without obs")
            return None
        ref_dt, factor = _parse_time_units(ds[coords["time"]].attrs["units"])
        t = np.asarray(ds[coords["time"]].values, dtype=float)
        times = ref_dt + pd.to_timedelta(t * factor, unit="h")
        df = pd.DataFrame({
            "site_id": _decode_ids(ds[coords["site_id"]]),
            "obs_lat": np.asarray(ds[coords["latitude"]].values, dtype=float),
            "obs_lon": np.asarray(ds[coords["longitude"]].values, dtype=float),
            "time": times,
            "obs_value": np.asarray(ds[res["obs_var"]].values, dtype="float64")
                         * res["obs_scale"],
        })
    return df[np.isfinite(df["obs_value"])].reset_index(drop=True)


def _read_obs_window(res, obs_dir, base_date, n_days=6, cache=None):
    """Concatenate obs for base_date..base_date+n_days-1 (files read once via `cache`).

    Each day resolves its own subdirectory, so a window crossing a month or year
    boundary needs no special handling.

    The window is widened by `res['obs_pad_days']` either side (1 for AERONET, as
    ver0d does for mode_aod) to pick up records filed under a neighbouring day.
    Padding cannot pull in out-of-window observations: _aggregate_obs_to_steps
    bins every obs by its own timestamp and keeps only those within STEP_TOL_H of
    a real model step, so extra records either fill a step they belong to or are
    discarded.
    """
    cache = {} if cache is None else cache
    pad = res["obs_pad_days"]
    base_dt = datetime.strptime(base_date, "%Y%m%d")
    frames = []
    for k in range(-pad, n_days + pad):
        day = base_dt + timedelta(days=k)
        key = day.strftime("%Y%m%d")
        if key not in cache:
            cache[key] = _read_obs_day(res, obs_dir, day)
        if cache[key] is not None:
            frames.append(cache[key])
    if not frames:
        return None
    obs = pd.concat(frames, ignore_index=True)
    return obs.drop_duplicates(subset=["site_id", "time", "obs_value"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Site matching + temporal alignment (identical logic for every network)
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


def _align(obs, model1, model2, res):
    """Aggregate obs to model1's steps and attach values from both experiments."""
    base_dt, steps = model1["base_dt"], model1["steps"]
    agg = _aggregate_obs_to_steps(obs, base_dt, steps)

    col1 = f"{res['model_var']}_{model1['expid']}"
    agg = _attach_model_values(agg, model1, col1)

    col2 = f"{res['model_var']}_{model2['expid']}"
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

    fc1_name = paths["fc1_name"]
    fc2_name = paths["fc2_name"]
    fc1_dir = paths["fc1_path"]
    fc2_dir = paths["fc2_path"]
    obs_dir = paths["obs_path"]

    rd = config["read_data"]
    obs_cfg = rd.get("cams_netcdf", {}) or {}
    fc1_cfg = rd["forecast_model1"].get("cams_netcdf", {}) or {}
    fc2_cfg = rd["forecast_model2"].get("cams_netcdf", {}) or {}

    # Resolves + validates (network, variable); raises with the valid options listed.
    res = resolve(variable, obs_cfg.get("network"), overrides=obs_cfg)

    expid1 = fc1_cfg.get("expid", fc1_name)
    expid2 = fc2_cfg.get("expid", fc2_name)
    run_hour1 = fc1_cfg.get("run_hour", "00")
    run_hour2 = fc2_cfg.get("run_hour", run_hour1)

    # Warn (don't raise) when an expid looks like it belongs to another network:
    # model sites are extracted per network, so a mismatch matches few or no
    # stations. Only a warning, because the ver0d suffix is a settings-file
    # basename that legitimately varies (network 'china' has both 'icki_china'
    # and '<expid>_china_aq' dirs).
    suffix = res["model_suffix"]
    if suffix:
        for label, expid in (("forecast_model1", expid1), ("forecast_model2", expid2)):
            if not expid.endswith(suffix):
                print(f"  [WARN] {label} expid '{expid}' does not end with "
                      f"'{suffix}' expected for network '{res['network']}' — "
                      f"check it is the {res['network']} site extraction")

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
    print(f"  Variable        : {variable}  (network = {res['network']})")
    print(f"  Obs variable    : {res['obs_var']}  (model {res['model_var']}, "
          f"conversion {res['conv']})")
    if res["obs_scale"] != 1.0:
        print(f"  Obs scale       : x{res['obs_scale']:g}  (registry fix for obs units)")
    print(f"  Models          : {fc1_name} ({expid1}) vs {fc2_name} ({expid2})")
    print(f"  Model files     : <dir>/{{expid}}_YYYYMMDD_{run_hour1}.nc")
    for i, p in enumerate(obs_file_candidates(res, obs_dir, dates[0].to_pydatetime())):
        print(f"  {'Obs files' if i == 0 else '   (or)':<16}: {p}")
    print(f"  Steps (h)       : {step_hours}  →  forecast days {forecast_days}")
    print(f"  Dates           : {dates[0]:%Y%m%d}..{dates[-1]:%Y%m%d} ({len(dates)} days)")
    print(f"  Output          : {output_path}")

    obs_cache = {}
    obs_days_found = 0
    dates_attempted = 0
    for date_idx, date in enumerate(dates):
        date_str = date.strftime("%Y%m%d")

        if list(tmp_dir.glob(f"{date_str}_day*.parquet")):
            print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) ✓ already extracted — skipping")
            continue

        print(f"    [{date_str}] ({date_idx + 1}/{len(dates)}) extracting...", flush=True)
        dates_attempted += 1
        try:
            model1 = read_model_file(res, fc1_dir, expid1, date_str, run_hour1)
            model2 = read_model_file(res, fc2_dir, expid2, date_str, run_hour2)
            obs = _read_obs_window(res, obs_dir, date_str, n_days=6, cache=obs_cache)
            if obs is None:
                print(f"      [WARN] no obs window for {date_str} — skipping")
                continue
            obs_days_found += 1
            aligned, col1, col2 = _align(obs, model1, model2, res)
            if aligned.empty:
                print(f"      [WARN] no matched rows for {date_str} — skipping")
                continue
            pipe = _to_pipeline_format(aligned, model1["base_dt"], col1, col2)
            pipe = pipe[pipe["step"].isin(step_hours)]
        except FileNotFoundError as e:
            print(f"      [WARN] missing model file for {date_str}: {e} — skipping")
            continue
        except KeyError:
            # A missing variable is a config/data mismatch that every date will
            # hit — fail now rather than warn 365 times and write nothing.
            raise
        except Exception as e:
            print(f"      [WARN] extraction failed for {date_str}: {e}")
            continue

        for day, grp in pipe.groupby("forecast_day"):
            grp.to_parquet(tmp_dir / f"{date_str}_day{int(day)}.parquet",
                            compression="snappy", index=False)

        # evict obs days no longer needed by any future base date — the next base
        # date's window starts obs_pad_days before it, so keep that much history
        nxt_key = (date + timedelta(days=1 - res["obs_pad_days"])).strftime("%Y%m%d")
        for key in [k for k in obs_cache if k < nxt_key]:
            del obs_cache[key]

    if dates_attempted and not obs_days_found:
        # Every date found zero obs files. Almost always a wrong obs_path or a
        # network whose layout does not match — say so instead of writing nothing.
        example = obs_file_candidates(res, obs_dir, dates[0].to_pydatetime())
        raise FileNotFoundError(
            f"cams_extract: no {res['network']} observation files found for any of "
            f"the {dates_attempted} date(s) tried. Looked for e.g.\n    "
            + "\n    ".join(str(p) for p in example)
            + f"\nCheck read_data.cams_netcdf.obs_path and network ('{res['network']}').")

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
