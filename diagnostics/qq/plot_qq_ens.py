#!/usr/bin/env python3
"""
QQ Plot Tool (ENSEMBLE) — Quantile-Quantile Plots for Ensemble Forecasts
========================================================================
Parallel of ``plot_qq.py`` for ENSEMBLE forecast systems.

Where ``plot_qq.py`` works on a single deterministic forecast value per
station-day (``fc1_value`` / ``fc2_value``), this tool works on the wide
ensemble parquet produced by ``extract_points_ensemble.py`` which stores one
column per member:

    fc1_member_0 .. fc1_member_50      (member 0 = control, 1-50 = perturbed)
    fc2_member_0 .. fc2_member_50

The user chooses, via ``--member``, what forecast the Q-Q plot represents:

    --member full        the FULL ensemble system (default): every member
                         value across every station-day is pooled into one
                         forecast sample, so the forecast quantiles describe
                         the full predictive marginal distribution.
    --member control     only the control member (member 0), i.e. the
                         unperturbed forecast — behaves like a single
                         deterministic forecast.
    --member N           only one specific perturbed member N (1-50), or 0
                         for the control member.

Everything else (season / orography / lead-time subsetting, threshold lines,
2m-temperature cold-tail markers, the inset Δ-stats bar chart and the shared
colour-blind-safe style) mirrors ``plot_qq.py``.

USAGE
-----
  python plot_qq_ens.py --config config_2t_ens.yaml
  python plot_qq_ens.py --config config_2t_ens.yaml --member control
  python plot_qq_ens.py --config config_2t_ens.yaml --member 5 --season DJF
  python plot_qq_ens.py --config config_2t_ens.yaml --member full --orog low mid high
  python plot_qq_ens.py --help

By default the script loops over every (season, orography) combination you
specify and saves one PNG per combination.  Use --no-loop to collapse all
selected data into a single plot.

NOTE on statistics for the full ensemble
-----------------------------------------
Correlation / bias / RMSE are *paired* metrics and need one forecast value per
observation.  For ``--member full`` the inset stats box therefore uses the
ENSEMBLE MEAN (per station-day) against the observation, while the Q-Q markers
themselves use the pooled-member quantiles.  For a single member the stats and
the markers both use that one member (identical to the deterministic tool).
"""

import argparse
import re
import sys
import gc
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent            # this script's subfolder
_DIAG = _HERE.parent                               # diagnostics/  (shared _style)
_REPO = _DIAG.parent                               # repository root
for _p in (_REPO, _REPO / "src", _DIAG, _HERE):
    sys.path.insert(0, str(_p))
import _style

# Reuse the constants and config helpers from the deterministic tool so the two
# stay in lock-step (same variable labels, seasons, orography bins, areas and
# config parsing).  Importing the module is side-effect free (its main() is
# guarded by ``if __name__ == "__main__"``).
from plot_qq import (  # noqa: E402
    VARIABLE_LABELS,
    SEASON_MONTHS,
    OROGRAPHY_RANGES,
    AREAS,
    load_config,
    get_model_names,
    get_threshold_from_config,
    _format_threshold_suffix,
)

# Physically-impossible sentinel used by the extractor / other ensemble
# diagnostics for missing member values (e.g. a station outside a model's grid).
IMPOSSIBLE_VALUE = -200.0

# Non-member columns we need for filtering; read only these + the selected
# member columns to keep the (very wide) ensemble parquet light in memory.
BASE_COLUMNS = [
    "date", "step", "station_id", "lat", "lon",
    "obs_value", "forecast_day", "sdfor",
]


# ============================================================================
# MEMBER SELECTION
# ============================================================================

def _member_num(col: str) -> int:
    """Return the integer index N from a 'fcX_member_N' column name."""
    return int(col.rsplit("_", 1)[1])


def resolve_members(available_columns, selection: str):
    """
    Work out which member column(s) each model contributes for the requested
    selection.

    Parameters
    ----------
    available_columns : iterable of str
        All column names present in the ensemble parquet.
    selection : str
        'full', 'control', or an integer index (0 = control, 1-50 = perturbed).

    Returns
    -------
    (fc1_cols, fc2_cols, member_label, member_tag)
        fc1_cols / fc2_cols : list[str]   the member columns to use per model
        member_label        : str         human-readable label for titles
        member_tag          : str         short tag for the output filename
    """
    available = set(available_columns)
    fc1_all = sorted([c for c in available if c.startswith("fc1_member_")],
                     key=_member_num)
    fc2_all = sorted([c for c in available if c.startswith("fc2_member_")],
                     key=_member_num)

    if not fc1_all and not fc2_all:
        raise SystemExit(
            "\nERROR: No ensemble member columns (fc1_member_* / fc2_member_*) "
            "were found in the extracted data.\n"
            "This tool is for ENSEMBLE data — for a single deterministic "
            "forecast use plot_qq.py instead."
        )

    sel = selection.strip().lower()

    # ---- full ensemble --------------------------------------------------
    if sel == "full":
        if not fc1_all or not fc2_all:
            raise SystemExit(
                "\nERROR: --member full needs member columns for BOTH models, "
                f"but found {len(fc1_all)} fc1 and {len(fc2_all)} fc2 member "
                "columns."
            )
        return fc1_all, fc2_all, "full ensemble", "ensfull"

    # ---- single member --------------------------------------------------
    if sel == "control":
        n = 0
    else:
        try:
            n = int(sel)
        except ValueError:
            raise SystemExit(
                f"\nERROR: invalid --member value '{selection}'. "
                "Use 'full', 'control', or an integer member index (0-50)."
            )

    fc1_col, fc2_col = f"fc1_member_{n}", f"fc2_member_{n}"
    missing = [c for c in (fc1_col, fc2_col) if c not in available]
    if missing:
        raise SystemExit(
            f"\nERROR: requested member column(s) not present: {missing}.\n"
            f"  Available fc1 members: {[_member_num(c) for c in fc1_all]}\n"
            f"  Available fc2 members: {[_member_num(c) for c in fc2_all]}\n"
            + ("Note: member 0 (control) is not archived for every model."
               if n == 0 else "")
        )

    if n == 0:
        return [fc1_col], [fc2_col], "control member", "ctrl"
    return [fc1_col], [fc2_col], f"member {n}", f"mem{n:02d}"


# ============================================================================
# DATA SOURCE DISCOVERY (locate files + resolve members; no bulk load yet)
# ============================================================================

def discover_ens_source(config: dict, fc1_name: str, fc2_name: str, selection: str):
    """
    Locate the extracted ensemble data and resolve the member columns for the
    requested selection WITHOUT loading the (potentially huge) data.

    Loading is deferred to ``load_and_filter_ens`` which streams one
    forecast-day file at a time and filters it before accumulating, so the full
    per-member table (150M+ rows × 100 members for a season of ENS) never has to
    live in memory at once.

    Returns a dict describing the source:
      per-day  : {'kind':'perday', 'day_files', 'needed', 'fc1_cols', 'fc2_cols',
                  'member_label', 'member_tag'}
      combined : {'kind':'single', 'df', 'fc1_cols', 'fc2_cols',
                  'member_label', 'member_tag'}
    """
    var = config["variable"]
    point_data_path = Path(config["extract_points"]["output_path"])

    # -- Per-forecast-day files (preferred, ensemble suffix first) -----------
    day_pattern = f"{var}_{fc1_name}_vs_{fc2_name}_ens_day*.parquet"
    day_files = sorted(point_data_path.glob(day_pattern))
    if not day_files:
        day_pattern = f"{var}_{fc1_name}_vs_{fc2_name}_day*.parquet"
        day_files = sorted(point_data_path.glob(day_pattern))

    if day_files:
        available = pq.ParquetFile(str(day_files[0])).schema.names
        fc1_cols, fc2_cols, member_label, member_tag = resolve_members(
            available, selection
        )
        needed = [c for c in BASE_COLUMNS if c in available] + fc1_cols + fc2_cols
        return {
            "kind": "perday",
            "day_files": day_files,
            "needed": needed,
            "fc1_cols": fc1_cols,
            "fc2_cols": fc2_cols,
            "member_label": member_label,
            "member_tag": member_tag,
        }

    # -- Fallback: single combined file --------------------------------------
    base = f"{var}_{fc1_name}_vs_{fc2_name}{_format_threshold_suffix(config)}"
    for ext, reader in [(".parquet", pd.read_parquet), (".pkl", pd.read_pickle)]:
        fp = point_data_path / f"{base}{ext}"
        if fp.exists():
            print(f"  Reading {fp.name}")
            df = reader(fp)
            fc1_cols, fc2_cols, member_label, member_tag = resolve_members(
                df.columns, selection
            )
            keep = [c for c in BASE_COLUMNS if c in df.columns] + fc1_cols + fc2_cols
            return {
                "kind": "single",
                "df": df[keep].copy(),
                "fc1_cols": fc1_cols,
                "fc2_cols": fc2_cols,
                "member_label": member_label,
                "member_tag": member_tag,
            }

    raise FileNotFoundError(
        f"No extracted ENSEMBLE data found in '{point_data_path}'.\n"
        f"Expected files matching:\n"
        f"  {point_data_path / day_pattern}\n"
        "Run the ensemble pipeline first to generate the extracted point data."
    )


# ============================================================================
# FILTERING (member-aware) + STREAMING LOADER
# ============================================================================

def filter_chunk(
    df: pd.DataFrame,
    config: dict,
    member_cols,
    *,
    season=None,
    orog_type=None,
    lead_time=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Apply the same sub-setting filters as ``plot_qq.apply_filters`` to ONE chunk
    (adapted to ensemble member columns: obs-based NaN/QC, per-member sanity
    masking).  Quiet and side-effect free (returns a possibly-empty frame); the
    caller aggregates across chunks and checks for emptiness.

    Note: no Gaussian sigma-based outlier removal is applied here (see below);
    only physical validity caps are.
    """
    # -- Geographic area -----------------------------------------------------
    area_name = config.get("filter", {}).get("area")
    if area_name and area_name in AREAS:
        lat_north, lon_west, lat_south, lon_east = AREAS[area_name]
        df = df[
            (df["lat"] >= lat_south) & (df["lat"] <= lat_north) &
            (df["lon"] >= lon_west) & (df["lon"] <= lon_east)
        ]

    # -- Date range ----------------------------------------------------------
    sd = start_date or config.get("start_date")
    ed = end_date   or config.get("end_date")
    if sd and ed:
        sd_str = datetime.strptime(sd, "%Y-%m-%d").strftime("%Y%m%d")
        ed_str = datetime.strptime(ed, "%Y-%m-%d").strftime("%Y%m%d")
        df = df[
            (df["date"].astype(str) >= sd_str) &
            (df["date"].astype(str) <= ed_str)
        ]

    # -- Season --------------------------------------------------------------
    if season:
        months = SEASON_MONTHS.get(season.upper())
        if months is None:
            raise ValueError(
                f"Unknown season '{season}'. Choose from: {list(SEASON_MONTHS)}"
            )
        _month = df["date"].astype(str).str[4:6].astype(int)
        df = df[_month.isin(months)]

    # -- Lead time -----------------------------------------------------------
    if lead_time is not None:
        steps = [lead_time] if isinstance(lead_time, int) else list(lead_time)
        df = df[df["step"].isin(steps)]

    # -- Orography -----------------------------------------------------------
    if orog_type and "sdfor" in df.columns:
        bounds = OROGRAPHY_RANGES.get(orog_type.lower())
        if bounds is None:
            raise ValueError(
                f"Unknown orography type '{orog_type}'. "
                f"Choose from: {list(OROGRAPHY_RANGES)}"
            )
        lo, hi = bounds
        df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]

    if len(df) == 0:
        return df

    # From here we mutate; ensure we own the data (boolean-indexing above
    # already yields copies, but guard the no-filter path too).
    df = df.copy()

    # -- Physically-impossible member values → NaN ---------------------------
    if member_cols:
        bad = df[member_cols] < IMPOSSIBLE_VALUE
        if bad.to_numpy().any():
            df[member_cols] = df[member_cols].mask(bad)

    # -- NaN removal (obs + at least one valid member per model) -------------
    df = df.dropna(subset=["obs_value"])
    fc1_cols = [c for c in member_cols if c.startswith("fc1_member_")]
    fc2_cols = [c for c in member_cols if c.startswith("fc2_member_")]
    if fc1_cols:
        df = df[df[fc1_cols].notna().any(axis=1)]
    if fc2_cols:
        df = df[df[fc2_cols].notna().any(axis=1)]

    # -- Physical validity QC (config) ---------------------------------------
    # Applied to BOTH the observations and every forecast member so the two
    # tails are treated identically.  Only physically-impossible values are
    # dropped (e.g. a 2386 mm "24 h" precip report); the legitimate heavy tail
    # is preserved because that tail is exactly what these Q-Q plots examine.
    filter_cfg = config.get("filter", {})
    variable = config.get("variable", "")

    if variable == "tp24":
        max_precip = filter_cfg.get("max_valid_precipitation", None)
        if max_precip is not None:
            df = df[df["obs_value"] <= max_precip]
            if member_cols:
                df[member_cols] = df[member_cols].mask(df[member_cols] > max_precip)

    if variable == "2t":
        min_temp = filter_cfg.get("min_valid_temperature", -80.0)
        max_temp = filter_cfg.get("max_valid_temperature", 60.0)
        df = df[(df["obs_value"] >= min_temp) & (df["obs_value"] <= max_temp)]
        if member_cols:
            oob = (df[member_cols] < min_temp) | (df[member_cols] > max_temp)
            df[member_cols] = df[member_cols].mask(oob)

    # NOTE: the Gaussian mean±N·std outlier ("sigma clip") step has been REMOVED
    # from the pipeline entirely and is not applied here.  For a heavy-tailed field
    # the mean+Nσ cut (tp24: ≈39 mm) deletes genuine extreme observations, and
    # applying it to the observations only — never the pooled members — asymmetrically
    # truncates the observed tail and fabricates a spurious forecast-over-obs hook at p99.8–99.9.

    return df


def load_and_filter_ens(
    source: dict,
    config: dict,
    member_cols,
    *,
    season=None,
    orog_type=None,
    lead_time=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Build the filtered DataFrame for one (season, orography, …) combination.

    For per-day sources this streams one forecast-day parquet at a time,
    filters it (including the orography subset) and only keeps the survivors,
    so peak memory is bounded by the filtered subset rather than the whole
    multi-hundred-GB ensemble table.
    """
    if source["kind"] == "single":
        df = filter_chunk(
            source["df"].copy(), config, member_cols,
            season=season, orog_type=orog_type, lead_time=lead_time,
            start_date=start_date, end_date=end_date,
        )
        if len(df) == 0:
            raise RuntimeError(
                "No data remains after filtering. Adjust filter criteria or "
                "check that the data covers the selected conditions."
            )
        return df.reset_index(drop=True)

    day_files = source["day_files"]
    needed = source["needed"]
    parts = []
    total_raw = 0
    for fp in day_files:
        chunk = pd.read_parquet(fp, columns=needed)
        m = re.search(r"day(\d+)\.parquet", fp.name)
        if m:
            chunk["forecast_day"] = int(m.group(1))
        total_raw += len(chunk)
        chunk = filter_chunk(
            chunk, config, member_cols,
            season=season, orog_type=orog_type, lead_time=lead_time,
            start_date=start_date, end_date=end_date,
        )
        if len(chunk):
            parts.append(chunk)
        del chunk
        gc.collect()

    if not parts:
        raise RuntimeError(
            "No data remains after filtering. Adjust filter criteria or check "
            "that the data covers the selected conditions."
        )
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"  Streamed {len(day_files)} file(s): {total_raw:,} rows read → "
          f"{len(df):,} rows after filtering")
    return df


# ============================================================================
# FORECAST SAMPLE CONSTRUCTION
# ============================================================================

def _forecast_arrays(df: pd.DataFrame, member_cols):
    """
    Build the forecast member matrix and its pooled 1-D sample.

    Returns
    -------
    (dist, mat)
        dist : 1-D array used for the FORECAST quantiles (Q-Q markers) — every
               member value pooled (flattened). Single member → that member.
        mat  : 2-D (N, n_members) member values. Used for the paired stats,
               where obs is paired against EVERY member (no ensemble mean is
               taken). Single member → shape (N, 1).
    """
    # Kept in float32 — the parquet stores float32, so the default float64 cast
    # would double peak memory on the (potentially huge) pooled sample.
    mat = df[member_cols].to_numpy(dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[:, None]
    dist = mat.ravel()
    return dist, mat


def _paired_stats(mat: np.ndarray, obs: np.ndarray):
    """Correlation, bias and RMSE pooling obs against every member.

    Accumulates per member column so the temporaries stay at O(N) instead of
    materialising the O(N·members) pooled obs vector.
    """
    obs64 = obs.astype(np.float64)
    n = 0
    sf = sff = so = soo = sfo = 0.0
    for j in range(mat.shape[1]):
        col = mat[:, j]
        m = np.isfinite(col)
        if not m.any():
            continue
        f = col[m].astype(np.float64)
        o = obs64[m]
        n += f.size
        sf += f.sum(); sff += np.dot(f, f)
        so += o.sum(); soo += np.dot(o, o)
        sfo += np.dot(f, o)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mf, mo = sf / n, so / n
    cov = sfo / n - mf * mo
    vf = sff / n - mf * mf
    vo = soo / n - mo * mo
    corr = float(cov / np.sqrt(vf * vo)) if vf > 0.0 and vo > 0.0 else float("nan")
    bias = float(mf - mo)
    rmse = float(np.sqrt(max((sff - 2.0 * sfo + soo) / n, 0.0)))
    return corr, bias, rmse


# ============================================================================
# PLOTTING
# ============================================================================

def _condition_label(season, orog_type, lead_time) -> str:
    parts = []
    if season:
        parts.append(season.upper())
    if orog_type:
        parts.append(f"{orog_type.upper()} terrain")
    if lead_time is not None:
        steps = [lead_time] if isinstance(lead_time, int) else list(lead_time)
        parts.append("lt=" + "/".join(f"{s}h" for s in steps))
    return " | ".join(parts) if parts else "All conditions"


def plot_qq_ens(
    df: pd.DataFrame,
    config: dict,
    fc1_name: str,
    fc2_name: str,
    fc1_cols,
    fc2_cols,
    member_label: str,
    member_tag: str,
    *,
    season=None,
    orog_type=None,
    lead_time=None,
    n_quantiles: int = 200,
    output_dir=None,
    threshold=None,
    dpi: int = 150,
) -> Path:
    """
    Create an ensemble Q-Q plot (obs vs both models for the chosen member
    selection) and save it.  Mirrors ``plot_qq.plot_qq`` visually.
    """
    variable = config["variable"]
    var_label, unit = VARIABLE_LABELS.get(variable, (variable, ""))
    condition = _condition_label(season, orog_type, lead_time)
    is_full = len(fc1_cols) > 1 or len(fc2_cols) > 1

    obs = df["obs_value"].to_numpy(dtype=np.float32)
    fc1_dist, fc1_mat = _forecast_arrays(df, fc1_cols)
    fc2_dist, fc2_mat = _forecast_arrays(df, fc2_cols)

    # -- Dots at every integer percentile 0–99 ------------------------------
    # 2t starts at 1 (not 0): the 0th percentile is the single coldest raw
    # value (often one bad station reading); its tail is covered by cold_x.
    has_cold_tail = variable == "2t"
    dot_levels = np.arange(1, 99.5, 1.0) if has_cold_tail else np.arange(0, 99.5, 1.0)
    obs_dots  = np.nanpercentile(obs,      dot_levels)
    fc1_dots  = np.nanpercentile(fc1_dist, dot_levels)
    fc2_dots  = np.nanpercentile(fc2_dist, dot_levels)

    # -- X markers at 99.0, 99.1, …, 99.9 ----------------------------------
    x_levels = np.linspace(99.0, 99.9, 10)
    obs_x  = np.nanpercentile(obs,      x_levels)
    fc1_x  = np.nanpercentile(fc1_dist, x_levels)
    fc2_x  = np.nanpercentile(fc2_dist, x_levels)

    # -- 2t only: mirror the extreme-tail markers on the COLD side ----------
    if has_cold_tail:
        cold_x_levels = np.linspace(0.1, 1.0, 10)
        obs_cold_x = np.nanpercentile(obs,      cold_x_levels)
        fc1_cold_x = np.nanpercentile(fc1_dist, cold_x_levels)
        fc2_cold_x = np.nanpercentile(fc2_dist, cold_x_levels)

    # -- Statistics: obs paired against every member (pooled, no ens-mean) ---
    corr1, bias1, rmse1 = _paired_stats(fc1_mat, obs)
    corr2, bias2, rmse2 = _paired_stats(fc2_mat, obs)

    frac_rmse = (rmse2 - rmse1) / rmse1 * 100.0 if rmse1 != 0.0 else 0.0
    frac_corr = (corr2 - corr1) / abs(corr1) * 100.0 if corr1 != 0.0 else 0.0
    frac_bias = (abs(bias2) - abs(bias1)) / abs(bias1) * 100.0 if bias1 != 0.0 else 0.0

    COLOR_FC1 = _style.C_FC1
    COLOR_FC2 = _style.C_FC2
    COLOR_REF = _style.C_REF

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Q-Q Plot (ENS)  —  {var_label}  |  {member_label}  |  {condition}  "
        f"(N = {len(df):,})",
        fontsize=12, fontweight="bold",
    )

    # Axis limits capped at 99.9th percentile (0.1st for 2t)
    all_capped = [obs_dots, obs_x, fc1_dots, fc1_x, fc2_dots, fc2_x]
    if has_cold_tail:
        all_capped += [obs_cold_x, fc1_cold_x, fc2_cold_x]
    all_capped = np.concatenate(all_capped)
    vmin = np.nanmin(all_capped)
    vmax = np.nanmax(all_capped)
    margin = (vmax - vmin) * 0.05
    lim = (vmin - margin, vmax + margin)

    # 1:1 reference line (no legend entry)
    ax.plot(lim, lim, color=COLOR_REF, linestyle="--", linewidth=1.2, zorder=1)

    # QQ markers: dot per integer percentile + x every 0.1% from 99–99.9
    for fc_dots, fc_x, color, name in [
        (fc1_dots, fc1_x, COLOR_FC1, fc1_name),
        (fc2_dots, fc2_x, COLOR_FC2, fc2_name),
    ]:
        ax.scatter(obs_dots, fc_dots, color=color, s=20, zorder=3,
                   marker="o", label=name)
        ax.scatter(obs_x, fc_x, color=color, s=60, zorder=4,
                   marker="x", linewidths=2)

    if has_cold_tail:
        for fc_cold_x, color in [(fc1_cold_x, COLOR_FC1), (fc2_cold_x, COLOR_FC2)]:
            ax.scatter(obs_cold_x, fc_cold_x, color=color, s=60, zorder=4,
                       marker="x", linewidths=2)

    # Threshold lines (no legend entry)
    if threshold is not None:
        ax.axvline(threshold, color="green", linestyle=":", linewidth=1.3,
                   alpha=0.9, zorder=1)
        ax.axhline(threshold, color="green", linestyle=":", linewidth=1.3,
                   alpha=0.9, zorder=1)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed [{unit}]", fontsize=11)
    ax.set_ylabel(f"Forecast [{unit}]", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Legend (lower right), then place o/x text box flush on top of it
    legend = ax.legend(fontsize=13, loc="lower right")
    plt.tight_layout()
    fig.canvas.draw()  # force render so legend bbox is valid
    lb = legend.get_window_extent().transformed(ax.transAxes.inverted())
    marker_note = (
        "o = 0–99th percentile\nx = 0.1–1st & 99–99.9th (Δ 0.1%)"
        if has_cold_tail else
        "o = 0–99th percentile\nx = 99–99.9th (Δ 0.1%)"
    )
    if is_full:
        marker_note += "\nmarkers & stats: pooled members"
    ax.text(
        lb.x1, lb.y1 + 0.005,
        marker_note,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="lightgray", alpha=0.85),
    )

    # ----------------------------------------- inset stats box (top-left)
    ax_ins = ax.inset_axes([0.12, 0.65, 0.26, 0.18])
    bar_x    = [0.0, 0.27, 0.52]
    bar_vals = [frac_rmse, frac_corr, frac_bias]
    ax_ins.bar(bar_x, bar_vals, width=0.20, color=COLOR_FC2)
    ax_ins.axhline(0, color="black", linewidth=0.8)
    ax_ins.set_xticks(bar_x)
    ax_ins.set_xticklabels(["RMSE", "ρ", "|b|"], fontsize=10)
    ax_ins.set_ylabel("Δ [%]", fontsize=10, labelpad=2)
    ax_ins.tick_params(axis="y", labelsize=9, pad=1)

    trans = mtransforms.blended_transform_factory(ax_ins.transData, ax_ins.transAxes)
    for i, (v2, v1) in enumerate(zip([rmse2, corr2, bias2], [rmse1, corr1, bias1])):
        ax_ins.text(bar_x[i], 1.22, f"{v2:.2f}",
                    transform=trans, ha="center", va="bottom", fontsize=9,
                    color=COLOR_FC2, weight="bold", clip_on=False)
        ax_ins.text(bar_x[i], 1.05, f"{v1:.2f}",
                    transform=trans, ha="center", va="bottom", fontsize=9,
                    color=COLOR_FC1, weight="bold", clip_on=False)

    # ------------------------------------------------------------------ save
    if output_dir is None:
        output_dir = Path(
            config.get("save", {}).get("output_directory", "./results")
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fn_parts = [f"qq_{variable}_{fc1_name}_vs_{fc2_name}_{member_tag}"]
    if season:
        fn_parts.append(season.upper())
    if orog_type:
        fn_parts.append(orog_type.lower())
    if lead_time is not None:
        steps = [lead_time] if isinstance(lead_time, int) else list(lead_time)
        fn_parts.append("lt" + "_".join(str(s) for s in steps) + "h")
    filename = "_".join(fn_parts) + ".png"

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")
    return out_path


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="plot_qq_ens.py",
        description=(
            "Create ENSEMBLE Q-Q plots (observations vs forecasts) from "
            "extracted ensemble point data.\n"
            "The extracted point data must already exist — run the ensemble "
            "pipeline first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Full ensemble system (default), all data combined:
  python plot_qq_ens.py --config config_2t_ens.yaml

  # Control member only, per season:
  python plot_qq_ens.py --config config_2t_ens.yaml --member control \\
      --season DJF MAM JJA SON

  # A single perturbed member (member 5), low terrain, day-3 lead time:
  python plot_qq_ens.py --config config_2t_ens.yaml --member 5 \\
      --orog low --lead-time 72

  # Full ensemble over all 12 season×orography combinations:
  python plot_qq_ens.py --config config_2t_ens.yaml --member full \\
      --season DJF MAM JJA SON --orog low mid high
""",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (same file used for the ensemble pipeline).",
    )
    parser.add_argument(
        "--member", default="full", metavar="SEL",
        help=(
            "Which forecast the Q-Q plot represents:\n"
            "  full     – the full ensemble system, all members pooled (default)\n"
            "  control  – only the control member (member 0)\n"
            "  N        – only perturbed member N (1-50), or 0 for control"
        ),
    )
    parser.add_argument(
        "--season", nargs="+", choices=["DJF", "MAM", "JJA", "SON"],
        metavar="SEASON", default=None,
        help=(
            "Season(s) to plot.  Specify one or more of DJF MAM JJA SON. "
            "If omitted, all seasons are combined."
        ),
    )
    parser.add_argument(
        "--orog", dest="orog_type", nargs="+",
        choices=["low", "mid", "high"], metavar="OROG", default=None,
        help=(
            "Orography type(s): low (0–40 m sdfor), mid (40–120 m), "
            "high (>120 m).  If omitted, all terrain types are combined."
        ),
    )
    parser.add_argument(
        "--lead-time", type=int, nargs="+", dest="lead_time",
        metavar="HOURS", default=None,
        help=(
            "Lead time(s) in hours to include, e.g. --lead-time 24 48 72.  "
            "If omitted, all lead times are combined."
        ),
    )
    parser.add_argument(
        "--start-date", default=None, metavar="YYYY-MM-DD",
        help="Start date override (default: use config start_date).",
    )
    parser.add_argument(
        "--end-date", default=None, metavar="YYYY-MM-DD",
        help="End date override (default: use config end_date).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=(
            "Threshold value to draw as a reference line on the plot. "
            "Defaults to the fixed threshold in the config file if present."
        ),
    )
    parser.add_argument(
        "--n-quantiles", type=int, default=200, metavar="N",
        help="Number of quantile points to compute (default: 200).",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help=(
            "Directory to save PNG plots. "
            "Default: save.output_directory from the config file."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Image resolution in dots per inch (default: 150).",
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help=(
            "Collapse all selected seasons and orography types into a single "
            "combined plot instead of producing one plot per combination."
        ),
    )
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    _style.apply_style()
    config = load_config(args.config)
    fc1_name, fc2_name = get_model_names(config)

    print("\n" + "=" * 70)
    print("Q-Q PLOT TOOL (ENSEMBLE)  —  Scorecards4Extremes")
    print("=" * 70)
    print(f"  Config   : {args.config}")
    print(f"  Variable : {config['variable']}")
    print(f"  Models   : {fc1_name}  vs  {fc2_name}")
    print(f"  Member   : {args.member}")
    print(f"  Period   : {config.get('start_date', '?')} → {config.get('end_date', '?')}")

    threshold = (
        args.threshold
        if args.threshold is not None
        else get_threshold_from_config(config)
    )
    if threshold is not None:
        print(f"  Threshold: {threshold}")

    # Locate the data and resolve member columns (no bulk load yet — data is
    # streamed per (season, orography) combination to keep memory bounded).
    print("\nLocating extracted ensemble point data...")
    source = discover_ens_source(config, fc1_name, fc2_name, args.member)
    fc1_cols = source["fc1_cols"]
    fc2_cols = source["fc2_cols"]
    member_label = source["member_label"]
    member_tag = source["member_tag"]
    member_cols = fc1_cols + fc2_cols
    if source["kind"] == "perday":
        print(f"  {len(source['day_files'])} forecast-day file(s); reading "
              f"{len(source['needed'])} column(s)/file for '{member_label}' "
              f"({len(fc1_cols)} fc1 / {len(fc2_cols)} fc2 member col(s))")
    else:
        print(f"  Single combined file; '{member_label}' "
              f"({len(fc1_cols)} fc1 / {len(fc2_cols)} fc2 member col(s))")

    seasons   = args.season    or [None]
    orogs     = args.orog_type or [None]
    lead_time = args.lead_time

    def _run(season, orog):
        df = load_and_filter_ens(
            source, config, member_cols,
            season=season, orog_type=orog,
            lead_time=lead_time,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        plot_qq_ens(
            df, config, fc1_name, fc2_name,
            fc1_cols, fc2_cols, member_label, member_tag,
            season=season, orog_type=orog,
            lead_time=lead_time,
            n_quantiles=args.n_quantiles,
            output_dir=args.output_dir,
            threshold=threshold,
            dpi=args.dpi,
        )
        del df
        gc.collect()

    if args.no_loop:
        print("\n[1/1] Plotting: all selected conditions combined")
        try:
            _run(None, None)
        except RuntimeError as exc:
            print(f"  Skipping: {exc}")
    else:
        combos = [(s, o) for s in seasons for o in orogs]
        total = len(combos)
        for idx, (season, orog) in enumerate(combos, start=1):
            label = f"{season or 'ALL'} × {orog or 'ALL'}"
            print(f"\n[{idx}/{total}] Plotting: {label}")
            try:
                _run(season, orog)
            except RuntimeError as exc:
                print(f"  Skipping {label}: {exc}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
