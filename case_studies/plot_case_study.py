#!/usr/bin/env python3
"""
plot_case_study.py — Visualise a specific date/forecast-day identified by
find_case_studies.py as a case study.

Produces THREE figures:

Figure 1 — 4-panel overview (saved as <output>.png):
  Panel A (top-left)    : Station map — hit/miss/FA for Model 1
  Panel B (top-right)   : Station map — hit/miss/FA for Model 2
  Panel C (bottom-left) : Exceedance bar chart (top 25 stations by obs excess)
  Panel D (bottom-right): Score comparison table + narrative
  Maps include coastlines and country borders.

Figure 2 — 3-panel actual values (saved as <output>_values.png):
  Panel A : Observed values
  Panel B : Model 1 values
  Panel C : Model 2 values
  Coloured by value magnitude (p2-p98 shared range); black outlines = extremes.

Figure 3 — 2-panel GRIB field + obs overlay (saved as <output>_overlay.png):
  Panel A : Raw IFS GRIB field (pcolormesh, 0.25° grid) + obs dots overlaid
  Panel B : Raw AIFS GRIB field (pcolormesh, 0.25° grid) + obs dots overlaid
  GRIB is read from the raw paths in the config, averaged over all forecast
  steps that constitute the requested day, and interpolated to a 0.25° regular
  grid with scipy.  A discrete 12-band colourmap makes the obs / model colour
  differences immediately visible.  Extreme obs stations are highlighted.

Usage
-----
  python plot_case_study.py --config path/to/config.yaml \\
      --date 20260115 --day 7 [options]

Options
-------
  --config      YAML config file (required)
  --date        Date string YYYYMMDD (required)
  --day         Forecast day integer (required)
  --output-dir  Directory for output PNGs (default: case_study_output/<config_stem>)
  --output      Full PNG path for figure 1 (overrides --output-dir)
  --season      Filter season: DJF MAM JJA SON
  --orog        Filter orog: low mid high
  --title       Extra title string
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml

import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent.parent))
from case_studies.case_study_utils import (
    load_per_station_thresholds,
    get_event_type,
    extract_forecast_values,
    extract_exceedance_probability,
    classify_events,
    classify_events_probabilistic,
    compute_date_metrics,
)


# ─── Constants ────────────────────────────────────────────────────────────────

COLORS = {
    "hit":          "#2e7d32",  # dark green
    "miss":         "#e65100",  # deep orange
    "false_alarm":  "#c62828",  # dark red
    "correct_neg":  "#bdbdbd",  # light grey
}
LABELS = {
    "hit":          "Hit (obs & fc extreme)",
    "miss":         "Miss (obs extreme, fc normal)",
    "false_alarm":  "False Alarm (fc extreme, obs normal)",
    "correct_neg":  "Correct Negative",
}
MARKER_SIZE_SCALE = 55
EUROPE_LON = (-25, 40)
EUROPE_LAT = (35, 72)
PROJ = ccrs.PlateCarree()

# Named zoom regions: (lon_min, lon_max, lat_min, lat_max)
ZOOM_REGIONS = {
    "europe":         (-25, 40,  35, 72),
    "germany":        (  5, 16,  47, 56),
    "uk":             (-11,  3,  49, 61),
    "de_pl_cz":       (  5, 25,  47, 56),
    "central_europe": (  2, 26,  43, 59),  # BeNeLux, DE, PL, S.Scandinavia, CZ, HR, RS, HU
}

# Active extent — overridden at startup when --zoom is passed
_ACTIVE_LON = list(EUROPE_LON)
_ACTIVE_LAT = list(EUROPE_LAT)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      required=True)
    p.add_argument("--date",        required=True, help="YYYYMMDD")
    p.add_argument("--day",         required=True, type=int, help="Forecast day")
    p.add_argument("--output-dir",  default=None,
                   dest="output_dir",
                   help="Directory for output PNGs (default: case_study_output/<config_stem>)")
    p.add_argument("--output",      default=None,
                   help="Full path for figure 1 PNG (overrides --output-dir)")
    p.add_argument("--season",      default=None)
    p.add_argument("--orog",    default=None)
    p.add_argument("--title",   default=None)
    p.add_argument("--tag",     default=None,
                   help="Short label appended to output filenames to avoid overwriting (e.g. 'iekm_ifs')")
    p.add_argument("--no-ensemble-prob", action="store_true")
    p.add_argument("--zoom",    default=None,
                   choices=list(ZOOM_REGIONS.keys()),
                   help="Zoom into a named region (germany, uk, europe)")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Map feature helpers (cartopy) ───────────────────────────────────────────

def _add_map_features(ax, coastline_lw=0.7, border_lw=0.45):
    """Add land, ocean, coastlines and country borders to a cartopy GeoAxes."""
    ax.add_feature(cfeature.LAND,      facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6e8f5", zorder=0)
    ax.add_feature(cfeature.LAKES,     facecolor="#d6e8f5", alpha=0.6, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=coastline_lw, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS,   linewidth=border_lw,    edgecolor="0.35",
                   linestyle="-", zorder=5)
    ax.set_extent([_ACTIVE_LON[0], _ACTIVE_LON[1], _ACTIVE_LAT[0], _ACTIVE_LAT[1]],
                  crs=PROJ)


def _add_gridlines(ax, left_labels=True, bottom_labels=True):
    """Add lat/lon gridlines with axis labels."""
    gl = ax.gridlines(crs=PROJ, draw_labels=True, linewidth=0.4,
                      color="grey", alpha=0.5, linestyle="--",
                      x_inline=False, y_inline=False)
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style  = {"size": 7, "color": "0.4"}
    gl.ylabel_style  = {"size": 7, "color": "0.4"}
    return gl


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_date_slice(config: dict, date_str: str, day: int,
                    season: str = None, orog: str = None) -> pd.DataFrame:
    """Load the parquet for the given forecast day and filter to the target date."""
    parquet_dir = Path(config["extract_points"]["output_path"])
    # Find matching parquet
    candidates = list(parquet_dir.glob(f"*_day{day}.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No parquet for day {day} in {parquet_dir}"
        )
    df = pd.read_parquet(candidates[0])
    # Filter to requested date
    df = df[df["date"].astype(str) == str(date_str)]
    if df.empty:
        raise ValueError(f"Date {date_str} not found in day-{day} parquet")
    # Season filter
    if season:
        month_map = {"DJF": {12,1,2}, "MAM": {3,4,5},
                     "JJA": {6,7,8},  "SON": {9,10,11}}
        months = month_map.get(season.upper(), set())
        m = int(str(date_str)[4:6])
        if months and m not in months:
            raise ValueError(f"Date {date_str} not in season {season}")
    # Orog filter
    if orog and "sdfor" in df.columns:
        ranges = (config.get("filter", {}).get("orography_ranges", {}))
        lo, hi = ranges.get(orog, {"low":(0,40),"mid":(40,120),"high":(120,9999)}.get(orog,(0,9999)))
        df = df[(df["sdfor"] >= lo) & (df["sdfor"] <= hi)]
    return df.reset_index(drop=True)


# ─── Map panel ────────────────────────────────────────────────────────────────

def _compute_marker_sizes(values, T, event_type, masks, base=MARKER_SIZE_SCALE):
    """Return per-station marker sizes scaled by exceedance magnitude."""
    if event_type == "above":
        exc = np.maximum(values - T, 0)
    else:
        exc = np.maximum(T - values, 0)
    # Normalise to [1, 5] relative units then multiply by base
    max_exc = float(np.nanpercentile(exc, 95)) if np.any(exc > 0) else 1.0
    if max_exc < 1e-6:
        max_exc = 1.0
    sizes = base * (1 + 4 * np.minimum(exc / max_exc, 1.0))
    # correct_neg gets smallest size
    sizes[masks["correct_neg"]] = base * 0.4
    return sizes


def draw_station_map(ax, lats, lons, obs, fc, T, event_type, model_name,
                     title_prefix=""):
    """Draw hit/miss/FA scatter map with coastlines and country borders."""
    _add_map_features(ax)
    _add_gridlines(ax)

    masks = classify_events(obs, fc, T, event_type)
    sizes = _compute_marker_sizes(fc, T, event_type, masks)

    order = ["correct_neg", "hit", "miss", "false_alarm"]
    for cat in order:
        m = masks[cat]
        if m.sum() == 0:
            continue
        ax.scatter(
            lons[m], lats[m],
            c=COLORS[cat], s=sizes[m],
            alpha=0.70 if cat == "correct_neg" else 0.88,
            linewidths=0.2, edgecolors="white",
            label=f"{LABELS[cat]} (n={m.sum()})",
            transform=PROJ,
            zorder=6 + order.index(cat),
        )

    n_obs_ext = masks["hit"].sum() + masks["miss"].sum()
    n_fc_ext  = masks["hit"].sum() + masks["false_alarm"].sum()
    pod = masks["hit"].sum() / max(n_obs_ext, 1)
    far = masks["false_alarm"].sum() / max(n_fc_ext, 1)
    ax.set_title(
        f"{title_prefix}{model_name}\n"
        f"POD={pod:.2f}  FAR={far:.2f}  "
        f"Hits={masks['hit'].sum()}  "
        f"Misses={masks['miss'].sum()}  FA={masks['false_alarm'].sum()}",
        fontsize=9, pad=4,
    )
    ax.legend(loc="lower left", fontsize=7, markerscale=0.7,
              framealpha=0.88, edgecolor="grey", facecolor="white")
    return masks


def draw_exceedance_bars(ax, obs, fc1, fc2, T, event_type,
                          model1_name, model2_name, top_n=25):
    """Bar chart: top-N stations by obs exceedance, showing fc1 vs fc2 vs obs."""
    if event_type == "above":
        exc_obs = obs - T
        exc_fc1 = fc1 - T
        exc_fc2 = fc2 - T
    else:
        exc_obs = T - obs
        exc_fc1 = T - fc1
        exc_fc2 = T - fc2

    obs_ext_mask = exc_obs > 0
    if obs_ext_mask.sum() == 0:
        ax.text(0.5, 0.5, "No observed extreme events for this date",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        return

    idx = np.where(obs_ext_mask)[0]
    sorted_idx = idx[np.argsort(-exc_obs[idx])][:top_n]
    xi = np.arange(len(sorted_idx))
    w = 0.28

    ax.bar(xi - w,     exc_obs[sorted_idx], w, label="Obs",  color="#2196F3", alpha=0.85)
    ax.bar(xi,         exc_fc1[sorted_idx], w, label=model1_name[:20], color="#1565C0", alpha=0.85)
    ax.bar(xi + w,     exc_fc2[sorted_idx], w, label=model2_name[:20], color="#B71C1C", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(xi)
    ax.set_xticklabels([str(i+1) for i in range(len(sorted_idx))], fontsize=7)
    ax.set_xlabel(f"Station rank by obs exceedance (top {len(sorted_idx)})")
    ax.set_ylabel("Exceedance above threshold")
    ax.set_title(f"Station-level exceedance (top {len(sorted_idx)} obs extremes)\n"
                 f"Positive = above threshold; negative = model below threshold",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")


# ─── Stats table panel ────────────────────────────────────────────────────────

def draw_stats_table(ax, metrics: dict, model1_name: str, model2_name: str):
    """Draw a formatted comparison table and narrative text."""
    ax.set_axis_off()

    m1, m2 = model1_name[:18], model2_name[:18]
    def _w(v1, v2, low_better=True):
        if v1 is None or v2 is None or np.isnan(v1 or 0) or np.isnan(v2 or 0):
            return "—", "—"
        if low_better:
            return ("✓" if v1 <= v2 else " "), ("✓" if v2 < v1 else " ")
        return ("✓" if v1 >= v2 else " "), ("✓" if v2 > v1 else " ")

    rows = [
        ["Metric", m1, m2, "Better"],
        ["── Contingency ─────────────────────────────"],
        ["  Hits",        metrics["n_hit1"],   metrics["n_hit2"],   "↑ better"],
        ["  Misses",      metrics["n_miss1"],  metrics["n_miss2"],  "↓ better"],
        ["  False Alarms",metrics["n_fa1"],    metrics["n_fa2"],    "↓ better"],
        ["── Scores ──────────────────────────────────"],
        ["  POD",  f"{metrics['pod1']:.3f}",  f"{metrics['pod2']:.3f}",  "↑ better"],
        ["  FAR",  f"{metrics['far1']:.3f}",  f"{metrics['far2']:.3f}",  "↓ better"],
        ["  ETS",  f"{metrics['ets1']:.3f}" if metrics['ets1'] else "—",
                   f"{metrics['ets2']:.3f}" if metrics['ets2'] else "—", "↑ better"],
        ["── twMAE ───────────────────────────────────"],
        ["  Total",       f"{metrics['twmae1']:.5f}", f"{metrics['twmae2']:.5f}", "↓ better"],
        ["  FA severity", f"{metrics['fa_severity1']:.3f}" if metrics['fa_severity1'] else "—",
                          f"{metrics['fa_severity2']:.3f}" if metrics['fa_severity2'] else "—", "↓ better"],
        ["  Hit error",   f"{metrics['hit_err1']:.3f}" if metrics['hit_err1'] else "—",
                          f"{metrics['hit_err2']:.3f}" if metrics['hit_err2'] else "—", "↓ better"],
    ]

    y = 0.99; dy = 0.062
    ax.text(0.5, y + dy, "Score Comparison", ha="center", va="top",
            fontsize=11, weight="bold", transform=ax.transAxes)

    header_done = False
    for row in rows:
        y -= dy
        if len(row) == 1:  # section header
            ax.text(0.01, y, row[0], ha="left", va="top", fontsize=8,
                    weight="bold", color="#2C3E50",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#ECF0F1",
                              edgecolor="none"))
        elif not header_done:
            for xi, (txt, col) in enumerate(zip(row, [0.0, 0.43, 0.65, 0.87])):
                ax.text(col, y, txt, ha="left", va="top", fontsize=8.5,
                        weight="bold", transform=ax.transAxes)
            ax.axhline(y - 0.01, color="grey", linewidth=0.8,
                       xmin=0, xmax=1)
            header_done = True
        else:
            label = row[0]
            v1 = str(row[1]); v2 = str(row[2]); note = str(row[3])
            # Determine winner cell colour
            try:
                fv1, fv2 = float(v1.replace("✓","").strip()), float(v2.replace("✓","").strip())
                low_better = "↓" in note
                w1, w2 = _w(fv1, fv2, low_better)
            except (ValueError, TypeError):
                w1 = w2 = " "

            for xi, (txt, col_x) in enumerate(zip([label, v1, v2, note],
                                                   [0.0, 0.43, 0.65, 0.87])):
                color = "#155724" if (xi == 1 and w1 == "✓") or \
                                     (xi == 2 and w2 == "✓") else "black"
                ax.text(col_x, y, txt, ha="left", va="top", fontsize=8,
                        color=color, transform=ax.transAxes)

    # Auto-narrative
    y -= dy * 1.5
    delta_twmae = metrics.get("delta_twmae", 0)
    better_name = model2_name[:20] if delta_twmae > 0 else model1_name[:20]
    narrative = (
        f"► {better_name} has lower twMAE on this date "
        f"(Δ={abs(delta_twmae):.5f}).  Case: {metrics.get('case_type', '?')}.  "
        f"Dominant FA region: {metrics.get('fa_region1', '?')} ({model1_name[:12]}) / "
        f"{metrics.get('fa_region2', '?')} ({model2_name[:12]})."
    )
    ax.text(0.01, y, narrative, ha="left", va="top", fontsize=8,
            transform=ax.transAxes, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="darkorange", alpha=0.95),
            wrap=True)


# ─── Figure 1: 4-panel overview ──────────────────────────────────────────────

def make_case_study_figure(lats, lons, obs, fc1_cls, fc2_cls, T_v, event_type,
                            fc1_mean, fc2_mean, metrics,
                            config: dict, date_str: str, day: int,
                            model1_name: str, model2_name: str,
                            extra_title: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(22, 16))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22,
                            left=0.04, right=0.97, top=0.91, bottom=0.04)
    ax_m1   = fig.add_subplot(gs[0, 0], projection=PROJ)
    ax_m2   = fig.add_subplot(gs[0, 1], projection=PROJ)
    ax_bars = fig.add_subplot(gs[1, 0])
    ax_tbl  = fig.add_subplot(gs[1, 1])

    draw_station_map(ax_m1, lats, lons, obs, fc1_cls, T_v, event_type,
                     model1_name, title_prefix="Model 1: ")
    draw_station_map(ax_m2, lats, lons, obs, fc2_cls, T_v, event_type,
                     model2_name, title_prefix="Model 2: ")
    draw_exceedance_bars(ax_bars, obs, fc1_mean, fc2_mean, T_v, event_type,
                          model1_name, model2_name)
    draw_stats_table(ax_tbl, metrics, model1_name, model2_name)

    var   = config.get("variable", "")
    pct   = config.get("threshold", {}).get("local_obs_climatology", {}).get("percentile", "?")
    title = (f"Case Study: {model1_name}  vs  {model2_name}\n"
             f"Variable: {var}  |  Date: {date_str}  |  Forecast Day: {day}  "
             f"|  p{pct} per-station threshold")
    if extra_title:
        title += f"\n{extra_title}"
    fig.suptitle(title, fontsize=12, weight="bold", y=0.97)
    return fig


# ─── Figure 2: 3-panel actual values ─────────────────────────────────────────

def _choose_cmap(variable: str) -> str:
    v = variable.lower()
    if "2t" in v or "temp" in v:    return "seismic"   # deep blue=cold, deep red=warm
    if "tp" in v or "precip" in v:  return "YlGnBu"    # white=dry, blue=wet
    if "10ff" in v or "wind" in v:  return "YlOrRd"    # light=calm, red=strong
    return "viridis"


def make_values_figure(df_v: pd.DataFrame, T_v: np.ndarray,
                        fc1: np.ndarray, fc2: np.ndarray, obs: np.ndarray,
                        lats: np.ndarray, lons: np.ndarray,
                        config: dict, date_str: str, day: int,
                        model1_name: str, model2_name: str,
                        extra_title: str = "") -> plt.Figure:
    """3-panel scatter map showing raw values for obs / model1 / model2."""
    var        = config.get("variable", "unknown")
    cmap       = _choose_cmap(var)
    event_type = get_event_type(config)
    pct_cfg    = config.get("threshold", {}).get("local_obs_climatology", {}).get("percentile", "?")

    # Shared colour range: 2nd–98th percentile of all three arrays combined
    all_valid = np.concatenate([obs, fc1, fc2])
    all_valid = all_valid[np.isfinite(all_valid)]
    vmin = float(np.percentile(all_valid, 2))
    vmax = float(np.percentile(all_valid, 98))

    # For diverging temperature maps: centre on the day's median
    _DIVERGING = {"RdBu", "RdBu_r", "seismic", "bwr", "coolwarm"}
    if any(d in cmap for d in _DIVERGING):
        mid  = float(np.nanmedian(all_valid))
        half = max(abs(vmax - mid), abs(mid - vmin))
        vmin, vmax = mid - half, mid + half

    # Extreme event flags
    if event_type == "below":
        obs_ext = obs <= T_v;  fc1_ext = fc1 <= T_v;  fc2_ext = fc2 <= T_v
    else:
        obs_ext = obs >= T_v;  fc1_ext = fc1 >= T_v;  fc2_ext = fc2 >= T_v

    panels = [
        (obs, obs_ext, "Observations"),
        (fc1, fc1_ext, f"Model 1: {model1_name}"),
        (fc2, fc2_ext, f"Model 2: {model2_name}"),
    ]

    fig = plt.figure(figsize=(22, 9))
    gs  = fig.add_gridspec(1, 3, wspace=0.06,
                            left=0.03, right=0.90, top=0.88, bottom=0.06)
    cax = fig.add_axes([0.915, 0.14, 0.016, 0.64])

    unit = "°C" if ("2t" in var or "temp" in var) else \
           "mm" if ("tp" in var or "precip" in var) else ""

    sc_last = None
    for col, (vals, ext_mask, panel_title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col], projection=PROJ)
        _add_map_features(ax)
        _add_gridlines(ax, left_labels=(col == 0), bottom_labels=True)

        # All stations — small background dots
        sc = ax.scatter(
            lons, lats, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
            s=10, alpha=0.72, linewidths=0, edgecolors="none",
            transform=PROJ, zorder=3,
        )
        # Extreme stations — larger with black outline
        if ext_mask.sum() > 0:
            ax.scatter(
                lons[ext_mask], lats[ext_mask],
                c=vals[ext_mask], cmap=cmap, vmin=vmin, vmax=vmax,
                s=45, alpha=0.95, linewidths=0.6, edgecolors="black",
                transform=PROJ, zorder=4,
            )
        pct_ext = 100 * ext_mask.sum() / max(len(vals), 1)
        ax.set_title(f"{panel_title}\n"
                     f"{ext_mask.sum()} extreme stations ({pct_ext:.1f}%,  black outline)",
                     fontsize=9, pad=5)
        sc_last = sc

    # Single shared colourbar
    cbar = fig.colorbar(sc_last, cax=cax)
    cbar.set_label(f"{var}  ({unit})", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    event_lbl = "below" if event_type == "below" else "above"
    title = (f"Station values: {model1_name}  vs  {model2_name}\n"
             f"Variable: {var}  |  Date: {date_str}  |  Day: {day}  "
             f"|  Black outline = p{pct_cfg} extreme events ({event_lbl} threshold)")
    if extra_title:
        title += f"\n{extra_title}"
    fig.suptitle(title, fontsize=11, weight="bold", y=0.97)
    return fig


# ─── Figure 3: GRIB field + obs overlay ─────────────────────────────────────

def _read_grib_europe(grib_path: Path, steps_set: set) -> tuple:
    """Read raw GRIB values over Europe, averaged over the given forecast steps.
    Returns (lats_1d, lons_1d, values_celsius).
    Values are converted from Kelvin to Celsius if the median > 100.
    """
    import eccodes

    lat_min, lat_max = _ACTIVE_LAT
    lon_min, lon_max = _ACTIVE_LON

    mask     = None
    eur_lats = eur_lons = None
    vals_sum = None
    n_steps  = 0

    with open(grib_path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            step = eccodes.codes_get(h, "step")
            if step in steps_set:
                if mask is None:
                    all_lats = eccodes.codes_get_array(h, "latitudes")
                    all_lons = eccodes.codes_get_array(h, "longitudes")
                    # normalise 0-360 longitudes to -180..180
                    all_lons = np.where(all_lons > 180, all_lons - 360, all_lons)
                    mask = ((all_lats >= lat_min) & (all_lats <= lat_max) &
                            (all_lons >= lon_min) & (all_lons <= lon_max))
                    eur_lats = all_lats[mask].astype(np.float32)
                    eur_lons = all_lons[mask].astype(np.float32)

                chunk = eccodes.codes_get_array(h, "values").astype(np.float32)[mask]
                vals_sum = chunk.copy() if vals_sum is None else vals_sum + chunk
                n_steps += 1
            eccodes.codes_release(h)

    if n_steps == 0:
        raise ValueError(f"No GRIB messages for steps {steps_set} in {grib_path}")

    vals = vals_sum / n_steps
    if float(np.nanmedian(vals)) > 100:   # Kelvin → Celsius
        vals -= 273.15
    return eur_lats, eur_lons, vals


def _interpolate_europe(lats: np.ndarray, lons: np.ndarray,
                         vals: np.ndarray, res: float = 0.25) -> tuple:
    """Interpolate scattered GRIB points to a regular lat/lon grid.
    Returns (lat2d, lon2d, grid_values).
    """
    from scipy.interpolate import griddata
    lat_g  = np.arange(_ACTIVE_LAT[0], _ACTIVE_LAT[1] + res, res)
    lon_g  = np.arange(_ACTIVE_LON[0], _ACTIVE_LON[1] + res, res)
    lon2d, lat2d = np.meshgrid(lon_g, lat_g)
    grid = griddata((lons, lats), vals, (lon2d, lat2d), method="linear")
    return lat2d, lon2d, grid


def make_grib_overlay_figure(df_v: pd.DataFrame, T_v: np.ndarray,
                              obs: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                              config: dict, date_str: str, day: int,
                              model1_name: str, model2_name: str,
                              fc1_mean: np.ndarray = None,
                              fc2_mean: np.ndarray = None,
                              extra_title: str = "") -> plt.Figure:
    """2-panel figure: raw GRIB model field (pcolormesh) + obs stations (dots).

    Reads the original GRIB files from the raw paths defined in the config,
    averages over all forecast steps that constitute the requested forecast day,
    interpolates to a 0.25-degree regular grid with scipy.griddata, and plots
    as a filled pcolormesh.  A discrete 12-band colourmap makes colour
    differences between the field and the overlaid obs dots immediately
    readable.  Extreme obs stations are highlighted with bold black outlines.
    """
    import eccodes  # noqa: F401 — confirms eccodes is available

    var        = config.get("variable", "unknown")
    event_type = get_event_type(config)
    pct_cfg    = config.get("threshold", {}).get("local_obs_climatology", {}).get("percentile", "?")

    # Forecast steps for this day from the parquet data
    steps_set = set(int(s) for s in df_v["step"].unique())

    # GRIB file paths from config
    src1 = config["read_data"]["forecast_model1"].get("source", "")
    src2 = config["read_data"]["forecast_model2"].get("source", "")
    if src1 != "local_grib" or src2 != "local_grib":
        raise ValueError("make_grib_overlay_figure requires source='local_grib' for both models")
    grib_dir1 = Path(config["read_data"]["forecast_model1"]["local_grib"]["path"])
    grib_dir2 = Path(config["read_data"]["forecast_model2"]["local_grib"]["path"])
    grib1 = grib_dir1 / f"{var}_{date_str}.grib"
    grib2 = grib_dir2 / f"{var}_{date_str}.grib"
    if not grib1.exists():
        raise FileNotFoundError(f"Model 1 GRIB not found: {grib1}")
    if not grib2.exists():
        raise FileNotFoundError(f"Model 2 GRIB not found: {grib2}")

    print(f"  Reading GRIB (model 1): {grib1.name}  steps={sorted(steps_set)}")
    glats1, glons1, gvals1 = _read_grib_europe(grib1, steps_set)
    print(f"  Reading GRIB (model 2): {grib2.name}")
    glats2, glons2, gvals2 = _read_grib_europe(grib2, steps_set)
    print(f"  Interpolating {len(glats1)} + {len(glats2)} points → 0.25° grid ...")
    lat2d1, lon2d1, grid1 = _interpolate_europe(glats1, glons1, gvals1)
    lat2d2, lon2d2, grid2 = _interpolate_europe(glats2, glons2, gvals2)

    # Shared colour range across both grids + obs
    all_vals = np.concatenate([gvals1, gvals2, obs])
    all_vals  = all_vals[np.isfinite(all_vals)]
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    base_cmap_name = _choose_cmap(var)
    _DIVERGING2 = {"RdBu", "RdBu_r", "seismic", "bwr", "coolwarm"}
    if any(d in base_cmap_name for d in _DIVERGING2):
        mid  = float(np.nanmedian(all_vals))
        half = max(abs(vmax - mid), abs(mid - vmin))
        vmin, vmax = mid - half, mid + half

    # 20-band discrete colourmap for striking visual discrimination
    N_LEVELS  = 20
    base_cmap = plt.get_cmap(base_cmap_name)
    cmap_disc = mcolors.ListedColormap(
        [base_cmap(i / (N_LEVELS - 1)) for i in range(N_LEVELS)]
    )
    bounds = np.linspace(vmin, vmax, N_LEVELS + 1)
    norm   = mcolors.BoundaryNorm(bounds, ncolors=N_LEVELS)

    # Extreme obs flags
    obs_ext = obs <= T_v if event_type == "below" else obs >= T_v

    unit   = "°C" if ("2t" in var or "temp" in var) else \
             "mm" if ("tp" in var or "precip" in var) else ""

    # ── Difference grid and station-level forecast difference ─────────────────
    grid_diff = grid1 - grid2  # positive where M1 > M2
    diff_vals_finite = grid_diff[np.isfinite(grid_diff)]
    diff_half = max(abs(float(np.percentile(diff_vals_finite,  2))),
                    abs(float(np.percentile(diff_vals_finite, 98))), 0.1)
    N_DIFF = 20
    diff_cmap_obj  = plt.get_cmap("seismic")
    diff_cmap_disc = mcolors.ListedColormap(
        [diff_cmap_obj(i / (N_DIFF - 1)) for i in range(N_DIFF)]
    )
    diff_bounds = np.linspace(-diff_half, diff_half, N_DIFF + 1)
    diff_norm   = mcolors.BoundaryNorm(diff_bounds, ncolors=N_DIFF)

    # Station-level forecast difference (from parquet values if available)
    if fc1_mean is not None and fc2_mean is not None:
        fc_diff = fc1_mean.astype(np.float32) - fc2_mean.astype(np.float32)
    else:
        fc_diff = None

    panels = [
        (lat2d1, lon2d1, grid1,    cmap_disc, norm,      f"Model 1: {model1_name}"),
        (lat2d2, lon2d2, grid2,    cmap_disc, norm,      f"Model 2: {model2_name}"),
        (lat2d1, lon2d1, grid_diff, diff_cmap_disc, diff_norm,
         f"Difference: {model1_name} − {model2_name}  (red = M1 warmer)"),
    ]

    fig = plt.figure(figsize=(28, 10))
    gs  = fig.add_gridspec(1, 3, wspace=0.06,
                            left=0.02, right=0.99, top=0.88, bottom=0.16)
    # Horizontal colourbar below panels 0+1 (temperature) and below panel 2 (diff)
    cax_main = fig.add_axes([0.03, 0.05, 0.60, 0.04])   # below M1 + M2 panels
    cax_diff = fig.add_axes([0.69, 0.05, 0.29, 0.04])   # below diff panel

    for col, (lat2d, lon2d, grid_vals, col_cmap, col_norm, panel_title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col], projection=PROJ)
        _add_map_features(ax, coastline_lw=1.8, border_lw=0.9)

        ax.pcolormesh(
            lon2d, lat2d, grid_vals,
            cmap=col_cmap, norm=col_norm,
            transform=PROJ, zorder=2, alpha=0.88,
            shading="auto",
        )

        _add_gridlines(ax, left_labels=(col == 0), bottom_labels=True)

        if col < 2:
            # Panels 0 and 1: all obs stations + extreme highlighted
            ax.scatter(
                lons, lats, c=obs, cmap=cmap_disc, norm=norm,
                s=14, alpha=0.92, marker="o",
                linewidths=0.7, edgecolors="white",
                transform=PROJ, zorder=6,
            )
            if obs_ext.sum() > 0:
                ax.scatter(
                    lons[obs_ext], lats[obs_ext],
                    c=obs[obs_ext], cmap=cmap_disc, norm=norm,
                    s=55, alpha=1.0, marker="o",
                    linewidths=1.3, edgecolors="black",
                    transform=PROJ, zorder=7,
                )
            ax.set_title(
                f"{panel_title}  (GRIB) + obs\n"
                f"obs extreme (black outline): n={obs_ext.sum()}",
                fontsize=9, pad=5,
            )
        else:
            # Panel 2: difference field only — no observation dots
            ax.set_title(
                f"{panel_title}",
                fontsize=9, pad=5,
            )

    sm_main = plt.cm.ScalarMappable(norm=norm, cmap=cmap_disc)
    sm_main.set_array([])
    cbar_main = fig.colorbar(sm_main, cax=cax_main, ticks=bounds[::4],
                             orientation="horizontal")
    cbar_main.set_label(f"2m Temperature  ({unit})  — Model fields (panels 1 & 2)",
                        fontsize=10, fontweight="bold")
    cbar_main.ax.tick_params(labelsize=8)

    sm_diff = plt.cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap_disc)
    sm_diff.set_array([])
    cbar_diff = fig.colorbar(sm_diff, cax=cax_diff, ticks=diff_bounds[::4],
                             orientation="horizontal")
    cbar_diff.set_label(f"M1 − M2  ({unit})  — Difference (panel 3)",
                        fontsize=10, fontweight="bold")
    cbar_diff.ax.tick_params(labelsize=8)

    event_lbl = "below" if event_type == "below" else "above"
    step_str  = "+".join(str(s) for s in sorted(steps_set))
    title = (
        f"Raw GRIB field (pcolormesh, 0.25°) + Observations (dots):  "
        f"{model1_name}  vs  {model2_name}\n"
        f"Variable: {var}  |  Date: {date_str}  |  Day {day}  (steps: {step_str}h)  "
        f"|  p{pct_cfg} threshold ({event_lbl})  |  Black circle = extreme obs"
    )
    if extra_title:
        title += f"\n{extra_title}"
    fig.suptitle(title, fontsize=11, weight="bold", y=0.97)
    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    # Apply zoom region if requested
    if args.zoom and args.zoom in ZOOM_REGIONS:
        lon0, lon1, lat0, lat1 = ZOOM_REGIONS[args.zoom]
        _ACTIVE_LON[0], _ACTIVE_LON[1] = lon0, lon1
        _ACTIVE_LAT[0], _ACTIVE_LAT[1] = lat0, lat1
        print(f"  Zoom region: {args.zoom}  "
              f"(lon {lon0}–{lon1}, lat {lat0}–{lat1})")

    model1 = config["read_data"]["forecast_model1"]["name"]
    model2 = config["read_data"]["forecast_model2"]["name"]
    mode   = config.get("mode", "deterministic")
    use_prob = (mode == "ensemble") and (not args.no_ensemble_prob)

    print(f"\nPlotting case study: {args.date}  day {args.day}")
    print(f"  Models  : {model1}  vs  {model2}")
    print(f"  Config  : {args.config}")

    df = load_date_slice(config, args.date, args.day,
                         season=args.season, orog=args.orog)
    print(f"  Stations: {len(df)}")

    # Shared preprocessing — computed once for both figures
    event_type = get_event_type(config)
    T_full = load_per_station_thresholds(config, df)
    valid  = ~np.isnan(T_full) & ~np.isnan(df["obs_value"].values)
    df_v   = df[valid].copy()
    T_v    = T_full[valid]
    obs    = df_v["obs_value"].values.astype(np.float32)
    lats   = df_v["lat"].values.astype(np.float32)
    lons   = df_v["lon"].values.astype(np.float32)

    if mode == "ensemble" and use_prob:
        prob1, prob2       = extract_exceedance_probability(df_v, T_v, event_type)
        fc1_mean, fc2_mean = extract_forecast_values(df_v, "ensemble")
        fc1_cls, fc2_cls   = prob1, prob2
    else:
        fc1_mean, fc2_mean = extract_forecast_values(df_v, mode)
        fc1_cls, fc2_cls   = fc1_mean, fc2_mean

    metrics = compute_date_metrics(obs, fc1_mean, fc2_mean, T_v, event_type,
                                    lats, lons)

    # Figure 1 — 4-panel hit/miss/FA overview
    fig1 = make_case_study_figure(
        lats, lons, obs, fc1_cls, fc2_cls, T_v, event_type,
        fc1_mean, fc2_mean, metrics,
        config, args.date, args.day, model1, model2,
        extra_title=args.title or "",
    )
    zoom_suffix = f"_{args.zoom}" if args.zoom and args.zoom != "europe" else ""
    if args.output:
        out1 = args.output
    else:
        out_dir = Path(args.output_dir) if args.output_dir else \
                  Path("case_study_output") / Path(args.config).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        tag_suffix = f"_{args.tag}" if args.tag else ""
        out1 = str(out_dir / f"case_study_{args.date}_day{args.day}{zoom_suffix}{tag_suffix}.png")
    fig1.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ Saved (overview): {out1}")

    # Figure 2 — 3-panel values map
    out2 = str(out1).replace(".png", "_values.png")
    fig2 = make_values_figure(
        df_v, T_v, fc1_mean, fc2_mean, obs, lats, lons,
        config, args.date, args.day, model1, model2,
        extra_title=args.title or "",
    )
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved (values):   {out2}")

    # Figure 3 — 3-panel raw GRIB field + obs overlay + difference
    out3 = str(out1).replace(".png", "_overlay.png")
    try:
        fig3 = make_grib_overlay_figure(
            df_v, T_v, obs, lats, lons,
            config, args.date, args.day, model1, model2,
            fc1_mean=fc1_mean, fc2_mean=fc2_mean,
            extra_title=args.title or "",
        )
        fig3.savefig(out3, dpi=180, bbox_inches="tight")
        plt.close(fig3)
        print(f"  ✓ Saved (overlay):  {out3}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ! Skipped overlay figure: {exc}")


if __name__ == "__main__":
    main()
