#!/usr/bin/env python3
"""
QQ Plot Tool - Quantile-Quantile Plots for Forecast Verification
=================================================================
Creates QQ plots (observations vs forecasts) from extracted point data.

The extracted point data must already exist (run the main pipeline first).

USAGE
-----
  python plot_qq.py --config config_tp24_precipitation.yaml
pi  python plot_qq.py --config config_tp24_precipitation.yaml --season DJF MAM --orog low mid high
  python plot_qq.py --config config_tp24_precipitation.yaml --lead-time 24 48
  python plot_qq.py --help

By default the script loops over every (season, orography) combination you
specify and saves one PNG per combination.  Use --no-loop to collapse all
selected data into a single plot.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling _style module
import _style

# ============================================================================
# CONSTANTS
# ============================================================================

VARIABLE_LABELS = {
    "2t":     ("2m Temperature",     "°C"),
    "10ff":   ("10m Wind Speed",     "m/s"),
    "tp24":   ("24h Precipitation",  "mm"),
    "aod500": ("AOD 500 nm",         ""),
    "go3":    ("Ozone (O3)",         "ppb"),
    "pm2p5":  ("PM2.5",              "µg/m³"),
}

SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

OROGRAPHY_RANGES = {
    "low":  (0,   40),
    "mid":  (40,  120),
    "high": (120, 3000),
}

# Predefined geographic areas [North, West, South, East] — MUST mirror filter.py
AREAS = {
    "europe":          [68, -15, 27, 50],
    "nh_extratropics": [90, -180, 20, 180],
    "tropics":         [20, -180, -20, 180],
}


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def load_config(config_file: str) -> dict:
    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"\nERROR: Config file not found: {config_file}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR loading config: {exc}")
        sys.exit(1)


def get_model_names(config: dict):
    fc1 = config["read_data"]["forecast_model1"]["name"]
    fc2 = config["read_data"]["forecast_model2"]["name"]
    return fc1, fc2


def get_threshold_from_config(config: dict):
    """Extract the threshold value from config if it is a fixed threshold."""
    thresh_cfg = config.get("threshold", {})
    method = thresh_cfg.get("method")
    if method == "fixed":
        return thresh_cfg["fixed"]["value"]
    return None


def _format_threshold_suffix(config: dict) -> str:
    """
    Reproduce the filename suffix used by filter.py / extract_points.py,
    so we can locate the correct parquet/csv file.
    """
    if "threshold" not in config:
        return ""
    thresh_cfg = config["threshold"]
    method = thresh_cfg.get("method", "")
    variable = config.get("variable", "")

    if method == "fixed":
        value = thresh_cfg["fixed"]["value"]
        if variable == "tp24":
            return f"_{value:.0f}mm"
        if variable == "2t":
            return f"_{value:.1f}C"
        if variable == "10ff":
            return f"_{value:.1f}ms"
        return f"_{value:.1f}"

    if method in ("dataset_climatology", "station_climatology"):
        percentile = thresh_cfg[method]["percentile"]
        suffixes = {1: "1st", 2: "2nd", 3: "3rd", 99: "99th"}
        pct_str = suffixes.get(percentile, f"{percentile}th")
        return f"_{pct_str}"

    return ""


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config: dict, fc1_name: str, fc2_name: str) -> pd.DataFrame:
    """
    Load extracted point data from the path specified in config.
    Supports both the per-forecast-day format (memory-efficient) and
    the legacy single-file format (parquet / pickle / csv).
    """
    var = config["variable"]
    point_data_path = Path(config["extract_points"]["output_path"])
    threshold_suffix = _format_threshold_suffix(config)

    # --- Per-forecast-day files (preferred) --------------------------------
    # Try plain pattern first, then ensemble (_ens) suffix variant
    day_pattern = f"{var}_{fc1_name}_vs_{fc2_name}_day*.parquet"
    day_files = sorted(point_data_path.glob(day_pattern))
    if not day_files:
        day_pattern = f"{var}_{fc1_name}_vs_{fc2_name}_ens_day*.parquet"
        day_files = sorted(point_data_path.glob(day_pattern))

    if day_files:
        print(f"  Found {len(day_files)} forecast-day file(s) — memory-efficient load")
        parts = []
        for fp in day_files:
            df_day = pd.read_parquet(fp)
            m = re.search(r"day(\d+)\.parquet", fp.name)
            if m:
                df_day["forecast_day"] = int(m.group(1))
            # Ensemble parquets: collapse member columns into means immediately
            # before concatenating, to avoid holding 100 columns × 10 files in RAM.
            fc1_members = sorted([c for c in df_day.columns if c.startswith("fc1_member_")])
            fc2_members = sorted([c for c in df_day.columns if c.startswith("fc2_member_")])
            if fc1_members and "fc1_value" not in df_day.columns:
                df_day["fc1_value"] = df_day[fc1_members].mean(axis=1)
                df_day = df_day.drop(columns=fc1_members)
            if fc2_members and "fc2_value" not in df_day.columns:
                df_day["fc2_value"] = df_day[fc2_members].mean(axis=1)
                df_day = df_day.drop(columns=fc2_members)
            parts.append(df_day)
            print(f"    {fp.name}: {len(df_day):,} rows")
        df = pd.concat(parts, ignore_index=True)
        return df

    # --- Fallback: single combined file ------------------------------------
    base = f"{var}_{fc1_name}_vs_{fc2_name}{threshold_suffix}"
    for ext, reader in [
        (".parquet", pd.read_parquet),
        (".pkl",     pd.read_pickle),
        (".csv",     pd.read_csv),
    ]:
        fp = point_data_path / f"{base}{ext}"
        if fp.exists():
            print(f"  Reading {fp.name}")
            return reader(fp)

    raise FileNotFoundError(
        f"No extracted data found in '{point_data_path}'.\n"
        f"Expected files matching:\n"
        f"  {point_data_path / day_pattern}  (per-day)\n"
        f"  {point_data_path / base}.parquet  (combined)\n"
        "Run the main pipeline first (python run.py --config ...) to generate "
        "the extracted point data."
    )


# ============================================================================
# FILTERING
# ============================================================================

def apply_filters(
    df: pd.DataFrame,
    config: dict,
    *,
    season=None,
    orog_type=None,
    lead_time=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Apply sub-setting filters that mirror filter.py logic.

    Parameters
    ----------
    season     : str or None  – 'DJF', 'MAM', 'JJA', 'SON'
    orog_type  : str or None  – 'low', 'mid', 'high'
    lead_time  : int / list of int / None  – step(s) in hours
    start_date : 'YYYY-MM-DD' or None
    end_date   : 'YYYY-MM-DD' or None
    """
    print(f"  Data size before filtering: {len(df):,} rows")
    df = df.copy()

    # -- Geographic area -------------------------------------------------------
    area_name = config.get("filter", {}).get("area")
    if area_name:
        if area_name in AREAS:
            lat_north, lon_west, lat_south, lon_east = AREAS[area_name]
            df = df[
                (df["lat"] >= lat_south) & (df["lat"] <= lat_north) &
                (df["lon"] >= lon_west) & (df["lon"] <= lon_east)
            ]
            print(f"  After area filter ({area_name}): {len(df):,} rows")
        else:
            print(f"  Warning: Unknown area '{area_name}', skipping area filter")

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
        print(f"  After date filter  ({sd} → {ed}): {len(df):,} rows")

    # -- Season --------------------------------------------------------------
    if season:
        months = SEASON_MONTHS.get(season.upper())
        if months is None:
            raise ValueError(
                f"Unknown season '{season}'. Choose from: {list(SEASON_MONTHS)}"
            )
        df["_month"] = df["date"].astype(str).str[4:6].astype(int)
        df = df[df["_month"].isin(months)].drop(columns=["_month"])
        print(f"  After season filter ({season}): {len(df):,} rows")

    # -- Lead time -----------------------------------------------------------
    if lead_time is not None:
        steps = [lead_time] if isinstance(lead_time, int) else list(lead_time)
        df = df[df["step"].isin(steps)]
        print(f"  After lead-time filter (steps={steps}h): {len(df):,} rows")

    # -- Orography -----------------------------------------------------------
    if orog_type:
        if "sdfor" not in df.columns:
            print("  Warning: 'sdfor' column not found — skipping orography filter")
        else:
            bounds = OROGRAPHY_RANGES.get(orog_type.lower())
            if bounds is None:
                raise ValueError(
                    f"Unknown orography type '{orog_type}'. "
                    f"Choose from: {list(OROGRAPHY_RANGES)}"
                )
            lo, hi = bounds
            df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]
            print(
                f"  After orography filter ({orog_type}, sdfor=[{lo}, {hi})): "
                f"{len(df):,} rows"
            )

    # -- NaN removal ---------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=["obs_value", "fc1_value", "fc2_value"])
    if len(df) < before:
        print(f"  After NaN removal: {len(df):,} rows (dropped {before - len(df):,})")

    # -- Quality control from config (mirrors filter.py) ---------------------
    filter_cfg = config.get("filter", {})
    variable = config.get("variable", "")

    # Precipitation: remove obs above max_valid_precipitation
    if variable == "tp24":
        max_precip = filter_cfg.get("max_valid_precipitation", None)
        if max_precip is not None:
            before = len(df)
            df = df[df["obs_value"] <= max_precip]
            removed = before - len(df)
            if removed:
                print(
                    f"  Precipitation QC (obs <= {max_precip} mm): "
                    f"{len(df):,} rows (removed {removed:,} suspect obs)"
                )

    # Temperature: apply valid range filter
    if variable == "2t":
        min_temp = filter_cfg.get("min_valid_temperature", -80.0)
        max_temp = filter_cfg.get("max_valid_temperature", 60.0)
        before = len(df)
        df = df[
            (df["obs_value"] >= min_temp) & (df["obs_value"] <= max_temp)
        ]
        removed = before - len(df)
        if removed:
            print(
                f"  Temperature QC ({min_temp}–{max_temp} °C): "
                f"{len(df):,} rows (removed {removed:,})"
            )

    # General outlier removal (std-based)
    if filter_cfg.get("remove_outliers", False):
        threshold_std = filter_cfg.get("outlier_threshold_std", 5.0)
        before = len(df)
        for col in ["obs_value", "fc1_value", "fc2_value"]:
            mean, std = df[col].mean(), df[col].std()
            df = df[np.abs(df[col] - mean) <= threshold_std * std]
        removed = before - len(df)
        if removed:
            print(
                f"  Outlier removal (±{threshold_std}σ): "
                f"{len(df):,} rows (removed {removed:,})"
            )

    if len(df) == 0:
        raise RuntimeError(
            "No data remains after filtering. "
            "Adjust filter criteria or check that the data covers the selected conditions."
        )

    return df


# ============================================================================
# QQ COMPUTATION
# ============================================================================

def compute_quantiles(series: np.ndarray, n_quantiles: int = 200) -> np.ndarray:
    """Return *n_quantiles* evenly-spaced percentiles of *series*."""
    q = np.linspace(0, 100, n_quantiles)
    return np.nanpercentile(series, q)


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


def plot_qq(
    df: pd.DataFrame,
    config: dict,
    fc1_name: str,
    fc2_name: str,
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
    Create a QQ plot comparing observations vs both forecast models and save it.

    Single panel showing both models with:
      - Dots at each integer percentile (0–99th).
      - 'x' markers every 0.1 percentile from 99th to 99.9th.
      - An inset bar chart showing % change of RMSE / correlation / |bias|
        for fc2 relative to fc1, with absolute values annotated.
      - Optional threshold marker lines.
    """
    variable = config["variable"]
    var_label, unit = VARIABLE_LABELS.get(variable, (variable, ""))
    condition = _condition_label(season, orog_type, lead_time)

    obs = df["obs_value"].values
    fc1 = df["fc1_value"].values
    fc2 = df["fc2_value"].values

    # -- Continuous QQ line (n_quantiles points, for smooth visual guide) ---
    q_levels = np.linspace(0, 100, n_quantiles)
    obs_q  = np.nanpercentile(obs,  q_levels)
    fc1_q  = np.nanpercentile(fc1,  q_levels)
    fc2_q  = np.nanpercentile(fc2,  q_levels)

    # -- Dots at every integer percentile 0–99 ------------------------------
    dot_levels = np.arange(0, 99.5, 1.0)
    obs_dots  = np.nanpercentile(obs,  dot_levels)
    fc1_dots  = np.nanpercentile(fc1,  dot_levels)
    fc2_dots  = np.nanpercentile(fc2,  dot_levels)

    # -- X markers at 99.0, 99.1, …, 99.9 ----------------------------------
    x_levels = np.linspace(99.0, 99.9, 10)
    obs_x  = np.nanpercentile(obs,  x_levels)
    fc1_x  = np.nanpercentile(fc1,  x_levels)
    fc2_x  = np.nanpercentile(fc2,  x_levels)

    # -- Statistics from raw paired values ----------------------------------
    corr1 = float(np.corrcoef(fc1, obs)[0, 1])
    bias1 = float(np.mean(fc1 - obs))
    rmse1 = float(np.sqrt(np.mean((fc1 - obs) ** 2)))

    corr2 = float(np.corrcoef(fc2, obs)[0, 1])
    bias2 = float(np.mean(fc2 - obs))
    rmse2 = float(np.sqrt(np.mean((fc2 - obs) ** 2)))

    frac_rmse = (rmse2 - rmse1) / rmse1 * 100.0 if rmse1 != 0.0 else 0.0
    frac_corr = (corr2 - corr1) / abs(corr1) * 100.0 if corr1 != 0.0 else 0.0
    frac_bias = (abs(bias2) - abs(bias1)) / abs(bias1) * 100.0 if bias1 != 0.0 else 0.0

    # Shared colourblind-safe scheme (blue = fc1, vermillion = fc2) — see _style.py
    COLOR_FC1 = _style.C_FC1
    COLOR_FC2 = _style.C_FC2
    COLOR_REF = _style.C_REF

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Q-Q Plot  —  {var_label}  |  {condition}  "
        f"(N = {len(df):,})",
        fontsize=12, fontweight="bold",
    )

    # Axis limits capped at 99.9th percentile — no extreme outliers displayed
    all_capped = np.concatenate([obs_dots, obs_x, fc1_dots, fc1_x, fc2_dots, fc2_x])
    vmin = np.nanmin(all_capped)
    vmax = np.nanmax(all_capped)
    margin = (vmax - vmin) * 0.05
    lim = (vmin - margin, vmax + margin)

    # 1:1 reference line (no legend entry)
    ax.plot(lim, lim, color=COLOR_REF, linestyle="--", linewidth=1.2,
            zorder=1)

    # QQ markers: dot per integer percentile + x every 0.1% from 99–99.9
    for fc_dots, fc_x, color, name in [
        (fc1_dots, fc1_x, COLOR_FC1, fc1_name),
        (fc2_dots, fc2_x, COLOR_FC2, fc2_name),
    ]:
        # Dot at each integer percentile (0–99)
        ax.scatter(obs_dots, fc_dots, color=color, s=20, zorder=3,
                   marker="o", label=name)
        # X marker every 0.1 % from 99th to 99.9th
        ax.scatter(obs_x, fc_x, color=color, s=60, zorder=4,
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
    ax.text(
        lb.x1, lb.y1 + 0.005,
        "o = 0–99th percentile\nx = 99–99.9th (Δ 0.1%)",
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

    # Value labels: placed *above* the inset (y > 1 in inset-axes fraction),
    # using clip_on=False so they are visible outside the inset boundary.
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

    # Build filename
    fn_parts = [f"qq_{variable}_{fc1_name}_vs_{fc2_name}"]
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
        prog="plot_qq.py",
        description=(
            "Create Q-Q plots (observations vs forecasts) from extracted point data.\n"
            "The extracted point data must already exist — run the main pipeline first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # All data combined (one QQ plot):
  python plot_qq.py --config config_tp24_precipitation.yaml

  # One QQ plot per season (4 plots):
  python plot_qq.py --config config_tp24_precipitation.yaml --season DJF MAM JJA SON

  # Specific season + orography (1 plot):
  python plot_qq.py --config config_tp24_precipitation.yaml --season DJF --orog low

  # All 12 season×orography combinations:
  python plot_qq.py --config config_tp24_precipitation.yaml \\
      --season DJF MAM JJA SON --orog low mid high

  # Only day-3 lead time, custom output folder:
  python plot_qq.py --config config_tp24_precipitation.yaml \\
      --lead-time 72 --output-dir ./plots/qq

  # Override threshold and number of quantiles:
  python plot_qq.py --config config_tp24_precipitation.yaml \\
      --threshold 20.0 --n-quantiles 500
""",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (same file used for the main pipeline).",
    )
    parser.add_argument(
        "--season", nargs="+", choices=["DJF", "MAM", "JJA", "SON"],
        metavar="SEASON", default=None,
        help=(
            "Season(s) to plot.  Specify one or more of DJF MAM JJA SON. "
            "If omitted, all seasons are combined into a single plot unless "
            "--no-loop is not set."
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
    print("Q-Q PLOT TOOL  —  Scorecards4Extremes")
    print("=" * 70)
    print(f"  Config   : {args.config}")
    print(f"  Variable : {config['variable']}")
    print(f"  Models   : {fc1_name}  vs  {fc2_name}")
    print(f"  Period   : {config.get('start_date', '?')} → {config.get('end_date', '?')}")

    # Threshold
    threshold = (
        args.threshold
        if args.threshold is not None
        else get_threshold_from_config(config)
    )
    if threshold is not None:
        print(f"  Threshold: {threshold}")

    # Load data once
    print("\nLoading extracted point data...")
    df_full = load_data(config, fc1_name, fc2_name)
    print(f"  Total rows loaded: {len(df_full):,}")

    # Resolve iteration space
    seasons   = args.season    or [None]
    orogs     = args.orog_type or [None]
    lead_time = args.lead_time  # list[int] or None

    if args.no_loop:
        # ------------------------------------------------------------------
        # Single combined plot
        # ------------------------------------------------------------------
        print("\n[1/1] Plotting: all selected conditions combined")
        filtered = apply_filters(
            df_full, config,
            season=None, orog_type=None,
            lead_time=lead_time,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        plot_qq(
            filtered, config, fc1_name, fc2_name,
            season=None, orog_type=None,
            lead_time=lead_time,
            n_quantiles=args.n_quantiles,
            output_dir=args.output_dir,
            threshold=threshold,
            dpi=args.dpi,
        )
    else:
        # ------------------------------------------------------------------
        # Loop over season × orography combinations
        # ------------------------------------------------------------------
        combos = [(s, o) for s in seasons for o in orogs]
        total = len(combos)

        for idx, (season, orog) in enumerate(combos, start=1):
            label = f"{season or 'ALL'} × {orog or 'ALL'}"
            print(f"\n[{idx}/{total}] Plotting: {label}")
            try:
                filtered = apply_filters(
                    df_full, config,
                    season=season, orog_type=orog,
                    lead_time=lead_time,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                plot_qq(
                    filtered, config, fc1_name, fc2_name,
                    season=season, orog_type=orog,
                    lead_time=lead_time,
                    n_quantiles=args.n_quantiles,
                    output_dir=args.output_dir,
                    threshold=threshold,
                    dpi=args.dpi,
                )
            except RuntimeError as exc:
                print(f"  Skipping {label}: {exc}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
