#!/usr/bin/env python3
"""
QQ Plot for Warm and Cold Extremes — split into two panels.

  Warm extremes: 90th–99.9th percentile (dots at each integer pct, x at 99–99.9)
  Cold extremes: 0.1th–10th percentile (dots at each integer pct, x at 0.1–1.0)

Usage:
  python plot_qq_extremes.py --config <yaml> [--day N] [--season DJF] [--orog low]
"""

import argparse
import sys
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

VARIABLE_LABELS = {
    "2t":     ("2m Temperature",     "°C"),
    "10ff":   ("10m Wind Speed",     "m/s"),
    "tp24":   ("24h Precipitation",  "mm"),
    "aod500": ("AOD 500 nm",         ""),
    "go3":    ("Ozone (O3)",         "ppb"),
    "pm2p5":  ("PM2.5",              "µg/m³"),
}

SEASON_MONTHS = {
    "DJF": [12, 1, 2], "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],  "SON": [9, 10, 11],
    "ASO": [8, 9, 10],
}

OROGRAPHY_RANGES = {
    "flat": (0, 40), "low": (0, 40),
    "hilly": (40, 120), "mid": (40, 120),
    "complex": (120, 3000), "high": (120, 3000),
}

# Predefined geographic areas [North, West, South, East] — MUST mirror filter.py
AREAS = {
    "europe":          [68, -15, 27, 50],
    "nh_extratropics": [90, -180, 20, 180],
    "tropics":         [20, -180, -20, 180],
}

# Shared colourblind-safe palette (see diagnostics/_style.py)
COLOR_FC1 = _style.C_FC1   # model1 — blue
COLOR_FC2 = _style.C_FC2   # model2 — vermillion


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_day(config, day):
    """Load a single forecast-day parquet."""
    var = config["variable"]
    fc1 = config["read_data"]["forecast_model1"]["name"]
    fc2 = config["read_data"]["forecast_model2"]["name"]
    base = Path(config["extract_points"]["output_path"])

    # Try plain pattern, then with _99th suffix (symlinked), then _ens
    for pat in [
        f"{var}_{fc1}_vs_{fc2}_day{day}.parquet",
        f"{var}_{fc1}_vs_{fc2}_99th_day{day}.parquet",
        f"{var}_{fc1}_vs_{fc2}_ens_day{day}.parquet",
    ]:
        fp = base / pat
        if fp.exists():
            print(f"  Loading {fp.name}")
            df = pd.read_parquet(fp)
            # Collapse ensemble members if present
            for prefix, target in [("fc1_member_", "fc1_value"),
                                   ("fc2_member_", "fc2_value")]:
                members = [c for c in df.columns if c.startswith(prefix)]
                if members and target not in df.columns:
                    df[target] = df[members].mean(axis=1)
                    df = df.drop(columns=members)
            return df, fc1, fc2

    raise FileNotFoundError(f"No day-{day} parquet found in {base}")


def filter_data(df, config, season=None, orog=None):
    """Apply date/season/orog/QC filters."""
    area_name = config.get("filter", {}).get("area")
    if area_name:
        if area_name in AREAS:
            lat_north, lon_west, lat_south, lon_east = AREAS[area_name]
            before = len(df)
            df = df[(df["lat"] >= lat_south) & (df["lat"] <= lat_north) &
                    (df["lon"] >= lon_west) & (df["lon"] <= lon_east)]
            print(f"  Area filter ({area_name}): {len(df):,} rows (removed {before - len(df):,})")
        else:
            print(f"  Warning: Unknown area '{area_name}', skipping area filter")

    sd = config.get("start_date", "")
    ed = config.get("end_date", "")
    if sd and ed:
        from datetime import datetime
        sd_str = datetime.strptime(sd, "%Y-%m-%d").strftime("%Y%m%d")
        ed_str = datetime.strptime(ed, "%Y-%m-%d").strftime("%Y%m%d")
        df = df[(df["date"].astype(str) >= sd_str) &
                (df["date"].astype(str) <= ed_str)]

    if season:
        months = SEASON_MONTHS[season.upper()]
        df = df[df["date"].astype(str).str[4:6].astype(int).isin(months)]

    if orog and "sdfor" in df.columns:
        lo, hi = OROGRAPHY_RANGES[orog.lower()]
        df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]

    # QC
    df = df.dropna(subset=["obs_value", "fc1_value", "fc2_value"])
    var = config.get("variable", "")
    fcfg = config.get("filter", {})
    if var == "2t":
        lo = fcfg.get("min_valid_temperature", -60.0)
        hi = fcfg.get("max_valid_temperature", 60.0)
        df = df[(df["obs_value"] >= lo) & (df["obs_value"] <= hi)]
    if var == "tp24":
        mx = fcfg.get("max_valid_precipitation")
        if mx is not None:
            df = df[df["obs_value"] <= mx]

    print(f"  After filtering: {len(df):,} rows")
    return df


def _stats_text(obs, fc, label, color):
    """Compute RMSE, corr, bias for a subset."""
    mask = np.isfinite(obs) & np.isfinite(fc)
    o, f = obs[mask], fc[mask]
    if len(o) < 10:
        return ""
    rmse = np.sqrt(np.mean((f - o) ** 2))
    corr = np.corrcoef(f, o)[0, 1]
    bias = np.mean(f - o)
    return f"{label}: RMSE={rmse:.2f}  ρ={corr:.3f}  bias={bias:+.2f}"


def plot_warm(ax, obs, fc1, fc2, fc1_name, fc2_name, var_label, unit):
    """Warm extremes: 90th–99.9th percentile."""
    # Dots at each integer percentile 90–99
    dot_levels = np.arange(90, 100, 1.0)
    obs_dots = np.nanpercentile(obs, dot_levels)
    fc1_dots = np.nanpercentile(fc1, dot_levels)
    fc2_dots = np.nanpercentile(fc2, dot_levels)

    # X markers at 99.0, 99.1, ..., 99.9
    x_levels = np.linspace(99.0, 99.9, 10)
    obs_x = np.nanpercentile(obs, x_levels)
    fc1_x = np.nanpercentile(fc1, x_levels)
    fc2_x = np.nanpercentile(fc2, x_levels)

    all_vals = np.concatenate([obs_dots, obs_x, fc1_dots, fc1_x, fc2_dots, fc2_x])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (vmax - vmin) * 0.06
    lim = (vmin - margin, vmax + margin)

    ax.plot(lim, lim, color="black", ls="--", lw=1.2, zorder=1)

    for fd, fx, color, name in [
        (fc1_dots, fc1_x, COLOR_FC1, fc1_name),
        (fc2_dots, fc2_x, COLOR_FC2, fc2_name),
    ]:
        ax.scatter(obs_dots, fd, color=color, s=25, zorder=3, marker="o", label=name)
        ax.scatter(obs_x, fx, color=color, s=70, zorder=4, marker="x", linewidths=2)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed [{unit}]", fontsize=11)
    ax.set_ylabel(f"Forecast [{unit}]", fontsize=11)
    ax.set_title("Warm extremes (90th–99.9th pct)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    legend = ax.legend(fontsize=10, loc="lower right")
    ax.figure.canvas.draw()
    lb = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(lb.x1, lb.y1 + 0.005,
            "o = 90–99th pct\nx = 99–99.9th (Δ 0.1%)",
            transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85))


def plot_cold(ax, obs, fc1, fc2, fc1_name, fc2_name, var_label, unit):
    """Cold extremes: 0.1th–10th percentile."""
    # Dots at each integer percentile 1–10
    dot_levels = np.arange(1, 11, 1.0)
    obs_dots = np.nanpercentile(obs, dot_levels)
    fc1_dots = np.nanpercentile(fc1, dot_levels)
    fc2_dots = np.nanpercentile(fc2, dot_levels)

    # X markers at 0.1, 0.2, ..., 1.0 (bottom percentile in 10 chunks)
    x_levels = np.linspace(0.1, 1.0, 10)
    obs_x = np.nanpercentile(obs, x_levels)
    fc1_x = np.nanpercentile(fc1, x_levels)
    fc2_x = np.nanpercentile(fc2, x_levels)

    all_vals = np.concatenate([obs_dots, obs_x, fc1_dots, fc1_x, fc2_dots, fc2_x])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (vmax - vmin) * 0.06
    lim = (vmin - margin, vmax + margin)

    ax.plot(lim, lim, color="black", ls="--", lw=1.2, zorder=1)

    for fd, fx, color, name in [
        (fc1_dots, fc1_x, COLOR_FC1, fc1_name),
        (fc2_dots, fc2_x, COLOR_FC2, fc2_name),
    ]:
        ax.scatter(obs_dots, fd, color=color, s=25, zorder=3, marker="o", label=name)
        ax.scatter(obs_x, fx, color=color, s=70, zorder=4, marker="x", linewidths=2)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed [{unit}]", fontsize=11)
    ax.set_ylabel(f"Forecast [{unit}]", fontsize=11)
    ax.set_title("Cold extremes (0.1th–10th pct)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    legend = ax.legend(fontsize=10, loc="lower right")
    ax.figure.canvas.draw()
    lb = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(lb.x1, lb.y1 + 0.005,
            "o = 1–10th pct\nx = 0.1–1.0th (Δ 0.1%)",
            transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85))


def plot_cold_zoom(ax, obs, fc1, fc2, fc1_name, fc2_name, var_label, unit):
    """Cold extremes zoom: connected line from p10 down to p0.1.

    Percentile grid:
      - p10 → p1  : every 1 percentile  (dots)
      - p1  → p0.1: every 0.1 percentile (x markers)
    """
    # Bulk of cold zoom: 1st to 10th, every 1 pct
    dot_levels = np.arange(1.0, 11.0, 1.0)          # 1, 2, …, 10
    # Fine extreme tail: 0.1 to 0.9 (not overlapping p1)
    x_levels   = np.arange(0.1, 1.0,  0.1)          # 0.1, 0.2, …, 0.9

    # Sorted ascending so the line flows left→right
    all_levels = np.sort(np.concatenate([x_levels, dot_levels]))
    obs_q  = np.nanpercentile(obs, all_levels)
    fc1_q  = np.nanpercentile(fc1, all_levels)
    fc2_q  = np.nanpercentile(fc2, all_levels)

    all_vals = np.concatenate([obs_q, fc1_q, fc2_q])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (vmax - vmin) * 0.06
    lim = (vmin - margin, vmax + margin)

    ax.plot(lim, lim, color="black", ls="--", lw=1.2, zorder=1, label="Perfect (y=x)")

    # p1 threshold line (vertical on obs axis)
    p1_val = np.nanpercentile(obs, 1.0)
    ax.axvline(p1_val, color="gray", lw=0.8, ls=":", zorder=2)
    ax.text(p1_val, lim[1], " p1", color="gray", fontsize=8, va="top", ha="left")

    is_dot = np.isin(all_levels, dot_levels)   # p1–p10  → dots
    is_x   = ~is_dot                           # p0.1–p0.9 → x markers

    for fq, color, name in [
        (fc1_q, COLOR_FC1, fc1_name),
        (fc2_q, COLOR_FC2, fc2_name),
    ]:
        # Connected line through all levels
        ax.plot(obs_q, fq, color=color, lw=1.5, zorder=3)
        # Dot markers for p1–p10
        ax.scatter(obs_q[is_dot], fq[is_dot], color=color, s=28,
                   zorder=4, marker="o", label=name)
        # X markers for p0.1–p0.9
        ax.scatter(obs_q[is_x], fq[is_x], color=color, s=60,
                   zorder=5, marker="x", linewidths=2)

    # Annotate a few key percentile levels
    label_levels = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
    for lvl, ov in zip(all_levels, obs_q):
        if lvl in label_levels:
            ax.annotate(
                f"p{lvl:g}",
                xy=(ov, np.nanpercentile(fc1, lvl)),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="gray", zorder=6,
            )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed [{unit}]", fontsize=11)
    ax.set_ylabel(f"Forecast [{unit}]", fontsize=11)
    ax.set_title("Cold extremes zoom (p0.1–p10)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    ax.figure.canvas.draw()
    ax.text(0.02, 0.98,
            "o = p1–p10 (Δ 1%)\nx = p0.1–p0.9 (Δ 0.1%)",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85))


def main():
    parser = argparse.ArgumentParser(description="QQ plots for warm & cold extremes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--day", type=int, default=3, help="Forecast day (default: 3)")
    parser.add_argument("--season", default=None)
    parser.add_argument("--orog", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    _style.apply_style(save_dpi=args.dpi)
    config = load_config(args.config)
    variable = config["variable"]
    var_label, unit = VARIABLE_LABELS.get(variable, (variable, ""))

    print(f"\nLoading day {args.day}...")
    df, fc1_name, fc2_name = load_day(config, args.day)
    df = filter_data(df, config, season=args.season, orog=args.orog)

    obs = df["obs_value"].values
    fc1_vals = df["fc1_value"].values
    fc2_vals = df["fc2_value"].values

    # Build condition string for title
    parts = [f"Day {args.day}"]
    if args.season:
        parts.append(args.season.upper())
    if args.orog:
        parts.append(f"{args.orog} terrain")
    condition = " | ".join(parts)

    # --- Create figure with three panels ---
    fig, (ax_warm, ax_cold, ax_cold_zoom) = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Q-Q Extremes  —  {var_label}  |  {condition}  (N = {len(df):,})",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plot_warm(ax_warm, obs, fc1_vals, fc2_vals, fc1_name, fc2_name, var_label, unit)
    plot_cold(ax_cold, obs, fc1_vals, fc2_vals, fc1_name, fc2_name, var_label, unit)
    plot_cold_zoom(ax_cold_zoom, obs, fc1_vals, fc2_vals, fc1_name, fc2_name, var_label, unit)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    out_dir = Path(args.output_dir or config.get("save", {}).get(
        "output_directory", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fn_parts = [f"qq_extremes_{variable}_{fc1_name}_vs_{fc2_name}_day{args.day}"]
    if args.season:
        fn_parts.append(args.season.upper())
    if args.orog:
        fn_parts.append(args.orog.lower())
    filename = "_".join(fn_parts) + ".png"

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
