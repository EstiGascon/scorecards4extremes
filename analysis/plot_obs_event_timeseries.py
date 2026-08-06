#!/usr/bin/env python3
"""
Daily count of observation-based extreme events — time series plot.

For each config:
  - Load extracted parquet obs values
  - Apply per-station, per-month climatology threshold
  - Count how many stations per day exceed (or fall below) the threshold
  - Plot the daily time series (one panel per config)

Usage:
  python analysis/plot_obs_event_timeseries.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE     = Path(__file__).parent.parent
CLIM_DIR      = WORKSPACE / "obs_clim_local"
PARQUET_BASE  = Path("/perm/moeg/scorecards4extremes/extracted_points")
PLOTS_DIR     = WORKSPACE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config table: (label, param, clim_param, percentile, event_type, parquet_dir)
# ---------------------------------------------------------------------------
CONFIGS = [
    dict(
        label      = "2t — cold extreme (p1)",
        param      = "2t",
        clim_param = "2t",
        percentile = 1,
        event_type = "below",
        parquet_dir= PARQUET_BASE / "2t_local_p1obsclim_ifs_oper_aifs1.0_oper_new",
    ),
    dict(
        label      = "2t — warm extreme (p99)",
        param      = "2t",
        clim_param = "2t",
        percentile = 99,
        event_type = "above",
        parquet_dir= PARQUET_BASE / "2t_local_p99obsclim_ifs_oper_aifs1.0_oper_new",
    ),
    dict(
        label      = "tp24 — heavy precip (p99)",
        param      = "tp24",
        clim_param = "tp",
        percentile = 99,
        event_type = "above",
        parquet_dir= PARQUET_BASE / "tp24_local_p99obsclim_ifs_oper_aifs1.0_oper_new",
    ),
    dict(
        label      = "10ff — wind extreme (p98)",
        param      = "10ff",
        clim_param = "10ff",
        percentile = 98,
        event_type = "above",
        parquet_dir= PARQUET_BASE / "10ff_local_p98obsclim_ifs_oper_aifs1.0_oper_new",
    ),
]

import re as _re
from scipy.spatial import cKDTree

MISSING_VAL = 3e+38


# ---------------------------------------------------------------------------
# Parse GEO NCOLS climatology file → DataFrame with lat/lon + quantile columns
# ---------------------------------------------------------------------------
def _parse_clim_file(fpath: Path) -> pd.DataFrame:
    rows, header, in_data = [], None, False
    with open(fpath) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#DATA"):
                in_data = True
                continue
            if not in_data:
                if not line.startswith("#") and header is None and "stnid" in line:
                    header = line.split()
                continue
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.split())
    if header is None or not rows:
        raise ValueError(f"Could not parse: {fpath}")
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    df["latitude"]  = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    quant_cols = [c for c in df.columns
                  if c.startswith("q") and _re.match(r"^\d+$", c[1:])]
    df[quant_cols] = df[quant_cols].astype(float).replace(3e38, np.nan)
    for col in quant_cols:
        df.loc[df[col] > 1e37, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Pre-load all 12 months of climatology (DataFrame + KDTree) for a param/pct
# ---------------------------------------------------------------------------
def load_all_clim(clim_param: str, percentile: int) -> dict:
    """Returns dict {month: (df_clim, kd_tree, quant_col)}"""
    cache = {}
    for m in range(1, 13):
        fname = CLIM_DIR / f"clim_{clim_param}_1_{m:02d}_20years_2005_2024_65"
        if not fname.exists():
            continue
        df = _parse_clim_file(fname)
        quant_col = f"q{percentile}"
        if quant_col not in df.columns:
            continue
        tree = cKDTree(np.column_stack([df["latitude"].values,
                                        df["longitude"].values]))
        cache[m] = (df, tree, quant_col)
    return cache


# ---------------------------------------------------------------------------
# Load day1 parquet, apply lat/lon KDTree threshold matching, count events
# ---------------------------------------------------------------------------
def daily_counts_from_day1(parquet_dir: Path, all_clim: dict,
                            event_type: str) -> pd.DataFrame:
    # Prefer non-batched day1 file; fall back to first batched day1 file
    candidates = [f for f in sorted(parquet_dir.glob("*day1*.parquet"))
                  if "batch" not in f.name]
    if not candidates:
        candidates = sorted(parquet_dir.glob("*day1*.parquet"))
    if not candidates:
        candidates = sorted(parquet_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    f = candidates[0]
    print(f"    reading {f.name} ...")

    df = pd.read_parquet(f, columns=["valid_time", "date", "step",
                                     "lat", "lon", "obs_value"])
    df["valid_time"] = df["valid_time"].astype(str)

    # Parse month from valid time (init date + step), matching threshold.py logic
    init_dates  = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    valid_dates = init_dates + pd.to_timedelta(df["step"].astype(int), unit="h")
    df["valid_date"] = valid_dates.dt.date
    df["month"]      = valid_dates.dt.month

    # Aggregate to daily mean per station before comparing against daily-mean
    # climatology threshold (matching the pipeline's Step 5b daily aggregation).
    # Use lat/lon as station proxy (consistent across steps within a day).
    daily_mean = (df.groupby(["valid_date", "lat", "lon", "month"])
                    .agg(obs_value=("obs_value", "mean"))
                    .reset_index())

    # Apply per-station monthly threshold via nearest-neighbour KDTree
    max_dist = 0.1   # degrees, same default as threshold.py
    daily_mean["threshold"] = np.nan

    for month, (df_clim, tree, quant_col) in all_clim.items():
        mask = (daily_mean["month"] == month).values
        if not mask.any():
            continue
        lats = daily_mean.loc[mask, "lat"].values
        lons = daily_mean.loc[mask, "lon"].values
        dists, nn_idxs = tree.query(np.column_stack([lats, lons]))
        thresholds = np.where(
            dists <= max_dist,
            df_clim[quant_col].values[nn_idxs],
            np.nan,
        )
        daily_mean.loc[mask, "threshold"] = thresholds

    df = daily_mean[daily_mean["threshold"].notna() & daily_mean["obs_value"].notna()].copy()

    if event_type == "above":
        df["is_event"] = (df["obs_value"] > df["threshold"]).astype(int)
    else:
        df["is_event"] = (df["obs_value"] < df["threshold"]).astype(int)

    result = (df.groupby("valid_date")
                .agg(n_events=("is_event", "sum"), n_total=("is_event", "count"))
                .reset_index()
                .rename(columns={"valid_date": "date"}))
    result["date"]       = pd.to_datetime(result["date"])
    result["event_rate"] = result["n_events"] / result["n_total"]
    return result.sort_values("date")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n = len(CONFIGS)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Daily count of extreme events (observation-based climatology threshold)\n"
                 "Europe stations — IFS oper / AIFS 1.0 oper evaluation period",
                 fontsize=13, fontweight="bold")

    for ax, cfg in zip(axes, CONFIGS):
        label      = cfg["label"]
        clim_param = cfg["clim_param"]
        percentile = cfg["percentile"]
        event_type = cfg["event_type"]
        parquet_dir= cfg["parquet_dir"]

        print(f"\nProcessing: {label}")

        if not parquet_dir.exists():
            ax.text(0.5, 0.5, f"No data found:\n{parquet_dir.name}",
                    ha="center", va="center", transform=ax.transAxes, color="red")
            ax.set_title(label, fontweight="bold")
            continue

        print(f"  Loading climatology ({clim_param}, p{percentile}) ...")
        all_clim = load_all_clim(clim_param, percentile)

        print(f"  Processing day1 obs from {parquet_dir.name} ...")
        daily = daily_counts_from_day1(parquet_dir, all_clim, event_type)
        print(f"  Date range: {daily['date'].min().date()} – {daily['date'].max().date()}")
        print(f"  Mean daily events: {daily['n_events'].mean():.0f} / {daily['n_total'].mean():.0f} stations")

        # ---- plot ----
        color_bar  = "#4a90d9"
        color_rate = "#d9534f"

        ax2 = ax.twinx()

        ax.bar(daily["date"], daily["n_events"], width=0.8, color=color_bar,
               alpha=0.7, label="N events (left)")
        ax2.plot(daily["date"], daily["event_rate"] * 100, color=color_rate,
                 linewidth=1.2, alpha=0.85, label="Event rate % (right)")

        # Expected climatological rate line
        expected_pct = (100 - percentile) if event_type == "above" else percentile
        ax2.axhline(expected_pct, color=color_rate, linewidth=0.8,
                    linestyle="--", alpha=0.5,
                    label=f"Climatological mean rate ({expected_pct}%)\n(higher on synoptic-scale events)")

        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_ylabel("N stations exceeding threshold", color=color_bar)
        ax2.set_ylabel("Event rate (%)", color=color_rate)
        ax.tick_params(axis="y", labelcolor=color_bar)
        ax2.tick_params(axis="y", labelcolor=color_rate)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(daily["date"].min(), daily["date"].max())

        # Combined legend
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)

    outfile = PLOTS_DIR / "obs_event_timeseries_aifs_ifs.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {outfile}")


if __name__ == "__main__":
    main()
