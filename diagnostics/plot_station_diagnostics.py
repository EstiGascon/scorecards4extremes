#!/usr/bin/env python3
"""
Station-level diagnostic maps for extracted points.
Works for both ensemble and deterministic forecasts — mode is auto-detected.

Standalone script — run directly, not via run.py / submit_job.sh.

Usage:
    python3 plot_station_diagnostics.py <extracted_points_dir> [options]

Examples:
    # Ensemble
    python3 plot_station_diagnostics.py ./extracted_points/2t_ens \
        --threshold 30 --event-type above --variable 2t \
        --output-dir ./plots/station_diagnostics/2t_warm \
        --forecast-day 1 --model1-name ifs_ens --model2-name aifs_ens

    # Deterministic
    python3 plot_station_diagnostics.py ./extracted_points/2t \
        --threshold -5 --event-type below --variable 2t \
        --output-dir ./plots/station_diagnostics/2t_cold \
        --forecast-day 3 --model1-name ifs_oper --model2-name iekm
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def _find_parquet_files(data_dir, forecast_day=None):
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        print(f"ERROR: No parquet files found in {data_dir}")
        sys.exit(1)
    if forecast_day is not None:
        files = [f for f in files if f"day{forecast_day}." in f or f"day{forecast_day}_" in f]
        if not files:
            print(f"ERROR: No files found for forecast day {forecast_day}")
            sys.exit(1)
    return files


def peek_columns(data_dir, forecast_day=None):
    """Return column names by reading only parquet metadata (no data loaded)."""
    files = _find_parquet_files(data_dir, forecast_day)
    return pq.read_schema(files[0]).names


def detect_mode(columns):
    """Return 'ensemble' or 'deterministic' based on column names."""
    if any(c.startswith("fc1_member_") for c in columns):
        return "ensemble"
    if "fc1_value" in columns:
        return "deterministic"
    raise ValueError("Cannot detect forecast mode: expected fc1_member_* or fc1_value columns")


def load_data(data_dir, forecast_day=None, columns=None):
    """Load parquet files, optionally selecting only specific columns."""
    files = _find_parquet_files(data_dir, forecast_day)
    dfs = [pd.read_parquet(f, columns=columns) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    col_desc = f"{len(columns)} columns" if columns else "all columns"
    print(f"Loaded {len(df):,} rows from {len(files)} files ({col_desc})")
    return df


def get_member_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    return cols


def _add_extreme_flags(df, threshold, event_type):
    if event_type == "above":
        df["obs_extreme"] = (df["obs_value"] >= threshold).astype(int)
    else:
        df["obs_extreme"] = (df["obs_value"] <= threshold).astype(int)
    return df


# Rounding precision for lat/lon grouping key (~100 m, fine enough to keep distinct
# nearby stations but robust against floating-point jitter across dates)
_LAT_LON_ROUND = 3


def _add_location_key(df):
    """Add a stable station location key from rounded lat/lon.

    station_id in the parquet files is a positional index (S0, S1 ...) assigned
    per-date, NOT a persistent station identifier, so the same index can refer to
    different physical stations on different dates.  Rounding lat/lon to 3 decimal
    places (~100 m) gives a reliable grouping key.
    """
    df["_loc"] = (
        df["lat"].round(_LAT_LON_ROUND).astype(str)
        + "_"
        + df["lon"].round(_LAT_LON_ROUND).astype(str)
    )
    return df


def compute_station_stats_ens(df, threshold, event_type, fc1_cols, fc2_cols):
    """Per-station statistics for ensemble forecasts."""
    df = df.dropna(subset=["obs_value"]).copy()
    df = _add_extreme_flags(df, threshold, event_type)
    df = _add_location_key(df)
    extreme_mask = df["obs_extreme"] == 1

    df["fc1_mean"] = df[fc1_cols].mean(axis=1)
    df["fc2_mean"] = df[fc2_cols].mean(axis=1)
    df["fc1_spread"] = df[fc1_cols].std(axis=1)
    df["fc2_spread"] = df[fc2_cols].std(axis=1)

    if event_type == "above":
        df["fc1_prob"] = df[fc1_cols].ge(threshold, axis=0).mean(axis=1)
        df["fc2_prob"] = df[fc2_cols].ge(threshold, axis=0).mean(axis=1)
    else:
        df["fc1_prob"] = df[fc1_cols].le(threshold, axis=0).mean(axis=1)
        df["fc2_prob"] = df[fc2_cols].le(threshold, axis=0).mean(axis=1)

    df["fc1_bias"] = df["fc1_mean"] - df["obs_value"]
    df["fc2_bias"] = df["fc2_mean"] - df["obs_value"]
    df["fc1_bias_extreme"] = np.where(extreme_mask, df["fc1_bias"], np.nan)
    df["fc2_bias_extreme"] = np.where(extreme_mask, df["fc2_bias"], np.nan)

    stats = df.groupby("_loc").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        n_cases=("obs_value", "count"),
        n_extreme_obs=("obs_extreme", "sum"),
        obs_mean=("obs_value", "mean"),
        obs_std=("obs_value", "std"),
        fc1_mean_bias=("fc1_bias", "mean"),
        fc2_mean_bias=("fc2_bias", "mean"),
        fc1_mean_spread=("fc1_spread", "mean"),
        fc2_mean_spread=("fc2_spread", "mean"),
        fc1_mean_prob=("fc1_prob", "mean"),
        fc2_mean_prob=("fc2_prob", "mean"),
        fc1_bias_extreme=("fc1_bias_extreme", "mean"),
        fc2_bias_extreme=("fc2_bias_extreme", "mean"),
    ).reset_index(drop=True)

    stats["extreme_frac"] = stats["n_extreme_obs"] / stats["n_cases"]
    stats["fc1_prob_bias"] = stats["fc1_mean_prob"] - stats["extreme_frac"]
    stats["fc2_prob_bias"] = stats["fc2_mean_prob"] - stats["extreme_frac"]
    stats["bias_diff"] = stats["fc2_mean_bias"] - stats["fc1_mean_bias"]
    stats["spread_diff"] = stats["fc2_mean_spread"] - stats["fc1_mean_spread"]
    return stats


_DET_COLS = ["lat", "lon", "obs_value", "fc1_value", "fc2_value"]


def _process_det_batch(chunk, threshold, event_type, area=None):
    """Compute per-station intermediate sums for one batch. Returns None if empty."""
    chunk = chunk.dropna(subset=["obs_value"])
    if area is not None:
        lat_n, lon_w, lat_s, lon_e = area
        chunk = chunk[(chunk["lat"] >= lat_s) & (chunk["lat"] <= lat_n) &
                      (chunk["lon"] >= lon_w) & (chunk["lon"] <= lon_e)]
    if len(chunk) == 0:
        return None
    chunk = _add_location_key(chunk)

    if event_type == "above":
        obs_x = chunk["obs_value"] >= threshold
        fc1_x = (chunk["fc1_value"] >= threshold).astype(float)
        fc2_x = (chunk["fc2_value"] >= threshold).astype(float)
    else:
        obs_x = chunk["obs_value"] <= threshold
        fc1_x = (chunk["fc1_value"] <= threshold).astype(float)
        fc2_x = (chunk["fc2_value"] <= threshold).astype(float)

    non_x = ~obs_x
    fc1_bias = chunk["fc1_value"] - chunk["obs_value"]
    fc2_bias = chunk["fc2_value"] - chunk["obs_value"]

    chunk = chunk.assign(
        n_x=obs_x.astype(int),
        obs_sq=chunk["obs_value"] ** 2,
        f1bs=fc1_bias,
        f2bs=fc2_bias,
        f1as=fc1_bias.abs(),
        f2as=fc2_bias.abs(),
        f1bxs=np.where(obs_x, fc1_bias, np.nan),
        f2bxs=np.where(obs_x, fc2_bias, np.nan),
        f1axs=np.where(obs_x, fc1_bias.abs(), np.nan),
        f2axs=np.where(obs_x, fc2_bias.abs(), np.nan),
        f1hs=np.where(obs_x, fc1_x, np.nan),
        f2hs=np.where(obs_x, fc2_x, np.nan),
        f1fs=np.where(non_x, fc1_x, np.nan),
        f2fs=np.where(non_x, fc2_x, np.nan),
    )

    return chunk.groupby("_loc").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        n=("obs_value", "count"),
        n_x=("n_x", "sum"),
        obs_s=("obs_value", "sum"),
        obs_sq=("obs_sq", "sum"),
        f1bs=("f1bs", "sum"),  f2bs=("f2bs", "sum"),
        f1as=("f1as", "sum"),  f2as=("f2as", "sum"),
        f1bxs=("f1bxs", "sum"), f2bxs=("f2bxs", "sum"),
        f1axs=("f1axs", "sum"), f2axs=("f2axs", "sum"),
        f1bxn=("f1bxs", "count"), f2bxn=("f2bxs", "count"),
        f1hs=("f1hs", "sum"),  f2hs=("f2hs", "sum"),
        f1hn=("f1hs", "count"), f2hn=("f2hs", "count"),
        f1fs=("f1fs", "sum"),  f2fs=("f2fs", "sum"),
        f1fn=("f1fs", "count"), f2fn=("f2fs", "count"),
    ).reset_index()


def compute_station_stats_det(files, threshold, event_type, batch_size=500_000, area=None):
    """Streaming per-station statistics for deterministic forecasts (memory-efficient).

    Processes data in batches to avoid OOM on large files (tens of millions of rows).
    """
    partials = []
    total_rows = 0
    for fpath in files:
        pfile = pq.ParquetFile(fpath)
        for batch in pfile.iter_batches(batch_size=batch_size, columns=_DET_COLS):
            chunk = batch.to_pandas()
            total_rows += len(chunk)
            partial = _process_det_batch(chunk, threshold, event_type, area=area)
            if partial is not None:
                partials.append(partial)

    print(f"  Processed {total_rows:,} rows in {len(partials)} batches")

    # Reduce: sum all accumulated counters per station
    all_p = pd.concat(partials, ignore_index=True)
    sum_cols = [c for c in all_p.columns if c not in ("_loc", "lat", "lon")]
    agg = {c: "sum" for c in sum_cols}
    agg["lat"] = "mean"
    agg["lon"] = "mean"
    g = all_p.groupby("_loc").agg(agg).reset_index(drop=True)

    # Derive final metrics from accumulated sums
    g["extreme_frac"] = g["n_x"] / g["n"]
    g["n_extreme_obs"] = g["n_x"]
    g["obs_mean"] = g["obs_s"] / g["n"]
    # Variance via E[X^2] - E[X]^2 (biased, fine for diagnostics)
    g["obs_std"] = np.sqrt(np.maximum((g["obs_sq"] / g["n"]) - g["obs_mean"] ** 2, 0))
    g["fc1_mean_bias"] = g["f1bs"] / g["n"]
    g["fc2_mean_bias"] = g["f2bs"] / g["n"]
    g["fc1_mae"] = g["f1as"] / g["n"]
    g["fc2_mae"] = g["f2as"] / g["n"]
    g["fc1_bias_extreme"] = np.where(g["f1bxn"] > 0, g["f1bxs"] / g["f1bxn"], np.nan)
    g["fc2_bias_extreme"] = np.where(g["f2bxn"] > 0, g["f2bxs"] / g["f2bxn"], np.nan)
    g["fc1_mae_extreme"] = np.where(g["f1bxn"] > 0, g["f1axs"] / g["f1bxn"], np.nan)
    g["fc2_mae_extreme"] = np.where(g["f2bxn"] > 0, g["f2axs"] / g["f2bxn"], np.nan)
    g["fc1_hit_rate"] = np.where(g["f1hn"] > 0, g["f1hs"] / g["f1hn"], np.nan)
    g["fc2_hit_rate"] = np.where(g["f2hn"] > 0, g["f2hs"] / g["f2hn"], np.nan)
    g["fc1_false_alarm_rate"] = np.where(g["f1fn"] > 0, g["f1fs"] / g["f1fn"], np.nan)
    g["fc2_false_alarm_rate"] = np.where(g["f2fn"] > 0, g["f2fs"] / g["f2fn"], np.nan)
    g["bias_diff"] = g["fc2_mean_bias"] - g["fc1_mean_bias"]
    g["mae_diff"] = g["fc2_mae"] - g["fc1_mae"]
    g["hit_rate_diff"] = g["fc2_hit_rate"] - g["fc1_hit_rate"]
    return g


def _shared_sym_lim(stats, *cols, pct=95):
    """Symmetric vmax computed jointly across multiple columns (for comparable paired maps)."""
    vals = np.concatenate([stats[c].dropna().values for c in cols])
    vmax = np.percentile(np.abs(vals), pct) if len(vals) else 1.0
    return -vmax, vmax


def _shared_lim(stats, *cols, lo_pct=2, hi_pct=98, vmin_floor=None):
    """Shared vmin/vmax computed jointly across multiple columns."""
    vals = np.concatenate([stats[c].dropna().values for c in cols])
    vmin = np.percentile(vals, lo_pct) if len(vals) else 0.0
    vmax = np.percentile(vals, hi_pct) if len(vals) else 1.0
    if vmin_floor is not None:
        vmin = max(vmin, vmin_floor)
    if vmin == vmax:
        vmax = vmin + 1
    return vmin, vmax


def _get_extent(lats, lons, margin=2):
    return [lons.min() - margin, lons.max() + margin,
            lats.min() - margin, lats.max() + margin]


def plot_map(stats, col, title, output_path, cmap="RdBu_r", vmin=None, vmax=None,
             symmetric=False, units="", marker_size=8):
    """Plot a single station-level map."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    extent = _get_extent(stats["lat"], stats["lon"])
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff", alpha=0.3)

    vals = stats[col].values
    valid = ~np.isnan(vals)

    if symmetric and vmin is None:
        vmax_abs = np.nanpercentile(np.abs(vals[valid]), 95)
        vmin, vmax = -vmax_abs, vmax_abs

    if vmin is None:
        vmin = np.nanpercentile(vals[valid], 2)
    if vmax is None:
        vmax = np.nanpercentile(vals[valid], 98)
    if vmin == vmax:
        vmax = vmin + 1

    sc = ax.scatter(
        stats["lon"].values[valid], stats["lat"].values[valid],
        c=vals[valid], cmap=cmap, s=marker_size,
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        edgecolors="none", alpha=0.8,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    if units:
        cbar.set_label(units, fontsize=10)

    ax.set_title(title, fontsize=13, weight="bold")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Station-level diagnostic maps")
    parser.add_argument("data_dir", help="Path to extracted_points directory (e.g. ./extracted_points/2t_ens)")
    parser.add_argument("--threshold", type=float, default=None, help="Extreme event threshold")
    parser.add_argument("--event-type", choices=["above", "below"], default="above")
    parser.add_argument("--variable", default="2t", help="Variable name for titles")
    parser.add_argument("--output-dir", default="./plots/station_diagnostics")
    parser.add_argument("--forecast-day", type=int, default=None, help="Forecast day to plot (default: all)")
    parser.add_argument("--model1-name", default="model1")
    parser.add_argument("--model2-name", default="model2")
    parser.add_argument("--marker-size", type=float, default=8)
    parser.add_argument(
        "--area", nargs=4, type=float, metavar=("LAT_N", "LON_W", "LAT_S", "LON_E"),
        default=None,
        help="Bounding box to filter stations, e.g. --area 75 -25 25 50 for Europe",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    units = {"2t": "°C", "10ff": "m/s", "tp24": "mm"}.get(args.variable, "")

    day_label = f"day{args.forecast_day}" if args.forecast_day else "all_days"
    m1, m2 = args.model1_name, args.model2_name

    # Peek at schema to detect mode before loading data
    all_columns = peek_columns(args.data_dir, forecast_day=args.forecast_day)
    mode = detect_mode(all_columns)
    print(f"Mode: {mode}")

    ge_le = '≥' if args.event_type == 'above' else '≤'
    ms = args.marker_size

    if mode == "ensemble":
        # Ensemble files are small enough to load fully
        member_cols = [c for c in all_columns if c.startswith(("fc1_member_", "fc2_member_"))]
        load_cols = ["lat", "lon", "obs_value"] + member_cols
        df = load_data(args.data_dir, forecast_day=args.forecast_day, columns=load_cols)
        if args.area is not None:
            lat_n, lon_w, lat_s, lon_e = args.area
            df = df[(df["lat"] >= lat_s) & (df["lat"] <= lat_n) &
                    (df["lon"] >= lon_w) & (df["lon"] <= lon_e)].copy()
            print(f"Area filter applied: {len(df):,} rows remaining")

        if args.threshold is None:
            pct = 99 if args.event_type == "above" else 1
            args.threshold = float(np.nanpercentile(df["obs_value"].dropna(), pct))
            print(f"Auto threshold ({args.event_type}): {args.threshold:.2f} {units}")

        thr_str = f"{args.threshold:.1f}{units}"
        fc1_cols = get_member_cols(df, "fc1_member_")
        fc2_cols = get_member_cols(df, "fc2_member_")
        print(f"Computing per-station statistics ({len(fc1_cols)} members)...")
        stats = compute_station_stats_ens(df, args.threshold, args.event_type, fc1_cols, fc2_cols)
        print(f"  {len(stats)} stations")

        # Pre-compute shared limits for comparable paired maps
        bias_vmin, bias_vmax = _shared_sym_lim(stats, "fc1_mean_bias", "fc2_mean_bias")
        bx_vmin, bx_vmax = _shared_sym_lim(stats, "fc1_bias_extreme", "fc2_bias_extreme")
        sp_vmin, sp_vmax = _shared_lim(stats, "fc1_mean_spread", "fc2_mean_spread", vmin_floor=0)
        prob_vmin, prob_vmax = _shared_lim(stats, "fc1_mean_prob", "fc2_mean_prob", vmin_floor=0)
        pb_vmin, pb_vmax = _shared_sym_lim(stats, "fc1_prob_bias", "fc2_prob_bias")

        plot_map(stats, "n_extreme_obs",
                 f"Number of observed extreme events (obs {ge_le} {thr_str})\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_n_extreme_obs_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=0, units="count", marker_size=ms)
        plot_map(stats, "extreme_frac",
                 f"Fraction of cases that are extreme\n{args.variable} | {day_label} | threshold={thr_str}",
                 os.path.join(args.output_dir, f"map_extreme_frac_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=0, vmax=min(stats["extreme_frac"].max() * 1.1, 1.0),
                 units="fraction", marker_size=ms)
        plot_map(stats, "obs_mean",
                 f"Mean observed {args.variable}\n{day_label}",
                 os.path.join(args.output_dir, f"map_obs_mean_{args.variable}_{day_label}.png"),
                 cmap="coolwarm", units=units, marker_size=ms)
        plot_map(stats, "fc1_mean_bias",
                 f"Ensemble mean bias — {m1}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_bias_{m1}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bias_vmin, vmax=bias_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc2_mean_bias",
                 f"Ensemble mean bias — {m2}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_bias_{m2}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bias_vmin, vmax=bias_vmax, units=units, marker_size=ms)
        plot_map(stats, "bias_diff",
                 f"Bias difference ({m2} − {m1})\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_bias_diff_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", symmetric=True, units=units, marker_size=ms)
        plot_map(stats, "fc1_bias_extreme",
                 f"Ens mean bias on extremes only — {m1}\n{args.variable} | {day_label} | obs {ge_le} {thr_str}",
                 os.path.join(args.output_dir, f"map_bias_extreme_{m1}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bx_vmin, vmax=bx_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc2_bias_extreme",
                 f"Ens mean bias on extremes only — {m2}\n{args.variable} | {day_label} | obs {ge_le} {thr_str}",
                 os.path.join(args.output_dir, f"map_bias_extreme_{m2}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bx_vmin, vmax=bx_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc1_mean_spread",
                 f"Mean ensemble spread — {m1}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_spread_{m1}_{args.variable}_{day_label}.png"),
                 cmap="viridis", vmin=sp_vmin, vmax=sp_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc2_mean_spread",
                 f"Mean ensemble spread — {m2}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_spread_{m2}_{args.variable}_{day_label}.png"),
                 cmap="viridis", vmin=sp_vmin, vmax=sp_vmax, units=units, marker_size=ms)
        plot_map(stats, "spread_diff",
                 f"Spread difference ({m2} − {m1})\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_spread_diff_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", symmetric=True, units=units, marker_size=ms)
        plot_map(stats, "fc1_mean_prob",
                 f"Mean P(extreme) from ensemble — {m1}\n{args.variable} | {day_label} | threshold={thr_str}",
                 os.path.join(args.output_dir, f"map_prob_{m1}_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=prob_vmin, vmax=prob_vmax, units="probability", marker_size=ms)
        plot_map(stats, "fc2_mean_prob",
                 f"Mean P(extreme) from ensemble — {m2}\n{args.variable} | {day_label} | threshold={thr_str}",
                 os.path.join(args.output_dir, f"map_prob_{m2}_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=prob_vmin, vmax=prob_vmax, units="probability", marker_size=ms)
        plot_map(stats, "fc1_prob_bias",
                 f"Probability bias — {m1}\nP(forecast extreme) − obs frequency | {args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_prob_bias_{m1}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=pb_vmin, vmax=pb_vmax, units="prob difference", marker_size=ms)
        plot_map(stats, "fc2_prob_bias",
                 f"Probability bias — {m2}\nP(forecast extreme) − obs frequency | {args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_prob_bias_{m2}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=pb_vmin, vmax=pb_vmax, units="prob difference", marker_size=ms)

    else:  # deterministic — use chunked streaming (files can be tens of millions of rows)
        files = _find_parquet_files(args.data_dir, forecast_day=args.forecast_day)

        if args.threshold is None:
            # Load only lat/lon/obs_value to compute percentile on area-filtered subset
            area_cols = ["lat", "lon", "obs_value"] if args.area else ["obs_value"]
            obs_s = pd.concat(
                [pd.read_parquet(f, columns=area_cols) for f in files],
                ignore_index=True,
            )
            if args.area is not None:
                lat_n, lon_w, lat_s, lon_e = args.area
                obs_s = obs_s[(obs_s["lat"] >= lat_s) & (obs_s["lat"] <= lat_n) &
                              (obs_s["lon"] >= lon_w) & (obs_s["lon"] <= lon_e)]
            obs_s = obs_s["obs_value"].dropna()
            pct = 99 if args.event_type == "above" else 1
            args.threshold = float(np.percentile(obs_s, pct))
            print(f"Auto threshold ({args.event_type}): {args.threshold:.2f} {units}")
            del obs_s

        thr_str = f"{args.threshold:.1f}{units}"
        print(f"Computing per-station statistics (chunked streaming, {len(files)} files)...")
        stats = compute_station_stats_det(files, args.threshold, args.event_type, area=args.area)
        print(f"  {len(stats)} stations")

        # Pre-compute shared limits for comparable paired maps
        bias_vmin, bias_vmax = _shared_sym_lim(stats, "fc1_mean_bias", "fc2_mean_bias")
        bx_vmin, bx_vmax = _shared_sym_lim(stats, "fc1_bias_extreme", "fc2_bias_extreme")
        mae_vmin, mae_vmax = _shared_lim(stats, "fc1_mae", "fc2_mae", vmin_floor=0)
        maex_vmin, maex_vmax = _shared_lim(stats, "fc1_mae_extreme", "fc2_mae_extreme", vmin_floor=0)

        plot_map(stats, "n_extreme_obs",
                 f"Number of observed extreme events (obs {ge_le} {thr_str})\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_n_extreme_obs_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=0, units="count", marker_size=ms)
        plot_map(stats, "extreme_frac",
                 f"Fraction of cases that are extreme\n{args.variable} | {day_label} | threshold={thr_str}",
                 os.path.join(args.output_dir, f"map_extreme_frac_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=0, vmax=min(stats["extreme_frac"].max() * 1.1, 1.0),
                 units="fraction", marker_size=ms)
        plot_map(stats, "obs_mean",
                 f"Mean observed {args.variable}\n{day_label}",
                 os.path.join(args.output_dir, f"map_obs_mean_{args.variable}_{day_label}.png"),
                 cmap="coolwarm", units=units, marker_size=ms)
        plot_map(stats, "fc1_mean_bias",
                 f"Mean bias — {m1}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_bias_{m1}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bias_vmin, vmax=bias_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc2_mean_bias",
                 f"Mean bias — {m2}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_bias_{m2}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bias_vmin, vmax=bias_vmax, units=units, marker_size=ms)
        plot_map(stats, "bias_diff",
                 f"Bias difference ({m2} − {m1})\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_bias_diff_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", symmetric=True, units=units, marker_size=ms)
        plot_map(stats, "fc1_bias_extreme",
                 f"Bias on extremes only — {m1}\n{args.variable} | {day_label} | obs {ge_le} {thr_str}",
                 os.path.join(args.output_dir, f"map_bias_extreme_{m1}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bx_vmin, vmax=bx_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc2_bias_extreme",
                 f"Bias on extremes only — {m2}\n{args.variable} | {day_label} | obs {ge_le} {thr_str}",
                 os.path.join(args.output_dir, f"map_bias_extreme_{m2}_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", vmin=bx_vmin, vmax=bx_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc1_mae",
                 f"Mean Absolute Error — {m1}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_mae_{m1}_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=mae_vmin, vmax=mae_vmax, units=units, marker_size=ms)
        plot_map(stats, "fc2_mae",
                 f"Mean Absolute Error — {m2}\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_mae_{m2}_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=mae_vmin, vmax=mae_vmax, units=units, marker_size=ms)
        plot_map(stats, "mae_diff",
                 f"MAE difference ({m2} − {m1})\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_mae_diff_{args.variable}_{day_label}.png"),
                 cmap="RdBu_r", symmetric=True, units=units, marker_size=ms)
        plot_map(stats, "fc1_hit_rate",
                 f"Hit rate — {m1}\nP(fc {ge_le} {thr_str} | obs {ge_le} {thr_str}) | {args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_hit_rate_{m1}_{args.variable}_{day_label}.png"),
                 cmap="RdYlGn", vmin=0, vmax=1, units="fraction", marker_size=ms)
        plot_map(stats, "fc2_hit_rate",
                 f"Hit rate — {m2}\nP(fc {ge_le} {thr_str} | obs {ge_le} {thr_str}) | {args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_hit_rate_{m2}_{args.variable}_{day_label}.png"),
                 cmap="RdYlGn", vmin=0, vmax=1, units="fraction", marker_size=ms)
        plot_map(stats, "hit_rate_diff",
                 f"Hit rate difference ({m2} − {m1})\n{args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_hit_rate_diff_{args.variable}_{day_label}.png"),
                 cmap="RdBu", symmetric=True, units="fraction", marker_size=ms)
        plot_map(stats, "fc1_false_alarm_rate",
                 f"False alarm rate — {m1}\nP(fc {ge_le} {thr_str} | obs not extreme) | {args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_far_{m1}_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=0, vmax=1, units="fraction", marker_size=ms)
        plot_map(stats, "fc2_false_alarm_rate",
                 f"False alarm rate — {m2}\nP(fc {ge_le} {thr_str} | obs not extreme) | {args.variable} | {day_label}",
                 os.path.join(args.output_dir, f"map_far_{m2}_{args.variable}_{day_label}.png"),
                 cmap="YlOrRd", vmin=0, vmax=1, units="fraction", marker_size=ms)

    print(f"\n✓ All maps saved to {args.output_dir}")


if __name__ == "__main__":
    main()
