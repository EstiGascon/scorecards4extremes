"""
diagnose_twcrps_simple.py
─────────────────────────────────────────────────────────────────────────────
Intuitive diagnostic: when a warm extreme occurs, what do IFS-ENS and AIFS-ENS
actually forecast, and how does that explain the twCRPS difference?

Three simple rows:
  Row 1 – "When obs exceeded T, where were the ensemble members?"
           KDE distributions of obs (green), IFS median (red), AIFS median (blue)
           normalised as (temperature − threshold T).
           Intuition: if red/blue peaks are LEFT of green → model is too cold.

  Row 2 – "How biased was each model during extreme events?"
           Box plots of (ensemble_median − obs) per extreme event.
           0 = perfect; negative = model too cold; positive = model too warm.

  Row 3 – "Which model scored better? (twCRPS)"
           Bar chart, lower = better, annotated with % difference.

Usage:
  python diagnose_twcrps_simple.py \\
      --config config_2t_ens_local_p99obsclim_aifsvsifs.yaml \\
      --orog low --day 5
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))
from case_studies.case_study_utils import load_per_station_thresholds

# ── Constants ─────────────────────────────────────────────────────────────────
SEASON_MONTHS = {"DJF": {12, 1, 2}, "MAM": {3, 4, 5},
                 "JJA": {6, 7, 8},  "SON": {9, 10, 11}}
C1   = "#d7191c"   # IFS-ENS (red)
C2   = "#2c7bb6"   # AIFS-ENS (blue)
COBS = "#1a9641"   # Observations (green)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _month(d):
    return int(str(int(d))[4:6])


def load_season_data(parquet_path, season, orog_range, config,
                     max_samples=300_000, seed=42):
    """Load parquet, filter by season + orography, aggregate to daily means,
    return (df, n_raw).

    The pipeline aggregates 6-hourly ensemble data to daily means before scoring.
    We replicate that here so that extreme events and scores match the heatmap.
    """
    months = SEASON_MONTHS[season]
    lo, hi = orog_range
    pf = pq.ParquetFile(str(parquet_path))
    chunks, n_raw = [], 0
    for batch in pf.iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        n_raw += len(chunk)
        chunk = chunk[chunk["date"].apply(_month).isin(months)]
        if "sdfor" in chunk.columns:
            chunk = chunk[(chunk["sdfor"] >= lo) & (chunk["sdfor"] < hi)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return None, 0
    df = pd.concat(chunks, ignore_index=True)
    # Remove physically impossible member values
    fc_cols = [c for c in df.columns if c.startswith("fc")]
    df = df[~(df[fc_cols] < -200).any(axis=1)].reset_index(drop=True)

    # ── Aggregate 6-hourly → daily means ─────────────────────────────────────
    # Group by (date, station_id); take mean of obs and each member column.
    # Static fields (lat, lon, sdfor, lsm, forecast_day) are constant per group.
    member_cols  = [c for c in df.columns if c.startswith("fc1_member_") or c.startswith("fc2_member_")]
    static_cols  = ["lat", "lon", "sdfor", "lsm", "forecast_day"]
    static_cols  = [c for c in static_cols if c in df.columns]
    agg_dict     = {c: "mean" for c in ["obs_value"] + member_cols}
    agg_dict.update({c: "first" for c in static_cols})
    df = df.groupby(["date", "station_id"], as_index=False).agg(agg_dict)

    if max_samples and len(df) > max_samples:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(df), size=max_samples, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
    return df, n_raw


def twcrps_score(ens_np, obs, T_arr, event_type="above"):
    """Fair threshold-weighted CRPS."""
    M = ens_np.shape[1]
    if event_type == "above":
        vfc  = np.maximum(ens_np - T_arr[:, None], 0.0)
        vobs = np.maximum(obs   - T_arr,            0.0)
    else:
        vfc  = np.maximum(T_arr[:, None] - ens_np, 0.0)
        vobs = np.maximum(T_arr          - obs,     0.0)
    T1           = np.abs(vfc - vobs[:, None]).mean(axis=1)
    vfc_sorted   = np.sort(vfc, axis=1)
    k            = np.arange(1, M + 1)
    fair_bonus   = (vfc_sorted * (2 * k - M - 1)).sum(axis=1) / (M * (M - 1))
    return float(np.mean(T1 - fair_bonus))


def kde_plot(ax, data, color, label, lw=2.0, alpha_fill=0.18, bw=0.3):
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return
    kde  = gaussian_kde(data, bw_method=bw)
    xs   = np.linspace(data.min() - 0.3, data.max() + 0.3, 400)
    ys   = kde(xs)
    ax.plot(xs, ys, color=color, lw=lw, label=label)
    ax.fill_between(xs, ys, alpha=alpha_fill, color=color)


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure(results, m1_name, m2_name, day, orog, event_type,
                variable, output_path, results_dir=None):
    """
    results  – dict  {season: {obs_ext, fc1_ext, fc2_ext, T_ext, n_total}}
    """
    seasons  = list(results.keys())
    n_s      = len(seasons)
    var_map  = {"2t": "2m Temperature", "10ff": "10m Wind Speed", "tp24": "Precipitation"}
    var_disp = var_map.get(variable, variable)

    fig, axes = plt.subplots(3, n_s, figsize=(8 * n_s, 19))
    if n_s == 1:
        axes = axes[:, np.newaxis]
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{var_disp}  |  Warm extreme diagnostics  |  Day {day}  |  {orog.upper()} terrain\n"
        f"Threshold: 99th percentile of per-station obs climatology",
        fontsize=14, fontweight="bold", y=1.00
    )

    score_row = {}   # {season: (tw1, tw2)}

    for col, season in enumerate(seasons):
        d        = results[season]
        obs_ext  = d["obs_ext"]
        fc1_ext  = d["fc1_ext"]
        fc2_ext  = d["fc2_ext"]
        T_ext    = d["T_ext"]
        n_ext    = len(obs_ext)
        n_total  = d["n_total"]
        exc_freq = 100.0 * n_ext / n_total
        T_mean   = float(np.mean(T_ext))
        obs_mean = float(np.mean(obs_ext))

        # Per-case ensemble median (extreme events only for visual panels)
        fc1_med = np.median(fc1_ext, axis=1)
        fc2_med = np.median(fc2_ext, axis=1)

        # Bias = median − obs  (normalised to threshold)
        fc1_bias_norm = fc1_med - obs_ext   # each case
        fc2_bias_norm = fc2_med - obs_ext

        fc1_mean_bias = float(np.mean(fc1_bias_norm))
        fc2_mean_bias = float(np.mean(fc2_bias_norm))

        # Spread: mean std across all extreme events
        fc1_spread = float(np.mean(np.std(fc1_ext, axis=1, ddof=1)))
        fc2_spread = float(np.mean(np.std(fc2_ext, axis=1, ddof=1)))

        # twCRPS: use pipeline CSV if available (matches heatmap exactly),
        # otherwise fall back to computing over extreme events only
        tw1, tw2 = None, None
        if results_dir:
            orog_key_csv = orog.lower()
            csv_path = Path(results_dir) / f"scores_by_leadtime_{variable}_{season}_{orog_key_csv}.csv"
            if csv_path.exists():
                try:
                    csv_df = pd.read_csv(csv_path)
                    row_csv = csv_df[csv_df["forecast_day"] == day]
                    if not row_csv.empty and "twCRPS_fc1" in csv_df.columns:
                        tw1 = float(row_csv["twCRPS_fc1"].iloc[0])
                        tw2 = float(row_csv["twCRPS_fc2"].iloc[0])
                        print(f"  {season}: pipeline twCRPS  IFS={tw1:.5f}  AIFS={tw2:.5f}")
                except Exception as e:
                    print(f"  Warning: could not read pipeline CSV: {e}")
        if tw1 is None:
            tw1 = twcrps_score(fc1_ext, obs_ext, T_ext, event_type)
            tw2 = twcrps_score(fc2_ext, obs_ext, T_ext, event_type)
            print(f"  {season}: computed twCRPS (extreme only)  IFS={tw1:.5f}  AIFS={tw2:.5f}")
        score_row[season] = (tw1, tw2)

        # Normalise everything to (x - T) so "0 = at threshold"
        obs_norm  = obs_ext - T_ext
        fc1_norm  = (fc1_med - T_ext)
        fc2_norm  = (fc2_med - T_ext)

        # ── ROW 1 ── "When extremes happened, where were the forecasts?" ───────
        ax1 = axes[0, col]

        # Clip x-range to 1st–99th percentile for clean display
        all_vals  = np.concatenate([obs_norm, fc1_norm, fc2_norm])
        xlo = max(np.percentile(all_vals, 1) - 0.5, -3)
        xhi = np.percentile(all_vals, 99) + 0.5

        kde_plot(ax1, obs_norm,  COBS, "Observed temperature", lw=2.5, alpha_fill=0.25)
        kde_plot(ax1, fc1_norm,  C1,   f"IFS-ENS (ensemble median)",    lw=2.0)
        kde_plot(ax1, fc2_norm,  C2,   f"AIFS-ENS (ensemble median)",   lw=2.0)

        ax1.axvline(0, color="black", lw=1.5, ls="--", zorder=3, label="Extreme threshold T")

        # Mean lines
        for val, col_c in [(float(np.mean(obs_norm)), COBS),
                           (float(np.mean(fc1_norm)), C1),
                           (float(np.mean(fc2_norm)), C2)]:
            ax1.axvline(val, color=col_c, lw=1.2, ls=":", alpha=0.8)

        ax1.set_xlim(xlo, xhi)
        ax1.set_xlabel("Temperature − threshold T  (°C)", fontsize=10)
        ax1.set_ylabel("Density", fontsize=10)
        ax1.set_title(
            f"{season}  ·  {n_ext} extremes out of {n_total:,} total ({exc_freq:.1f}%)\n"
            f"Ensemble median distribution during obs > T  (T̄ = {T_mean:.1f} °C)",
            fontsize=11, fontweight="bold"
        )
        ax1.legend(fontsize=9, loc="upper right")
        ax1.set_facecolor("#f8f8f8")

        # Annotation: mean biases
        ylim = ax1.get_ylim()
        yt   = ylim[1] * 0.72
        ax1.annotate(
            f"IFS-ENS mean: {float(np.mean(fc1_norm)):+.2f} °C",
            xy=(float(np.mean(fc1_norm)), yt), xytext=(float(np.mean(fc1_norm)) - 0.05, yt),
            ha="right", fontsize=8, color=C1, fontweight="bold"
        )
        ax1.annotate(
            f"AIFS-ENS mean: {float(np.mean(fc2_norm)):+.2f} °C",
            xy=(float(np.mean(fc2_norm)), yt * 0.88), xytext=(float(np.mean(fc2_norm)) + 0.05, yt * 0.88),
            ha="left", fontsize=8, color=C2, fontweight="bold"
        )
        ax1.annotate(
            f"Obs mean: {float(np.mean(obs_norm)):+.2f} °C",
            xy=(float(np.mean(obs_norm)), yt * 0.76), xytext=(float(np.mean(obs_norm)) + 0.05, yt * 0.76),
            ha="left", fontsize=8, color=COBS, fontweight="bold"
        )

        # ── ROW 2 ── "How far off was each model? (bias per case)" ─────────────
        ax2 = axes[1, col]

        bp_data = [fc1_bias_norm, fc2_bias_norm]
        labels  = [f"IFS-ENS\n(spread={fc1_spread:.2f}°C)",
                   f"AIFS-ENS\n(spread={fc2_spread:.2f}°C)"]
        colors  = [C1, C2]

        for i, (data, lbl, clr) in enumerate(zip(bp_data, labels, colors)):
            bp = ax2.boxplot(
                data, positions=[i], widths=0.45,
                patch_artist=True, notch=False, sym="",
                whiskerprops=dict(color=clr, lw=1.5),
                capprops=dict(color=clr, lw=1.5),
                medianprops=dict(color="black", lw=2),
                boxprops=dict(facecolor=clr, alpha=0.45, linewidth=1.5),
                flierprops=dict(marker=".", markersize=1, color=clr, alpha=0.3),
            )
            ax2.annotate(
                f"mean: {np.mean(data):+.2f}°C",
                xy=(i, np.percentile(data, 75)),
                xytext=(i + 0.28, np.percentile(data, 75)),
                fontsize=9, color=clr, fontweight="bold", va="center"
            )

        ax2.axhline(0, color="black", lw=1.5, ls="--", zorder=3)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel("Ensemble median  −  Observation  (°C)", fontsize=10)
        ax2.set_title(
            f"{season}  ·  Forecast error during extreme events\n"
            f"Negative = model too cold  ·  Positive = model too warm",
            fontsize=11, fontweight="bold"
        )
        ax2.set_facecolor("#f8f8f8")

        # Color the zero-crossing region
        ax2.axhspan(-0.5, 0.5, alpha=0.07, color="green", label="±0.5°C tolerance")
        ax2.legend(fontsize=8, loc="upper right")

        # ── ROW 3 ── "Overall score (twCRPS) — split extreme vs non-extreme" ──
        ax3 = axes[2, col]

        # Compute twCRPS on extreme-only subset (from data) for comparison
        tw1_ext = twcrps_score(fc1_ext, obs_ext, T_ext, event_type)
        tw2_ext = twcrps_score(fc2_ext, obs_ext, T_ext, event_type)
        pct_ext  = 100.0 * (tw2_ext - tw1_ext) / tw1_ext
        pct_all  = 100.0 * (tw2 - tw1) / tw1

        x      = np.array([0.0, 1.0, 2.5, 3.5])
        labels = ["IFS-ENS\n(extreme\nevents only)",
                  "AIFS-ENS\n(extreme\nevents only)",
                  "IFS-ENS\n(all events,\npipeline score)",
                  "AIFS-ENS\n(all events,\npipeline score)"]
        vals   = [tw1_ext, tw2_ext, tw1, tw2]

        # colour: green = winner in that group, grey = loser
        c_ext = [C1 if tw1_ext < tw2_ext else "#bbbbbb",
                 C2 if tw2_ext < tw1_ext else "#bbbbbb"]
        c_all = [C1 if tw1 < tw2 else "#bbbbbb",
                 C2 if tw2 < tw1 else "#bbbbbb"]
        bar_colors = c_ext + c_all

        bars = ax3.bar(x, vals, color=bar_colors, alpha=0.88,
                       width=0.7, edgecolor="black", lw=0.8, zorder=2)

        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     val + max(vals) * 0.01,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color="black")

        # Bracket labels for the two groups
        y_bracket = max(vals) * 1.12
        for xmid, label, pct, better in [
            (0.5,  f"During extremes\n(obs > T)", pct_ext,
             "AIFS" if pct_ext < 0 else "IFS"),
            (3.0,  f"All cases\n(pipeline twCRPS)", pct_all,
             "AIFS" if pct_all < 0 else "IFS"),
        ]:
            clr_txt = C2 if better == "AIFS" else C1
            ax3.annotate(
                f"{better} wins ({abs(pct):.0f}%)",
                xy=(xmid, y_bracket), ha="center", fontsize=9,
                fontweight="bold", color=clr_txt
            )

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=8)
        ax3.set_ylabel("twCRPS  ←  lower is better", fontsize=10)
        ax3.set_facecolor("#f8f8f8")
        ax3.set_ylim(0, max(vals) * 1.3)

        # Vertical separator between the two groups
        ax3.axvline(1.75, color="gray", lw=1.0, ls="--", alpha=0.6)

        winner_all = "AIFS-ENS" if tw2 < tw1 else "IFS-ENS"
        winner_ext = "AIFS-ENS" if tw2_ext < tw1_ext else "IFS-ENS"
        ax3.set_title(
            f"{season}  ·  twCRPS comparison\n"
            f"Extreme events → {winner_ext} wins  |  All events → {winner_all} wins",
            fontsize=11, fontweight="bold"
        )

        # Explanation box
        if winner_ext != winner_all:
            msg = (f"⚠ Paradox: {winner_ext} is better during extreme events,\n"
                   f"but {winner_all} wins the overall score.\n"
                   f"Reason: the overall twCRPS is dominated by non-extreme days\n"
                   f"(~{100*(n_total-n_ext)/n_total:.0f}% of cases). On those days,\n"
                   f"{'AIFS-ENS' if winner_all=='IFS-ENS' else 'IFS-ENS'} generates more false alarms\n"
                   f"(members > T when obs is not extreme), which adds penalty.")
        else:
            msg = (f"{winner_all} wins both during extreme events and overall.\n"
                   f"Consistent performance across all conditions.")
        ax3.text(0.5, -0.38, msg, transform=ax3.transAxes,
                 fontsize=8.5, ha="center", va="top", style="italic",
                 color="#333333",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", alpha=0.95,
                           edgecolor="#f0c040", lw=1.2))

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      required=True)
    p.add_argument("--orog",        default="low")
    p.add_argument("--day",         type=int, default=5)
    p.add_argument("--seasons",     nargs="+", default=["DJF", "JJA"])
    p.add_argument("--max-samples", type=int, default=300_000, dest="max_samples")
    p.add_argument("--output-dir",  default="case_study_output/twcrps_diagnostic",
                   dest="output_dir")
    p.add_argument("--results-dir", default=None, dest="results_dir",
                   help="Pipeline results directory to read actual twCRPS from CSVs")
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    m1_name    = config["read_data"]["forecast_model1"]["name"]
    m2_name    = config["read_data"]["forecast_model2"]["name"]
    variable   = config["variable"]
    event_type = config.get("threshold", {}).get("event_type", "above")

    parquet_dir  = Path(config["extract_points"]["output_path"])
    parquet_path = next(parquet_dir.glob(f"*_day{args.day}.parquet"), None)
    if parquet_path is None:
        print(f"ERROR: no day{args.day} parquet in {parquet_dir}")
        sys.exit(1)

    # Resolve orography range from config
    raw_ranges = config.get("filter", {}).get("orography_ranges", {})
    orog_key   = args.orog.lower()
    aliases    = {"flat": "low", "hilly": "mid", "complex": "high"}
    orog_key   = aliases.get(orog_key, orog_key)
    if orog_key in raw_ranges:
        orog_range = tuple(raw_ranges[orog_key])
    else:
        orog_range = {"low": (0, 40), "mid": (40, 120), "high": (120, 3000)}.get(orog_key, (0, 40))

    print(f"Config : {args.config}")
    print(f"Variable: {variable}  |  Day {args.day}  |  Orog: {args.orog} {orog_range}")
    print(f"Models  : {m1_name}  vs  {m2_name}")

    results = {}
    for season in args.seasons:
        print(f"\n── Loading {season} ──")
        df, n_raw = load_season_data(parquet_path, season, orog_range,
                                     config, args.max_samples)
        if df is None or df.empty:
            print(f"  No data for {season}")
            continue

        fc1_cols = sorted([c for c in df.columns if c.startswith("fc1_member_")],
                          key=lambda c: int(c.split("_")[-1]))
        fc2_cols = sorted([c for c in df.columns if c.startswith("fc2_member_")],
                          key=lambda c: int(c.split("_")[-1]))
        n_total = len(df)

        T_arr = load_per_station_thresholds(config, df)
        valid = ~np.isnan(T_arr)
        df    = df[valid].reset_index(drop=True)
        T_arr = T_arr[valid]

        obs    = df["obs_value"].values.astype(float)
        fc1_np = df[fc1_cols].values.astype(float)
        fc2_np = df[fc2_cols].values.astype(float)

        extreme = obs > T_arr
        n_ext   = int(extreme.sum())
        print(f"  {n_ext} extreme events ({100*n_ext/len(obs):.1f}%) from {len(obs):,} cases")

        if n_ext < 30:
            print(f"  ⚠ Too few extreme events, skipping {season}")
            continue

        results[season] = dict(
            obs_ext  = obs[extreme],
            fc1_ext  = fc1_np[extreme],
            fc2_ext  = fc2_np[extreme],
            T_ext    = T_arr[extreme],
            n_total  = n_total,
        )

    if not results:
        print("No valid data loaded, exiting.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons_str  = "_".join(results.keys())
    output_path  = out_dir / f"twcrps_simple_{variable}_day{args.day}_{seasons_str}_{args.orog}.png"

    make_figure(results, m1_name, m2_name, args.day, args.orog,
                event_type, variable, output_path,
                results_dir=args.results_dir)


if __name__ == "__main__":
    main()
