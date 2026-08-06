"""
diagnose_det_extremes_simple.py
───────────────────────────────────────────────────────────────────────────────
Intuitive diagnostic for deterministic extreme-event verification.
Pools data across multiple forecast days and shows:

  Row 1 – "When obs exceeded T, where was the deterministic forecast?"
           KDE of (fc − T) and (obs − T).  0 = exactly at threshold.
           Left of 0 = model forecast non-extreme when obs was extreme (miss).

  Row 2 – "Forecast error during extreme events"
           Box plot of (fc − obs) for each model.  0 = perfect.
           Left = model too cold (underprediction); right = too warm.

  Row 3 – "Score breakdown: hits, misses, false alarms"
           Stacked bar chart of event counts: Hits / Misses / False Alarms,
           read from pipeline CSV scores (ETS, POD, FAR).
           Consistent with the heatmap values.

Usage:
  python diagnose_det_extremes_simple.py \\
      --config config_2t_local_p99obsclim_aifs_ifs_single.yaml \\
      --orog flat --days 1 3 --season JJA
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import gaussian_kde

# Put the repo root (parent of diagnostics/) on sys.path so `case_studies` is
# importable when the script is run as `python diagnostics/<this>.py` from the
# repo root (the invocation documented in docs/USER_GUIDE.md).  Running Python on
# a file puts the file's own directory on sys.path[0], not the repo root, so
# inserting diagnostics/ here (as before) never resolved the import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling _style module
from case_studies.case_study_utils import load_per_station_thresholds
import _style

# ── Colours (shared, colourblind-safe — see diagnostics/_style.py) ─────────────
C1   = _style.C_FC1   # model1 / IFS   — blue
C2   = _style.C_FC2   # model2 / AIFS  — vermillion
COBS = _style.C_OBS   # observations   — green
SEASON_MONTHS = {"DJF": {12,1,2}, "MAM": {3,4,5}, "JJA": {6,7,8}, "SON": {9,10,11}}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def _month(d):
    return int(str(int(d))[4:6])

def kde_plot(ax, data, color, label, lw=2.0, alpha_fill=0.18, bw=0.3):
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return
    kde  = gaussian_kde(data, bw_method=bw)
    xs   = np.linspace(np.percentile(data, 0.5), np.percentile(data, 99.5), 400)
    ys   = kde(xs)
    ax.plot(xs, ys, color=color, lw=lw, label=label)
    ax.fill_between(xs, ys, alpha=alpha_fill, color=color)


def load_and_filter(parquet_path, season, orog_range, config,
                    max_samples=None, seed=42):
    """Load parquet, filter by season + orog + coastal, aggregate 6h→daily.

    Note: max_samples/seed are accepted for backward compatibility but are no
    longer used — the data is returned in full so that the extreme-event counts
    and exceedance frequencies reported downstream are exact rather than computed
    on a random subsample.  After daily aggregation the per-season/per-orography
    subset is small enough to hold in memory.
    """
    months = SEASON_MONTHS[season]
    lo, hi = orog_range
    coastal_thresh = config.get("filter", {}).get("coastal_lsm_threshold", 0.9)
    remove_coastal = config.get("filter", {}).get("remove_coastal_stations", False)

    pf = pq.ParquetFile(str(parquet_path))
    chunks = []
    for batch in pf.iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        # Vectorised month filter (dates are stored as YYYYMMDD ints/strings);
        # much faster than a per-row .apply(_month) over every 100k-row batch.
        month_of = chunk["date"].astype(str).str[4:6].astype(int)
        chunk = chunk[month_of.isin(months)]
        if "sdfor" in chunk.columns:
            chunk = chunk[(chunk["sdfor"] >= lo) & (chunk["sdfor"] < hi)]
        if remove_coastal and "lsm" in chunk.columns:
            chunk = chunk[chunk["lsm"] >= coastal_thresh]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)

    # QC — this diagnostic is only ever run for 2t (temperature), so the fixed
    # [-60, 60] °C validity window is appropriate.  It would silently drop valid
    # extremes if reused for 10ff (>60 m/s) or tp24 (>60 mm); make it
    # variable-aware before using this script for those variables.
    for col in ["obs_value", "fc1_value", "fc2_value"]:
        if col in df.columns:
            df = df[(df[col] > -60) & (df[col] < 60)]

    # Aggregate 6-hourly → daily mean (matches pipeline)
    static_cols = [c for c in ["lat", "lon", "sdfor", "lsm", "forecast_day"] if c in df.columns]
    agg = {c: "mean" for c in ["obs_value", "fc1_value", "fc2_value"] if c in df.columns}
    agg.update({c: "first" for c in static_cols})
    df = df.groupby(["date", "station_id"], as_index=False).agg(agg)
    return df


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure(all_dfs, T_arrays, m1_name, m2_name, season, orog,
                days, variable, results_dir, config, output_path):

    # Pool all days together
    obs_all  = np.concatenate([d["obs_value"].values for d in all_dfs])
    fc1_all  = np.concatenate([d["fc1_value"].values for d in all_dfs])
    fc2_all  = np.concatenate([d["fc2_value"].values for d in all_dfs])
    T_all    = np.concatenate(T_arrays)

    extreme  = obs_all > T_all
    n_ext    = int(extreme.sum())
    n_total  = len(obs_all)
    T_mean   = float(np.mean(T_all))

    obs_ext  = obs_all[extreme]
    fc1_ext  = fc1_all[extreme]
    fc2_ext  = fc2_all[extreme]
    T_ext    = T_all[extreme]

    days_str = " & ".join(f"Day {d}" for d in days)
    var_map  = {"2t": "2m Temperature", "10ff": "10m Wind Speed", "tp24": "Precipitation"}
    var_disp = var_map.get(variable, variable)
    unit     = {"2t": "°C", "10ff": "m/s", "tp24": "mm"}.get(variable, "")

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{var_disp}  |  Warm extreme diagnostics  |  {days_str} (pooled)  |  "
        f"{season}  |  {orog.upper()} terrain\n"
        f"Threshold: 99th percentile of per-station obs climatology  "
        f"(T̄ = {T_mean:.1f}{unit})  |  "
        f"{n_ext} extreme events out of {n_total:,} total ({100*n_ext/n_total:.1f}%)",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── ROW 1 ── KDE of forecast vs obs during extreme events ─────────────────
    ax1 = axes[0]
    obs_norm = obs_ext - T_ext
    fc1_norm = fc1_ext - T_ext
    fc2_norm = fc2_ext - T_ext

    all_vals = np.concatenate([obs_norm, fc1_norm, fc2_norm])
    xlo = max(np.percentile(all_vals, 1) - 0.5, -8)
    xhi = np.percentile(all_vals, 99) + 0.5

    kde_plot(ax1, obs_norm,  COBS, "Observed temperature", lw=2.5, alpha_fill=0.25)
    kde_plot(ax1, fc1_norm,  C1,   f"IFS-oper (deterministic)",  lw=2.0)
    kde_plot(ax1, fc2_norm,  C2,   f"AIFS-oper (deterministic)", lw=2.0)
    ax1.axvline(0, color="black", lw=1.5, ls="--", zorder=3, label="Threshold T")

    for val, clr in [(float(np.mean(obs_norm)), COBS),
                     (float(np.mean(fc1_norm)), C1),
                     (float(np.mean(fc2_norm)), C2)]:
        ax1.axvline(val, color=clr, lw=1.0, ls=":", alpha=0.8)

    ax1.set_xlim(xlo, xhi)
    ax1.set_xlabel(f"Temperature − threshold T  ({unit})", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        f"Where was the forecast when an extreme event occurred?  (obs > T)\n"
        f"Values left of 0 = model forecast below the extreme threshold (miss territory)",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.set_facecolor("#f8f8f8")

    ylim = ax1.get_ylim()
    yt = ylim[1] * 0.80
    for lbl, val, clr, align in [
        (f"IFS mean: {np.mean(fc1_norm):+.2f}{unit}",  np.mean(fc1_norm), C1,   "right"),
        (f"AIFS mean: {np.mean(fc2_norm):+.2f}{unit}", np.mean(fc2_norm), C2,   "left"),
        (f"Obs mean: {np.mean(obs_norm):+.2f}{unit}",  np.mean(obs_norm), COBS, "left"),
    ]:
        offset = -0.08 if align == "right" else 0.08
        ax1.annotate(lbl, xy=(val, yt), xytext=(val + offset, yt),
                     ha=align, fontsize=9, color=clr, fontweight="bold")
        yt *= 0.88

    ax1.axvspan(xlo, 0, alpha=0.04, color="red")
    ax1.text((xlo + 0) / 2, ylim[1] * 0.95, "← model too cold\n(misses)",
             ha="center", va="top", fontsize=8, color="darkred", style="italic")

    # ── ROW 2 ── Forecast error box plots (extreme events only) ───────────────
    ax2 = axes[1]
    err1 = fc1_ext - obs_ext
    err2 = fc2_ext - obs_ext
    mean1 = float(np.mean(err1)); mean2 = float(np.mean(err2))
    spread1 = float(np.std(err1)); spread2 = float(np.std(err2))

    for i, (data, clr, lbl) in enumerate([
        (err1, C1,   f"IFS-oper\n(std={spread1:.2f}{unit})"),
        (err2, C2,   f"AIFS-oper\n(std={spread2:.2f}{unit})"),
    ]):
        ax2.boxplot(data, positions=[i], widths=0.45,
                    patch_artist=True, notch=False, sym="",
                    whiskerprops=dict(color=clr, lw=1.5),
                    capprops=dict(color=clr, lw=1.5),
                    medianprops=dict(color="black", lw=2.0),
                    boxprops=dict(facecolor=clr, alpha=0.45, linewidth=1.5),
                    flierprops=dict(marker=".", markersize=1, color=clr, alpha=0.2))
        ax2.annotate(f"mean: {np.mean(data):+.2f}{unit}",
                     xy=(i, np.percentile(data, 75)),
                     xytext=(i + 0.30, np.percentile(data, 75)),
                     fontsize=10, color=clr, fontweight="bold", va="center")

    ax2.axhline(0, color="black", lw=1.5, ls="--", zorder=3)
    ax2.axhspan(-0.5, 0.5, alpha=0.07, color="green")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"IFS-oper\n(std={spread1:.2f}{unit})",
                          f"AIFS-oper\n(std={spread2:.2f}{unit})"], fontsize=11)
    ax2.set_ylabel(f"Forecast − Observation  ({unit})", fontsize=11)
    ax2.set_title(
        f"Forecast error when obs exceeded threshold  (fc − obs)\n"
        f"Negative = model too cold  ·  Positive = model too warm",
        fontsize=11, fontweight="bold"
    )
    ax2.set_facecolor("#f8f8f8")

    # ── ROW 3 ── Score breakdown from pipeline CSVs ────────────────────────────
    ax3 = axes[2]

    # Read scores from CSV, average over requested days
    csv_rows = {"fc1": {}, "fc2": {}}
    csv_sig = {}   # {score: bool} — bootstrap significance of the model difference
    score_cols = ["ETS", "PSS", "POD", "FAR", "twMAE"]
    csv_path = Path(results_dir) / f"scores_by_leadtime_{variable}_{season}_{orog}.csv"
    csv_available = False
    if csv_path.exists():
        try:
            csv_df = pd.read_csv(csv_path)
            sub = csv_df[csv_df["forecast_day"].isin(days)]
            if not sub.empty:
                csv_available = True
                for sc in score_cols:
                    if f"{sc}_fc1" in sub.columns:
                        csv_rows["fc1"][sc] = float(sub[f"{sc}_fc1"].mean())
                        csv_rows["fc2"][sc] = float(sub[f"{sc}_fc2"].mean())
                        # Significant only if the diff is significant on every
                        # selected day (conservative when pooling days).
                        if f"{sc}_is_significant" in sub.columns:
                            csv_sig[sc] = bool(sub[f"{sc}_is_significant"].all())
        except Exception as e:
            print(f"  Warning reading CSV: {e}")

    if csv_available:
        sc_names = [s for s in score_cols if s in csv_rows["fc1"]]
        x       = np.arange(len(sc_names))
        width   = 0.35

        for i, (model, clr) in enumerate([(m1_name, C1), (m2_name, C2)]):
            key = "fc1" if i == 0 else "fc2"
            vals = [csv_rows[key][s] for s in sc_names]
            bars = ax3.bar(x + i * width, vals, width, label=model,
                           color=clr, alpha=0.80, edgecolor="black", lw=0.7)
            for bar, val in zip(bars, vals):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.005,
                         f"{val:.3f}", ha="center", va="bottom",
                         fontsize=8, color=clr, fontweight="bold")

        # Mark winners per score + bootstrap significance of the difference
        y_top = max(max(csv_rows["fc1"][s], csv_rows["fc2"][s]) for s in sc_names)
        for j, sc in enumerate(sc_names):
            v1 = csv_rows["fc1"][sc]; v2 = csv_rows["fc2"][sc]
            higher_better = sc not in ["FAR", "twMAE"]
            winner = (v2 > v1) if higher_better else (v2 < v1)
            ymax = max(csv_rows["fc1"][sc], csv_rows["fc2"][sc])
            # One star over the winning model's bar (position + colour encode
            # which model won; the glyph is always a star).
            ax3.annotate("★",
                         xy=(j + width / 2 + (width if winner else 0), ymax + 0.025),
                         ha="center", fontsize=12,
                         color=C2 if winner else C1)
            # Significance of the model difference (n.s. → the winner is not robust)
            marker = _style.significance_marker(csv_sig.get(sc)) if sc in csv_sig else ""
            if marker:
                ax3.annotate(marker, xy=(j + width / 2, y_top * 1.10),
                             ha="center", fontsize=8, style="italic",
                             color="#333333" if marker.startswith("✓") else "#999999")

        # Reference lines for skill scores
        ax3.axhline(0, color=_style.C_REF, lw=0.8, ls="--")
        ax3.set_xticks(x + width / 2)
        ax3.set_xticklabels(sc_names, fontsize=11)
        ax3.set_ylabel("Score value", fontsize=11)
        ax3.legend(fontsize=10)
        ax3.set_title(
            f"Pipeline verification scores  ({days_str} averaged)\n"
            f"ETS/PSS/POD: higher = better  ·  FAR/twMAE: lower = better  ·  "
            f"★ = winner  ·  ✓ sig. / n.s. = bootstrap significance",
            fontsize=11, fontweight="bold"
        )

        # Note: explain consistency with heatmap
        ets1 = csv_rows["fc1"].get("ETS", np.nan)
        ets2 = csv_rows["fc2"].get("ETS", np.nan)
        pct_ets = 100 * (ets2 - ets1) / abs(ets1) if ets1 != 0 else 0
        twmae1 = csv_rows["fc1"].get("twMAE", np.nan)
        twmae2 = csv_rows["fc2"].get("twMAE", np.nan)
        pct_twmae = 100 * (twmae2 - twmae1) / twmae1 if twmae1 != 0 else 0
        winner_ets   = m2_name if ets2 > ets1 else m1_name
        winner_twmae = m2_name if twmae2 < twmae1 else m1_name
        msg = (f"ETS: {winner_ets} better ({pct_ets:+.1f}%)  ·  "
               f"twMAE: {winner_twmae} better ({pct_twmae:+.1f}%)\n"
               f"These values match the heatmap for {season} / {orog} / {days_str}.")
        ax3.text(0.5, -0.14, msg, transform=ax3.transAxes,
                 fontsize=9, ha="center", va="top", style="italic",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#eeeeee", alpha=0.9))
    else:
        ax3.text(0.5, 0.5, f"Pipeline CSV not found:\n{csv_path}",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=11, color="red")

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      required=True)
    p.add_argument("--season",      default="JJA")
    p.add_argument("--orog",        default="flat")
    p.add_argument("--days",        nargs="+", type=int, default=[1, 3])
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                   help="Deprecated/ignored — full data is now used so extreme "
                        "counts are exact (retained for CLI compatibility).")
    p.add_argument("--output-dir",  default="case_study_output/twcrps_diagnostic",
                   dest="output_dir")
    return p.parse_args()


def main():
    args   = parse_args()
    _style.apply_style()
    config = load_config(args.config)

    m1_name  = config["read_data"]["forecast_model1"]["name"]
    m2_name  = config["read_data"]["forecast_model2"]["name"]
    variable = config["variable"]

    parquet_dir = Path(config["extract_points"]["output_path"])
    raw_ranges  = config.get("filter", {}).get("orography_ranges", {})
    orog_key    = args.orog.lower()
    aliases     = {"flat": "flat", "low": "flat", "hilly": "hilly",
                   "mid": "hilly", "complex": "complex", "high": "complex"}
    orog_key    = aliases.get(orog_key, orog_key)
    # Fail loudly rather than silently defaulting to (0, 40): a mismatch between
    # the requested terrain and the config's orography_ranges keys would otherwise
    # produce a plot filtered to the wrong terrain with no warning.
    if orog_key in raw_ranges:
        orog_range = tuple(raw_ranges[orog_key])
    else:
        _fallback = {"flat": (0, 40), "hilly": (40, 120), "complex": (120, 3000)}
        if orog_key not in _fallback:
            sys.exit(
                f"ERROR: orography '{args.orog}' → '{orog_key}' not found in config "
                f"filter.orography_ranges ({list(raw_ranges)}) nor in the built-in "
                f"fallback ({list(_fallback)}). Use one of flat/low, hilly/mid, complex/high."
            )
        orog_range = _fallback[orog_key]
        print(f"  Note: '{orog_key}' not in config orography_ranges; "
              f"using built-in fallback {orog_range}")

    save_cfg  = config.get("save", {})
    results_dir = save_cfg.get("output_directory", str(parquet_dir))

    print(f"Config:   {args.config}")
    print(f"Variable: {variable}  |  {args.season}  |  {orog_key} {orog_range}  |  Days: {args.days}")
    print(f"Models:   {m1_name}  vs  {m2_name}")

    all_dfs   = []
    T_arrays  = []

    for day in args.days:
        cands = list(parquet_dir.glob(f"*_day{day}.parquet"))
        # Prefer files without a fixed-threshold suffix
        cands = [c for c in cands if "_35." not in c.name and "_8." not in c.name] or cands
        if not cands:
            print(f"  [day {day}] No parquet found")
            continue
        parquet_path = cands[0]
        print(f"\n── Day {day}: {parquet_path.name} ──")
        df = load_and_filter(parquet_path, args.season, orog_range, config,
                             args.max_samples)
        if df is None or df.empty:
            print(f"  [day {day}] No data after filters")
            continue

        T_arr = load_per_station_thresholds(config, df)
        valid = ~np.isnan(T_arr)
        df    = df[valid].reset_index(drop=True)
        T_arr = T_arr[valid]

        extreme = df["obs_value"].values > T_arr
        print(f"  {int(extreme.sum())} extreme events ({100*extreme.mean():.1f}%) "
              f"from {len(df):,} daily cases")
        if extreme.sum() < 20:
            print(f"  ⚠ Too few extremes, skipping day {day}")
            continue

        all_dfs.append(df)
        T_arrays.append(T_arr)

    if not all_dfs:
        print("No data loaded. Exiting.")
        sys.exit(1)

    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days_str   = "_".join(f"day{d}" for d in args.days)
    output_path = out_dir / f"det_extremes_{variable}_{args.season}_{orog_key}_{days_str}.png"

    make_figure(all_dfs, T_arrays, m1_name, m2_name, args.season,
                orog_key, args.days, variable, results_dir, config, output_path)


if __name__ == "__main__":
    main()
