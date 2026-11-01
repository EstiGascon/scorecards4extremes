#!/usr/bin/env python3
"""
Count evolution plot (fig 19) zoomed to p90-p99.
Usage:
  python3 plot_count_evolution_zoomed.py \
      --config config_2t_local_p99obsclim_aifs_ifs_single.yaml \
      --day 3 --season DJF --orog complex
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Import data-loading helpers from diagnose_extremes
import diagnose_extremes as de

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True)
    p.add_argument("--day",      type=int, default=3)
    p.add_argument("--season",   default=None)
    p.add_argument("--orog",     default=None)
    p.add_argument("--pct-min",  type=int, default=90,
                   help="Start of percentile sweep (default: 90)")
    p.add_argument("--fig-num",  type=str, default=None,
                   help="Figure number prefix in filename (e.g. '19b', '20')")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()

def main():
    args  = parse_args()
    PCT_MIN = args.pct_min
    config = de.load_config(args.config)
    fc1_name, fc2_name = de.get_model_names(config)
    variable = config['variable']

    # Mirror the output path logic from diagnose_extremes.main()
    base_out = config.get("save", {}).get("output_directory", "./results/diagnostics")
    thr_tag  = "pct99"
    run_tag  = f"day{args.day}_{thr_tag}"
    if args.season:
        run_tag += f"_{args.season}"
    if args.orog:
        run_tag += f"_{args.orog}"
    out_dir = Path(args.output_dir or (str(Path(base_out) / run_tag)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set module globals so _savefig works
    de.OUTPUT_PATH  = str(out_dir)
    de.SAVE_FIGURES = True

    # ── Load & filter data ────────────────────────────────────────────────────
    print(f"Loading day {args.day} data ...")
    df, _, _ = de.load_day(config, args.day)
    df = de.filter_data(df, config, season=args.season, orog=args.orog)

    thr_method = config.get('threshold', {}).get('method', '')
    lead_freq  = config.get('lead_time_frequency', 24)
    if thr_method == 'local_obs_climatology' and lead_freq < 24:
        print("  Aggregating sub-daily data to daily means ...")
        df = de._aggregate_to_daily_mean_local(df, config)

    obs = df['obs_value'].values.astype(np.float32)
    fc1 = df['fc1_value'].values.astype(np.float32)
    fc2 = df['fc2_value'].values.astype(np.float32)
    print(f"  N = {len(obs):,}")

    # ── Compute counts across p90–p99 ────────────────────────────────────────
    event_type  = config.get('threshold', {}).get('event_type', 'above')
    percentiles = np.arange(PCT_MIN, 100, 1)
    thresholds  = np.array([np.percentile(obs, p) for p in percentiles])

    hits1, miss1, fa1 = [], [], []
    hits2, miss2, fa2 = [], [], []
    for T in thresholds:
        if event_type == 'below':
            h1 = int(np.sum((obs <= T) & (fc1 <= T))); m1 = int(np.sum((obs <= T) & (fc1 >  T))); f1 = int(np.sum((obs >  T) & (fc1 <= T)))
            h2 = int(np.sum((obs <= T) & (fc2 <= T))); m2 = int(np.sum((obs <= T) & (fc2 >  T))); f2 = int(np.sum((obs >  T) & (fc2 <= T)))
        else:
            h1 = int(np.sum((obs >= T) & (fc1 >= T))); m1 = int(np.sum((obs >= T) & (fc1 <  T))); f1 = int(np.sum((obs <  T) & (fc1 >= T)))
            h2 = int(np.sum((obs >= T) & (fc2 >= T))); m2 = int(np.sum((obs >= T) & (fc2 <  T))); f2 = int(np.sum((obs <  T) & (fc2 >= T)))
        hits1.append(h1); miss1.append(m1); fa1.append(f1)
        hits2.append(h2); miss2.append(m2); fa2.append(f2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    c1, c2 = "#1565C0", "#B71C1C"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (axes[0], hits1, hits2, "Hits"),
        (axes[1], miss1, miss2, "Misses"),
        (axes[2], fa1,   fa2,   "False Alarms"),
    ]
    for ax, d1, d2, title in panels:
        ax.plot(percentiles, d1, color=c1, lw=2, label=fc1_name, marker='o', ms=4)
        ax.plot(percentiles, d2, color=c2, lw=2, label=fc2_name, marker='s', ms=4)
        ax.fill_between(percentiles, d1, d2,
                        where=[a > b for a, b in zip(d1, d2)],
                        alpha=0.13, color=c1, interpolate=True)
        ax.fill_between(percentiles, d1, d2,
                        where=[a < b for a, b in zip(d1, d2)],
                        alpha=0.13, color=c2, interpolate=True)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(f"Percentile (p{PCT_MIN}–p99)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(percentiles[::2])
        ax.tick_params(labelsize=10)

    season_lbl = args.season or "All seasons"
    orog_lbl   = (args.orog or "all").capitalize()
    fig.suptitle(
        f"Count Evolution (p{PCT_MIN}–p99) — 2m Temperature  |  "
        f"Day {args.day}  |  {season_lbl}  |  {orog_lbl} orography",
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    fig_num = args.fig_num if args.fig_num else ("19b" if PCT_MIN >= 90 else "19b")
    fname = (f"{fig_num}_count_evolution_p{PCT_MIN}plus_{fc1_name}_vs_{fc2_name}"
             f"_{variable}_day{args.day}.png")
    out = out_dir / fname
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Plot 20: Count difference (M1 − M2) ──────────────────────────────────
    dh  = [h1 - h2 for h1, h2 in zip(hits1, hits2)]
    dm  = [m1 - m2 for m1, m2 in zip(miss1, miss2)]
    dfa = [f1 - f2 for f1, f2 in zip(fa1,   fa2)]

    bar_w = (percentiles[1] - percentiles[0]) * 0.8 if len(percentiles) > 1 else 1
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    # positive_is_good: True=Hits (more=better for M1), False=Misses/FAs (more=worse for M1)
    diff_panels = [
        (axes2[0], dh,  "ΔHits (M1−M2)",        "M1 better: more hits →",    "M2 better: more hits →",    True),
        (axes2[1], dm,  "ΔMisses (M1−M2)",       "M1 worse: more misses →",   "M2 worse: more misses →",   False),
        (axes2[2], dfa, "ΔFalse Alarms (M1−M2)", "M1 worse: more FAs →",      "M2 worse: more FAs →",      False),
    ]
    for ax, vals, title, pos_lbl, neg_lbl, pos_good in diff_panels:
        v = np.array(vals, dtype=float)
        # Blue = M1 winning, Red = M2 winning — consistent with plots 10/11
        bar_colors = ["#1565C0" if (x >= 0) == pos_good else "#B71C1C" for x in v]
        ax.bar(percentiles, v, color=bar_colors, alpha=0.75, width=bar_w)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(f"Percentile (p{PCT_MIN}–p99)", fontsize=11)
        ax.set_ylabel("Count difference (M1 − M2)", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(percentiles[::2])
        ax.tick_params(labelsize=10)
        ax.text(0.97, 0.93, pos_lbl, transform=ax.transAxes, ha='right',
                fontsize=9, color="#1565C0" if pos_good else "#B71C1C")
        ax.text(0.97, 0.06, neg_lbl, transform=ax.transAxes, ha='right',
                fontsize=9, color="#B71C1C" if pos_good else "#1565C0")

    fig2.suptitle(
        f"Count Difference M1−M2 (p{PCT_MIN}–p99) — 2m Temperature  |  "
        f"Day {args.day}  |  {season_lbl}  |  {orog_lbl} orography\n"
        f"M1={fc1_name}   M2={fc2_name}",
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    fname2 = (f"20_count_difference_p{PCT_MIN}plus_{fc1_name}_vs_{fc2_name}"
              f"_{variable}_day{args.day}.png")
    out2 = out_dir / fname2
    fig2.savefig(out2, dpi=160, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out2}")

if __name__ == "__main__":
    main()
