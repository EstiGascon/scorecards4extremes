#!/usr/bin/env python3
"""
plot_bias_crossover.py — visualise the IFS-cold-bias / AIFS-warm-bias
crossover that explains the "hilly is red, mountain is blue" twCRPS
scorecard pattern for 2t cold extremes (p1 obs climatology).

Reads the already-computed production score CSVs
(scores_by_leadtime_2t_{low,mid,high}.csv) — no re-scoring needed.

Panel 1 (top): ens_mean_bias (fc - obs) for IFS (dashed) vs AIFS (solid),
  one colour per orography bin, vs forecast day. Shows IFS's cold bias
  growing much faster with terrain roughness than AIFS's warm bias, and
  where the two cross in magnitude.
Panel 2 (bottom): twCRPS_diff (AIFS - IFS) for the same bins/lead times,
  for direct visual correlation with the bias crossover above.

Usage
-----
  python analysis/plot_bias_crossover.py \\
      --results-dir results/2t_ens_local_p1obsclim_aifsvsifs_commonperiod \\
      --output-dir case_study_output/hilly_month_geo_2t_p1cold_aifsvsifs_commonperiod
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BIN_COLORS = {"low": "#2c7bb6", "mid": "#d7191c", "high": "#5e3c99"}
BIN_LABELS = {"low": "LOW (flat, sdfor<40)", "mid": "MID (hilly, 40-120)", "high": "HIGH (mountain, \u2265120)"}


def _unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    i = 2
    while True:
        cand = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="Directory with scores_by_leadtime_2t_{low,mid,high}.csv")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model1-label", default="IFS")
    p.add_argument("--model2-label", default="AIFS")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for b in ("low", "mid", "high"):
        f = results_dir / f"scores_by_leadtime_2t_{b}.csv"
        if not f.exists():
            print(f"  WARNING: {f} not found — skipping {b}")
            continue
        data[b] = pd.read_csv(f)

    if not data:
        raise SystemExit(f"No scores_by_leadtime_2t_*.csv found in {results_dir}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for b, df in data.items():
        col = BIN_COLORS[b]
        x = df["forecast_day"]
        ax1.plot(x, df["ens_mean_bias_fc1"], ls="--", color=col, lw=1.8,
                  marker="o", ms=6, mfc="white", mec=col, mew=1.5)
        ax1.plot(x, df["ens_mean_bias_fc2"], ls="-", color=col, lw=2.4,
                  marker="s", ms=6, mfc=col, mec="k", mew=0.5)
        ax2.plot(x, df["twCRPS_diff"], ls="-", color=col, lw=2.4,
                  marker="o", ms=6, mec="k", mew=0.5, label=BIN_LABELS[b])

    ax1.axhline(0, color="k", lw=1.0, ls=":", alpha=0.7)
    ax1.set_ylabel("ens_mean_bias = fc \u2212 obs  (\u00b0C)\n(all conditions, not just extremes)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Bias crossover: IFS cold bias vs AIFS warm bias, by orography bin", fontsize=12, weight="bold")

    bin_handles = [Line2D([0], [0], color=BIN_COLORS[b], lw=2.4, label=BIN_LABELS[b]) for b in data]
    model_handles = [
        Line2D([0], [0], color="0.25", ls="--", lw=1.8, marker="o", ms=6,
               mfc="white", mec="0.25", mew=1.5, label=f"{args.model1_label} (fc1)"),
        Line2D([0], [0], color="0.25", ls="-", lw=2.4, marker="s", ms=6,
               mfc="0.25", mec="k", mew=0.5, label=f"{args.model2_label} (fc2)"),
    ]
    leg1 = ax1.legend(handles=bin_handles, title="Orography bin", fontsize=9,
                       title_fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       borderaxespad=0., framealpha=0.95)
    ax1.add_artist(leg1)
    ax1.legend(handles=model_handles, title="Model (line style)", fontsize=9,
               title_fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 0.55),
               borderaxespad=0., framealpha=0.95)

    ax2.axhline(0, color="k", lw=1.0, ls=":", alpha=0.7)
    ax2.set_ylabel(f"twCRPS_diff = {args.model2_label} \u2212 {args.model1_label}\n(negative = {args.model2_label} better)", fontsize=11)
    ax2.set_xlabel("Forecast day", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Actual scorecard metric (twCRPS_diff) for the same bins/lead times", fontsize=12, weight="bold")
    ax2.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., framealpha=0.95)

    fig.subplots_adjust(right=0.78, hspace=0.15)
    out = _unique_path(out_dir / "bias_crossover_by_orography.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [bias-crossover] {out}")


if __name__ == "__main__":
    main()
