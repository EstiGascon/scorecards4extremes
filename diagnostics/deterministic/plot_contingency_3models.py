#!/usr/bin/env python3
"""
Hits / misses / false alarms of IFS-oper, iekm and AIFS-oper for extreme 10 m wind.
==================================================================================
Reads the contingency counts written by threshold_sweep_3models.py (raw forecasts,
identical cases for the three models, forecast days 1-5 pooled) and draws, per model,
two bars normalised by the number of observed events:

  observed bar = hits + misses       (height 1; hit share = POD)
  forecast bar = hits + false alarms (height = frequency bias; FA share = FAR)

Rows: orography class. Columns: event threshold (station percentile or fixed m/s).

Usage:
    python diagnostics/deterministic/plot_contingency_3models.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style  # noqa: E402

MODELS = ["IFS-oper", "iekm", "AIFS-oper"]
COLORS = {"IFS-oper": _style.C_FC1, "AIFS-oper": _style.C_FC2, "iekm": "#CC79A7"}
OROG = {"flat": (0, 40), "hilly": (40, 120), "complex": (120, 3000)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/10ff_3model_extremes_d1to5")
    p.add_argument("--percentiles", type=float, nargs="+", default=[95, 98, 99, 99.5])
    p.add_argument("--thresholds", type=float, nargs="+", default=[10, 12, 16])
    p.add_argument("--slide-percentiles", type=float, nargs="+", default=[98, 99])
    return p.parse_args()


def slide_figure(df, pct, out_png):
    """Presentation version: hits % and false-alarm % per terrain, one station percentile."""
    sub = df[(df.lead == "D1-5") & (df.kind == "raw_pct") & (df.level == pct)]
    terrains = list(OROG)
    x = np.arange(len(terrains))
    w = 0.26
    panels = (("Observed extremes forecast\n(hits, %; the rest are misses)",
               lambda t: 100 * t.hits / (t.hits + t.misses)),
              ("Forecast extremes that did not occur\n(false alarms, %)",
               lambda t: 100 * t.false_alarms / (t.hits + t.false_alarms)))
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, (title, fn) in zip(axes, panels):
            for i, m in enumerate(MODELS):
                t = sub[sub.model == m].set_index("terrain").reindex(terrains)
                vals = fn(t).values
                bars = ax.bar(x + (i - 1) * w, vals, w, color=COLORS[m], label=m)
                ax.bar_label(bars, labels=[f"{v:.0f}" for v in vals], padding=3, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([t.capitalize() for t in terrains])
            ax.set_title(title)
            ax.set_ylim(0, 100)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(loc="upper right", frameon=False)
        fig.suptitle(f"10 m wind above the station {pct:g}th percentile, forecast days 1-5",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, "Europe, 6-hourly, 2025-03-01 to 2026-02-28; identical cases for the three models",
                 ha="center", fontsize=11, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def contingency_figure(df, kind, levels, label, out_png):
    sub = df[(df.lead == "D1-5") & (df.kind == kind)]
    fig, axes = plt.subplots(len(OROG), len(levels), figsize=(3.3 * len(levels), 9.5), sharey="row")
    axes = np.atleast_2d(axes)
    x = np.arange(len(MODELS))
    w = 0.36
    for r, terrain in enumerate(OROG):
        lo, hi = OROG[terrain]
        ymax = 1.0
        for c, lvl in enumerate(levels):
            ax = axes[r, c]
            t = sub[(sub.terrain == terrain) & (sub.level == lvl)].set_index("model").reindex(MODELS)
            n_obs = (t.hits + t.misses).astype(float)
            hit, miss, fa = t.hits / n_obs, t.misses / n_obs, t.false_alarms / n_obs
            ax.bar(x - w / 2, hit, w, color=_style.C_HIT, edgecolor="white", lw=0.5)
            ax.bar(x - w / 2, miss, w, bottom=hit, color=_style.C_MISS, edgecolor="white", lw=0.5)
            ax.bar(x + w / 2, hit, w, color=_style.C_HIT, edgecolor="white", lw=0.5)
            ax.bar(x + w / 2, fa, w, bottom=hit, color=_style.C_FA, edgecolor="white", lw=0.5)
            ax.axhline(1.0, color=_style.C_REF, lw=0.8, ls="--")
            for i, m in enumerate(MODELS):
                pod = hit[m]
                far = fa[m] / (hit[m] + fa[m]) if (hit[m] + fa[m]) > 0 else np.nan
                top = max(1.0, hit[m] + fa[m])
                ymax = max(ymax, top)
                ax.text(x[i], top + 0.03, f"POD {pod:.2f}\nFAR {far:.2f}", ha="center", va="bottom", fontsize=7)
                ax.text(x[i] - w / 2, -0.02, "obs", ha="center", va="top", fontsize=6.5, color=_style.C_REF)
                ax.text(x[i] + w / 2, -0.02, "fc", ha="center", va="top", fontsize=6.5, color=_style.C_REF)
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=8)
            ax.tick_params(axis="x", pad=11)
            for tick, m in zip(ax.get_xticklabels(), MODELS):
                tick.set_color(COLORS[m])
                tick.set_fontweight("bold")
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{terrain}, {label(lvl)}\n{int(n_obs.iloc[0]):,} observed events", fontsize=9)
        for ax in axes[r]:
            ax.set_ylim(0, ymax * 1.28)
        axes[r, 0].set_ylabel(f"Fraction of observed events\n({terrain}, sdfor {lo}-{hi} m)")
    handles = [Patch(color=_style.C_HIT, label="hits"), Patch(color=_style.C_MISS, label="misses"),
               Patch(color=_style.C_FA, label="false alarms")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle("Extreme 6-hourly 10 m wind, Europe, 2025-03-01 to 2026-02-28, forecast days 1-5 pooled\n"
                 "obs bar = hits + misses (= 1); fc bar = hits + false alarms (height = frequency bias)",
                 fontsize=10, y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png)
    plt.close(fig)
    print(f"✓ {out_png}")


def main():
    args = parse_args()
    _style.apply_style()
    d = Path(args.dir)
    df = pd.read_csv(d / "scores_3models.csv")
    contingency_figure(df, "raw_pct", sorted(args.percentiles), lambda v: f"> station p{v:g}",
                       d / "contingency_station_percentiles_3models.png")
    contingency_figure(df, "fixed", sorted(args.thresholds), lambda v: f">= {v:g} m/s",
                       d / "contingency_fixed_thresholds_3models.png")
    for pct in args.slide_percentiles:
        slide_figure(df, pct, d / f"slide_hits_falsealarms_p{pct:g}_3models.png")

    s = df[(df.lead == "D1-5") & (df.kind.isin(["raw_pct", "fixed"]))].copy()
    s["POD"] = s.hits / (s.hits + s.misses)
    s["FAR"] = s.false_alarms / (s.hits + s.false_alarms)
    s["miss_rate"] = s.misses / (s.hits + s.misses)
    s["FA_per_obs_event"] = s.false_alarms / (s.hits + s.misses)
    cols = ["kind", "terrain", "level", "model", "n_obs_events", "hits", "misses", "false_alarms",
            "POD", "miss_rate", "FAR", "FA_per_obs_event", "freq_bias"]
    s[cols].round(4).to_csv(d / "contingency_rates_3models.csv", index=False)
    print(f"✓ {d / 'contingency_rates_3models.csv'}")


if __name__ == "__main__":
    main()
