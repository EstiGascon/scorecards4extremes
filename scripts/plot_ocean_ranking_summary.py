#!/usr/bin/env python
"""Summary plots ranking the four ocean-coupling experiments against Control (twCRPS).

Aggregates the 160 scorecard cells per experiment (4 variables x NH flat/hilly/complex
+ tropics x days 1-10) from the same CSVs as plot_ocean_twcrps_panels.py:
  1. summary_overall_*.png  — significant better/worse cells, mean twCRPS change, mean rank
  2. summary_by_variable_*.png — mean twCRPS change per variable x experiment, with
     the significant better/worse cell counts written in each cell
Blue = better than Control, red = worse (same polarity as the scorecards).

Usage:  python scripts/plot_ocean_ranking_summary.py [output_dir]   (default plots/ocean_ens_FebAug2026)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_ocean_twcrps_panels import COLUMNS, MODELS, PALETTE, VARIABLES, _csv  # noqa: E402

BETTER, WORSE = "#3a5ad6", "#c14329"           # scorecard blue/red (CVD-checked pair)
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
VAR_LABELS = {"tp24": "tp24  p99", "10ff": "10ff  p99", "mx2t": "mx2t  p99", "mn2t": "mn2t  p1"}
SUBTITLE = ("twCRPS vs Control, 8-member ensembles, 2026-02-02..04-28 + 05-13..08-23; "
            "160 cells per experiment = 4 variables × (NH flat, hilly, complex, tropics) × days 1-10")


def load_cells():
    rows = []
    for var, thr, _ in VARIABLES:
        for model, _ in MODELS:
            for region, orog, _ in COLUMNS:
                df = _csv(var, thr, model, region, orog)
                pct = (df["twCRPS_fc2"] - df["twCRPS_fc1"]) / df["twCRPS_fc1"] * 100
                sig = df["twCRPS_is_significant"].fillna(False).astype(bool)
                for day in df.index:
                    rows.append(dict(var=var, model=model, col=orog or "tropics", day=day,
                                     pct=pct[day], sig=sig[day]))
    d = pd.DataFrame(rows)
    d["better"] = d.sig & (d.pct < 0)
    d["worse"] = d.sig & (d.pct > 0)
    d["rank"] = d.groupby(["var", "col", "day"]).pct.rank()   # 1 = lowest twCRPS of the 4
    return d


def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_overall(d, out):
    models = [m for m, _ in MODELS]
    labels = [l for _, l in MODELS]
    g = d.groupby("model").agg(mean_pct=("pct", "mean"), better=("better", "sum"),
                               worse=("worse", "sum"), rank=("rank", "mean")).reindex(models)
    y = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.70, bottom=0.17, wspace=0.35)

    ax = axes[0]
    ax.barh(y, g.better, color=BETTER, height=0.55, label="significantly better")
    ax.barh(y, -g.worse, color=WORSE, height=0.55, label="significantly worse")
    for yi, b, w in zip(y, g.better, g.worse):
        ax.text(b + 1.5, yi, f"{b:.0f}", va="center", color=INK, fontsize=10,
                fontweight="bold" if b == g.better.max() else "normal")
        ax.text(-w - 1.5, yi, f"{w:.0f}", va="center", ha="right", color=INK, fontsize=10,
                fontweight="bold" if w == g.worse.min() else "normal")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-40, 90)
    ax.set_xticks([-30, -15, 0, 15, 30, 45, 60, 75])
    ax.set_xticklabels(["30", "15", "0", "15", "30", "45", "60", "75"])
    ax.set_xlabel("number of cells   (← worse | better →)", color=MUTED)
    ax.set_title("Significant cells vs Control (of 160)", loc="left", fontsize=11, color=INK, pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
              handlelength=1.2, borderaxespad=0.2)

    ax = axes[1]
    ax.barh(y, g.mean_pct, color=BETTER, height=0.55)
    for yi, v in zip(y, g.mean_pct):
        ax.text(v - 0.1, yi, f"{v:.2f}%", va="center", ha="right", color=INK, fontsize=10,
                fontweight="bold" if v == g.mean_pct.min() else "normal")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-5.6, 0)
    ax.set_xlabel("mean twCRPS change (%)   ← better", color=MUTED)
    ax.set_title("Mean change over all cells", loc="left", fontsize=11, color=INK, pad=22)

    ax = axes[2]
    ax.hlines(y, 1, g["rank"], color=GRID, lw=2)
    ax.scatter(g["rank"], y, s=90, color=BETTER, zorder=3, edgecolor="white", linewidth=2)
    for yi, r in zip(y, g["rank"]):
        ax.text(r + 0.06, yi + 0.18, f"{r:.2f}", color=INK, fontsize=10,
                fontweight="bold" if r == g["rank"].min() else "normal")
    ax.set_xlim(1, 4)
    ax.set_xticks([1, 2, 2.5, 3, 4])
    ax.set_xticklabels(["1\nbest", "2", "2.5", "3", "4\nworst"])
    ax.set_xlabel("mean rank among the 4 per cell", color=MUTED)
    ax.set_title("Average rank", loc="left", fontsize=11, color=INK, pad=22)

    for ax in axes:
        _style(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=11, color=INK)
    fig.suptitle("Which experiment is better overall?  (bold = best in each panel)",
                 x=0.08, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.08, 0.86, SUBTITLE, fontsize=9, color=MUTED)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_by_variable(d, out, limit=15.0):
    models = [m for m, _ in MODELS]
    vars_ = [v for v, _, _ in VARIABLES]
    g = d.groupby(["var", "model"]).agg(mean_pct=("pct", "mean"), better=("better", "sum"),
                                        worse=("worse", "sum"))
    mean = g.mean_pct.unstack().reindex(index=vars_, columns=models)
    cmap = LinearSegmentedColormap.from_list("scorecard", PALETTE)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.13, right=0.88, top=0.80, bottom=0.06)
    im = ax.imshow(np.clip(mean.values, -limit, limit), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    for i, v in enumerate(vars_):
        row_best = mean.loc[v].min()
        for j, m in enumerate(models):
            val = mean.loc[v, m]
            b, w = int(g.loc[(v, m), "better"]), int(g.loc[(v, m), "worse"])
            txt_col = "white" if abs(val) > 0.6 * limit else INK
            ax.text(j, i - 0.12, f"{val:+.1f}%", ha="center", va="center", fontsize=13,
                    color=txt_col, fontweight="bold" if val == row_best else "normal")
            ax.text(j, i + 0.22, f"{b} better · {w} worse", ha="center", va="center",
                    fontsize=9, color=txt_col)
    for k in range(1, len(models)):
        ax.axvline(k - 0.5, color="white", lw=3)
    for k in range(1, len(vars_)):
        ax.axhline(k - 0.5, color="white", lw=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([l for _, l in MODELS], fontsize=11, color=INK)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(vars_)))
    ax.set_yticklabels([VAR_LABELS[v] for v in vars_], fontsize=11, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.90, 0.06, 0.02, 0.74])
    cbar = fig.colorbar(im, cax=cax, extend="both")
    cbar.set_label("mean twCRPS change vs Control (%)", color=MUTED)
    cbar.outline.set_visible(False)
    fig.suptitle("Mean twCRPS change by variable  (bold = best in the row)",
                 x=0.13, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.13, 0.885, "Blue = better than Control, red = worse. Counts = cells (of 40) "
             "significantly better / worse than Control at 95%.", fontsize=9, color=MUTED)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "plots" / "ocean_ens_FebAug2026"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_cells()
    plot_overall(d, out_dir / "summary_overall_ocean_vs_control_twCRPS_FebAug2026.png")
    plot_by_variable(d, out_dir / "summary_by_variable_ocean_vs_control_twCRPS_FebAug2026.png")


if __name__ == "__main__":
    main()
