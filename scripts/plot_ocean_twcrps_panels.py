#!/usr/bin/env python
"""One score panel per variable (default twCRPS): Ocean A | Ocean B | GT | T, each vs Control.

Each sub-scorecard has the columns NH FLAT | NH HILLY | NH COMPLEX | TROPICS
(rows = forecast day 1-10), drawn from the scores_by_leadtime CSVs of the
ocean-coupling ensemble runs (configs/ensemble/*vsctrl_FebAug2026_*.yaml).
Colours, colour limit and significance boxes follow plot.create_smooth_multicolumn_heatmap:
cell value = (score_exp - score_control) / score_control * 100 (lower-is-better scores:
twCRPS, Brier, fCRPS, twQS, ...), so negative
(blue) means the experiment is better; a black box marks a significant difference.

Usage:  python scripts/plot_ocean_twcrps_panels.py [--score Brier] [output_dir]
        (default score twCRPS, default output plots/ocean_ens_FebAug2026)
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
sys.path.insert(0, str(REPO_ROOT / "src"))
import plot  # noqa: E402

SCORE = "twCRPS"
MODELS = [("oceanA", "Ocean A"), ("oceanB", "Ocean B"), ("GT", "GT"), ("T", "T")]
VARIABLES = [  # (variable, threshold tag, description)
    ("tp24", "p99obsclim", "24h Precipitation — p99 obs climatology (per station)"),
    ("10ff", "p99obsclim", "10m Wind Speed — p99 obs climatology (per station)"),
    ("mx2t", "p99obsclim", "2m Max Temperature — p99 obs climatology (per station)"),
    ("mn2t", "p1obsclim", "2m Min Temperature — p1 obs climatology (per station, cold tail)"),
]
COLUMNS = [("nhextrop", "flat", "NH\nFLAT"), ("nhextrop", "hilly", "NH\nHILLY"),
           ("nhextrop", "complex", "NH\nCOMPLEX"), ("tropics", None, "TROPICS")]
DAYS = list(range(1, 11))
COLOR_LIMIT = 30.0  # % — colour bar runs from -30 to +30 in every figure

# Error-score palette of plot.create_smooth_multicolumn_heatmap (blue = model 2 better)
PALETTE = ["#0044cf", "#274ed3", "#3a5ad6", "#4865da", "#546fdd", "#5f7be1",
           "#6886e4", "#7291e7", "#7a9ceb", "#83a8ee", "#8ab3f1", "#92c0f4",
           "#99ccf7", "#a0d8fa", "#a7e4fd", "#d3d3d3",
           "#fbd5c0", "#f8c7b1", "#f4bba2", "#f0ad94", "#eba086", "#e69478",
           "#e1876a", "#db7a5d", "#d56d4f", "#cf6043", "#c85235", "#c14329",
           "#b9341c", "#b2210e", "#aa0000"]


def _csv(var, thr, model, region, orog):
    d = REPO_ROOT / "results" / f"{var}_ens_quaver_{thr}_{model}vsctrl_FebAug2026_{region}"
    name = f"scores_by_leadtime_{var}_{orog}.csv" if orog else f"scores_by_leadtime_{var}.csv"
    return pd.read_csv(d / name).set_index("forecast_day").reindex(DAYS)


def _matrices(var, thr, model, score=SCORE):
    pct = np.full((len(DAYS), len(COLUMNS)), np.nan)
    sig = np.zeros_like(pct, dtype=bool)
    for j, (region, orog, _) in enumerate(COLUMNS):
        df = _csv(var, thr, model, region, orog)
        f1, f2 = df[f"{score}_fc1"], df[f"{score}_fc2"]
        pct[:, j] = ((f2 - f1) / f1.replace(0, np.nan) * 100).values
        sig[:, j] = df[f"{score}_is_significant"].fillna(False).astype(bool).values
    return pct, sig


def draw_variable(var, thr, desc, out_dir, score=SCORE):
    mats = {model: _matrices(var, thr, model, score) for model, _ in MODELS}
    limit = COLOR_LIMIT  # same symmetric scale for every variable and experiment
    cmap = LinearSegmentedColormap.from_list("smooth", PALETTE)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.5 * len(MODELS) + 1.2, 7.4))
    fig.subplots_adjust(left=0.06, right=0.91, top=0.80, bottom=0.03, wspace=0.12)

    for k, (ax, (model, label)) in enumerate(zip(axes, MODELS)):
        pct, sig = mats[model]
        im = ax.imshow(np.clip(pct, -limit, limit), cmap=cmap, vmin=-limit, vmax=limit,
                       aspect="equal", interpolation="nearest")
        for i in range(len(DAYS)):
            for j in range(len(COLUMNS)):
                if not np.isnan(pct[i, j]):
                    ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center",
                            fontsize=7.5, fontweight="bold")
                if sig[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="black", lw=2.0, zorder=10))
        for j in range(1, len(COLUMNS)):
            ax.axvline(j - 0.5, color="white", lw=2, zorder=5)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), len(COLUMNS), len(DAYS), fill=False,
                                   edgecolor="black", lw=1.5, zorder=11))
        ax.set_xticks(range(len(COLUMNS)))
        ax.set_xticklabels([c[2] for c in COLUMNS], fontsize=7, fontweight="bold")
        ax.xaxis.set_ticks_position("top")
        ax.set_yticks(range(len(DAYS)))
        ax.set_yticklabels([f"Day {d}" for d in DAYS] if k == 0 else [], fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{label} vs Control", fontsize=11, fontweight="bold", pad=34)

    cax = fig.add_axes([0.93, 0.03, 0.015, 0.77])
    cbar = fig.colorbar(im, cax=cax, extend="both")
    cbar.set_label(f"{score} difference vs Control (%)", fontsize=9)
    fig.suptitle(f"{score} — {desc}\n8-member ensembles, init 00 UTC, "
                 "2026-02-02..04-28 + 05-13..08-23 — blue: experiment better, boxed: significant (95%)",
                 fontsize=11, fontweight="bold")
    out = out_dir / f"{score}_panel_{var}_{thr}_ocean_vs_control_FebAug2026.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    args = sys.argv[1:]
    score = SCORE
    if args[:1] == ["--score"]:
        score, args = args[1], args[2:]
    if score not in plot._ERROR_SCORES | plot._BOUNDED_ERROR_SCORES:
        sys.exit(f"{score}: only lower-is-better scores are supported (blue = experiment better)")
    out_dir = Path(args[0]) if args else REPO_ROOT / "plots" / "ocean_ens_FebAug2026"
    out_dir.mkdir(parents=True, exist_ok=True)
    for var, thr, desc in VARIABLES:
        draw_variable(var, thr, desc, out_dir, score)


if __name__ == "__main__":
    main()
