#!/usr/bin/env python3
"""
Generic panel plot: arrange individual score heatmaps into a grid.

Usage:
  python analysis/panel_plot.py <variable> <threshold_label> <score> [score2 ...]

Examples:
  python analysis/panel_plot.py 2t fixed35warm twMAE ETS PSS
  python analysis/panel_plot.py 2t fixedm5cold twMAE ETS PSS
  python analysis/panel_plot.py tp24 fixed30 ETS PSS twMAE
  python analysis/panel_plot.py 10ff fixed12 twMAE ETS PSS

Discovers all experiments automatically from results/.
"""

import sys
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

REF = "j5vo"

# Human-readable labels for suptitle
_VAR_LABELS = {
    "2t":   "2m Temperature",
    "10ff": "10m Wind Speed",
    "tp24": "24h Precipitation",
}
_THRESH_LABELS = {
    "fixed35warm": "> 35 °C (warm extreme)",
    "fixedm5cold": "< −5 °C (cold extreme)",
    "fixed12":     "> 12 m/s (wind extreme)",
    "fixed30":     "> 30 mm/24h",
    "fixed50":     "> 50 mm/24h",
    "p99obsclim":  "p99 obs climatology (above)",
    "p1obsclim":   "p1 obs climatology (below)",
    "p98obsclim":  "p98 obs climatology (above)",
    "p95obsclim":  "p95 obs climatology (above)",
}


def _discover_experiments(variable: str, threshold_label: str, score: str) -> list[str]:
    pattern = f"{variable}_local_{threshold_label}_{REF}_*"
    found = []
    for d in sorted(RESULTS_DIR.glob(pattern)):
        exp = d.name.replace(f"{variable}_local_{threshold_label}_{REF}_", "")
        png = d / f"heatmap_smooth_{score}_{variable}_{REF}_vs_{exp}_all_conditions.png"
        if png.exists():
            found.append(exp)
    return found


def make_panel(variable: str, threshold_label: str, score: str) -> None:
    experiments = _discover_experiments(variable, threshold_label, score)
    if not experiments:
        print(f"  ⚠  No heatmaps found for {variable} {threshold_label} {score} — skipping.")
        return
    print(f"  Found {len(experiments)} experiments: {experiments}")

    # Load images
    images = {}
    for exp in experiments:
        p = (RESULTS_DIR
             / f"{variable}_local_{threshold_label}_{REF}_{exp}"
             / f"heatmap_smooth_{score}_{variable}_{REF}_vs_{exp}_all_conditions.png")
        images[exp] = mpimg.imread(str(p))

    # Grid layout
    n = len(experiments)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    # Figure sizing — preserve exact image aspect ratio
    sample = next(iter(images.values()))
    img_h_px, img_w_px = sample.shape[:2]
    aspect = img_h_px / img_w_px

    cell_w   = 4.0
    cell_h   = cell_w * aspect
    title_h  = 0.55       # space for per-panel title
    sup_h    = 0.45       # tight suptitle strip at top
    legend_h = 0.55       # horizontal legend strip at bottom

    fig_w = ncols * cell_w
    fig_h = nrows * (cell_h + title_h) + sup_h + legend_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    var_label    = _VAR_LABELS.get(variable, variable)
    thresh_label = _THRESH_LABELS.get(threshold_label, threshold_label)
    suptitle = (
        f"{score}  —  {var_label},  threshold {thresh_label}  |  "
        f"Reference: {REF.upper()}  vs  experiments"
    )
    # Place suptitle very close to panels: y fraction = panels top edge
    panels_top = 1.0 - sup_h / fig_h
    fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=panels_top + 0.01)

    for idx, exp in enumerate(experiments):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.imshow(images[exp], interpolation='lanczos', aspect='auto')
        ax.axis('off')
        ax.set_title(
            f"Ref: {REF.upper()}   vs   {exp.upper()}",
            fontsize=12, fontweight='bold', pad=5,
        )

    # Hide unused cells
    for idx in range(len(experiments), nrows * ncols):
        fig.add_subplot(nrows, ncols, idx + 1).axis('off')

    plt.tight_layout(rect=[0, legend_h / fig_h, 1, panels_top])

    # Horizontal legend below all panels
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='#2166ac', label=f'Experiment better than {REF.upper()}')
    red_patch  = mpatches.Patch(color='#d6604d', label=f'Experiment worse than {REF.upper()}')
    white_patch = mpatches.Patch(color='#f7f7f7', edgecolor='grey',
                                 label='No significant difference')
    fig.legend(
        handles=[blue_patch, red_patch, white_patch],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        fontsize=12,
        frameon=True,
        framealpha=0.9,
        edgecolor='grey',
        handlelength=2.0,
        handleheight=1.2,
    )

    out = PLOTS_DIR / f"panel_{score}_{variable}_{threshold_label}_{REF}_vs_exps.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓  Saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    variable        = sys.argv[1]
    threshold_label = sys.argv[2]
    scores          = sys.argv[3:]

    for score in scores:
        print(f"\n--- {variable} / {threshold_label} / {score} ---")
        make_panel(variable, threshold_label, score)

    print("\nDone.")
