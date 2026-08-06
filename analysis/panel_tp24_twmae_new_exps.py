#!/usr/bin/env python3
"""
Panel plots for any score × threshold combination — tp24, j5vo reference.
Usage:
  python analysis/panel_tp24_twmae_new_exps.py          # all scores, all thresholds
  python analysis/panel_tp24_twmae_new_exps.py ETS 30   # specific score + threshold
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

REF = "j5vo"


def _discover_experiments(threshold_mm: int, score: str) -> list[str]:
    """Return all experiments that have a ready heatmap for this score, sorted."""
    label = f"fixed{threshold_mm}"
    found = []
    for d in sorted(RESULTS_DIR.glob(f"tp24_local_{label}_{REF}_*")):
        exp = d.name.replace(f"tp24_local_{label}_{REF}_", "")
        png = d / f"heatmap_smooth_{score}_tp24_{REF}_vs_{exp}_all_conditions.png"
        if png.exists():
            found.append(exp)
    return found


def make_panel(threshold_mm: int, score: str) -> None:
    label = f"fixed{threshold_mm}"
    results_subdir = f"tp24_local_{label}_{REF}_{{exp}}"

    experiments = _discover_experiments(threshold_mm, score)
    if not experiments:
        print(f"  No heatmaps found for {score} {threshold_mm} mm — skipping.")
        return
    print(f"  Found {len(experiments)} experiments: {experiments}")

    # ------------------------------------------------------------------ load
    images = {}
    for exp in experiments:
        p = (RESULTS_DIR / results_subdir.format(exp=exp)
             / f"heatmap_smooth_{score}_tp24_{REF}_vs_{exp}_all_conditions.png")
        images[exp] = mpimg.imread(str(p))

    # Auto-layout: prefer square-ish grid
    n = len(experiments)
    import math
    NCOLS = math.ceil(math.sqrt(n))
    NROWS = math.ceil(n / NCOLS)

    # ------------------------------------------------------------------ figure
    # Preserve original image aspect ratio (H/W) in each cell.
    sample = next(iter(images.values()))
    img_h_px, img_w_px = sample.shape[:2]
    aspect = img_h_px / img_w_px   # ~1.87 (taller than wide)

    cell_w = 4.0                   # inches per cell width
    cell_h = cell_w * aspect       # inches per cell height (keeps ratio)
    title_h = 0.5                  # inches for per-panel title
    suptitle_h = 0.8               # inches for figure super-title

    fig_w = NCOLS * cell_w
    fig_h = NROWS * (cell_h + title_h) + suptitle_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    suptitle = (
        f"{score} — 24h Precipitation, threshold {threshold_mm} mm/24h  |  "
        f"Reference: {REF.upper()}  vs  experiments\n"
        f"(blue = experiment better than {REF.upper()},  red = experiment worse)"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight='bold',
                 y=1.0 - suptitle_h / (2 * fig_h))

    for idx, exp in enumerate(experiments):
        ax = fig.add_subplot(NROWS, NCOLS, idx + 1)
        ax.imshow(images[exp], interpolation='lanczos')
        ax.axis('off')
        ax.set_title(
            f"Ref: {REF.upper()}  vs  {exp.upper()}",
            fontsize=11, fontweight='bold', pad=4,
        )

    plt.tight_layout(rect=[0, 0, 1, 1 - suptitle_h / fig_h])

    out = PLOTS_DIR / f"panel_{score}_tp24_{label}_{REF}_vs_new_exps.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓  Saved: {out}")


if __name__ == "__main__":
    import sys
    scores_to_run = [sys.argv[1]] if len(sys.argv) > 1 else ['twMAE', 'ETS', 'PSS']
    thrs_to_run  = [int(sys.argv[2])] if len(sys.argv) > 2 else [30, 50]
    for score in scores_to_run:
        for thr in thrs_to_run:
            print(f"\n--- Building panel: {score}  tp24 {thr} mm ---")
            make_panel(thr, score)
    print("\nDone.")
