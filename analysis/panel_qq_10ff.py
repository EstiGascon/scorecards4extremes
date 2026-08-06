#!/usr/bin/env python3
"""
Panel QQ plots for 10ff: 3 rows (day 1 / day 5 / day 10) × 3 cols (flat / hilly / complex).
One figure per experiment (9 figures total).

Usage:
  python analysis/panel_qq_10ff.py
"""

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

WORKSPACE = Path(__file__).parent.parent
QQ_DIR    = WORKSPACE / "plots" / "qq"
OUT_DIR   = WORKSPACE / "plots"
OUT_DIR.mkdir(exist_ok=True)

REF  = "j5vo"
EXPERIMENTS = ["j3d0", "j5zr", "j6uz", "j6zg", "j78d", "j78e", "j7ba", "j7bc", "j7bd"]

# row → (day label, day number, step string in filename)
ROWS = [
    ("Day 1",  1,  "6_12_18_24"),
    ("Day 5",  5,  "102_108_114_120"),
    ("Day 10", 10, "222_228_234_240"),
]

# col → (terrain label, orog key used in filename)
COLS = [
    ("Flat",    "low"),
    ("Hilly",   "mid"),
    ("Complex", "high"),
]


def make_panel(exp: str) -> None:
    nrows, ncols = len(ROWS), len(COLS)

    # Load all images first — skip if any missing
    images = {}
    for row_label, daynum, step_str in ROWS:
        for col_label, orog in COLS:
            f = (QQ_DIR
                 / f"10ff_{REF}_vs_{exp}_day{daynum}"
                 / f"qq_10ff_{REF}_vs_{exp}_{orog}_lt{step_str}h.png")
            if not f.exists():
                print(f"  ⚠  Missing: {f}")
                return
            img = mpimg.imread(str(f))
            # Crop top ~5% to remove the baked-in per-plot title
            crop = int(img.shape[0] * 0.05)
            images[(daynum, orog)] = img[crop:, :, :]

    # Figure sizing — preserve image aspect ratio
    sample = next(iter(images.values()))
    img_h, img_w = sample.shape[:2]
    aspect = img_h / img_w

    cell_w    = 5.0          # inches per cell
    cell_h    = cell_w * aspect
    title_h   = 0.55         # per-panel title
    row_lbl_w = 0.7          # left strip for row labels
    sup_h     = 0.9          # suptitle — extra space to avoid overlap
    legend_h  = 0.0          # no extra legend (already in each QQ image)

    fig_w = ncols * cell_w + row_lbl_w
    fig_h = nrows * (cell_h + title_h) + sup_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        f"Q-Q plots — 10m Wind Speed  |  Reference: {REF.upper()}  vs  {exp.upper()}\n"
        f"Threshold: 12 m/s  |  Europe  |  Lead times: Day 1, Day 5, Day 10",
        fontsize=13, fontweight="bold",
        y=1.0 - sup_h / (2 * fig_h),
    )

    # GridSpec: nrows × ncols panels, with a narrow left column for row labels
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        nrows, ncols,
        left   = row_lbl_w / fig_w,
        right  = 0.99,
        top    = 1.0 - sup_h / fig_h,
        bottom = 0.01,
        hspace = title_h / cell_h,
        wspace = 0.03,
    )

    for r, (row_label, daynum, step_str) in enumerate(ROWS):
        for c, (col_label, orog) in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(images[(daynum, orog)], interpolation="lanczos", aspect="auto")
            ax.axis("off")
            if r == 0:
                ax.set_title(col_label, fontsize=12, fontweight="bold", pad=6)

        # Row label on the left
        fig.text(
            row_lbl_w / fig_w / 2,
            1.0 - (sup_h + (r + 0.5) * (cell_h + title_h)) / fig_h,
            row_label,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            rotation=90,
        )

    out = OUT_DIR / f"panel_qq_10ff_{REF}_vs_{exp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  Saved: {out}")


if __name__ == "__main__":
    print("Creating QQ panel plots for 10ff...\n")
    for exp in EXPERIMENTS:
        print(f"--- {exp} ---")
        make_panel(exp)
    print("\nDone.")
