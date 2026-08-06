#!/usr/bin/env python3
"""
Panel QQ plots: all 9 experiments vs j5vo, one per orography type.

Usage:
  python analysis/panel_qq_all_exps.py --day 5 --orog flat
  python analysis/panel_qq_all_exps.py --day 5 --orog complex
  python analysis/panel_qq_all_exps.py --day 1 --orog hilly
"""

import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

WORKSPACE   = Path(__file__).parent.parent
QQ_DIR      = WORKSPACE / "plots" / "qq"
OUT_DIR     = WORKSPACE / "plots"
REF         = "j5vo"
EXPERIMENTS = ["j3d0", "j5zr", "j6uz", "j6zg", "j78d", "j78e", "j7ba", "j7bc", "j7bd"]

DAY_STEPS = {
    1:  "6_12_18_24",
    5:  "102_108_114_120",
    10: "222_228_234_240",
}

OROG_MAP = {
    "flat":    ("low",  "Flat terrain (sdfor 0–40 m)"),
    "hilly":   ("mid",  "Hilly terrain (sdfor 40–120 m)"),
    "complex": ("high", "Complex terrain (sdfor > 120 m)"),
}

LEAD_LABELS = {
    1:  "Day 1 (lead times 6–24h)",
    5:  "Day 5 (lead times 102–120h)",
    10: "Day 10 (lead times 222–240h)",
}


def make_panel(day: int, orog_key: str) -> None:
    steps            = DAY_STEPS[day]
    orog_id, orog_lbl = OROG_MAP[orog_key]

    images = {}
    for exp in EXPERIMENTS:
        f = (QQ_DIR / f"10ff_{REF}_vs_{exp}_day{day}"
             / f"qq_10ff_{REF}_vs_{exp}_{orog_id}_lt{steps}h.png")
        if not f.exists():
            print(f"  ⚠  Missing: {f}")
            return
        img  = mpimg.imread(str(f))
        crop = int(img.shape[0] * 0.05)     # remove baked-in per-plot title
        images[exp] = img[crop:, :, :]

    ncols = 3
    nrows = math.ceil(len(EXPERIMENTS) / ncols)
    sample = next(iter(images.values()))
    aspect = sample.shape[0] / sample.shape[1]

    cell_w = 5.0
    cell_h = cell_w * aspect
    sup_h  = 1.4          # generous top margin so suptitle doesn't overlap

    fig_w = ncols * cell_w
    fig_h = nrows * cell_h + sup_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    panels_top = 1.0 - sup_h / fig_h
    fig.suptitle(
        f"Q-Q plots — 10m Wind Speed  |  {REF.upper()} vs all experiments\n"
        f"{LEAD_LABELS[day]}  |  {orog_lbl}  |  Europe",
        fontsize=13, fontweight="bold",
        y=panels_top + 0.03,
    )

    gs = GridSpec(nrows, ncols,
                  left=0.01, right=0.99,
                  top=panels_top, bottom=0.01,
                  hspace=0.08, wspace=0.03)

    for idx, exp in enumerate(EXPERIMENTS):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(images[exp], interpolation="lanczos", aspect="auto")
        ax.axis("off")
        ax.set_title(f"{REF.upper()} vs {exp.upper()}", fontsize=11,
                     fontweight="bold", pad=4)

    out = OUT_DIR / f"panel_qq_10ff_all_exps_day{day}_{orog_key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day",  type=int, choices=[1, 5, 10], default=5)
    parser.add_argument("--orog", choices=["flat", "hilly", "complex"], default="flat")
    args = parser.parse_args()
    make_panel(args.day, args.orog)
