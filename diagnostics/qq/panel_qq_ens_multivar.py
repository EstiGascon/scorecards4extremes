#!/usr/bin/env python3
"""
panel_qq_ens_multivar.py
========================
Compose a 2×2 panel of already-rendered ensemble Q-Q plots — {2t, 10ff} ×
{low, high orography} — for one member selection (full ensemble / control /
a single perturbed member). Image composition only (mirrors
diagnostics/qq/panel_qq_all_exps.py); it does NOT recompute any statistics, it just
crops and arranges the PNGs produced by diagnostics/plot_qq_ens.py.

Usage
-----
  python diagnostics/qq/panel_qq_ens_multivar.py --member full
  python diagnostics/qq/panel_qq_ens_multivar.py --member control
  python diagnostics/qq/panel_qq_ens_multivar.py --member 1
  python diagnostics/qq/panel_qq_ens_multivar.py --member full control 1   # all at once

Prerequisite: for each member selection, the 4 individual PNGs must already
exist (produced by plot_qq_ens.py --config <cfg> --member <member> --orog low high).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

WORKSPACE = Path(__file__).resolve().parents[2]    # repository root
OUT_DIR = WORKSPACE / "plots"

# (variable, results_dir, fc1_name, fc2_name) — the two datasets sharing the
# ifsens4aifs vs aifs_ens model pair.
DATASETS = [
    ("2t", WORKSPACE / "results" / "2t_ens_local_p99obsclim_aifsvsifs",
     "ifsens4aifs", "aifs_ens"),
    ("10ff", WORKSPACE / "results" / "10ff_ens_local_p98obsclim_aifsvsifs",
     "ifsens4aifs", "aifs_ens"),
]

OROGS = ["low", "high"]

VAR_LABELS = {"2t": "2m Temperature", "10ff": "10m Wind Speed"}
OROG_LABELS = {"low": "Low terrain (sdfor 0\u201340 m)", "high": "Complex terrain (sdfor > 120 m)"}

# member selection -> (filename tag used by plot_qq_ens.py, human label)
MEMBER_TAGS = {
    "full": ("ensfull", "Full ensemble (all members)"),
    "control": ("ctrl", "Control member"),
}


def _member_tag_and_label(member: str):
    if member in MEMBER_TAGS:
        return MEMBER_TAGS[member]
    try:
        n = int(member)
    except ValueError:
        raise SystemExit(f"ERROR: invalid --member value '{member}'")
    return f"mem{n:02d}", f"Member {n}"


def make_panel(member: str) -> None:
    tag, label = _member_tag_and_label(member)

    images = {}
    for var, results_dir, fc1_name, fc2_name in DATASETS:
        for orog in OROGS:
            f = results_dir / f"qq_{var}_{fc1_name}_vs_{fc2_name}_{tag}_{orog}.png"
            if not f.exists():
                print(f"  \u26a0  Missing: {f}")
                print("     Generate it first with diagnostics/plot_qq_ens.py "
                      f"--member {member} --orog {orog}")
                return
            img = mpimg.imread(str(f))
            crop = int(img.shape[0] * 0.075)  # trim the per-plot title, panel adds its own
            images[(var, orog)] = img[crop:, :, :]

    nrows, ncols = len(OROGS), len(DATASETS)
    sample = next(iter(images.values()))
    aspect = sample.shape[0] / sample.shape[1]

    cell_w = 6.0
    cell_h = cell_w * aspect
    sup_h = 1.1

    fig_w = ncols * cell_w
    fig_h = nrows * cell_h + sup_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    panels_top = 1.0 - sup_h / fig_h
    fig.suptitle(
        f"Ensemble Q-Q plots \u2014 {label}  |  IFS-ENS (ifsens4aifs) vs AIFS-ENS (aifs_ens)\n"
        f"2m Temperature (p99) \u00b7 10m Wind Speed (p98)  |  Europe",
        fontsize=13, fontweight="bold", y=0.99,
    )

    gs = GridSpec(nrows, ncols,
                  left=0.01, right=0.99,
                  top=panels_top, bottom=0.01,
                  hspace=0.18, wspace=0.03)

    for r, orog in enumerate(OROGS):
        for c, (var, _, _, _) in enumerate(DATASETS):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(images[(var, orog)], interpolation="lanczos", aspect="auto")
            ax.axis("off")
            ax.set_title(f"{VAR_LABELS[var]}  \u00b7  {OROG_LABELS[orog]}",
                         fontsize=11, fontweight="bold", pad=6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"panel_qq_ens_multivar_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  \u2713  Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", nargs="+", default=["full"],
                        help="One or more of: full, control, <member index>")
    args = parser.parse_args()
    for m in args.member:
        make_panel(m)
