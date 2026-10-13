"""Compose the existing twMAE heatmap PNGs of the three p99 runs into one panel.

Nothing is recomputed — this just tiles the already-saved
heatmap_smooth_twMAE_*_all_conditions.png images side by side.

Run:  python diagnostics/panels/panel_twmae_variable_comparison.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_REPO = Path(__file__).resolve().parents[2]

PANELS = [
    ("results/mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear/"
     "heatmap_smooth_twMAE_mx2t_ifs_oper_vs_aifs1.0_oper_all_conditions.png",
     "mx2t (daily max) — fullyear (JJA = 2025), Europe"),
    ("results/mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_JJA2026_europe/"
     "heatmap_smooth_twMAE_mx2t_ifs_oper_vs_aifs1.0_oper_all_conditions.png",
     "mx2t (daily max) — JJA 2026, Europe"),
    ("results/2t_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear/"
     "heatmap_smooth_twMAE_2t_ifs_oper_vs_aifs1.0_oper_all_conditions.png",
     "2t (daily mean) — fullyear (JJA = 2025), Europe"),
]

fig, axes = plt.subplots(1, len(PANELS), figsize=(7.2 * len(PANELS), 7.2))

for ax, (rel, title) in zip(axes, PANELS):
    ax.imshow(mpimg.imread(_REPO / rel))
    ax.set_title(title, fontsize=13)
    ax.axis("off")

fig.tight_layout()
out = _REPO / "plots" / "panel_twMAE_mx2t_2t_p99_comparison.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
