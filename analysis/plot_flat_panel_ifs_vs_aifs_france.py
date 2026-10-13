"""Flat-area summary panel — IFS vs AIFS, JJA2026 mx2t p99, France.

Left  : POD (hit rate) vs lead.
Middle: frequency bias (forecast / observed count) vs lead.
Right : twMAE decomposition (hits / misses / false alarms), IFS vs AIFS bars.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = "results/mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_JJA2026_france"
OUT = f"{BASE}/diag_flat_panel_ifs_vs_aifs.png"

C_HIT = "#5b8c5a"
C_MISS = "#e08b3c"
C_FA = "#c0392b"

df = pd.read_csv(f"{BASE}/scores_by_leadtime_mx2t_JJA_flat.csv")
day = df["forecast_day"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- POD ---
ax = axes[0]
ax.plot(day, df["POD_fc1"], "o-", color="#1f4e79", label="IFS")
ax.plot(day, df["POD_fc2"], "s--", color="#e67e22", label="AIFS")
ax.set_ylabel("POD  (hit rate)")
ax.set_xlabel("Forecast day")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
ax.set_title("Hit rate", fontsize=12, fontweight="bold")

# --- frequency bias ---
ax = axes[1]
fb_ifs = df["POD_fc1"] / (1 - df["FAR_fc1"])
fb_aifs = df["POD_fc2"] / (1 - df["FAR_fc2"])
ax.plot(day, fb_ifs, "o-", color="#1f4e79", label="IFS")
ax.plot(day, fb_aifs, "s--", color="#e67e22", label="AIFS")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_ylabel("Frequency bias\n(forecast / observed count)")
ax.set_xlabel("Forecast day")
ax.set_ylim(0.4, 1.4)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
ax.set_title("Frequency bias", fontsize=12, fontweight="bold")

# --- twMAE decomposition ---
ax = axes[2]
x = np.arange(len(day), dtype=float)
w = 0.38
for off, m in [(-0.5 * w - 0.02, "fc1"), (0.5 * w + 0.02, "fc2")]:
    h = df[f"twMAE_hits_{m}"].to_numpy()
    mi = df[f"twMAE_misses_{m}"].to_numpy()
    fa = df[f"twMAE_FA_{m}"].to_numpy()
    ax.bar(x + off, h, w, color=C_HIT)
    ax.bar(x + off, mi, w, bottom=h, color=C_MISS)
    ax.bar(x + off, fa, w, bottom=h + mi, color=C_FA)
for xi in x:
    ax.text(xi - 0.5 * w - 0.02, -0.006, "I", ha="center", va="top", fontsize=7, color="#333")
    ax.text(xi + 0.5 * w + 0.02, -0.006, "A", ha="center", va="top", fontsize=7, color="#333")
ax.set_xticks(x)
ax.set_xticklabels([int(d) for d in day])
ax.set_xlabel("Forecast day   (I = IFS,  A = AIFS)")
ax.set_ylabel("twMAE  (tail-weighted error)")
ax.grid(axis="y", alpha=0.3)
ax.set_title("twMAE decomposition", fontsize=12, fontweight="bold")
ax.legend(handles=[
    Patch(facecolor=C_HIT, label="hits"),
    Patch(facecolor=C_MISS, label="misses"),
    Patch(facecolor=C_FA, label="false alarms"),
], loc="upper left", fontsize=9)

fig.suptitle("Warm extremes (mx2t > p99), JJA 2026, France — flat terrain, IFS vs AIFS",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
