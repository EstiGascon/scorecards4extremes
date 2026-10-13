"""Where does AIFS's lower tail error come from? — IFS vs AIFS, JJA2026 mx2t p99, Europe.

twMAE (tail-weighted MAE, lower is better) splits exactly into three parts:
    twMAE = hits  +  misses  +  false alarms

For every lead we draw two stacked bars side by side: IFS (left) and AIFS (right).
The message is visual: the green (hits) and orange (misses) segments are almost
the same for both models, but the RED false-alarm segment is far smaller for AIFS.
So AIFS's better twMAE is not from catching more extremes — it is only from making
smaller over-forecasts when no extreme occurs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = "results/mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_JJA2026_europe"
OUT = f"{BASE}/diag_twMAE_decomposition_ifs_vs_aifs.png"

C_HIT = "#5b8c5a"   # correct warm-extreme forecasts
C_MISS = "#e08b3c"  # observed extreme the model missed
C_FA = "#c0392b"    # forecast extreme that did not happen (over-forecast)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

for ax, oro in zip(axes, ["flat", "hilly"]):
    df = pd.read_csv(f"{BASE}/scores_by_leadtime_mx2t_JJA_{oro}.csv")
    day = df["forecast_day"].to_numpy()
    x = np.arange(len(day), dtype=float)
    w = 0.38

    for off, m in [(-0.5 * w - 0.02, "fc1"), (0.5 * w + 0.02, "fc2")]:
        h = df[f"twMAE_hits_{m}"].to_numpy()
        mi = df[f"twMAE_misses_{m}"].to_numpy()
        fa = df[f"twMAE_FA_{m}"].to_numpy()
        ax.bar(x + off, h, w, color=C_HIT)
        ax.bar(x + off, mi, w, bottom=h, color=C_MISS)
        ax.bar(x + off, fa, w, bottom=h + mi, color=C_FA)

    # I / A labels under each pair of bars
    for xi in x:
        ax.text(xi - 0.5 * w - 0.02, -0.006, "I", ha="center", va="top",
                fontsize=7, color="#333")
        ax.text(xi + 0.5 * w + 0.02, -0.006, "A", ha="center", va="top",
                fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([int(d) for d in day])
    ax.set_xlabel("Forecast day   (I = IFS,  A = AIFS)")
    ax.set_title(f"{oro.capitalize()} terrain", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if oro == "flat":
        ax.set_ylabel("twMAE  (tail-weighted error, lower = better)")

legend = [
    Patch(facecolor=C_HIT, label="hits — correct warm extremes"),
    Patch(facecolor=C_MISS, label="misses — observed extreme not forecast"),
    Patch(facecolor=C_FA, label="false alarms — extreme forecast, did not happen"),
]
axes[1].legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.95)

fig.suptitle("Why AIFS has the lower tail error (twMAE): it is the false-alarm part, not the hits\n"
             "mx2t > p99, JJA 2026, Europe (IFS vs AIFS deterministic)",
             fontsize=12, fontweight="bold")
fig.text(0.5, 0.005,
         "The green (hits) and orange (misses) blocks are almost identical for IFS and AIFS — AIFS does "
         "NOT detect more extremes.\nThe whole gap is the red block: IFS makes much larger over-forecasts "
         "when no extreme occurs, and this dominates the tail-weighted error.",
         ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
