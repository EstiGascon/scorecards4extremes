"""Compare the AIFS-vs-IFS twCRPS pattern across thresholds (p99 ~32C, 35C, 38C).

mx2t warm extremes, JJA2026 Europe, flat terrain. Shows (A) the twCRPS % difference
vs lead time collapses from a mid-range red hump at the moderate p99 threshold to
uniformly negative (AIFS better) at the more extreme fixed thresholds, and (B) that
this happens because the accuracy penalty (dA) shrinks below the spread credit (dS)
as the threshold moves deeper into the tail.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES = "results"
CASES = [
    ("p99 (~32C)", f"{RES}/mx2t_ens_quaver_p99obsclim_aifsvsifs_JJA2026_europe/scores_by_leadtime_mx2t_JJA_flat.csv", "#c0392b"),
    ("fixed 35C",  f"{RES}/mx2t_ens_quaver_fixed35_aifsvsifs_JJA2026_europe/scores_by_leadtime_mx2t_35.0C_JJA_flat.csv", "#e67e22"),
    ("fixed 38C",  f"{RES}/mx2t_ens_quaver_fixed38_aifsvsifs_JJA2026_europe/scores_by_leadtime_mx2t_38.0C_JJA_flat.csv", "#2471a3"),
]

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("Effect of threshold level on the AIFS-vs-IFS twCRPS — mx2t, JJA2026 Europe, FLAT",
             fontsize=12, fontweight="bold")

# --- Panel A: twCRPS % difference vs lead time, one line per threshold ---
day5 = {}
for label, path, col in CASES:
    d = pd.read_csv(path)
    day = d["forecast_day"].to_numpy()
    pct = (d["twCRPS_fc2"] - d["twCRPS_fc1"]) / d["twCRPS_fc1"] * 100.0
    a.plot(day, pct, "o-", color=col, lw=2, label=label)
    i5 = np.where(day == 5)[0][0]
    day5[label] = (d["twCRPS_accuracy_fc2"][i5] - d["twCRPS_accuracy_fc1"][i5],
                   d["twCRPS_spread_fc2"][i5] - d["twCRPS_spread_fc1"][i5], col)
a.axhline(0, color="k", lw=1)
a.text(9.5, 1.5, "AIFS worse", color="#c0392b", ha="right", fontsize=9)
a.text(9.5, -1.5, "AIFS better", color="#2471a3", ha="right", va="top", fontsize=9)
a.set_title("A) twCRPS %diff (AIFS-IFS) vs lead time")
a.set_xlabel("Forecast day"); a.set_ylabel("% difference")
a.legend(title="threshold"); a.grid(alpha=0.3)

# --- Panel B: dA vs dS at day 5, grouped bars per threshold ---
labels = list(day5.keys())
x = np.arange(len(labels)); w = 0.36
dA = [day5[l][0] for l in labels]
dS = [day5[l][1] for l in labels]
b.bar(x - w/2, dA, w, color="#c0392b", label="dA  accuracy")
b.bar(x + w/2, dS, w, color="#2471a3", label="dS  spread")
for xi, (va, vs) in enumerate(zip(dA, dS)):
    winner = "red\n(AIFS worse)" if va > vs else "blue\n(AIFS better)"
    b.text(xi, max(va, vs) + 0.002, winner, ha="center", fontsize=8,
           color="#c0392b" if va > vs else "#2471a3")
b.set_xticks(x); b.set_xticklabels(labels)
b.set_title("B) day 5: which term dominates?\n(twCRPS = A - S, larger gap wins)")
b.set_ylabel("AIFS - IFS component"); b.legend(); b.grid(alpha=0.3, axis="y")

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{RES}/mx2t_ens_quaver_p99obsclim_aifsvsifs_JJA2026_europe/diag_threshold_comparison_flat.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
