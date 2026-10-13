"""Diagnostic figure explaining the twCRPS lead-time crossover (IFS-ENS vs AIFS-ENS).

mx2t p99 JJA2026 Europe, flat terrain. Shows that the sign flip of the heatmap
(blue at short/long lead, red in the middle) is a tug-of-war between the CRPS
accuracy term (AIFS penalty) and the CRPS spread term (AIFS credit).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESDIR = "results/mx2t_ens_quaver_p99obsclim_aifsvsifs_JJA2026_europe"
df = pd.read_csv(f"{RESDIR}/scores_by_leadtime_mx2t_JJA_flat.csv")

day = df["forecast_day"].to_numpy()
ifs = df["twCRPS_fc1"].to_numpy()
aifs = df["twCRPS_fc2"].to_numpy()
pct = (aifs - ifs) / ifs * 100.0

acc_i, acc_a = df["twCRPS_accuracy_fc1"].to_numpy(), df["twCRPS_accuracy_fc2"].to_numpy()
spr_i, spr_a = df["twCRPS_spread_fc1"].to_numpy(), df["twCRPS_spread_fc2"].to_numpy()
d_acc = acc_a - acc_i          # AIFS accuracy penalty (positive = AIFS worse)
d_spr = spr_a - spr_i          # AIFS spread credit (positive = AIFS gets more subtracted)
d_net = d_acc - d_spr          # == aifs - ifs twCRPS

RED, BLUE = "#c0392b", "#2471a3"
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("twCRPS lead-time crossover diagnostic — mx2t p99, JJA2026 Europe, FLAT\n"
             "IFS-ENS (reference) vs AIFS-ENS", fontsize=13, fontweight="bold")

# --- Panel A: twCRPS curves (the crossover) ---
a = ax[0]
a.plot(day, ifs, "o-", color="k", label="IFS-ENS", lw=2)
a.plot(day, aifs, "s--", color="darkorange", label="AIFS-ENS", lw=2)
a.set_title("A) twCRPS vs lead time")
a.set_xlabel("Forecast day"); a.set_ylabel("twCRPS (lower = better)")
a.legend(); a.grid(alpha=0.3)

# --- Panel B: % difference bars (the heatmap row) ---
b = ax[1]
colors = [RED if p > 0 else BLUE for p in pct]
b.bar(day, pct, color=colors)
b.axhline(0, color="k", lw=0.8)
for x, p in zip(day, pct):
    b.text(x, p + (0.6 if p >= 0 else -0.6), f"{p:+.1f}", ha="center",
           va="bottom" if p >= 0 else "top", fontsize=8)
b.set_title("B) AIFS-IFS twCRPS %diff  (red = AIFS worse)")
b.set_xlabel("Forecast day"); b.set_ylabel("% difference")
b.grid(alpha=0.3, axis="y")

# --- Panel C: the tug-of-war (penalty vs credit) ---
c = ax[2]
c.plot(day, d_acc, "^-", color=RED, label="Accuracy  ΔA = A$_{AIFS}$−A$_{IFS}$", lw=2)
c.plot(day, d_spr, "v-", color=BLUE, label="Spread  ΔS = S$_{AIFS}$−S$_{IFS}$", lw=2)
c.plot(day, d_net, "o-", color="k", label="Net twCRPS diff = ΔA − ΔS", lw=2)
c.axhline(0, color="k", lw=0.8)
c.fill_between(day, d_net, 0, where=(d_net > 0), color=RED, alpha=0.15)
c.fill_between(day, d_net, 0, where=(d_net <= 0), color=BLUE, alpha=0.15)
c.set_title("C) twCRPS = A − S")
c.set_xlabel("Forecast day"); c.set_ylabel("AIFS − IFS component")
c.legend(fontsize=8); c.grid(alpha=0.3)

fig.tight_layout(rect=[0, 0.06, 1, 0.92])
fig.text(0.5, 0.015,
         "twCRPS = A \u2212 S (A: member distance to obs; S: ensemble spread). AIFS-ENS has larger A and S than IFS-ENS at all "
         "leads; the larger A dominates at mid leads (red, AIFS worse), the larger S dominates at short/long leads (blue, AIFS better).",
         ha="center", va="bottom", fontsize=9, wrap=True)
out = f"{RESDIR}/diag_twCRPS_crossover_flat.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
