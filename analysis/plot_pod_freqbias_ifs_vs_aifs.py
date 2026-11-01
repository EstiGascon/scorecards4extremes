"""POD and frequency-bias vs lead time, IFS vs AIFS — JJA2026 mx2t p99, Europe.

Shows that at long lead times IFS keeps a realistic NUMBER of warm extremes
(frequency bias ~1) but with a lower hit rate (misplaced events), while AIFS
under-forecasts the warm tail (frequency bias < 1) and also detects fewer
events. Frequency bias FB = (events forecast)/(events observed) = POD/(1-FAR).
"""
import pandas as pd
import matplotlib.pyplot as plt

BASE = "results/mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_JJA2026_europe"
OUT = f"{BASE}/diag_POD_freqbias_ifs_vs_aifs.png"

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for col, oro in enumerate(["flat", "hilly"]):
    df = pd.read_csv(f"{BASE}/scores_by_leadtime_mx2t_JJA_{oro}.csv")
    day = df["forecast_day"]
    fb_ifs = df["POD_fc1"] / (1 - df["FAR_fc1"])
    fb_aifs = df["POD_fc2"] / (1 - df["FAR_fc2"])

    # top row: POD
    ax = axes[0, col]
    ax.plot(day, df["POD_fc1"], "o-", color="#1f4e79", label="IFS")
    ax.plot(day, df["POD_fc2"], "s--", color="#e67e22", label="AIFS")
    ax.set_title(f"{oro.capitalize()} terrain", fontsize=12, fontweight="bold")
    ax.set_ylabel("POD  (hit rate)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    if col == 0:
        ax.legend(loc="lower left")

    # bottom row: frequency bias
    ax = axes[1, col]
    ax.plot(day, fb_ifs, "o-", color="#1f4e79", label="IFS")
    ax.plot(day, fb_aifs, "s--", color="#e67e22", label="AIFS")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.text(day.iloc[0], 1.01, "forecasts = observed count", fontsize=8, va="bottom")
    ax.set_ylabel("Frequency bias\n(forecast / observed count)")
    ax.set_xlabel("Forecast day")
    ax.set_ylim(0.4, 1.4)
    ax.grid(alpha=0.3)

fig.suptitle("Warm extremes (mx2t > p99), JJA 2026, Europe — IFS vs AIFS deterministic",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005,
         "Top: hit rate (POD). Bottom: how many warm extremes each model forecasts "
         "relative to the observed number (1 = same count).\n"
         "At long lead IFS keeps a near-observed count (misplaced -> lower POD), "
         "while AIFS under-forecasts the warm tail.",
         ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
