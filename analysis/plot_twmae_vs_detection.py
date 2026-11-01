"""Why twMAE and detection scores disagree — IFS vs AIFS, JJA2026 mx2t p99, Europe.

All three curves are drawn so that POSITIVE = AIFS better than IFS:
  - twMAE skill   : magnitude score (lower is better) -> -twMAE_diff / twMAE_fc1
  - PSS skill     : detection score (higher is better) ->  PSS_diff  / |PSS_fc1|
  - POD (hit rate): detection      (higher is better) ->  POD_diff  / POD_fc1

The point: at long lead the magnitude score (twMAE) says AIFS is much better,
while the detection scores (PSS, POD) say AIFS is WORSE. They measure different
things — twMAE rewards AIFS for avoiding large false-alarm overshoots, not for
detecting more events.
"""
import pandas as pd
import matplotlib.pyplot as plt

BASE = "results/mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_JJA2026_europe"
OUT = f"{BASE}/diag_twMAE_vs_detection_ifs_vs_aifs.png"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

for ax, oro in zip(axes, ["flat", "hilly"]):
    df = pd.read_csv(f"{BASE}/scores_by_leadtime_mx2t_JJA_{oro}.csv")
    day = df["forecast_day"]
    twmae = -100 * df["twMAE_diff"] / df["twMAE_fc1"]
    pss = 100 * df["PSS_diff"] / df["PSS_fc1"].abs()
    pod = 100 * df["POD_diff"] / df["POD_fc1"]

    ax.axhline(0, color="k", lw=0.8)
    ax.plot(day, twmae, "o-", color="#2471a3", lw=2, label="twMAE (magnitude) — lower is better")
    ax.plot(day, pss, "s--", color="#c0392b", label="PSS (detection skill)")
    ax.plot(day, pod, "^:", color="#e67e22", label="POD (hit rate)")
    ax.fill_between(day, 0, twmae, where=(twmae > 0), color="#2471a3", alpha=0.08)

    ax.set_title(f"{oro.capitalize()} terrain", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast day")
    ax.grid(alpha=0.3)
    if oro == "flat":
        ax.set_ylabel("AIFS relative to IFS  (%)\n↑ AIFS better   ↓ AIFS worse")
        ax.legend(loc="upper left", fontsize=9)

fig.suptitle("Magnitude vs detection: why twMAE favours AIFS while detection does not\n"
             "mx2t > p99, JJA 2026, Europe (IFS vs AIFS deterministic)",
             fontsize=12, fontweight="bold")
fig.text(0.5, 0.005,
         "At long lead the magnitude score (twMAE, blue) rises — AIFS better — while the detection "
         "scores (PSS red, POD orange) fall below zero — AIFS worse.\n"
         "twMAE improves because AIFS avoids large false-alarm overshoots, not because it detects "
         "more warm extremes.",
         ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
