"""Diagnostic evidence for the medium-range twCRPS flip (AIFS-ENS vs IFS-ENS).

mx2t, p99 obs-climatology threshold, JJA 2026, Europe. Works from the raw
member-level parquet (51 members per model) and the SAME per-station p99
thresholds the scorecard used. Produces four evidence panels:

  A) Reliability diagram for P(mx2t > p99) at medium range (D+5):
     demonstrates AIFS over-forecasting (curve below the diagonal).
  B) Reliability diagram at long range (D+9):
     demonstrates convergence toward climatology / removal of the wet-warm bias.
  C) Forecast exceedance frequency vs observed base rate, per lead time:
     demonstrates the over-forecast bias maximising in the medium range.
  D) Ensemble spread vs lead with the climatological spread of the observations:
     demonstrates relaxation toward the climatological attractor and AIFS's
     systematically larger dispersion.
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from threshold import _compute_local_obs_climatology_threshold

EXTR = "extracted_points/mx2t_ens_quaver_p99obsclim_aifsvsifs_JJA2026_europe"
RES = "results/mx2t_ens_quaver_p99obsclim_aifsvsifs_JJA2026_europe"
PATTERN = "mx2t_ifsens4aifs_vs_aifs_ens_ens_day{d}.parquet"

FC1_COLS = [f"fc1_member_{i}" for i in list(range(1, 51)) + [0]]  # IFS-ENS, 51 members
FC2_COLS = [f"fc2_member_{i}" for i in list(range(1, 51)) + [0]]  # AIFS-ENS, 51 members
BASE_COLS = ["date", "step", "lat", "lon", "obs_value", "station_id"]

THR_CFG = {"threshold": {"local_obs_climatology": {
    "path": "./obs_clim_local", "parameter": "mx2t", "window_days": 1,
    "n_years": 20, "first_year": 2005, "last_year": 2024,
    "min_availability_pct": 65, "percentile": 99}}}

REL_BINS = np.linspace(0, 1, 11)          # 10 probability bins for reliability
NRANK = 52                                 # 51 members -> 52 possible ranks
DAYS = list(range(1, 11))
MED, LONG = 5, 9

# accumulators
lead = {d: {} for d in DAYS}
rel = {MED: {}, LONG: {}}   # per day: model -> (sum_obs per bin, count per bin)


def exceed_prob(members, thr):
    return (members > thr[:, None]).mean(axis=1)


for d in DAYS:
    fp = os.path.join(EXTR, PATTERN.format(d=d))
    cols = BASE_COLS + FC1_COLS + FC2_COLS
    df = pd.read_parquet(fp, columns=cols)
    thr = _compute_local_obs_climatology_threshold(THR_CFG, df).to_numpy()
    obs = df["obs_value"].to_numpy(dtype=float)
    m1 = df[FC1_COLS].to_numpy(dtype=float)
    m2 = df[FC2_COLS].to_numpy(dtype=float)
    del df
    ok = np.isfinite(thr) & np.isfinite(obs) & np.isfinite(m1).all(1) & np.isfinite(m2).all(1)
    thr, obs, m1, m2 = thr[ok], obs[ok], m1[ok], m2[ok]

    obs_event = (obs > thr).astype(float)
    p1 = exceed_prob(m1, thr)
    p2 = exceed_prob(m2, thr)

    lead[d]["base_rate"] = float(obs_event.mean())
    lead[d]["Pexc_fc1"] = float(p1.mean())
    lead[d]["Pexc_fc2"] = float(p2.mean())
    lead[d]["spread_fc1"] = float(m1.std(axis=1, ddof=1).mean())
    lead[d]["spread_fc2"] = float(m2.std(axis=1, ddof=1).mean())
    lead[d]["obs_clim_std"] = float(obs.std(ddof=1))
    lead[d]["n"] = int(ok.sum())

    if d in (MED, LONG):
        for name, p in (("IFS-ENS", p1), ("AIFS-ENS", p2)):
            idx = np.clip(np.digitize(p, REL_BINS) - 1, 0, len(REL_BINS) - 2)
            sum_obs = np.bincount(idx, weights=obs_event, minlength=len(REL_BINS) - 1)
            cnt = np.bincount(idx, minlength=len(REL_BINS) - 1)
            mean_p = np.bincount(idx, weights=p, minlength=len(REL_BINS) - 1)
            rel[d][name] = (sum_obs, cnt, mean_p)

    del m1, m2, p1, p2, obs, thr, obs_event
    gc.collect()
    print(f"day {d}: n={lead[d]['n']} base={lead[d]['base_rate']:.4f} "
          f"P1={lead[d]['Pexc_fc1']:.4f} P2={lead[d]['Pexc_fc2']:.4f}", flush=True)

# ------------------------------------------------------------------ plotting
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
C1, C2 = "#2471a3", "#c0392b"   # IFS blue, AIFS red


def plot_reliability(ax, day, title):
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="1:1 line")
    for name, col in (("IFS-ENS", C1), ("AIFS-ENS", C2)):
        sum_obs, cnt, mean_p = rel[day][name]
        m = cnt > 0
        fx = mean_p[m] / cnt[m]
        fy = sum_obs[m] / cnt[m]
        ax.plot(fx, fy, "-o", color=col, lw=2.2, ms=6, label=name)
    br = lead[day]["base_rate"]
    ax.axhline(br, color="gray", ls=":", lw=1, label=f"observed base rate ({br:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability  P(mx2t > p99)")
    ax.set_ylabel("Observed relative frequency")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)


plot_reliability(axes[0, 0], MED, f"(A) Reliability diagram, D+{MED}")
plot_reliability(axes[0, 1], LONG, f"(B) Reliability diagram, D+{LONG}")

# Panel C: forecast exceedance frequency vs observed base rate
ax = axes[1, 0]
dd = np.array(DAYS)
base = np.array([lead[d]["base_rate"] for d in DAYS])
P1 = np.array([lead[d]["Pexc_fc1"] for d in DAYS])
P2 = np.array([lead[d]["Pexc_fc2"] for d in DAYS])
ax.plot(dd, base, "-k", lw=2.4, marker="D", ms=6, label="observed base rate")
ax.plot(dd, P1, "-o", color=C1, lw=2.2, ms=6, label="IFS-ENS forecast freq.")
ax.plot(dd, P2, "-o", color=C2, lw=2.2, ms=6, label="AIFS-ENS forecast freq.")
ax.fill_between(dd, base, P2, where=(P2 >= base), color=C2, alpha=0.15)
ax.set_xlabel("Forecast lead time (days)")
ax.set_ylabel("Mean forecast P(exceed p99)")
ax.set_title("(C) Mean forecast exceedance probability vs lead time", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3); ax.set_xticks(dd)

# Panel D: ensemble spread vs lead + climatological spread of obs
ax = axes[1, 1]
S1 = np.array([lead[d]["spread_fc1"] for d in DAYS])
S2 = np.array([lead[d]["spread_fc2"] for d in DAYS])
climstd = np.array([lead[d]["obs_clim_std"] for d in DAYS])
ax.plot(dd, S1, "-o", color=C1, lw=2.2, ms=6, label="IFS-ENS spread")
ax.plot(dd, S2, "-o", color=C2, lw=2.2, ms=6, label="AIFS-ENS spread")
ax.axhline(climstd.mean(), color="gray", ls="--", lw=1.5,
           label=f"observed climatological std ({climstd.mean():.2f} K)")
ax.set_xlabel("Forecast lead time (days)")
ax.set_ylabel("Mean ensemble spread (std, K)")
ax.set_title("(D) Mean ensemble spread vs lead time", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3); ax.set_xticks(dd)

fig.suptitle("mx2t exceedance of p99 obs-climatology threshold "
             "\u2014 IFS-ENS vs AIFS-ENS, JJA 2026, Europe", fontsize=13, fontweight="bold", y=1.005)
fig.tight_layout()
out = os.path.join(RES, "diag_flip_mechanism.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out, flush=True)
