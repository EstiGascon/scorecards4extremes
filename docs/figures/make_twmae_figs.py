"""
Two explainer figures for twMAE hedgeability, built on the repo's real
src/det_scores.calculate_twmae.

Fig 1 — mechanism: twMAE applies v(x)=max(x,T) to BOTH obs and forecast, so a
        false alarm is charged. An 'obs-only' score ignores it -> hedgeable.
Fig 2 — consequence: a no-skill forecaster sweeping a constant value. twMAE is
        minimised by honest (<=T); the obs-only score is minimised by crying wolf.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import os
# Portable paths: this script lives in <repo>/docs/figures/ and writes PNGs next to itself.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import det_scores as D

OUT = HERE

# Okabe-Ito colourblind-safe palette
BLUE   = "#0072B2"   # forecast / symmetric twMAE
ORANGE = "#E69F00"   # obs-only score
VERM   = "#D55E00"   # false-alarm penalty highlight
GREEN  = "#009E73"   # winner / good
INK    = "#1A1A1A"   # observation / text
GREY   = "#8A8A8A"
TAIL   = "#FBE9D0"   # light tail shading

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#444", "ytick.color": "#444",
})

T = 15.0

def twmae_one(fc, obs):
    return D.calculate_twmae(np.array([fc], float), np.array([obs], float), T, 'above')

def obsonly_one(fc, obs):
    return abs(fc - obs) if obs >= T else 0.0

# ----------------------------------------------------------------------------
# FIGURE 1 — worked examples
# ----------------------------------------------------------------------------
cases = [
    # label,           obs,  fc,  note
    ("HIT\nobs & fc both extreme",         18, 17, "both > T  →  error = |17−18|"),
    ("MISS\nobs extreme, fc calm",         19, 12, "fc snapped up to T  →  |T−19|"),
    ("FALSE ALARM\nfc extreme, obs calm",  12, 19, "fc charged for crying wolf  →  |19−T|"),
    ("CORRECT NEGATIVE\nneither extreme",  11, 12, "both snapped to T  →  0"),
]

fig1, axes = plt.subplots(len(cases), 1, figsize=(13.8, 8.6))
fig1.suptitle("How twMAE scores one case:  v(x) = max(x, T) is applied to BOTH the forecast and the observation",
              fontsize=13.5, fontweight="bold", y=0.985)
fig1.text(0.5, 0.935,
          "The chaining transform 'snaps' any value on the safe side of the threshold up to T, then takes |forecast − observation|.",
          ha="center", fontsize=10.5, color="#555")

XLO, XHI = 8.0, 20.6
for ax, (label, obs, fc, note) in zip(axes, cases):
    is_fa = label.startswith("FALSE ALARM")
    ax.set_xlim(XLO, 34.0); ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    # two lanes: raw (y=0.55) and chained (y=-0.55)
    for y in (0.55, -0.55):
        ax.hlines(y, XLO, XHI, color="#D5D5D5", lw=1.2, zorder=1)
    # tail shading on both lanes
    ax.add_patch(Rectangle((T, -0.95), XHI - T, 1.9, color=TAIL, zorder=0))
    ax.vlines(T, -0.95, 0.95, color=GREY, lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.text(T, 1.0, "T", ha="center", va="bottom", fontsize=10, color=GREY, fontweight="bold")

    vobs, vfc = max(obs, T), max(fc, T)
    # raw markers
    ax.scatter([obs], [0.55], s=130, marker="o", color=INK, zorder=5)
    ax.scatter([fc],  [0.55], s=150, marker="s", color=BLUE, zorder=5)
    ax.text(obs, 0.95, f"obs={obs}", ha="center", fontsize=9.5, color=INK)
    ax.text(fc, 0.95, f"fc={fc}", ha="center", fontsize=9.5, color=BLUE)
    # chained markers
    ax.scatter([vobs], [-0.55], s=130, marker="o", facecolor="white", edgecolor=INK, linewidth=1.8, zorder=5)
    ax.scatter([vfc],  [-0.55], s=150, marker="s", facecolor="white", edgecolor=BLUE, linewidth=1.8, zorder=5)
    # snap arrows (only when value was below T)
    for raw, chd, col in ((obs, vobs, INK), (fc, vfc, BLUE)):
        if raw < T:
            ax.add_patch(FancyArrowPatch((raw, 0.3), (chd, -0.3), arrowstyle="-|>",
                         mutation_scale=11, color=col, lw=1.3, ls=":", zorder=4,
                         connectionstyle="arc3,rad=0.0"))
        else:
            ax.plot([raw, chd], [0.3, -0.3], color=col, lw=0.9, ls=":", zorder=3)
    # contribution bracket on chained lane (neutral colour for every row)
    lo, hi = sorted([vfc, vobs])
    ybr = -0.92
    ax.annotate("", xy=(lo, ybr), xytext=(hi, ybr),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.6))
    contrib = twmae_one(fc, obs)
    ax.text((lo + hi) / 2, ybr - 0.02, f"twMAE cost = {contrib:.0f}",
            ha="center", va="top", fontsize=9.5, color="#555", fontweight="bold")
    # lane labels (first row only)
    if ax is axes[0]:
        ax.text(XLO - 0.15, 0.55, "raw", ha="right", va="center", fontsize=8.5, color="#777")
        ax.text(XLO - 0.15, -0.55, "after\nv=max(x,T)", ha="right", va="center", fontsize=8.5, color="#777")

    # case label (far-left margin, axes-fraction coords) — neutral ink for every row
    ax.text(-0.095, 0.5, label, ha="left", va="center", fontsize=9.6, fontweight="bold",
            color=INK, transform=ax.transAxes, clip_on=False)
    # right-hand cost comparison
    xbox = 21.4
    tw = twmae_one(fc, obs); oo = obsonly_one(fc, obs)
    ax.text(xbox, 0.42, f"twMAE (symmetric):  {tw:.0f}", fontsize=9.8, color=BLUE, va="center")
    ax.text(xbox, -0.1, f"obs-only score:  {oo:.0f}", fontsize=9.8, color=ORANGE, va="center")
    if is_fa:
        # pivotal row: twMAE PENALISES the false alarm (protective, green ✓);
        # the obs-only score IGNORES it (the free pass, red ✗)
        xstat = xbox + 5.0
        ax.text(xstat, 0.42, "✓ penalised → not hedgeable", fontsize=8.9,
                color=GREEN, va="center", ha="left", fontweight="bold")
        ax.text(xstat, -0.1, "✗ ignored → hedgeable", fontsize=8.9,
                color=VERM, va="center", ha="left", fontweight="bold")
        ax.add_patch(Rectangle((xbox - 0.45, -0.42), 12.5, 1.02, fill=False,
                     edgecolor="#999", lw=1.4, ls=(0, (4, 2)), zorder=10))
        ax.text(xbox - 0.45, -0.58, "the pivotal case — the false-alarm penalty is what blocks hedging",
                ha="left", va="top", fontsize=8.4, color="#555", style="italic")

fig1.text(0.5, 0.032,
          "twMAE CHARGES the false alarm (cost 4)  →  over-forecasting back-fires  →  NOT hedgeable",
          ha="center", fontsize=10, color=GREEN, fontweight="bold")
fig1.text(0.5, 0.009,
          "the obs-only score IGNORES it (cost 0)  →  over-forecasting is free  →  hedgeable",
          ha="center", fontsize=10, color=VERM, fontweight="bold")
fig1.subplots_adjust(left=0.115, right=0.99, top=0.905, bottom=0.078, hspace=0.38)
fig1.savefig(f"{OUT}/fig1_twmae_mechanism.png", dpi=200, facecolor="white")
print("saved fig1")

# ----------------------------------------------------------------------------
# FIGURE 2 — the hedging test (no-skill forecaster sweeps a constant forecast)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(1)
mu, sd, N = 10.0, 3.0, 2_000_000
obs = rng.normal(mu, sd, N)
Tsweep = float(np.percentile(obs, 95))
cs = np.linspace(6, 24, 100)
tw = np.array([D.calculate_twmae(np.full(N, c), obs, Tsweep, 'above') for c in cs])
mask = obs >= Tsweep
oo = np.array([np.mean(np.abs(c - obs[mask])) for c in cs])

fig2, ax = plt.subplots(figsize=(11.2, 6.6))
ax.set_xlim(6, 24); ax.set_ylim(0, 11.0)
# honest zone
ax.axvspan(6, Tsweep, color="#EAF3F8", zorder=0)
ax.text(6.25, 0.35, "forecast ≤ T  (honest / 'no event')", fontsize=9, color="#6B8CA6", va="bottom")
ax.axvline(Tsweep, color=GREY, lw=1.3, ls=(0, (4, 3)), zorder=1)
ax.text(Tsweep - 0.18, 9.7, "threshold T (p95)", color=GREY, fontsize=9.5, va="top", ha="right")

ax.plot(cs, tw, color=BLUE, lw=2.8, zorder=4, label="twMAE (symmetric — your code)")
ax.plot(cs, oo, color=ORANGE, lw=2.8, ls="--", zorder=4, label="obs-only score (asymmetric)")

# honest & crier vertical guides + top labels
def nearest(cv):
    return int(np.argmin(np.abs(cs - cv)))
iH, iC = nearest(10), nearest(16)
for i, name in ((iH, "HONEST  (c=10)"), (iC, "CRIER  (c=16)")):
    ax.axvline(cs[i], color="#BBB", lw=1.0, ls=":", zorder=1)
    ax.text(cs[i], 10.78, name, ha="center", va="top", fontsize=9.3, color="#555", fontweight="bold")

# four point markers + inline value labels
ax.scatter([cs[iH]], [tw[iH]], s=95, color=BLUE, zorder=6, ec="white", lw=1.6)
ax.scatter([cs[iH]], [oo[iH]], s=95, color=ORANGE, zorder=6, ec="white", lw=1.6)
ax.scatter([cs[iC]], [tw[iC]], s=95, color=BLUE, zorder=6, ec="white", lw=1.6)
ax.scatter([cs[iC]], [oo[iC]], s=95, color=ORANGE, zorder=6, ec="white", lw=1.6)
ax.text(cs[iH] - 0.25, tw[iH] + 0.45, f"twMAE ≈ 0\n(great)", ha="right", va="bottom", fontsize=8.8, color=BLUE)
ax.text(cs[iH] - 0.25, oo[iH], f"obs-only = {oo[iH]:.1f}\n(bad)", ha="right", va="center", fontsize=8.8, color=ORANGE)
ax.text(cs[iC] + 0.3, tw[iC] + 0.5, f"twMAE = {tw[iC]:.1f}\n(worse, and rising)", ha="left", va="bottom", fontsize=8.8, color=BLUE)
ax.text(cs[iC] + 0.3, oo[iC] - 0.15, f"obs-only = {oo[iC]:.1f}  ← best!", ha="left", va="top", fontsize=8.8, color=ORANGE, fontweight="bold")

# curve takeaways in clear areas
ax.text(17.2, 8.6,
        "twMAE: lowest when honest (≤ T);\nover-forecasting is strictly PENALISED\n→  cannot be hedged",
        fontsize=10.2, color=BLUE, fontweight="bold", va="top", linespacing=1.4)
ax.annotate("obs-only: lowest when you shout 'extreme'\n→ REWARDS over-forecasting  →  hedgeable",
            xy=(cs[iC] - 0.1, oo[iC] + 0.15), xytext=(9.3, 3.1), fontsize=10.2, color=ORANGE,
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3), fontweight="bold",
            va="center", linespacing=1.4)

ax.set_xlabel("Constant value issued by a no-skill forecaster trying to game the score", fontsize=11)
ax.set_ylabel("Score  (lower = better, same units as the variable)", fontsize=11)
ax.set_title("The hedging test: can a forecaster improve the score by over-forecasting?",
             fontsize=13.5, fontweight="bold", pad=26)
ax.text(0.5, 1.012, "No-skill forecaster, event base rate ≈ 5%. The honest best guess (climatological median) sits below T.",
        transform=ax.transAxes, ha="center", fontsize=10, color="#555")
ax.legend(loc="upper left", frameon=False, fontsize=10.2, bbox_to_anchor=(0.012, 0.80))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig2_hedging_test.png", dpi=200, facecolor="white")
print("saved fig2")
print("DONE")
