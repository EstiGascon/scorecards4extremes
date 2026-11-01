#!/usr/bin/env python
"""
Detailed deep-dive into the tp24 p99 ensemble twCRPS decomposition (MJJ 2026),
IFS-ENS vs AIFS-ENS.

The fair twCRPS of an M-member ensemble against observation o, in the
extreme-tail "chained" space v(x)=max(x, threshold), decomposes EXACTLY as

    twCRPS  =  member_error  -  ensemble_spread
            =  <|v(x_j) - v(o)|>_j    -    (1/2)<|v(x_j) - v(x_k)|>_{j,k}

Both terms are >= 0. Lower member_error is better (members close to the
observed extreme); larger spread is *rewarded* (subtracted) because a wider
ensemble hedges better — but only if that spread is warranted. A perfectly
reliable ensemble has spread == error; spread < error means the ensemble is
UNDER-dispersive (over-confident) in the tail.

This script visualises, per orography (low/mid/high) and lead time:
  (A,B,C)  member_error (solid) vs ensemble_spread (dashed) for both models —
           the vertical GAP between a model's two curves IS its twCRPS.
  (D)      the resulting twCRPS for both models.
  (E)      the tail dispersion ratio spread/error (=1 perfectly reliable,
           <1 under-dispersive) — the calibration diagnostic.
  (F)      lead-averaged waterfall: how the IFS->AIFS twCRPS change splits
           into a member-error change and a spread change (they nearly cancel).

Reuses the validated decomposition cache written by
analysis/plot_twmae_vs_twcrps_MJJ.py (which reproduces the production
scorecard twCRPS bit-for-bit). Output: plots/twcrps_decomposition_detail_MJJ.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
CACHE = os.path.join(ROOT, "plots/.twmae_decomp_cache.npz")
OUT = os.path.join(ROOT, "plots/twcrps_decomposition_detail_MJJ.png")

CONDS = ["low", "mid", "high"]
COL = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#2ca02c"}
IFS_C, AIFS_C = "#1f77b4", "#d62728"


def get_data():
    if not os.path.exists(CACHE):
        raise SystemExit(
            "Cache not found. Run analysis/plot_twmae_vs_twcrps_MJJ.py first "
            "to generate plots/.twmae_decomp_cache.npz")
    z = np.load(CACHE, allow_pickle=True)
    return list(z["days"]), z["rec"].item()


def main():
    days, rec = get_data()
    x = np.array(days)

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2))
    fig.suptitle(
        "tp24  p99  MJJ 2026 (Europe)  —  ensemble twCRPS decomposition:  "
        "twCRPS = member-error \u2212 ensemble-spread   (fair CRPS, tail-chained)",
        fontsize=14, fontweight="bold")

    # ---- Row 1: error vs spread gap per orography ----
    for j, c in enumerate(CONDS):
        ax = axes[0, j]
        r = rec[c]
        # IFS
        ax.plot(x, r["t1_ifs"], "-", color=IFS_C, lw=2.2, label="IFS member-error")
        ax.plot(x, r["t2_ifs"], "--", color=IFS_C, lw=1.8, label="IFS spread")
        ax.fill_between(x, r["t2_ifs"], r["t1_ifs"], color=IFS_C, alpha=0.10)
        # AIFS
        ax.plot(x, r["t1_aifs"], "-", color=AIFS_C, lw=2.2, label="AIFS member-error")
        ax.plot(x, r["t2_aifs"], "--", color=AIFS_C, lw=1.8, label="AIFS spread")
        ax.fill_between(x, r["t2_aifs"], r["t1_aifs"], color=AIFS_C, alpha=0.10)
        ax.set_title(f"({'ABC'[j]})  {c.upper()} terrain", fontsize=12, fontweight="bold")
        ax.set_xlabel("forecast day")
        if j == 0:
            ax.set_ylabel("tail-chained magnitude (mm)")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=1)
        ax.text(0.97, 0.03,
                "gap (solid\u2212dashed)\n= twCRPS",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))

    # ---- (D) resulting twCRPS both models ----
    ax = axes[1, 0]
    for c in CONDS:
        r = rec[c]
        ax.plot(x, r["c_ifs"], "-o", color=COL[c], ms=4, lw=2, label=f"IFS {c}")
        ax.plot(x, r["c_aifs"], "--s", color=COL[c], ms=4, mfc="white", label=f"AIFS {c}")
    ax.set_title("(D)  Resulting twCRPS  (lower = better)", fontsize=12, fontweight="bold")
    ax.set_xlabel("forecast day"); ax.set_ylabel("twCRPS (mm)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.text(0.03, 0.03, "solid = IFS   ·   dashed = AIFS",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))

    # ---- (E) tail dispersion ratio  E|X-X'| / E|X-o| = 2*term2/term1 ----
    # Reference = 1.0 at perfect calibration (obs exchangeable with members);
    # verified numerically that a calibrated ensemble gives 2*term2/term1 = 1
    # even after the max(x,T) tail-chaining. <1 under-dispersive, >1 over.
    ax = axes[1, 1]
    for c in CONDS:
        r = rec[c]
        ax.plot(x, 2 * r["t2_ifs"] / r["t1_ifs"], "-o", color=COL[c], ms=4, lw=2, label=f"IFS {c}")
        ax.plot(x, 2 * r["t2_aifs"] / r["t1_aifs"], "--s", color=COL[c], ms=4, mfc="white", label=f"AIFS {c}")
    ax.axhline(1.0, color="k", lw=1.2)
    ax.text(0.5, 1.01, "perfectly calibrated tail  (spread = error)", transform=ax.get_yaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="0.3")
    ax.set_title("(E)  Tail dispersion ratio  E|X\u2212X'| / E|X\u2212o|", fontsize=12, fontweight="bold")
    ax.set_xlabel("forecast day")
    ax.set_ylabel("spread / error   (<1 under-dispersive, >1 over-dispersive)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")

    # ---- (F) lead-averaged contributions to the AIFS-IFS twCRPS change ----
    # twCRPS_AIFS - twCRPS_IFS = (+d_err) + (-d_spr).  Each bar = effect on twCRPS.
    ax = axes[1, 2]
    xpos = np.arange(len(CONDS))
    w = 0.36
    for i, c in enumerate(CONDS):
        r = rec[c]
        eff_err = np.nanmean(r["t1_aifs"] - r["t1_ifs"])      # + raises AIFS twCRPS (worse)
        eff_spr = -np.nanmean(r["t2_aifs"] - r["t2_ifs"])     # - lowers AIFS twCRPS (better)
        net = eff_err + eff_spr
        ax.bar(i - w/2, eff_err, w, color="#c62828", alpha=0.9,
               label="member-error effect (AIFS worse \u2192 +)" if i == 0 else None)
        ax.bar(i + w/2, eff_spr, w, color="#2e7d32", alpha=0.9,
               label="spread effect (AIFS more spread \u2192 \u2212)" if i == 0 else None)
        ax.plot(i, net, "kD", ms=9, zorder=5,
                label="net twCRPS change (AIFS\u2212IFS)" if i == 0 else None)
        ax.text(i, net + (0.0006 if net >= 0 else -0.0006), f"{net:+.4f}",
                ha="center", va="bottom" if net >= 0 else "top",
                fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(xpos); ax.set_xticklabels([c.upper() for c in CONDS])
    ax.set_title("(F)  Lead-avg contributions to AIFS\u2212IFS twCRPS",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("effect on twCRPS (mm)   (<0 = AIFS better)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7.8, loc="upper center")
    ax.text(0.02, 0.02,
            "the two large opposing effects\nnearly cancel \u2192 small net",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=140)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
