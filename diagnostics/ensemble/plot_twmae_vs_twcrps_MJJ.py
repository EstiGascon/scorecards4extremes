#!/usr/bin/env python
"""
Explain why deterministic tw-MAE favours AIFS everywhere while ensemble twCRPS
is mixed/neutral for tp24 p99 (MJJ 2026).

Panels (2x2):
  (A) Deterministic tw-MAE diff        (green=AIFS better, red=IFS better)
  (B) Ensemble twCRPS diff (+95% CI)   (green=AIFS better, red=IFS better)
  (D) twCRPS decomposition, lead-avg   (reliability vs tail-spread contributions)
  (E) Tail reach: % members > p99      (IFS solid vs AIFS dashed)

Exact scores come from the production by-leadtime CSVs; the twCRPS decomposition
is recomputed from the parquets using EXACT per-station p99 thresholds (pipeline
threshold function). The decomposition is cached to a .npz for fast re-runs.

Output: plots/twmae_vs_twcrps_MJJ_explained.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pf

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(ROOT, "src"))
from threshold import _compute_local_obs_climatology_threshold  # noqa: E402

DET = os.path.join(ROOT, "results/tp24_method2_marsstvl_p99obsclim_ifsvsaifs_MJJ")
ENS = os.path.join(ROOT, "results/tp24_method2_marsstvl_p99obsclim_ifsensvsaifsens_MJJ")
PARQ = os.path.join(ROOT, "extracted_points/tp24_method2_marsstvl_p99obsclim_ifsensvsaifsens_MJJ")
OUT = os.path.join(ROOT, "plots/twmae_vs_twcrps_MJJ_explained.png")
CACHE = os.path.join(ROOT, "plots/.twmae_decomp_cache.npz")

CONDS = ["low", "mid", "high"]
OROG_RANGE = {"low": (0, 40), "mid": (40, 120), "high": (120, 3000)}
COL = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#2ca02c"}
GREEN, RED = "#2e7d32", "#c62828"

CFG = {"threshold": {"local_obs_climatology": {
    "path": os.path.join(ROOT, "obs_clim_local"), "parameter": "tp", "percentile": 99,
    "window_days": 1, "n_years": 20, "first_year": 2005, "last_year": 2024,
    "min_availability_pct": 65}}}


def load_csv(run, cond):
    return pd.read_csv(os.path.join(run, f"scores_by_leadtime_tp24_MJJ_{cond}.csv"))


def fair_terms(X, obs, T):
    M = X.shape[1]
    vX = np.maximum(X, T[:, None]); vy = np.maximum(obs, T)
    term1 = np.abs(vX - vy[:, None]).mean(axis=1)
    Xs = np.sort(vX, axis=1)
    w = (2 * np.arange(M) - M + 1).astype(np.float64)
    term2 = (Xs * w[None, :]).sum(axis=1) / (M * (M - 1))
    return term1.mean(), term2.mean(), (term1 - term2).mean(), (X > T[:, None]).mean()


def _aggregate_to_daily(od, thr, fc1c, fc2c):
    """Replicate run._aggregate_to_daily_mean (Step 5b, local_obs_climatology):
    group by (lat, lon, date, forecast_day) and average obs + all ensemble
    members to daily means; propagate the per-station threshold via first().
    For 1-step-per-day tp24 this only merges duplicate (lat,lon,date) rows."""
    od = od.copy()
    od["_thr"] = thr
    gcols = ["lat", "lon", "date", "forecast_day"]
    aggc = ["obs_value"] + fc1c + fc2c
    agg = {c: (c, "mean") for c in aggc}
    agg["_thr"] = ("_thr", "first")
    a = od.groupby(gcols, sort=False).agg(**agg).reset_index()
    obs = a["obs_value"].values.astype(np.float64)
    X1 = a[fc1c].values.astype(np.float64)
    X2 = a[fc2c].values.astype(np.float64)
    T = a["_thr"].values.astype(np.float64)
    return X1, X2, obs, T


def decompose():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return list(z["days"]), z["rec"].item()
    days = list(range(1, 11))
    keys = ["t1_ifs", "t2_ifs", "c_ifs", "f_ifs", "t1_aifs", "t2_aifs", "c_aifs", "f_aifs"]
    rec = {c: {k: [] for k in keys} for c in CONDS}
    for d in days:
        df = pf.read_table(os.path.join(PARQ, f"tp24_ifs_ens_oper_vs_aifs_ens_ens_day{d}.parquet")).to_pandas()
        df["forecast_day"] = d
        fc1c = sorted([c for c in df.columns if c.startswith("fc1_member_")], key=lambda c: int(c.split("_")[-1]))
        fc2c = sorted([c for c in df.columns if c.startswith("fc2_member_")], key=lambda c: int(c.split("_")[-1]))
        # --- Ensemble scoring path (run.py per-day streaming), NOT filter.run_step4 ---
        # (1) tp24 QC mirrors the fixed pipeline: the Gaussian 5-sigma obs cut is
        #     SKIPPED for precipitation (it strips the genuine heavy tail); only the
        #     physical max_valid_precipitation cap is applied.
        df = df[df["obs_value"] <= 800.0]
        # (2) season month filter (MJJ = 5,6,7). NO coastal filter.
        mon = pd.to_datetime(df["date"], format="%Y%m%d").dt.month
        df = df[mon.isin([5, 6, 7])].reset_index(drop=True)
        for c in CONDS:
            lo, hi = OROG_RANGE[c]
            # (3) orography filter, then require finite members.
            od = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]
            od = od[np.isfinite(od[fc1c].values).all(axis=1) & np.isfinite(od[fc2c].values).all(axis=1)]
            if len(od) == 0:
                for k in keys:
                    rec[c][k].append(np.nan)
                continue
            # (4) Step 5 threshold on the filtered subset, then (5) Step 5b daily aggregation.
            thr = _compute_local_obs_climatology_threshold(CFG, od).values
            X1, X2, obs, T = _aggregate_to_daily(od, thr, fc1c, fc2c)
            keep = np.isfinite(T)
            X1, X2, obs, T = X1[keep], X2[keep], obs[keep], T[keep]
            if len(T) == 0:
                for k in keys:
                    rec[c][k].append(np.nan)
                continue
            t1a, t2a, ca, fa = fair_terms(X1, obs, T)
            t1b, t2b, cb, fb = fair_terms(X2, obs, T)
            for k, v in zip(keys, [t1a, t2a, ca, fa, t1b, t2b, cb, fb]):
                rec[c][k].append(v)
        print(f"  day{d} done")
    for c in CONDS:
        rec[c] = {k: np.array(v) for k, v in rec[c].items()}
    np.savez(CACHE, days=np.array(days), rec=rec)
    return days, rec


def better_worse_bg(ax):
    """Shade axes: values<0 => AIFS better (green), values>0 => IFS better (red).
    Bold corner labels + a green up-arrow make the winner obvious at a glance."""
    ymin, ymax = ax.get_ylim()
    ymin, ymax = min(ymin, 0), max(ymax, 0)
    pad = 0.12 * (ymax - ymin)
    ymin, ymax = ymin - pad * (ymin < 0), ymax + pad * (ymax > 0)
    ax.set_ylim(ymin, ymax)
    ax.axhspan(ymin, 0, color=GREEN, alpha=0.09, zorder=0)
    ax.axhspan(0, ymax, color=RED, alpha=0.09, zorder=0)
    ax.axhline(0, color="k", lw=1.3, zorder=1)
    ax.text(0.015, 0.03, "AIFS better", transform=ax.transAxes, ha="left", va="bottom",
            color=GREEN, fontweight="bold", fontsize=11,
            bbox=dict(fc="white", ec=GREEN, alpha=0.9, pad=1.5))
    ax.text(0.015, 0.97, "IFS better", transform=ax.transAxes, ha="left", va="top",
            color=RED, fontweight="bold", fontsize=11,
            bbox=dict(fc="white", ec=RED, alpha=0.9, pad=1.5))


def main():
    det = {c: load_csv(DET, c) for c in CONDS}
    ens = {c: load_csv(ENS, c) for c in CONDS}
    lead = det["low"]["forecast_day"].values
    print("twCRPS decomposition (exact thresholds)...")
    days, rec = decompose()

    # ---- VALIDATION: recomputed net twCRPS must match production by-leadtime CSV ----
    print("\nValidation — recomputed twCRPS vs production scorecard CSV (twCRPS_fc1 / twCRPS_fc2):")
    print(f"  {'cond':>4} {'day':>3} {'mine_fc1':>9} {'csv_fc1':>9} {'mine_fc2':>9} {'csv_fc2':>9}")
    max_err = 0.0
    for c in CONDS:
        e = ens[c]
        for i, d in enumerate(days):
            row = e[e["forecast_day"] == d]
            if row.empty:
                continue
            cf1, cf2 = float(row["twCRPS_fc1"].iloc[0]), float(row["twCRPS_fc2"].iloc[0])
            mf1, mf2 = rec[c]["c_ifs"][i], rec[c]["c_aifs"][i]
            max_err = max(max_err, abs(mf1 - cf1), abs(mf2 - cf2))
            if d <= 3 or d == 10:
                print(f"  {c:>4} {d:>3} {mf1:9.5f} {cf1:9.5f} {mf2:9.5f} {cf2:9.5f}")
    print(f"  max |recomputed - CSV| twCRPS over all conds/days = {max_err:.6f}")
    if max_err > 5e-4:
        print("  ⚠ WARNING: recomputed twCRPS does NOT match production within tolerance.")
    else:
        print("  ✓ recomputed twCRPS matches production scorecard.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("tp24  p99  MJJ 2026  (Europe)   —   AIFS vs IFS   (diff = AIFS − IFS)",
                 fontsize=14, fontweight="bold")

    # ---- (A) deterministic tw-MAE ----
    ax = axes[0, 0]
    for c in CONDS:
        ax.plot(lead, det[c]["twMAE_diff"].values, "-o", color=COL[c], label=c, ms=5, zorder=3)
    ax.set_title("(A)  Deterministic  tw-MAE", fontsize=13, fontweight="bold")
    ax.set_xlabel("forecast day"); ax.set_ylabel("Δ tw-MAE (mm)")
    ax.legend(title="orography", fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, zorder=0)
    better_worse_bg(ax)

    # ---- (B) ensemble twCRPS + CI ----
    ax = axes[0, 1]
    for c in CONDS:
        y = ens[c]["twCRPS_diff"].values
        lo = ens[c]["twCRPS_diff_ci_low"].values; hi = ens[c]["twCRPS_diff_ci_high"].values
        ax.plot(lead, y, "-o", color=COL[c], label=c, ms=5, zorder=3)
        ax.fill_between(lead, lo, hi, color=COL[c], alpha=0.15, zorder=2)
    ax.set_title("(B)  Ensemble  twCRPS   (95% CI)", fontsize=13, fontweight="bold")
    ax.set_xlabel("forecast day"); ax.set_ylabel("Δ twCRPS (mm)")
    ax.legend(title="orography", fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, zorder=0)
    better_worse_bg(ax)

    # ---- (D) twCRPS decomposition ----
    ax = axes[1, 0]
    x = np.arange(len(CONDS)); bw = 0.27
    d_rel = [np.nanmean(rec[c]["t1_aifs"] - rec[c]["t1_ifs"]) for c in CONDS]
    d_spr = [-np.nanmean(rec[c]["t2_aifs"] - rec[c]["t2_ifs"]) for c in CONDS]
    d_net = [np.nanmean(rec[c]["c_aifs"] - rec[c]["c_ifs"]) for c in CONDS]
    ax.bar(x - bw, d_rel, bw, label="reliability", color="#4c72b0", zorder=3)
    ax.bar(x, d_spr, bw, label="tail spread", color="#dd8452", zorder=3)
    ax.bar(x + bw, d_net, bw, label="NET twCRPS", color="#333333", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in CONDS])
    ax.set_title("(D)  twCRPS decomposition  (lead-avg)", fontsize=13, fontweight="bold")
    ax.set_ylabel("contribution to Δ twCRPS (mm)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, axis="y", zorder=0)
    better_worse_bg(ax)

    # ---- (E) tail reach ----
    ax = axes[1, 1]
    for c in CONDS:
        ax.plot(days, 100 * rec[c]["f_ifs"], "-o", color=COL[c], ms=5, lw=2, label=f"IFS {c}")
        ax.plot(days, 100 * rec[c]["f_aifs"], "--s", color=COL[c], ms=5, mfc="white", label=f"AIFS {c}")
    ax.set_title("(E)  Tail reach:  % of members > p99", fontsize=13, fontweight="bold")
    ax.set_xlabel("forecast day"); ax.set_ylabel("% members exceeding p99 threshold")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.text(0.015, 0.03, "solid = IFS   ·   dashed = AIFS\nhigher = reaches extremes more",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox=dict(fc="white", ec="0.5", alpha=0.85, pad=2))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=140)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
