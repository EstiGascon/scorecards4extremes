#!/usr/bin/env python3
"""
TAIL DISTRIBUTION DIAGNOSTICS
==============================
Investigates *why* twCRPS and Brier scores differ between two ensemble models
by examining how their forecast distributions fit the observed extreme tail.

Four panels
-----------
  (A) Rank histogram (Talagrand diagram) — conditioned on extreme events
      Rank of obs among all 51 members.
      Flat = calibrated;  peak at 51 = obs often above all members (under-dispersive /
      ensemble under-predicts extremes);  peak at 0 = ensemble over-predicts.

  (B) Tail CDF — conditioned on extreme events
      Empirical CDF of pooled ensemble member values vs observed values, across
      extreme days only.  Shows whether the ensemble places mass in the right part
      of the tail, not just whether its 99th percentile clears the observation.

  (C) Quantile score by quantile level — ALL events
      Mean pinball loss QS_τ at each quantile τ ∈ [0.01, 0.99].
      The shaded tail region (where ensemble quantile ≥ threshold) directly
      contributes to twCRPS = 2∫ QS_τ · w(q_τ) dτ over the tail.
      Wherever the fc2 line is below fc1, fc2 is more accurate at that quantile.

  (D) Reliability diagram — ALL events
      Forecast probability of exceeding the threshold (k/51 members beyond
      threshold) vs observed frequency.  Diagonal = perfect reliability.
      Over-dispersive models will be spread across many probability bins;
      under-dispersive ones cluster near 0.

Usage examples
--------------
  # tp24: heavy precipitation ≥ 30 mm
  python3 plot_tail_diagnostics.py --variable tp24 --event-type warm --threshold-value 30

  # 2t: cold extremes, DJF only, high terrain
  python3 plot_tail_diagnostics.py --variable 2t --event-type cold --percentile 1 \\
      --season DJF --orog-types high

  # 10ff: high-wind extremes, days 1 and 5 only
  python3 plot_tail_diagnostics.py --variable 10ff --event-type warm --percentile 99 \\
      --lead-days 1,5

  # Low terrain only (to match the twCRPS result discussed)
  python3 plot_tail_diagnostics.py --variable tp24 --event-type warm --threshold-value 30 \\
      --orog-types low

Options
-------
  --data-dir DIR        Root of extracted_points/        [default: ./extracted_points]
  --variable VAR        2t | 10ff | tp24                 [default: first found]
  --event-type TYPE     cold | warm                      [default: warm]
  --percentile N        Obs percentile for threshold     [default: 1 cold / 99 warm]
  --threshold-value V   Fixed threshold (overrides percentile)
  --season S            DJF | MAM | JJA | SON | ALL     [default: ALL]
  --orog-types LIST     low,mid,high (comma-separated)   [default: low,mid,high]
  --lead-days LIST      e.g. 1,3,5                       [default: all available]
  --output FILE         Output path
  --dpi N               Figure resolution                [default: 150]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ─────────────────────────────────────────────────────────────────
OROG_BINS = {"low": (0, 40), "mid": (40, 120), "high": (120, 9999)}
SEASON_MONTHS = {
    "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},  "SON": {9, 10, 11},
    "ALL": set(range(1, 13)),
}
VARIABLE_UNITS  = {"2t": "K",  "10ff": "m/s", "tp24": "mm"}
VARIABLE_LABELS = {"2t": "2m Temperature", "10ff": "10m Wind Speed",
                   "tp24": "24h Precipitation"}
FC1_CLR = "#1f77b4"   # blue
FC2_CLR = "#ff7f0e"   # orange
OBS_CLR = "#2ca02c"   # green


# ── Data helpers ───────────────────────────────────────────────────────────────
def discover(data_dir: Path) -> dict:
    ds = {}
    for d in sorted(data_dir.iterdir()):
        if not (d.is_dir() and d.name.endswith("_ens")):
            continue
        pq = sorted(d.glob("*.parquet"))
        if not pq:
            continue
        m = re.match(r"^(.+?)_(.+?)_vs_(.+?)_ens_day\d+\.parquet$", pq[0].name)
        if not m:
            continue
        var, m1, m2 = m.group(1), m.group(2), m.group(3)
        days = sorted(int(re.search(r"_day(\d+)", f.name).group(1)) for f in pq)
        ds[var] = {"models": (m1, m2), "days": days, "dir": d}
    return ds


def load_parquets(ens_dir: Path, lead_days: list) -> pd.DataFrame:
    dfs = []
    for f in sorted(ens_dir.glob("*.parquet")):
        day = int(re.search(r"_day(\d+)", f.name).group(1))
        if lead_days and day not in lead_days:
            continue
        df = pd.read_parquet(f)
        df["_ld"] = day
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No parquet files loaded from {ens_dir}")
    df = pd.concat(dfs, ignore_index=True)
    df["_month"] = pd.to_datetime(df["valid_time"], format="%Y%m%d").dt.month
    return df.dropna(subset=["obs_value"])


def member_cols(df: pd.DataFrame, prefix: str) -> list:
    return sorted(
        [c for c in df.columns if c.startswith(f"{prefix}_member_")],
        key=lambda c: int(c.split("_")[-1]),
    )


# ── Diagnostic computations ────────────────────────────────────────────────────
def rank_histogram(obs: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Relative frequency of obs rank among members (0 = below all, K = above all)."""
    K = members.shape[1]
    ranks = np.sum(members < obs[:, None], axis=1)   # 0 .. K
    counts, _ = np.histogram(ranks, bins=np.arange(K + 2))
    return counts / counts.sum()


def tail_cdf(
    obs_ext: np.ndarray, members_ext: np.ndarray,
    threshold: float, event_type: str,
) -> tuple:
    """Empirical CDFs of pooled members and obs, strictly in the extreme tail.
    For warm: x spans [threshold, max_value].
    For cold: x spans [min_value, threshold].
    """
    pm = np.sort(members_ext.flatten())
    po = np.sort(obs_ext)
    if event_type == "warm":
        lo = threshold
        hi_raw = max(pm[-1], po[-1])
        pm_range = max(hi_raw - pm[0], 0.5)
        hi = hi_raw + pm_range * 0.02
        xx = np.linspace(lo, hi, 400)
        # fraction of pooled members / obs that are ≤ x
        cdf_m = np.searchsorted(pm, xx, side="right") / len(pm)
        cdf_o = np.searchsorted(po, xx, side="right") / len(po)
        # re-normalise so both start at 0 at the threshold
        base_m = np.searchsorted(pm, lo, side="right") / len(pm)
        base_o = np.searchsorted(po, lo, side="right") / len(po)
        cdf_m = cdf_m - base_m
        cdf_o = cdf_o - base_o
    else:
        pm_range = pm[-1] - pm[0]
        margin = max(pm_range * 0.02, 0.5)
        lo = min(pm[0], po[0]) - margin
        hi = threshold
        xx = np.linspace(lo, hi, 400)
        cdf_m = np.searchsorted(pm, xx, side="right") / len(pm)
        cdf_o = np.searchsorted(po, xx, side="right") / len(po)
    return xx, cdf_m, cdf_o


def quantile_scores(
    obs: np.ndarray, m1: np.ndarray, m2: np.ndarray,
    n_taus: int = 99, chunk: int = 8000,
) -> tuple:
    """Mean pinball loss at each quantile level, chunked for memory efficiency."""
    taus  = np.linspace(0.01, 0.99, n_taus)
    q_lvl = taus * 100
    n     = len(obs)
    qs1   = np.zeros(n_taus)
    qs2   = np.zeros(n_taus)
    for s in range(0, n, chunk):
        e  = min(s + chunk, n)
        ob = obs[s:e][None, :]                          # (1, sz)
        q1 = np.percentile(m1[s:e], q_lvl, axis=1)    # (n_taus, sz)
        q2 = np.percentile(m2[s:e], q_lvl, axis=1)
        tc = taus[:, None]                              # (n_taus, 1)
        for qs, qq in [(qs1, q1), (qs2, q2)]:
            err = ob - qq
            qs += np.sum(
                np.where(err >= 0, tc * err, (tc - 1) * err), axis=1
            )
    return taus, qs1 / n, qs2 / n


def reliability_diagram(
    obs: np.ndarray, m1: np.ndarray, m2: np.ndarray,
    threshold: float, event_type: str, n_bins: int = 10,
) -> tuple:
    """Forecast probability vs observed frequency at the threshold."""
    if event_type == "warm":
        p1  = np.mean(m1 >= threshold, axis=1)
        p2  = np.mean(m2 >= threshold, axis=1)
        hit = (obs >= threshold).astype(float)
    else:
        p1  = np.mean(m1 <= threshold, axis=1)
        p2  = np.mean(m2 <= threshold, axis=1)
        hit = (obs <= threshold).astype(float)

    edges   = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r1   = np.full(n_bins, np.nan)
    r2   = np.full(n_bins, np.nan)
    cnt1 = np.zeros(n_bins, int)
    cnt2 = np.zeros(n_bins, int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mk1 = (p1 >= lo) & (p1 < hi)
        mk2 = (p2 >= lo) & (p2 < hi)
        cnt1[i] = mk1.sum()
        cnt2[i] = mk2.sum()
        if cnt1[i] > 10:
            r1[i] = hit[mk1].mean()
        if cnt2[i] > 10:
            r2[i] = hit[mk2].mean()
    clim = hit.mean()
    return centers, r1, cnt1, r2, cnt2, clim


def murphy_decomposition(
    obs: np.ndarray, members: np.ndarray,
    threshold: float, event_type: str, n_bins: int = 20,
) -> dict:
    """Brier Score Murphy (1973) decomposition:  BS = REL − RES + UNC

    REL (Reliability)  : weighted mean squared distance from reliability diagonal
                         lower is better (= more calibrated)
    RES (Resolution)   : weighted mean squared distance from climatology frequency
                         higher is better (= more discriminating)
    UNC (Uncertainty)  : climatological variance — same for both models, irreducible
    BS                 : total Brier Score = REL − RES + UNC  (lower is better)
    BSS (Brier SS)     : 1 − BS/UNC  (1 = perfect, 0 = climatology, <0 = worse)
    """
    if event_type == "warm":
        prob = np.mean(members >= threshold, axis=1)
        hit  = (obs >= threshold).astype(float)
    else:
        prob = np.mean(members <= threshold, axis=1)
        hit  = (obs <= threshold).astype(float)

    clim = float(hit.mean())
    n    = len(obs)

    edges = np.linspace(0, 1, n_bins + 1)
    rel = 0.0
    res = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi)
        n_k  = int(mask.sum())
        if n_k < 2:
            continue
        p_bar_k = float(prob[mask].mean())
        y_bar_k = float(hit[mask].mean())
        rel += n_k * (p_bar_k - y_bar_k) ** 2
        res += n_k * (y_bar_k - clim) ** 2
    rel /= n
    res /= n
    unc  = clim * (1.0 - clim)
    bs   = rel - res + unc
    bss  = 1.0 - bs / unc if unc > 0 else float("nan")
    return {"REL": rel, "RES": res, "UNC": unc, "BS": bs, "BSS": bss, "clim": clim}


# ── Figure ─────────────────────────────────────────────────────────────────────
def build_figure(
    obs_ext: np.ndarray, m1_ext: np.ndarray, m2_ext: np.ndarray,
    obs_all: np.ndarray, m1_all: np.ndarray, m2_all: np.ndarray,
    threshold: float, event_type: str,
    m1_name: str, m2_name: str,
    variable: str, season: str, orog_label: str, units: str,
    out_path: Path, dpi: int,
):
    ev_sign = "≥" if event_type == "warm" else "≤"

    # 2t data is stored in Celsius (unit label 'K' is a codebase convention but values are °C).
    # No conversion needed — just rename the display unit for clarity.
    _display_offset = 0.0
    _disp_units     = "°C" if variable == "2t" else units
    _disp_threshold = threshold + _display_offset

    print("  Computing rank histograms ...", flush=True)
    rh1 = rank_histogram(obs_ext, m1_ext)
    rh2 = rank_histogram(obs_ext, m2_ext)

    print("  Computing tail CDFs ...", flush=True)
    xx, cdf_m1, cdf_o = tail_cdf(obs_ext, m1_ext, threshold, event_type)
    _,  cdf_m2, _     = tail_cdf(obs_ext, m2_ext, threshold, event_type)

    print("  Computing quantile scores (chunked) ...", flush=True)
    taus, qs1, qs2 = quantile_scores(obs_all, m1_all, m2_all)

    print("  Computing reliability diagram ...", flush=True)
    centers, r1, cnt1, r2, cnt2, clim = reliability_diagram(
        obs_all, m1_all, m2_all, threshold, event_type
    )

    print("  Computing Murphy decomposition ...", flush=True)
    mur1 = murphy_decomposition(obs_all, m1_all, threshold, event_type)
    mur2 = murphy_decomposition(obs_all, m2_all, threshold, event_type)

    # Threshold quantile: fraction of all-event obs on the non-extreme side
    if event_type == "warm":
        thr_tau = float(np.mean(obs_all < threshold))
    else:
        thr_tau = float(np.mean(obs_all > threshold))

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        f"Tail Distribution Diagnostics:  {m2_name}  vs  {m1_name}\n"
        f"{VARIABLE_LABELS.get(variable, variable)} ({units})  |  "
        f"Extremes: obs {ev_sign} {threshold:.1f} {units}  |  "
        f"Season: {season}  |  Terrain: {orog_label}\n"
        f"Extreme events: {len(obs_ext):,}   |   All events (for QS & reliability): {len(obs_all):,}",
        fontsize=12, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.58, wspace=0.36,
        top=0.90, bottom=0.05, left=0.08, right=0.97,
        height_ratios=[1, 1, 0.85],
    )
    ax_rh    = fig.add_subplot(gs[0, 0])
    ax_cdf   = fig.add_subplot(gs[0, 1])
    ax_qs    = fig.add_subplot(gs[1, 0])
    ax_rel   = fig.add_subplot(gs[1, 1])
    ax_murph = fig.add_subplot(gs[2, :])

    # ── (A) Rank histogram ───────────────────────────────────────────────────
    ax = ax_rh
    n_bins  = len(rh1)
    perfect = 1.0 / n_bins
    xi  = np.arange(n_bins)
    bw  = 0.40
    ax.bar(xi - bw / 2, rh1, width=bw, color=FC1_CLR, alpha=0.80, label=m1_name)
    ax.bar(xi + bw / 2, rh2, width=bw, color=FC2_CLR, alpha=0.80, label=m2_name)
    ax.axhline(perfect, color="k", lw=1.2, ls="--",
               label=f"Uniform ({perfect:.4f})")
    # highlight the two diagnostic tails
    ax.axvspan(-0.5, 2.5,       color="#d62728", alpha=0.06, zorder=0,
               label="obs below most members")
    ax.axvspan(n_bins - 3, n_bins - 0.5, color="#ff7f0e", alpha=0.06, zorder=0,
               label="obs above most members")
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_xlabel("Rank of obs among 51 members\n"
                  "(0 = obs below all;  51 = obs above all)")
    ax.set_ylabel("Relative frequency")
    ax.set_title(
        "(A) Rank histogram — extreme events only\n"
        "(flat = calibrated;  peak at 51 → ensemble under-predicts cold/warm extremes;\n"
        " peak at 0 → ensemble over-predicts)",
        fontweight="bold",
    )
    ax.legend(fontsize=7.5, framealpha=0.85, ncol=2)

    # ── (B) Tail CDF ─────────────────────────────────────────────────────────
    ax = ax_cdf
    xx_disp = xx + _display_offset   # K→°C for 2t, no-op otherwise
    ax.axvline(_disp_threshold, color="gray", lw=1.0, ls=":",
               label=f"Threshold ({_disp_threshold:.1f} {_disp_units})", zorder=2)
    ax.plot(xx_disp, cdf_m1, color=FC1_CLR, lw=2.2, label=f"{m1_name} ensemble")
    ax.plot(xx_disp, cdf_m2, color=FC2_CLR, lw=2.2, label=f"{m2_name} ensemble")
    ax.plot(xx_disp, cdf_o,  color=OBS_CLR, lw=2.5, ls="--", label="Observed")

    # Clip x-axis to physically realistic range (display units)
    _REALISTIC = {
        "2t":   {"cold": (-60.0, _disp_threshold), "warm": (_disp_threshold, 55.0)},
        "tp24": {"cold": (0.0,   _disp_threshold), "warm": (_disp_threshold, 200.0)},
        "10ff": {"cold": (0.0,   _disp_threshold), "warm": (_disp_threshold, 60.0)},
    }
    if variable in _REALISTIC and event_type in _REALISTIC[variable]:
        _xlo, _xhi = _REALISTIC[variable][event_type]
        # don't over-clip: honour data extent within the realistic range
        _data_lo = float(xx_disp.min())
        _data_hi = float(xx_disp.max())
        ax.set_xlim(max(_xlo, _data_lo), min(_xhi, _data_hi))

    if event_type == "warm":
        ax.set_xlabel(f"Value ({_disp_units})  [only values ≥ threshold shown]")
        ax.set_ylabel("CDF contribution above threshold\n(re-normalised to 0 at threshold)")
        cdf_note = ("Ensemble CDF above obs CDF →\n"
                    "ensemble places less mass at high values\n"
                    "→ under-predicts warm extremes (causes misses)")
    else:
        ax.set_xlabel(f"Value ({_disp_units})  [only values ≤ threshold shown]")
        ax.set_ylabel("Empirical CDF  P(X ≤ x)")
        cdf_note = ("Ensemble CDF below obs CDF →\n"
                    "ensemble places less mass at cold values\n"
                    "→ under-predicts cold extremes (causes misses)")
    ax.text(0.03, 0.97, cdf_note, transform=ax.transAxes,
            fontsize=7.5, style="italic", color="#555", va="top")
    ax.set_title(
        f"(B) Tail CDF — extreme events only\n"
        f"(pooled ensemble members vs obs, restricted to {ev_sign} {_disp_threshold:.1f} {_disp_units})\n"
        "CDF gap = where ensemble probability mass is mis-placed",
        fontweight="bold",
    )
    ax.legend(fontsize=8, framealpha=0.85)

    # ── (C) Quantile score ────────────────────────────────────────────────────
    # Only show the relevant half: τ ≤ 0.50 for cold, τ ≥ 0.50 for warm/others
    ax = ax_qs
    if event_type == "cold":
        half_mask = taus <= 0.20
        qs_note   = "Showing cold tail only (τ ≤ 0.20)"
    else:
        half_mask = taus >= 0.80
        qs_note   = "Showing warm tail only (τ ≥ 0.80)"

    taus_h = taus[half_mask]
    qs1_h  = qs1[half_mask]
    qs2_h  = qs2[half_mask]

    # Shade tail region inside the displayed half
    if event_type == "warm":
        ax.axvspan(thr_tau, 1.0, color="gold", alpha=0.18,
                   label=f"Tail (τ > {thr_tau:.2f}) → twCRPS")
        ax.axvline(thr_tau, color="goldenrod", lw=1.0, ls="--")
        tmask = taus_h >= thr_tau
    else:
        ax.axvspan(0, thr_tau, color="gold", alpha=0.18,
                   label=f"Tail (τ < {thr_tau:.2f}) → twCRPS")
        ax.axvline(thr_tau, color="goldenrod", lw=1.0, ls="--")
        tmask = taus_h <= thr_tau

    ax.plot(taus_h, qs1_h, color=FC1_CLR, lw=2.2, label=m1_name)
    ax.plot(taus_h, qs2_h, color=FC2_CLR, lw=2.2, label=m2_name)

    # Summary box
    if tmask.sum() > 0:
        qs_m1_tail = np.mean(qs1_h[tmask])
        qs_m2_tail = np.mean(qs2_h[tmask])
        diff = qs_m1_tail - qs_m2_tail
        winner = m2_name if diff > 0 else m1_name
        ax.text(
            0.97, 0.97,
            f"Mean QS in shaded tail:\n"
            f"  {m1_name}: {qs_m1_tail:.4f} {units}\n"
            f"  {m2_name}: {qs_m2_tail:.4f} {units}\n"
            f"  fc1 − fc2: {diff:+.4f} {units}\n"
            f"  → {winner} better in tail",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9",
                      edgecolor="#bbb", alpha=0.95),
        )

    ax.set_xlabel(f"Quantile level τ  [{qs_note}]")
    ax.set_ylabel(f"Mean quantile score — pinball loss ({units})\n(lower = better)")
    ax.set_title(
        "(C) Quantile score by level — ALL events\n"
        "(shaded = tail region contributing to twCRPS)\n"
        "Where fc2 line < fc1: fc2 more accurate at that quantile",
        fontweight="bold",
    )
    ax.legend(fontsize=8, framealpha=0.85, loc="upper left")

    # ── (D) Reliability diagram ───────────────────────────────────────────────
    ax = ax_rel
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect reliability", zorder=3)
    ax.axhline(clim, color="#999", lw=1.0, ls=":",
               label=f"Climatology ({clim:.3f})", zorder=2)
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="#d62728", zorder=0)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color="#2ca02c", zorder=0)

    mask1 = ~np.isnan(r1)
    mask2 = ~np.isnan(r2)
    sz1 = np.sqrt(cnt1.astype(float)) * 3 + 20
    sz2 = np.sqrt(cnt2.astype(float)) * 3 + 20
    ax.scatter(centers[mask1], r1[mask1], s=sz1[mask1],
               color=FC1_CLR, zorder=5, label=m1_name, alpha=0.85)
    if mask1.sum() > 1:
        ax.plot(centers[mask1], r1[mask1], color=FC1_CLR, lw=1.5, alpha=0.6)
    ax.scatter(centers[mask2], r2[mask2], s=sz2[mask2],
               color=FC2_CLR, marker="s", zorder=5, label=m2_name, alpha=0.85)
    if mask2.sum() > 1:
        ax.plot(centers[mask2], r2[mask2], color=FC2_CLR, lw=1.5, alpha=0.6)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        f"Forecast probability of obs {ev_sign} {threshold:.1f} {units}\n"
        f"(fraction of 51 members {ev_sign} threshold)"
    )
    ax.set_ylabel(f"Observed frequency of obs {ev_sign} {threshold:.1f} {units}")
    ax.set_title(
        "(D) Reliability diagram — ALL events\n"
        "(dot size ∝ √n cases in bin;  diagonal = perfect;  dashed = climatology)\n"
        "above diagonal = over-forecast;  below = under-forecast",
        fontweight="bold",
    )
    ax.legend(fontsize=8, framealpha=0.85)
    ax.text(0.80, 0.12, "over-forecast", fontsize=7.5,
            color="#d62728", style="italic", ha="center", transform=ax.transAxes)
    ax.text(0.20, 0.82, "under-forecast", fontsize=7.5,
            color="#2ca02c", style="italic", ha="center", transform=ax.transAxes)

    # ── (E) Murphy decomposition ────────────────────────────────────────────
    # BS = REL − RES + UNC
    # REL: lower → better calibration (closer to reliability diagonal)
    # RES: higher → better resolution  (further from climatology)
    # UNC: same for both models (irreducible uncertainty of the event)
    ax = ax_murph
    _comps   = ["REL\n(lower=better)", "RES\n(higher=better)", "BS\n(lower=better)"]
    _keys    = ["REL", "RES", "BS"]
    _vals1   = [mur1[k] for k in _keys]
    _vals2   = [mur2[k] for k in _keys]
    _n_comp  = len(_comps)
    _xi      = np.arange(_n_comp)
    _bw      = 0.30
    _b1 = ax.bar(_xi - _bw / 2, _vals1, width=_bw, color=FC1_CLR, alpha=0.85,
                 label=m1_name, zorder=3)
    _b2 = ax.bar(_xi + _bw / 2, _vals2, width=_bw, color=FC2_CLR, alpha=0.85,
                 label=m2_name, zorder=3)
    # Value labels on bars
    for _bar, _val in [(b, v) for bars, vals in [(_b1, _vals1), (_b2, _vals2)]
                       for b, v in zip(bars, vals)]:
        _ypos = _bar.get_height()
        ax.text(_bar.get_x() + _bar.get_width() / 2, _ypos + 0.0003,
                f"{_val:.4f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    # UNC annotation (same for both)
    ax.axhline(mur1["UNC"], color="#999", lw=1.2, ls=":",
               label=f"UNC = {mur1['UNC']:.4f}  (climatological uncertainty, same for both)")
    # Difference arrows / annotations
    _diffs = {"REL": mur1["REL"] - mur2["REL"],
              "RES": mur2["RES"] - mur1["RES"],
              "BS":  mur1["BS"]  - mur2["BS"]}
    _labels_dir = {"REL": "lower=better", "RES": "higher=better", "BS": "lower=better"}
    for _i, _k in enumerate(_keys):
        _d = _diffs[_k]
        _col = FC2_CLR if _d > 0 else FC1_CLR
        _winner = m2_name if _d > 0 else m1_name
        ax.text(_i, -0.0025, f"{'+' if _d >= 0 else ''}{_d:.4f}\n→{_winner} better",
                ha="center", va="top", fontsize=7.5, color=_col, fontweight="bold",
                transform=ax.get_xaxis_transform())
    # BSS annotations
    _bss1 = mur1["BSS"]
    _bss2 = mur2["BSS"]
    ax.text(0.97, 0.97,
            f"Brier Skill Score (BSS = 1 − BS/UNC):\n"
            f"  {m1_name}: {_bss1:+.4f}\n"
            f"  {m2_name}: {_bss2:+.4f}\n"
            f"  (>0 = better than climatology)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9",
                      edgecolor="#bbb", alpha=0.95))
    ax.set_xticks(_xi)
    ax.set_xticklabels(_comps, fontsize=9)
    ax.set_ylabel("Score value")
    ax.set_title(
        "(E) Brier Score Murphy decomposition — ALL events\n"
        "BS = REL − RES + UNC  |  REL: calibration error (lower→better)  |  "
        "RES: sharpness beyond climatology (higher→better)  |  UNC: irreducible (dotted)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, framealpha=0.85, loc="upper left")
    ax.set_xlim(-0.6, _n_comp - 0.4)
    _ymax = max(max(_vals1), max(_vals2)) * 1.20
    ax.set_ylim(bottom=0, top=_ymax)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Tail distribution diagnostics for ensemble extreme scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--data-dir",        default="./extracted_points")
    ap.add_argument("--variable",        default=None)
    ap.add_argument("--event-type",      default="warm", choices=["cold", "warm"])
    ap.add_argument("--percentile",      type=float, default=None)
    ap.add_argument("--threshold-value", type=float, default=None)
    ap.add_argument("--season",          default="ALL",
                    choices=["DJF", "MAM", "JJA", "SON", "ALL"])
    ap.add_argument("--orog-types",      default="low,mid,high")
    ap.add_argument("--lead-days",       default=None)
    ap.add_argument("--output",          default=None)
    ap.add_argument("--dpi",             type=int, default=150)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    datasets = discover(data_dir)
    if not datasets:
        print("ERROR: No ensemble datasets found.")
        sys.exit(1)

    variable  = args.variable or next(iter(datasets))
    ds        = datasets[variable]
    m1, m2    = ds["models"]
    units     = VARIABLE_UNITS.get(variable, "?")
    percentile = args.percentile or (1.0 if args.event_type == "cold" else 99.0)
    lead_days  = ([int(d) for d in args.lead_days.split(",")]
                  if args.lead_days else ds["days"])
    orog_types = [o.strip() for o in args.orog_types.split(",")
                  if o.strip() in OROG_BINS]
    months     = SEASON_MONTHS.get(args.season, set(range(1, 13)))

    print(f"\n{'='*65}")
    print(f"Tail Distribution Diagnostics")
    print(f"  Variable  : {variable}  ({VARIABLE_LABELS.get(variable, variable)},  {units})")
    print(f"  Models    : {m1} (fc1)  vs  {m2} (fc2)")
    print(f"  Event     : {args.event_type}")
    if args.threshold_value is not None:
        print(f"  Threshold : fixed = {args.threshold_value} {units}")
    else:
        print(f"  Threshold : {percentile}th percentile of obs")
    print(f"  Season    : {args.season}")
    print(f"  Terrain   : {orog_types}")
    print(f"  Lead days : {lead_days}")
    print(f"{'='*65}\n")

    print("Loading parquet files ...", flush=True)
    df = load_parquets(ds["dir"], lead_days)
    df = df[df["_month"].isin(months)].copy()

    # Terrain filter
    omask = pd.Series(False, index=df.index)
    for t in orog_types:
        lo, hi = OROG_BINS[t]
        omask |= (df["sdfor"] >= lo) & (df["sdfor"] < hi)
    df = df[omask].copy()
    print(f"  Rows after filters: {len(df):,}", flush=True)

    # Model prefixes
    prefixes = sorted({
        c.split("_member_")[0] for c in df.columns if "_member_" in c
    })
    if len(prefixes) < 2:
        print(f"ERROR: expected 2 prefixes; found: {prefixes}")
        sys.exit(1)
    p1, p2  = prefixes[0], prefixes[1]

    obs_all = df["obs_value"].values
    m1_all  = df[member_cols(df, p1)].values   # (n, 51)
    m2_all  = df[member_cols(df, p2)].values

    # Threshold
    threshold = (args.threshold_value if args.threshold_value is not None
                 else float(np.nanpercentile(obs_all, percentile)))
    print(f"  Threshold : {threshold:.3f} {units}", flush=True)

    # Extreme mask
    if args.event_type == "warm":
        ext_mask = obs_all >= threshold
    else:
        ext_mask = obs_all <= threshold

    obs_ext = obs_all[ext_mask]
    m1_ext  = m1_all[ext_mask]
    m2_ext  = m2_all[ext_mask]
    print(f"  Extreme events: {ext_mask.sum():,}  ({ext_mask.mean() * 100:.2f}%)",
          flush=True)

    if ext_mask.sum() < 20:
        print("ERROR: too few extreme events — check threshold / filters.")
        sys.exit(1)

    # Output path
    orog_label = "+".join(orog_types)
    tag = (f"fixed{args.threshold_value}"
           if args.threshold_value is not None else f"pct{int(percentile)}")
    out = (Path(args.output) if args.output
           else Path(f"plots/tail_diag_{variable}_{args.event_type}_"
                     f"{args.season}_{orog_label}_{tag}.png"))

    print(f"\nBuilding figure → {out} ...", flush=True)
    build_figure(
        obs_ext, m1_ext, m2_ext,
        obs_all, m1_all, m2_all,
        threshold, args.event_type,
        m1, m2, variable, args.season,
        orog_label, units, out, args.dpi,
    )


if __name__ == "__main__":
    main()
