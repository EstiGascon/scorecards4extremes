"""
Diagnostic analysis for 10m wind speed extremes (DJF, fixed threshold 8.0 m/s)

Four analyses that test ensemble calibration and discrimination:
  1. Reliability diagram  – calibration
  2. Rank / Talagrand histogram – unconditional spread
  3. Conditional spread  – spread on event vs non-event days
  4. Brier score decomposition – reliability vs resolution vs uncertainty

Usage:
  /usr/local/apps/python3/3.11.10-01/bin/python3 analyse_10ff_discrimination.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE = Path(".")
PARQUET_DIR = Path("./extracted_points/10ff_ens")
OUT_DIR     = WORKSPACE / "plots/10ff_extremes_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_THRESHOLD = 8.0   # m/s (above = wind extreme)
EVENT_TYPE      = "above"
LEAD_DAYS       = [1, 3, 5]
N_MEMBERS       = 51
MODEL1_NAME     = "IFS-ENS"
MODEL2_NAME     = "AIFS-ENS"
VARIABLE_LABEL  = "10m Wind Speed"
UNITS           = "m/s"

SDFOR_BINS = {
    "LOW (sdfor<40)":   (0,    40),
    "MID (40-120)":     (40,  120),
    "HIGH (sdfor>120)": (120, 9999),
}

fc1_cols = [f"fc1_member_{i}" for i in range(N_MEMBERS)]
fc2_cols = [f"fc2_member_{i}" for i in range(N_MEMBERS)]


def is_djf(date_str):
    months = pd.to_datetime(date_str, format="%Y%m%d").dt.month
    return months.isin([12, 1, 2])


def ensemble_probability_above(members, threshold):
    return (members > threshold).mean(axis=1)


def reliability_diagram_data(prob, observed, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    centres, mean_p, obs_freq, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        mean_p.append(prob[mask].mean())
        obs_freq.append(observed[mask].mean())
        counts.append(mask.sum())
    return np.array(centres), np.array(mean_p), np.array(obs_freq), np.array(counts)


def brier_decomposition(prob, observed):
    n = len(prob)
    o_bar = observed.mean()
    unc = o_bar * (1 - o_bar)
    bins = np.linspace(0, 1, 11)
    rel, res = 0.0, 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi)
        if mask.sum() == 0:
            continue
        nk = mask.sum()
        pk = prob[mask].mean()
        ok = observed[mask].mean()
        rel += nk / n * (pk - ok) ** 2
        res += nk / n * (ok - o_bar) ** 2
    bs = ((prob - observed) ** 2).mean()
    return {"bs": bs, "reliability": rel, "resolution": res, "uncertainty": unc}


def rank_histogram(members, obs):
    return (members < obs[:, None]).sum(axis=1)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
for lead_day in LEAD_DAYS:
    print(f"\n{'='*60}")
    print(f"  Lead day {lead_day}")
    print(f"{'='*60}")

    parquet_path = PARQUET_DIR / f"10ff_ifs_ens_vs_aifs_ens_ens_day{lead_day}.parquet"
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    # Filter to DJF
    djf_mask = is_djf(df["date"])
    df_djf = df[djf_mask].copy()
    print(f"  DJF rows: {len(df_djf):,} / {len(df):,} total")

    obs   = df_djf["obs_value"].values
    fc1   = df_djf[fc1_cols].values
    fc2   = df_djf[fc2_cols].values
    sdfor = df_djf["sdfor"].values

    # Fixed threshold — same for all terrain
    thr   = FIXED_THRESHOLD
    event = (obs > thr).astype(float)
    print(f"  Event base rate (obs > {thr} {UNITS}): {event.mean()*100:.2f}%")

    p1 = ensemble_probability_above(fc1, thr)
    p2 = ensemble_probability_above(fc2, thr)
    spread1 = fc1.std(axis=1)
    spread2 = fc2.std(axis=1)

    # Pre-compute Brier decomposition per terrain x model
    terrain_labels = list(SDFOR_BINS.keys())
    decomp = {}
    for name, fc_arr in [(MODEL1_NAME, fc1), (MODEL2_NAME, fc2)]:
        for label, (lo, hi) in SDFOR_BINS.items():
            tmask = (sdfor >= lo) & (sdfor < hi)
            if tmask.sum() < 10:
                decomp[(name, label)] = None
                continue
            ev_t   = event[tmask]
            prob_t = ensemble_probability_above(fc_arr[tmask], thr)
            dec    = brier_decomposition(prob_t, ev_t)
            dec["bss"] = 1 - dec["bs"] / dec["uncertainty"] if dec["uncertainty"] > 0 else np.nan
            decomp[(name, label)] = dec

    # ----------------------------------------------------------------
    # Figure: 4 panels
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        f"{VARIABLE_LABEL} Extremes (DJF, threshold = {thr} {UNITS}) — Day {lead_day}\n"
        f"IFS-ENS vs AIFS-ENS  |  {len(df_djf):,} pairs  |  "
        f"Event base rate: {event.mean()*100:.1f}%",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ============================================================
    # Panel A: Reliability diagram
    # ============================================================
    ax_rel = fig.add_subplot(gs[0, 0])
    ax_rel.set_title("A — Reliability diagram (all terrain)", fontweight="bold")

    for prob, name, color, ls in [(p1, MODEL1_NAME, "#1f78b4", "-"),
                                   (p2, MODEL2_NAME, "#e31a1c", "--")]:
        centres, mean_p, obs_freq, counts = reliability_diagram_data(prob, event, n_bins=10)
        ax_rel.plot(mean_p, obs_freq, "o", color=color, lw=2, ms=6, label=name, ls=ls)

    ax_rel.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect reliability")
    ax_rel.axhline(event.mean(), color="gray", lw=1, ls=":", alpha=0.7,
                   label=f"Climatology ({event.mean():.3f})")
    ax_rel.set_xlabel("Forecast probability")
    ax_rel.set_ylabel("Observed frequency")
    ax_rel.set_xlim(-0.02, 1.02)
    ax_rel.set_ylim(-0.02, 1.02)
    ax_rel.legend(fontsize=8)
    ax_rel.grid(True, alpha=0.3)

    # ============================================================
    # Panel B: Rank histogram (Talagrand)
    # ============================================================
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_rank.set_title("B — Rank histogram (Talagrand)", fontweight="bold")

    for members, name, color in [(fc1, MODEL1_NAME, "#1f78b4"),
                                  (fc2, MODEL2_NAME, "#e31a1c")]:
        ranks = rank_histogram(members, obs)
        n_bins = N_MEMBERS + 1
        hist, edges = np.histogram(ranks, bins=n_bins, range=(0, n_bins))
        hist_norm = hist / hist.sum()
        centres_r = (edges[:-1] + edges[1:]) / 2
        ax_rank.plot(centres_r, hist_norm, lw=1.5, alpha=0.8, label=name, color=color)

    flat_line = 1 / (N_MEMBERS + 1)
    ax_rank.axhline(flat_line, color="k", lw=1, ls="--", label=f"Uniform ({flat_line:.4f})")
    ax_rank.set_xlabel("Rank of observation in ensemble")
    ax_rank.set_ylabel("Normalised frequency")
    ax_rank.legend(fontsize=8)
    ax_rank.grid(True, alpha=0.3)

    # ============================================================
    # Panel C: Conditional spread — event vs non-event
    # ============================================================
    ax_spr = fig.add_subplot(gs[1, 0])
    ax_spr.set_title("C — Ensemble spread: event vs non-event days", fontweight="bold")

    x = np.arange(len(terrain_labels))
    width = 0.18
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    bar_labels = ["IFS non-event", "IFS event", "AIFS non-event", "AIFS event"]
    bar_colors = ["#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c"]

    for i, (spr, emask, lbl) in enumerate(zip(
            [spread1, spread1, spread2, spread2],
            [event == 0, event == 1, event == 0, event == 1],
            bar_labels)):
        means = []
        for lo, hi in SDFOR_BINS.values():
            tmask = (sdfor >= lo) & (sdfor < hi)
            combined = tmask & emask
            means.append(spr[combined].mean() if combined.sum() > 0 else 0)
        ax_spr.bar(x + offsets[i], means, width, label=lbl, color=bar_colors[i])

    ax_spr.set_xticks(x)
    ax_spr.set_xticklabels(terrain_labels, fontsize=8)
    ax_spr.set_ylabel(f"Ensemble spread ({UNITS}, std of members)")
    ax_spr.legend(fontsize=7, ncol=2)
    ax_spr.grid(True, axis="y", alpha=0.3)

    # ============================================================
    # Panel D: Brier score decomposition — BSS + REL/RES
    # ============================================================
    gs_d = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.55,
                                   height_ratios=[1, 1])
    ax_d_bss = fig.add_subplot(gs_d[0])
    ax_d_rel = fig.add_subplot(gs_d[1])

    terrain_x = np.arange(len(terrain_labels))
    w = 0.35

    # Top: BSS
    ax_d_bss.set_title("D — Brier skill score & calibration by terrain",
                        fontweight="bold", fontsize=9)
    for j, (name, color) in enumerate([(MODEL1_NAME, "#1f78b4"), (MODEL2_NAME, "#e31a1c")]):
        bss_vals = [decomp[(name, lbl)]["bss"] if decomp[(name, lbl)] else np.nan
                    for lbl in terrain_labels]
        offset = (j - 0.5) * w
        bars = ax_d_bss.bar(terrain_x + offset, bss_vals, w, color=color,
                             label=name, alpha=0.85)
        for bar, val in zip(bars, bss_vals):
            if not np.isnan(val):
                ax_d_bss.text(bar.get_x() + bar.get_width() / 2,
                              val + (0.02 if val >= 0 else -0.05),
                              f"{val:+.2f}", ha='center', va='bottom', fontsize=7,
                              color=color, fontweight='bold')
    ax_d_bss.axhline(0, color="k", lw=1)
    ax_d_bss.set_xticks(terrain_x)
    ax_d_bss.set_xticklabels(terrain_labels, fontsize=8)
    ax_d_bss.set_ylabel("BSS  (>0 = better than climatology)", fontsize=8)
    ax_d_bss.legend(fontsize=7, loc='lower right')
    ax_d_bss.grid(True, axis="y", alpha=0.3)

    # Bottom: REL vs RES
    ax_d_rel.set_title("Reliability error vs Resolution  (both × 10³)", fontsize=8.5)
    sub_w = 0.8 / 4
    offsets_4 = [-1.5, -0.5, 0.5, 1.5]

    for k, (name, color) in enumerate([(MODEL1_NAME, "#1f78b4"), (MODEL2_NAME, "#e31a1c")]):
        rel_vals = [decomp[(name, lbl)]["reliability"] * 1000 if decomp[(name, lbl)] else np.nan
                    for lbl in terrain_labels]
        res_vals = [decomp[(name, lbl)]["resolution"]  * 1000 if decomp[(name, lbl)] else np.nan
                    for lbl in terrain_labels]
        ax_d_rel.bar(terrain_x + offsets_4[k * 2] * sub_w, rel_vals, sub_w,
                     color=color, alpha=0.5, hatch="//", label=f"{name} REL↓")
        ax_d_rel.bar(terrain_x + offsets_4[k * 2 + 1] * sub_w, res_vals, sub_w,
                     color=color, alpha=0.9, label=f"{name} RES↑")

    ax_d_rel.set_xticks(terrain_x)
    ax_d_rel.set_xticklabels(terrain_labels, fontsize=8)
    ax_d_rel.set_ylabel("Score × 10³", fontsize=8)
    ax_d_rel.legend(fontsize=6.5, ncol=2, loc='upper left')
    ax_d_rel.grid(True, axis="y", alpha=0.3)
    ax_d_rel.text(0.99, 0.97,
                  "REL (hatched) = calibration error — lower is better\n"
                  "RES (solid) = discrimination — higher is better",
                  transform=ax_d_rel.transAxes, fontsize=6.5, va='top', ha='right', color='gray')

    # ----------------------------------------------------------------
    # Console output: decomposition + conditional spread
    # ----------------------------------------------------------------
    print(f"\n  Brier decomposition (Day {lead_day}):")
    print(f"  {'Terrain':<22} {'Model':<12} {'Rate%':>6} {'BS':>8} {'REL':>8} {'RES':>8} {'UNC':>8} {'BSS':>7}")
    for label, (lo, hi) in SDFOR_BINS.items():
        tmask = (sdfor >= lo) & (sdfor < hi)
        rate = event[tmask].mean() * 100
        for name in [MODEL1_NAME, MODEL2_NAME]:
            dec = decomp[(name, label)]
            if dec is None:
                continue
            print(f"  {label:<22} {name:<12} {rate:>6.2f} {dec['bs']:.5f} "
                  f"{dec['reliability']:.5f} {dec['resolution']:.5f} "
                  f"{dec['uncertainty']:.5f} {dec['bss']:+.3f}")

    print(f"\n  Conditional spread (Day {lead_day}):")
    print(f"  {'Terrain':<22} {'Condition':<16} {'n':>7} {'IFS spread':>12} {'AIFS spread':>12} {'ratio':>8}")
    for label, (lo, hi) in SDFOR_BINS.items():
        tmask = (sdfor >= lo) & (sdfor < hi)
        for cond_name, emask in [("non-event", event == 0), ("event", event == 1)]:
            combined = tmask & emask
            if combined.sum() < 10:
                continue
            s1 = spread1[combined].mean()
            s2 = spread2[combined].mean()
            print(f"  {label:<22} {cond_name:<16} {combined.sum():>7,} {s1:>12.3f} {s2:>12.3f} {s2/s1:>8.2f}x")

    # ----------------------------------------------------------------
    out_path = OUT_DIR / f"10ff_extremes_diagnostics_day{lead_day}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

print("\nDone.")
