"""
Diagnostic analysis for 2m temperature warm extremes (DJF, 99th pct)

Four analyses that test why Brier/twCRPS are red but quantile_score/spread are blue:
  1. Reliability diagram  – calibration: are P% forecasts right P% of the time?
  2. Rank / Talagrand histogram – unconditional ensemble spread/calibration
  3. Conditional spread  – spread on event days vs non-event days separately
  4. Brier score decomposition – reliability vs resolution vs uncertainty

Usage:
  /usr/local/apps/python3/3.11.10-01/bin/python3 analyse_warm_extremes_discrimination.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE = Path(".")
PARQUET_DIR = Path("./extracted_points/2t_ens")
RESULTS_DIR = WORKSPACE / "results/2t_ens_ifs_vs_aifs_warm"
OUT_DIR = WORKSPACE / "plots/warm_extremes_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERCENTILE = 99        # warm extreme percentile
LEAD_DAYS = [1, 3, 5]  # lead times to analyse
N_MEMBERS = 51         # members per model (0..50)
MODEL1_NAME = "IFS-ENS"
MODEL2_NAME = "AIFS-ENS"

SDFOR_BINS = {
    "LOW (sdfor<40)":   (0,    40),
    "MID (40-120)":     (40,  120),
    "HIGH (sdfor>120)": (120, 9999),
}
TERRAIN_COLORS = {"LOW (sdfor<40)": "#1a9850", "MID (40-120)": "#f4a040", "HIGH (sdfor>120)": "#d73027"}

fc1_cols = [f"fc1_member_{i}" for i in range(N_MEMBERS)]
fc2_cols = [f"fc2_member_{i}" for i in range(N_MEMBERS)]


def is_djf(date_str):
    """Return boolean mask for DJF months from a Series of YYYYMMDD strings."""
    months = pd.to_datetime(date_str, format="%Y%m%d").dt.month
    return months.isin([12, 1, 2])


def compute_dataset_threshold(obs, pct=99):
    """
    Single dataset-level PERCENTILE (matches threshold.py: np.percentile(data['obs_value'], pct)).
    Returns a scalar.
    """
    return float(np.percentile(obs, pct))


def ensemble_probability_above(members, threshold):
    """
    Fraction of ensemble members exceeding a scalar threshold.
    members: (N, n_members) array
    threshold: scalar
    """
    return (members > threshold).mean(axis=1)  # (N,)


def reliability_diagram_data(prob, observed, n_bins=10):
    """
    Returns (bin_centres, mean_forecast_prob, observed_freq, bin_counts).
    """
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
    """
    Murphy (1973) decomposition: BS = REL - RES + UNC
    Returns dict with bs, reliability, resolution, uncertainty, n_bins used.
    """
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
    """
    Returns rank array (0..n_member). Obs rank among members.
    """
    # members: (N, M), obs: (N,)
    ranks = (members < obs[:, None]).sum(axis=1)  # 0 if below all, M if above all
    return ranks


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
for lead_day in LEAD_DAYS:
    print(f"\n{'='*60}")
    print(f"  Lead day {lead_day}")
    print(f"{'='*60}")

    parquet_path = PARQUET_DIR / f"2t_ifs_ens_vs_aifs_ens_ens_day{lead_day}.parquet"
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    # Filter to DJF
    djf_mask = is_djf(df["date"])
    df_djf = df[djf_mask].copy()
    print(f"  DJF rows: {len(df_djf):,} / {len(df):,} total")

    obs = df_djf["obs_value"].values
    fc1 = df_djf[fc1_cols].values   # (N, 51)
    fc2 = df_djf[fc2_cols].values
    sdfor = df_djf["sdfor"].values

    # Dataset-level threshold per terrain bin (matches threshold.py exactly)
    # We compute per-terrain-bin below, but store an overall one for the rank histogram
    thr_all = compute_dataset_threshold(obs, PERCENTILE)
    event_all = (obs > thr_all).astype(float)
    print(f"  Overall event base rate: {event_all.mean()*100:.2f}%  (threshold={thr_all:.2f}°C)")

    # Ensemble probabilities using overall threshold
    p1 = ensemble_probability_above(fc1, thr_all)  # IFS
    p2 = ensemble_probability_above(fc2, thr_all)  # AIFS

    # Ensemble spread
    spread1 = fc1.std(axis=1)
    spread2 = fc2.std(axis=1)

    # shorthand
    event = event_all
    thr = thr_all

    # ----------------------------------------------------------------
    # Figure: 4 panels
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        f"2m Temperature Warm Extremes (DJF, 99th pct) — Day {lead_day}\n"
        f"IFS-ENS vs AIFS-ENS  |  {len(df_djf):,} forecast-observation pairs",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ============================================================
    # Panel A: Reliability diagram (all terrain combined)
    # ============================================================
    ax_rel = fig.add_subplot(gs[0, 0])
    ax_rel.set_title("A — Reliability diagram (all terrain)", fontweight="bold")

    for prob, name, color, ls in [(p1, MODEL1_NAME, "#1f78b4", "-"),
                                   (p2, MODEL2_NAME, "#e31a1c", "--")]:
        centres, mean_p, obs_freq, counts = reliability_diagram_data(prob, event, n_bins=10)
        ax_rel.plot(mean_p, obs_freq, "o-", color=color, lw=2, ms=6, label=name, ls=ls)

    ax_rel.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect reliability")
    ax_rel.axhline(event.mean(), color="gray", lw=1, ls=":", alpha=0.7, label=f"Climatology ({event.mean():.3f})")
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
        n_bins = N_MEMBERS + 1   # 0..51
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
    # Panel C: Conditional spread — event vs non-event, per terrain
    # ============================================================
    ax_spr = fig.add_subplot(gs[1, 0])
    ax_spr.set_title("C — Ensemble spread: event vs non-event days", fontweight="bold")
    terrain_labels = list(SDFOR_BINS.keys())
    x = np.arange(len(terrain_labels))
    width = 0.18
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    bar_labels  = ["IFS non-event", "IFS event", "AIFS non-event", "AIFS event"]
    bar_colors  = ["#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c"]
    bar_spreads = [
        (spread1, event == 0, MODEL1_NAME + " non-event"),
        (spread1, event == 1, MODEL1_NAME + " event"),
        (spread2, event == 0, MODEL2_NAME + " non-event"),
        (spread2, event == 1, MODEL2_NAME + " event"),
    ]

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
    ax_spr.set_ylabel("Ensemble spread (°C, std of members)")
    ax_spr.legend(fontsize=7, ncol=2)
    ax_spr.grid(True, axis="y", alpha=0.3)

    # ============================================================
    # Panel D: Brier score decomposition by terrain
    # Split into two sub-axes: top = BSS, bottom = REL vs RES
    # ============================================================
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_d = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.55,
                                   height_ratios=[1, 1])
    ax_d_bss = fig.add_subplot(gs_d[0])
    ax_d_rel = fig.add_subplot(gs_d[1])

    # Pre-compute decomposition for all terrain x model combinations
    decomp = {}   # key: (name, label) -> dict with bs/rel/res/unc/bss
    for lo, hi in SDFOR_BINS.values():
        pass  # just to get labels in order

    for name, fc_arr, color in [(MODEL1_NAME, fc1, "#1f78b4"),
                                 (MODEL2_NAME, fc2, "#e31a1c")]:
        for label, (lo, hi) in SDFOR_BINS.items():
            tmask = (sdfor >= lo) & (sdfor < hi)
            if tmask.sum() < 10:
                decomp[(name, label)] = None
                continue
            thr_t  = compute_dataset_threshold(obs[tmask], PERCENTILE)
            ev_t   = (obs[tmask] > thr_t).astype(float)
            prob_t = ensemble_probability_above(fc_arr[tmask], thr_t)
            dec    = brier_decomposition(prob_t, ev_t)
            dec["bss"] = 1 - dec["bs"] / dec["uncertainty"] if dec["uncertainty"] > 0 else np.nan
            decomp[(name, label)] = dec

    terrain_x = np.arange(len(terrain_labels))
    w = 0.35

    # ---- Top sub-panel: Brier Skill Score ----
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

    # ---- Bottom sub-panel: REL and RES side by side ----
    ax_d_rel.set_title("Reliability error vs Resolution  (both × 10³)", fontsize=8.5)
    n_terrain = len(terrain_labels)
    group_w = 0.8
    sub_w = group_w / 4   # 4 bars per terrain group
    offsets_4 = [-1.5, -0.5, 0.5, 1.5]

    for k, (name, fc_arr, color) in enumerate([(MODEL1_NAME, fc1, "#1f78b4"),
                                                (MODEL2_NAME, fc2, "#e31a1c")]):
        rel_vals = [decomp[(name, lbl)]["reliability"] * 1000 if decomp[(name, lbl)] else np.nan
                    for lbl in terrain_labels]
        res_vals = [decomp[(name, lbl)]["resolution"]  * 1000 if decomp[(name, lbl)] else np.nan
                    for lbl in terrain_labels]

        # REL bars: hatched, lighter
        rel_off = offsets_4[k * 2] * sub_w
        ax_d_rel.bar(terrain_x + rel_off, rel_vals, sub_w, color=color, alpha=0.5,
                     hatch="//", label=f"{name} REL↓")
        # RES bars: solid, darker
        res_off = offsets_4[k * 2 + 1] * sub_w
        ax_d_rel.bar(terrain_x + res_off, res_vals, sub_w, color=color, alpha=0.9,
                     label=f"{name} RES↑")

    ax_d_rel.set_xticks(terrain_x)
    ax_d_rel.set_xticklabels(terrain_labels, fontsize=8)
    ax_d_rel.set_ylabel("Score × 10³", fontsize=8)
    ax_d_rel.legend(fontsize=6.5, ncol=2, loc='upper left')
    ax_d_rel.grid(True, axis="y", alpha=0.3)
    note = "REL (hatched) = calibration error — lower is better\n" \
           "RES (solid) = discrimination — higher is better"
    ax_d_rel.text(0.99, 0.97, note, transform=ax_d_rel.transAxes, fontsize=6.5,
                  va='top', ha='right', color='gray')

    # ----------------------------------------------------------------
    # Print numerical Brier decomposition to console (reuse decomp dict)
    # ----------------------------------------------------------------
    print(f"\n  Brier decomposition (Day {lead_day}):")
    print(f"  {'Terrain':<22} {'Model':<12} {'Threshold':>10} {'Rate%':>6} {'BS':>8} {'REL':>8} {'RES':>8} {'UNC':>8} {'BSS':>7}")
    for label, (lo, hi) in SDFOR_BINS.items():
        tmask = (sdfor >= lo) & (sdfor < hi)
        if tmask.sum() < 10:
            continue
        thr_t = compute_dataset_threshold(obs[tmask], PERCENTILE)
        ev_t  = (obs[tmask] > thr_t).astype(float)
        rate  = ev_t.mean() * 100
        for name in [MODEL1_NAME, MODEL2_NAME]:
            dec = decomp[(name, label)]
            if dec is None:
                continue
            print(f"  {label:<22} {name:<12} {thr_t:>10.2f} {rate:>6.2f} {dec['bs']:.5f} "
                  f"{dec['reliability']:.5f} {dec['resolution']:.5f} {dec['uncertainty']:.5f} {dec['bss']:+.3f}")

    # ----------------------------------------------------------------
    # Print conditional spread
    # ----------------------------------------------------------------
    print(f"\n  Conditional spread (Day {lead_day}) — event condition based on per-terrain threshold:")
    print(f"  {'Terrain':<22} {'Condition':<16} {'n':>7} {'IFS spread':>12} {'AIFS spread':>12} {'ratio':>8}")
    for label, (lo, hi) in SDFOR_BINS.items():
        tmask = (sdfor >= lo) & (sdfor < hi)
        if tmask.sum() < 10:
            continue
        thr_t = compute_dataset_threshold(obs[tmask], PERCENTILE)
        ev_t  = (obs[tmask] > thr_t).astype(float)
        full_event = np.zeros(len(obs))
        full_event[tmask] = ev_t
        for cond_name, emask in [("non-event", full_event == 0), ("event", full_event == 1)]:
            combined = tmask & emask
            if combined.sum() < 10:
                continue
            s1 = spread1[combined].mean()
            s2 = spread2[combined].mean()
            print(f"  {label:<22} {cond_name:<16} {combined.sum():>7,} {s1:>12.3f} {s2:>12.3f} {s2/s1:>8.2f}x")

    # ----------------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------------
    out_path = OUT_DIR / f"warm_extremes_diagnostics_day{lead_day}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

print("\nDone.")
