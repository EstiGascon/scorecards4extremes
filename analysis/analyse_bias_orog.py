"""
analyse_bias_orog.py — Bias and distribution analysis for a given season/orog class.

Designed to investigate *why* one model outperforms the other in specific
terrain / season combinations.  Produces 6 figures:

  1. QQ plot (obs vs IFS vs AIFS) across all lead times + cold-tail zoom inset
  2. Mean bias (ME) vs forecast lead time — all stations vs extreme-only
  3. Conditional bias — mean bias binned by observed temperature percentile
  4. Bias vs terrain roughness (sdfor bins)
  5. Cold/warm extreme skill scores (POD, FAR, ETS) vs lead time
  6. Error distribution violins per lead time — extreme stations only

Usage
-----
  python analyse_bias_orog.py --config config_2t_local_p1obsclim_aifs_ifs_single.yaml \\
      --season DJF --orog complex --output-dir results/cold_bias_DJF_complex

Typical SLURM submission:
  sbatch --export=CONFIG=...,SEASON=DJF,OROG=complex,OUTPUT=... \\
      case_studies/submit_diagnose.sh
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from case_studies.case_study_utils import load_per_station_thresholds

# ─── Palette ──────────────────────────────────────────────────────────────────
C_IFS  = "#d7191c"   # red
C_AIFS = "#2c7bb6"   # blue
C_OBS  = "#1a9641"   # green
DAYS   = [1, 3, 5, 7, 10]
DAY_ALPHAS = {1: 1.0, 3: 0.85, 5: 0.7, 7: 0.55, 10: 0.4}
DAY_LS     = {1: "-", 3: "--", 5: "-.", 7: ":", 10: (0, (3, 1, 1, 1))}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      required=True, help="YAML config file")
    p.add_argument("--season",      default=None,
                   help="Season filter: DJF | MAM | JJA | SON")
    p.add_argument("--orog",        default=None,
                   help="Orography class: flat | hilly | complex")
    p.add_argument("--output-dir",  default=None, dest="output_dir",
                   help="Output directory (default: results/bias_<season>_<orog>)")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

SEASON_MONTHS = {"DJF": {12, 1, 2}, "MAM": {3, 4, 5},
                 "JJA": {6, 7, 8},  "SON": {9, 10, 11}}

DEFAULT_OROG_RANGES = {"flat": (0, 40), "hilly": (40, 120), "complex": (120, 3000)}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_event_type(config):
    return config.get("threshold", {}).get("event_type", "above")


def month_of_date(date_int):
    return int(str(int(date_int))[4:6])


def apply_filters(df, season, orog, orog_ranges):
    if season and season in SEASON_MONTHS:
        months = SEASON_MONTHS[season]
        df = df[df["date"].astype(str).str[4:6].astype(int).isin(months)]
    if orog and orog in orog_ranges:
        lo, hi = orog_ranges[orog]
        df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]
    return df


def classify_events(obs, fc, T, event_type):
    """Return boolean arrays: hit, miss, false_alarm, correct_neg."""
    if event_type == "below":
        obs_ext = obs <= T
        fc_ext  = fc  <= T
    else:
        obs_ext = obs >= T
        fc_ext  = fc  >= T
    hit   = obs_ext &  fc_ext
    miss  = obs_ext & ~fc_ext
    fa    = ~obs_ext &  fc_ext
    cn    = ~obs_ext & ~fc_ext
    return hit, miss, fa, cn


def ets_score(nh, nm, nf, nn):
    N = nh + nm + nf + nn
    if N == 0:
        return np.nan
    hits_r = (nh + nm) * (nh + nf) / N
    denom = nh + nm + nf - hits_r
    return (nh - hits_r) / denom if denom > 0 else np.nan


def load_day(parquet_dir, day, season, orog, orog_ranges, config):
    """Load + filter one forecast day. Returns (df_filtered, T_array)."""
    candidates = list(Path(parquet_dir).glob(f"*_day{day}.parquet"))
    if not candidates:
        return None, None
    df = pd.read_parquet(candidates[0])
    df = apply_filters(df, season, orog, orog_ranges)
    if df.empty:
        return None, None
    # Per-station thresholds
    T = load_per_station_thresholds(config, df)
    valid = ~np.isnan(T) & ~np.isnan(df["obs_value"].values)
    df = df[valid].copy()
    T  = T[valid]
    return df.reset_index(drop=True), T


# ─── Figure 1: QQ plot ────────────────────────────────────────────────────────

def fig_qq(data_by_day, event_type, model1_name, model2_name, out_dir, label):
    """QQ plot: obs quantiles (x) vs IFS / AIFS quantiles (y).

    Day 5 only; both models on the same axes for direct comparison.
    Quantile grid: 0.1-resolution from 0.1–1 and 99–99.9, 1-resolution from 1–99.
    Dots at every quantile point; lines connecting them.
    """
    PLOT_DAY = 5
    tail_pct = 10 if event_type == "below" else 90

    # Dense grid at tails, coarser in the middle
    QUANTILES = np.unique(np.concatenate([
        np.arange(0.1, 1.0, 0.1),   # 0.1 … 0.9
        np.arange(1.0, 100.0, 1.0), # 1 … 99
        np.arange(99.1, 100.0, 0.1) # 99.1 … 99.9
    ]))

    # Fall back to closest available day if Day 5 not present
    available = sorted(data_by_day.keys())
    day = PLOT_DAY if PLOT_DAY in data_by_day else min(available, key=lambda d: abs(d - PLOT_DAY))

    df, _ = data_by_day[day]
    obs    = df["obs_value"].values
    fc1    = df["fc1_value"].values
    fc2    = df["fc2_value"].values

    obs_q  = np.percentile(obs, QUANTILES)
    fc1_q  = np.percentile(fc1, QUANTILES)
    fc2_q  = np.percentile(fc2, QUANTILES)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Lines + dots
    ax.plot(obs_q, fc1_q, color=C_IFS,  lw=1.5, zorder=3)
    ax.scatter(obs_q, fc1_q, color=C_IFS,  s=18, zorder=4, label=f"{model1_name}  (Day {day})")
    ax.plot(obs_q, fc2_q, color=C_AIFS, lw=1.5, zorder=3)
    ax.scatter(obs_q, fc2_q, color=C_AIFS, s=18, zorder=4, label=f"{model2_name}  (Day {day})")

    # 1:1 line extended to the full data range (min/max of obs, fc1, fc2)
    all_vals = np.concatenate([obs_q, fc1_q, fc2_q])
    lo = float(np.nanmin(all_vals))
    hi = float(np.nanmax(all_vals))
    margin = (hi - lo) * 0.02
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", lw=1.2, label="1:1 (perfect)", zorder=2)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    ax.set_xlabel("Observed quantile (°C)", fontsize=11)
    ax.set_ylabel("Forecast quantile (°C)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Zoomed tail inset ──
    if event_type == "below":
        tail_mask  = QUANTILES <= tail_pct
        inset_pos  = [0.55, 0.08, 0.42, 0.42]
        tail_label = f"Cold tail  (p0.1–p{tail_pct})"
    else:
        tail_mask  = QUANTILES >= tail_pct
        inset_pos  = [0.08, 0.55, 0.42, 0.42]
        tail_label = f"Warm tail  (p{tail_pct}–p99.9)"

    ax_in = ax.inset_axes(inset_pos)
    ax_in.set_facecolor("#f5f5f5")
    obs_tq  = obs_q[tail_mask]
    fc1_tq  = fc1_q[tail_mask]
    fc2_tq  = fc2_q[tail_mask]
    ax_in.plot(obs_tq, fc1_tq,  color=C_IFS,  lw=1.5, zorder=3)
    ax_in.scatter(obs_tq, fc1_tq, color=C_IFS,  s=14, zorder=4)
    ax_in.plot(obs_tq, fc2_tq,  color=C_AIFS, lw=1.5, zorder=3)
    ax_in.scatter(obs_tq, fc2_tq, color=C_AIFS, s=14, zorder=4)
    t_lo = float(obs_tq.min())
    t_hi = float(obs_tq.max())
    t_margin = (t_hi - t_lo) * 0.02
    ax_in.plot([t_lo - t_margin, t_hi + t_margin],
               [t_lo - t_margin, t_hi + t_margin], "k--", lw=1.2)
    ax_in.set_title(tail_label, fontsize=8)
    ax_in.grid(True, alpha=0.3)
    ax_in.tick_params(labelsize=7)

    fig.suptitle(
        f"QQ Plot — {model1_name} vs {model2_name}  |  Day {day}  |  {label}",
        fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "1_qq_plot.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] QQ plot → {out.name}")


# ─── Figure 2: Bias vs lead time ──────────────────────────────────────────────

def fig_bias_vs_leadtime(data_by_day, event_type, model1_name, model2_name,
                          out_dir, label):
    days_sorted = sorted(data_by_day.keys())

    me1_all, me2_all   = [], []
    me1_ext, me2_ext   = [], []
    std1_all, std2_all = [], []
    std1_ext, std2_ext = [], []
    mae1_all, mae2_all = [], []
    mae1_ext, mae2_ext = [], []

    for day in days_sorted:
        df, T = data_by_day[day]
        obs = df["obs_value"].values
        fc1 = df["fc1_value"].values
        fc2 = df["fc2_value"].values
        err1 = fc1 - obs
        err2 = fc2 - obs

        me1_all.append(np.mean(err1));  me2_all.append(np.mean(err2))
        std1_all.append(np.std(err1));  std2_all.append(np.std(err2))
        mae1_all.append(np.mean(np.abs(err1))); mae2_all.append(np.mean(np.abs(err2)))

        if event_type == "below":
            ext_mask = obs <= T
        else:
            ext_mask = obs >= T

        if ext_mask.sum() > 0:
            me1_ext.append(np.mean(err1[ext_mask]))
            me2_ext.append(np.mean(err2[ext_mask]))
            std1_ext.append(np.std(err1[ext_mask]))
            std2_ext.append(np.std(err2[ext_mask]))
            mae1_ext.append(np.mean(np.abs(err1[ext_mask])))
            mae2_ext.append(np.mean(np.abs(err2[ext_mask])))
        else:
            for lst in [me1_ext, me2_ext, std1_ext, std2_ext, mae1_ext, mae2_ext]:
                lst.append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    days_arr = np.array(days_sorted)

    # ME all
    ax = axes[0]
    ax.plot(days_arr, me1_all, "o-", color=C_IFS,  lw=2, label=f"{model1_name} all")
    ax.plot(days_arr, me2_all, "o-", color=C_AIFS, lw=2, label=f"{model2_name} all")
    ax.plot(days_arr, me1_ext, "s--", color=C_IFS,  lw=1.5, label=f"{model1_name} ext")
    ax.plot(days_arr, me2_ext, "s--", color=C_AIFS, lw=1.5, label=f"{model2_name} ext")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Forecast day"); ax.set_ylabel("Mean Error  fc − obs  (°C)")
    ax.set_title("Mean Bias (ME)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(days_arr)

    # MAE
    ax = axes[1]
    ax.plot(days_arr, mae1_all, "o-", color=C_IFS,  lw=2, label=f"{model1_name} all")
    ax.plot(days_arr, mae2_all, "o-", color=C_AIFS, lw=2, label=f"{model2_name} all")
    ax.plot(days_arr, mae1_ext, "s--", color=C_IFS,  lw=1.5, label=f"{model1_name} ext")
    ax.plot(days_arr, mae2_ext, "s--", color=C_AIFS, lw=1.5, label=f"{model2_name} ext")
    ax.set_xlabel("Forecast day"); ax.set_ylabel("MAE  |fc − obs|  (°C)")
    ax.set_title("Mean Absolute Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(days_arr)

    # Error std
    ax = axes[2]
    ax.plot(days_arr, std1_all, "o-", color=C_IFS,  lw=2, label=f"{model1_name} all")
    ax.plot(days_arr, std2_all, "o-", color=C_AIFS, lw=2, label=f"{model2_name} all")
    ax.plot(days_arr, std1_ext, "s--", color=C_IFS,  lw=1.5, label=f"{model1_name} ext")
    ax.plot(days_arr, std2_ext, "s--", color=C_AIFS, lw=1.5, label=f"{model2_name} ext")
    ax.set_xlabel("Forecast day"); ax.set_ylabel("Error std dev (°C)")
    ax.set_title("Error Spread"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(days_arr)

    fig.suptitle(f"Error Statistics vs Lead Time — {label}",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "2_bias_vs_leadtime.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2] Bias vs lead time → {out.name}")


# ─── Figure 3: Conditional bias ───────────────────────────────────────────────

def fig_conditional_bias(data_by_day, model1_name, model2_name, out_dir, label):
    """Mean bias binned by obs temperature percentile. All lead times overlaid."""
    PCT_EDGES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    pct_mids  = [(PCT_EDGES[i] + PCT_EDGES[i+1]) / 2 for i in range(len(PCT_EDGES)-1)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax_idx, (ax, fc_key, mname, col) in enumerate([
            (axes[0], "fc1_value", model1_name, C_IFS),
            (axes[1], "fc2_value", model2_name, C_AIFS)]):

        for day, (df, _) in sorted(data_by_day.items()):
            obs = df["obs_value"].values
            fc  = df[fc_key].values
            err = fc - obs
            bin_edges = np.percentile(obs, PCT_EDGES)
            # Force unique edges
            _, uniq_idx = np.unique(bin_edges, return_index=True)
            if len(uniq_idx) < 3:
                continue
            bin_edges_u = bin_edges[uniq_idx]
            pct_mid_u   = [pct_mids[i] for i in uniq_idx[:-1]]
            bin_indices = np.digitize(obs, bin_edges_u) - 1
            bin_means = []
            for b in range(len(bin_edges_u) - 1):
                m = bin_indices == b
                bin_means.append(np.mean(err[m]) if m.sum() > 0 else np.nan)
            ax.plot(pct_mid_u[:len(bin_means)], bin_means,
                    "o-", color=col, alpha=DAY_ALPHAS[day],
                    ls=DAY_LS[day], lw=1.5, label=f"Day {day}")

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvspan(0, 10, color="lightblue", alpha=0.2, label="Cold tail")
        ax.axvspan(90, 100, color="lightyellow", alpha=0.4, label="Warm tail")
        ax.set_xlabel("Observed temperature percentile bin")
        ax.set_ylabel("Mean bias  fc − obs  (°C)")
        ax.set_title(f"{mname}  conditional bias", fontsize=11, weight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

    fig.suptitle(f"Conditional Bias by Obs Percentile — {label}",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "3_conditional_bias.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] Conditional bias → {out.name}")


# ─── Figure 4: Bias vs terrain roughness ──────────────────────────────────────

def fig_bias_vs_terrain(data_by_day, model1_name, model2_name, out_dir, label):
    """Mean bias binned by sdfor (terrain roughness).  Uses all lead times pooled."""
    SDFOR_EDGES = [0, 40, 80, 120, 200, 300, 500, 800, 3000]
    sdfor_labels = ["0–40", "40–80", "80–120", "120–200",
                    "200–300", "300–500", "500–800", ">800"]

    all_obs, all_fc1, all_fc2, all_sdfor = [], [], [], []
    for df, _ in data_by_day.values():
        all_obs.append(df["obs_value"].values)
        all_fc1.append(df["fc1_value"].values)
        all_fc2.append(df["fc2_value"].values)
        all_sdfor.append(df["sdfor"].values)

    obs   = np.concatenate(all_obs)
    fc1   = np.concatenate(all_fc1)
    fc2   = np.concatenate(all_fc2)
    sdfor = np.concatenate(all_sdfor)

    me1, me2, mae1, mae2, counts, bin_labs = [], [], [], [], [], []
    for i in range(len(SDFOR_EDGES) - 1):
        m = (sdfor >= SDFOR_EDGES[i]) & (sdfor < SDFOR_EDGES[i+1])
        if m.sum() < 10:
            continue
        me1.append(np.mean(fc1[m] - obs[m]))
        me2.append(np.mean(fc2[m] - obs[m]))
        mae1.append(np.mean(np.abs(fc1[m] - obs[m])))
        mae2.append(np.mean(np.abs(fc2[m] - obs[m])))
        counts.append(m.sum())
        bin_labs.append(sdfor_labels[i])

    x = np.arange(len(bin_labs))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, y1, y2, ylabel, title in [
            (axes[0], me1,  me2,  "Mean Bias  fc−obs  (°C)", "Mean Error (ME)"),
            (axes[1], mae1, mae2, "MAE  |fc−obs|  (°C)",     "Mean Absolute Error (MAE)"),
    ]:
        ax.plot(x, y1, "o-", color=C_IFS,  lw=2, ms=7, label=model1_name)
        ax.plot(x, y2, "o-", color=C_AIFS, lw=2, ms=7, label=model2_name)
        if "ME" in title:
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(bin_labs, rotation=30, ha="right")
        ax.set_xlabel("sdfor bin (terrain roughness)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Secondary axis: station count
        ax2 = ax.twinx()
        ax2.bar(x, counts, width=0.4, alpha=0.15, color="grey")
        ax2.set_ylabel("Station-days (N)", fontsize=8, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=7)

    fig.suptitle(f"Error vs Terrain Roughness (sdfor) — all lead times pooled — {label}",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "4_bias_vs_terrain.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4] Bias vs terrain → {out.name}")


# ─── Figure 5: Extreme skill scores vs lead time ──────────────────────────────

def fig_skill_vs_leadtime(data_by_day, event_type, model1_name, model2_name,
                           out_dir, label):
    days_sorted = sorted(data_by_day.keys())
    pod1, pod2 = [], []
    far1, far2 = [], []
    ets1, ets2 = [], []
    n_ext = []

    for day in days_sorted:
        df, T = data_by_day[day]
        obs = df["obs_value"].values
        fc1 = df["fc1_value"].values
        fc2 = df["fc2_value"].values

        h1, m1, f1, c1 = classify_events(obs, fc1, T, event_type)
        h2, m2, f2, c2 = classify_events(obs, fc2, T, event_type)

        def _safe(a, b): return float(a) / b if b > 0 else np.nan
        pod1.append(_safe(h1.sum(), h1.sum() + m1.sum()))
        pod2.append(_safe(h2.sum(), h2.sum() + m2.sum()))
        far1.append(_safe(f1.sum(), h1.sum() + f1.sum()))
        far2.append(_safe(f2.sum(), h2.sum() + f2.sum()))
        ets1.append(ets_score(h1.sum(), m1.sum(), f1.sum(), c1.sum()))
        ets2.append(ets_score(h2.sum(), m2.sum(), f2.sum(), c2.sum()))
        n_ext.append(int(h1.sum() + m1.sum()))

    days_arr = np.array(days_sorted)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, y1, y2, ylab, title in [
            (axes[0], pod1, pod2, "Probability of Detection (POD)", "POD"),
            (axes[1], far1, far2, "False Alarm Ratio (FAR)", "FAR"),
            (axes[2], ets1, ets2, "Equitable Threat Score (ETS)", "ETS"),
    ]:
        ax.plot(days_arr, y1, "o-", color=C_IFS,  lw=2, ms=7, label=model1_name)
        ax.plot(days_arr, y2, "o-", color=C_AIFS, lw=2, ms=7, label=model2_name)
        ax.set_xlabel("Forecast day"); ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xticks(days_arr)

    # Add n_ext as text on POD panel
    for i, (d, n) in enumerate(zip(days_sorted, n_ext)):
        axes[0].annotate(f"n={n}", (d, min(pod1[i] or 0, pod2[i] or 0) - 0.03),
                         ha="center", fontsize=7, color="0.5")

    fig.suptitle(
        f"Extreme Event Skill ({event_type} threshold) vs Lead Time — {label}",
        fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "5_skill_vs_leadtime.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] Skill scores → {out.name}")


# ─── Figure 6: Error distribution violins ─────────────────────────────────────

def fig_error_violins(data_by_day, event_type, model1_name, model2_name,
                       out_dir, label):
    """Violin plot of fc−obs errors, restricted to observed extreme stations."""
    days_sorted = sorted(data_by_day.keys())
    fig, axes = plt.subplots(1, len(days_sorted), figsize=(4 * len(days_sorted), 7),
                              sharey=True)
    if len(days_sorted) == 1:
        axes = [axes]

    for ax, day in zip(axes, days_sorted):
        df, T = data_by_day[day]
        obs = df["obs_value"].values
        fc1 = df["fc1_value"].values
        fc2 = df["fc2_value"].values

        ext_mask = (obs <= T) if event_type == "below" else (obs >= T)
        if ext_mask.sum() < 5:
            ax.set_title(f"Day {day}\n(no extremes)")
            continue

        err1 = fc1[ext_mask] - obs[ext_mask]
        err2 = fc2[ext_mask] - obs[ext_mask]

        parts = ax.violinplot([err1, err2], positions=[1, 2],
                               showmedians=True, showextrema=False,
                               widths=0.7)
        for pc, col in zip(parts["bodies"], [C_IFS, C_AIFS]):
            pc.set_facecolor(col); pc.set_alpha(0.6)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([model1_name[:8], model2_name[:8]], rotation=30, ha="right")
        ax.set_title(f"Day {day}\n(n={ext_mask.sum():,})", fontsize=10, weight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # KS test p-value
        ks_stat, ks_p = ks_2samp(err1, err2)
        ax.annotate(f"KS p={ks_p:.3f}", xy=(0.5, 0.02), xycoords="axes fraction",
                    ha="center", fontsize=8,
                    color="darkgreen" if ks_p < 0.05 else "grey")

    axes[0].set_ylabel("Error  fc − obs  (°C)")
    fig.suptitle(
        f"Error Distribution — Observed Extreme Stations Only — {label}\n"
        f"KS p<0.05 (green) = distributions significantly different",
        fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "6_error_violins_extremes.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6] Error violins → {out.name}")


# ─── Figure 7: Bias vs elevation (obs_height) ─────────────────────────────────

def fig_bias_vs_elevation(data_by_day, model1_name, model2_name, out_dir, label):
    """Mean bias binned by station elevation (obs_height). All lead times pooled."""
    all_obs, all_fc1, all_fc2, all_elev = [], [], [], []
    for df, _ in data_by_day.values():
        if "obs_height" not in df.columns:
            continue
        all_obs.append(df["obs_value"].values)
        all_fc1.append(df["fc1_value"].values)
        all_fc2.append(df["fc2_value"].values)
        all_elev.append(df["obs_height"].values)

    if not all_obs:
        return

    obs   = np.concatenate(all_obs)
    fc1   = np.concatenate(all_fc1)
    fc2   = np.concatenate(all_fc2)
    elev  = np.concatenate(all_elev)

    ELEV_EDGES  = [0, 200, 400, 600, 800, 1000, 1500, 2000, 4000]
    elev_labels = ["0–200", "200–400", "400–600", "600–800",
                   "800–1000", "1000–1500", "1500–2000", ">2000"]

    me1, me2, mae1, mae2, counts, bin_labs = [], [], [], [], [], []
    for i in range(len(ELEV_EDGES) - 1):
        m = (elev >= ELEV_EDGES[i]) & (elev < ELEV_EDGES[i+1])
        if m.sum() < 5:
            continue
        me1.append(np.mean(fc1[m] - obs[m]))
        me2.append(np.mean(fc2[m] - obs[m]))
        mae1.append(np.mean(np.abs(fc1[m] - obs[m])))
        mae2.append(np.mean(np.abs(fc2[m] - obs[m])))
        counts.append(m.sum())
        bin_labs.append(elev_labels[i])

    if not bin_labs:
        return

    x = np.arange(len(bin_labs))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, y1, y2, ylabel, title in [
            (axes[0], me1,  me2,  "Mean Bias  fc−obs  (°C)", "Mean Error vs Elevation"),
            (axes[1], mae1, mae2, "MAE  |fc−obs|  (°C)",     "MAE vs Elevation"),
    ]:
        ax.plot(x, y1, "o-", color=C_IFS,  lw=2, ms=7, label=model1_name)
        ax.plot(x, y2, "o-", color=C_AIFS, lw=2, ms=7, label=model2_name)
        if "ME" in title or "Error" in title and "MAE" not in title:
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(bin_labs, rotation=30, ha="right")
        ax.set_xlabel("Station elevation (m a.s.l.)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.bar(x, counts, width=0.4, alpha=0.15, color="grey")
        ax2.set_ylabel("Station-days (N)", fontsize=8, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=7)

    fig.suptitle(f"Error vs Station Elevation — all lead times pooled — {label}",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "7_bias_vs_elevation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [7] Bias vs elevation → {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    model1_name = config["read_data"]["forecast_model1"]["name"]
    model2_name = config["read_data"]["forecast_model2"]["name"]
    event_type  = get_event_type(config)
    parquet_dir = config["extract_points"]["output_path"]
    orog_ranges = config.get("filter", {}).get("orography_ranges", DEFAULT_OROG_RANGES)

    season = args.season
    orog   = args.orog
    label  = f"{season or 'all'} / {orog or 'all orog'} | {model1_name} vs {model2_name}"

    # Build a unique auto-name: bias_{event}_{pNN}_{season}_{orog}_{m1}_vs_{m2}
    _pct  = config.get("threshold", {}).get(
                config.get("threshold", {}).get("method", ""), {}
            ).get("percentile", None)
    if _pct is None:
        # Try common sub-keys for any method
        for _sub in ("local_obs_climatology", "fixed", "quantile"):
            _pct = config.get("threshold", {}).get(_sub, {}).get("percentile", None)
            if _pct is not None:
                break
    _pct_str = f"p{int(_pct)}" if _pct is not None else "pX"
    _auto = (f"bias_{event_type}_{_pct_str}"
             f"_{season or 'all'}_{orog or 'all'}"
             f"_{model1_name}_vs_{model2_name}")
    out_dir = Path(args.output_dir) if args.output_dir else \
              Path("case_study_output") / _auto
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  BIAS / DISTRIBUTION ANALYSIS")
    print(f"  Config  : {args.config}")
    print(f"  Models  : {model1_name}  vs  {model2_name}")
    print(f"  Season  : {season or 'all'}")
    print(f"  Orog    : {orog or 'all'}")
    print(f"  Event   : {event_type}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*70}\n")

    # ── Load all forecast days ──
    data_by_day = {}
    for day in DAYS:
        print(f"  Loading day {day} ...", end=" ", flush=True)
        df, T = load_day(parquet_dir, day, season, orog, orog_ranges, config)
        if df is None or len(df) < 100:
            print("(skipped — insufficient data)")
            continue
        print(f"{len(df):,} rows  |  {df['date'].nunique()} dates  "
              f"|  {(~np.isnan(T)).sum():,} with threshold")
        data_by_day[day] = (df, T)

    if not data_by_day:
        print("ERROR: no data loaded. Check season/orog filters and parquet paths.")
        sys.exit(1)

    print(f"\n  Generating figures ...\n")

    fig_qq(data_by_day, event_type, model1_name, model2_name, out_dir, label)
    fig_bias_vs_leadtime(data_by_day, event_type, model1_name, model2_name, out_dir, label)
    fig_conditional_bias(data_by_day, model1_name, model2_name, out_dir, label)
    fig_bias_vs_terrain(data_by_day, model1_name, model2_name, out_dir, label)
    fig_skill_vs_leadtime(data_by_day, event_type, model1_name, model2_name, out_dir, label)
    fig_error_violins(data_by_day, event_type, model1_name, model2_name, out_dir, label)
    fig_bias_vs_elevation(data_by_day, model1_name, model2_name, out_dir, label)

    print(f"\n{'='*70}")
    print(f"  DONE — 7 figures saved to: {out_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
