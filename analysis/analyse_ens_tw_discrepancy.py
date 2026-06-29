"""
analyse_ens_tw_discrepancy.py
─────────────────────────────
Diagnostic tool for understanding why twQS_q099 and twCRPS may show
*opposite* relative performance between two ensemble models in the heatmap.

ROOT CAUSE
----------
twCRPS evaluates the full tail-weighted ensemble *distribution*.  It rewards
calibrated spread AND accuracy simultaneously (fair CRPS formula).

twQS_q099 evaluates only the 99th-percentile ensemble quantile with a strongly
asymmetric penalty (alpha=0.99 → 99x heavier penalty for underprediction than
for overprediction).  A model with wider spread can "earn" a lower twQS_q099
purely through a higher q99, even if its overall calibration is poor.

WHAT THIS TOOL PRODUCES (six diagnostic panels per lead day)
─────────────────────────────────────────────────────────────
 1. Reliability diagram – P(ens member > T) vs observed frequency(obs > T)
    Shows overall probability calibration at the extreme threshold.

 2. Rank histogram – rank of obs within ensemble (all events + extreme-only)
    Flat = calibrated; U-shape = under-dispersive; dome = over-dispersive.

 3. Ensemble spread distribution – violin plots of per-case spread for
    extreme (obs > T) vs non-extreme events.  Wider spread → higher q99.

 4. Conditional quantile fan – for extreme events (obs > T):  box plots of
    ensemble quantiles q10/q25/q50/q75/q90/q95/q99 vs observed values.
    Exposes whether one model's q99 actually brackets the extreme obs.

 5. twCRPS term decomposition – accuracy term vs spread bonus
    (T1 = E|v_T(fc)-v_T(obs)|,  T2 = E|v_T(fc)-v_T(fc')|/2,  twCRPS=T1-T2)
    Identifies whether a lower twCRPS comes from better accuracy or more spread.

 6. twQS_q099 contribution breakdown – split by event type:
    miss (obs>T, q99<T), hit (both>T), false-alarm (obs<T, q99>T).
    Reveals the "source" of the score difference between the two models.

Usage
─────
  python analyse_ens_tw_discrepancy.py \\
      --config config_2t_ens_local_p99obsclim_aifsvsifs.yaml \\
      --season DJF --orog low --days 1 3 5 7 10

SLURM (memory-heavy due to 50-member parquet)
  sbatch --mem=32G --time=00:30:00 --wrap="python analyse_ens_tw_discrepancy.py ..."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from case_studies.case_study_utils import load_per_station_thresholds

# ─── Colours & style ──────────────────────────────────────────────────────────
C1 = "#d7191c"    # model 1 (IFS-like)
C2 = "#2c7bb6"    # model 2 (AIFS-like)
C_OBS = "#1a9641"
ALPHA_FILL = 0.25

SEASON_MONTHS = {
    "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},  "SON": {9, 10, 11},
}
DEFAULT_OROG_RANGES = {"low": (0, 40), "mid": (40, 120), "high": (120, 3000),
                       "flat": (0, 40), "hilly": (40, 120), "complex": (120, 3000)}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",     required=True, help="YAML config file (ensemble mode)")
    p.add_argument("--season",     default=None,
                   help="Season filter: DJF | MAM | JJA | SON  (default: all)")
    p.add_argument("--orog",       default=None,
                   help="Orography class: low | mid | high  (default: all)")
    p.add_argument("--days",        nargs="+", type=int, default=[1, 3, 5, 7, 10],
                   help="Lead days to diagnose (default: 1 3 5 7 10)")
    p.add_argument("--max-samples", type=int,   default=200_000, dest="max_samples",
                   help="Random subsample after filters (default: 200000). "
                        "Use 0 to disable (full data, memory-intensive).")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for subsampling (default: 42)")
    p.add_argument("--output-dir",  default=None, dest="output_dir")
    return p.parse_args()


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_names(config):
    rd = config.get("read_data", {})
    m1 = rd.get("forecast_model1", {}).get("name", "model1")
    m2 = rd.get("forecast_model2", {}).get("name", "model2")
    return m1, m2


def get_event_type(config):
    return config.get("threshold", {}).get("event_type", "above")


def get_orog_ranges(config):
    raw = config.get("filter", {}).get("orography_ranges", None)
    if raw:
        return {k: tuple(v) for k, v in raw.items()}
    return DEFAULT_OROG_RANGES


# ─── Data loading / filtering ─────────────────────────────────────────────────

def _month(date_int):
    return int(str(int(date_int))[4:6])


def apply_filters(df, season, orog, orog_ranges):
    if season and season in SEASON_MONTHS:
        months = SEASON_MONTHS[season]
        df = df[df["date"].apply(_month).isin(months)]
    if orog:
        key = orog.lower()
        if key in orog_ranges:
            lo, hi = orog_ranges[key]
            df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]
    return df


def load_day_ensemble(parquet_dir, day, season, orog, orog_ranges, config,
                      max_samples=200_000, seed=42):
    """Load, filter, and optionally subsample one forecast day.

    Uses chunked reading (100k rows at a time) to avoid exhausting login-node
    memory on large parquet files with 50-member ensemble columns.

    Returns (df, T_arr, fc1_cols, fc2_cols) or (None, None, None, None).
    """
    candidates = list(Path(parquet_dir).glob(f"*_day{day}.parquet"))
    if not candidates:
        print(f"  [day {day}] No parquet file found in {parquet_dir}")
        return None, None, None, None

    pf = pq.ParquetFile(str(candidates[0]))

    # ── Chunked read + filter ────────────────────────────────────────────────
    chunks = []
    n_raw = 0
    for batch in pf.iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        n_raw += len(chunk)
        chunk = apply_filters(chunk, season, orog, orog_ranges)
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        print(f"  [day {day}] Empty after filters (season={season}, orog={orog})")
        return None, None, None, None

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    fc1_cols = sorted([c for c in df.columns if c.startswith("fc1_member_")],
                      key=lambda c: int(c.split("_")[-1]))
    fc2_cols = sorted([c for c in df.columns if c.startswith("fc2_member_")],
                      key=lambda c: int(c.split("_")[-1]))
    if not fc1_cols or not fc2_cols:
        print(f"  [day {day}] No member columns found")
        return None, None, None, None

    # Drop rows missing obs or any member
    valid_cols = ["obs_value"] + fc1_cols + fc2_cols
    df = df.dropna(subset=valid_cols).reset_index(drop=True)

    # Drop rows with clearly corrupt member values (sentinel ≈ -640 in some
    # parquet files; any value below -200 is physically impossible for 2t in °C)
    member_cols = fc1_cols + fc2_cols
    member_np   = df[member_cols].values
    corrupt     = (member_np < -200).any(axis=1)
    if corrupt.any():
        print(f"  [day {day}] Removing {corrupt.sum()} corrupt rows "
              f"(member value < -200)")
        df = df[~corrupt].reset_index(drop=True)
    if df.empty:
        return None, None, None, None

    # ── Optional random subsample ────────────────────────────────────────────
    if max_samples and len(df) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_samples, replace=False)
        idx.sort()
        df = df.iloc[idx].reset_index(drop=True)
        print(f"  [day {day}] Subsampled {max_samples:,} / {n_raw:,} raw rows")
    else:
        print(f"  [day {day}] {len(df):,} cases after filter  "
              f"(from {n_raw:,} raw rows)")

    T_arr = load_per_station_thresholds(config, df)
    valid_T = ~np.isnan(T_arr)
    df = df[valid_T].reset_index(drop=True)
    T_arr = T_arr[valid_T]

    print(f"  [day {day}] Final: {len(df):,} cases  "
          f"(season={season or 'all'}, orog={orog or 'all'})")
    return df, T_arr, fc1_cols, fc2_cols


# ─── Per-case helpers ─────────────────────────────────────────────────────────

def exceedance_prob(ens_np, T_arr):
    """Fraction of members exceeding T (per case).  Returns shape (n,)."""
    return (ens_np > T_arr[:, None]).mean(axis=1)


def ensemble_spread(ens_np):
    """Per-case std of ensemble members. Returns shape (n,)."""
    return ens_np.std(axis=1, ddof=1)


def rank_obs_in_ensemble(obs, ens_np):
    """Rank of obs within ensemble (0 = below all members, M = above all).
    Returned as rank / (M+1) → [0,1] for PIT."""
    M = ens_np.shape[1]
    ranks = (ens_np < obs[:, None]).sum(axis=1)
    return ranks / (M + 1)


def chaining_v(x, T, event_type):
    """Tail chaining function: max(x-T,0) for above, max(T-x,0) for below."""
    if event_type == "above":
        return np.maximum(x - T, 0.0)
    else:
        return np.maximum(T - x, 0.0)


def twcrps_terms(ens_np, obs, T_arr, event_type):
    """Returns (T1_per_case, T2_per_case) arrays for twCRPS decomposition.

    T1 = mean_m |v_T(m) - v_T(obs)| (accuracy / penalty term)
    T2 = fair spread bonus = sorted-trick pairwise term / 2
    twCRPS = mean(T1 - T2)
    """
    M = ens_np.shape[1]
    fc_v = chaining_v(ens_np, T_arr[:, None], event_type)
    obs_v = chaining_v(obs, T_arr, event_type)
    t1 = np.abs(fc_v - obs_v[:, None]).mean(axis=1)

    fc_sorted = np.sort(fc_v, axis=1)
    weights = 2 * np.arange(M) - M + 1
    pairwise = (fc_sorted * weights[None, :]).sum(axis=1) / (M * (M - 1))
    return t1, pairwise  # twCRPS_per_case = t1 - pairwise


def twqs_q099_contributions(ens_np, obs, T_arr, event_type, alpha=0.99):
    """Split twQS_q099 into miss / hit / FA / correct-negative contributions.

    Returns (contrib_miss, contrib_hit, contrib_fa, contrib_cn) each shape (n,).
    """
    q99 = np.quantile(ens_np, alpha, axis=1)
    obs_v  = chaining_v(obs,    T_arr,        event_type)
    q99_v  = chaining_v(q99,    T_arr,        event_type)

    # Event categories at obs / q99 level
    if event_type == "above":
        obs_ext = obs > T_arr
        q99_ext = q99 > T_arr
    else:
        obs_ext = obs < T_arr
        q99_ext = q99 < T_arr

    hit_m = obs_ext &  q99_ext
    miss_m = obs_ext & ~q99_ext
    fa_m  = ~obs_ext &  q99_ext
    cn_m  = ~obs_ext & ~q99_ext

    # Pinball loss in the tail-transformed space
    err = obs_v - q99_v
    loss = np.where(err >= 0, alpha * err, (alpha - 1.0) * err)

    return (loss * miss_m, loss * hit_m, loss * fa_m, loss * cn_m)


# ─── Six-panel figure ─────────────────────────────────────────────────────────

def make_figure(df, T_arr, fc1_cols, fc2_cols, event_type,
                m1_name, m2_name, season, orog, day, out_dir):

    obs    = df["obs_value"].values.astype(np.float64)
    fc1_np = df[fc1_cols].values.astype(np.float64)
    fc2_np = df[fc2_cols].values.astype(np.float64)
    T      = T_arr.astype(np.float64)

    # ── Derived quantities ──
    if event_type == "above":
        extreme_mask = obs > T
        direction = "above"
    else:
        extreme_mask = obs < T
        direction = "below"
    n_ext  = extreme_mask.sum()
    n_all  = len(obs)
    obs_freq = extreme_mask.mean()

    # Exceedance prob (P member exceeds T)
    exc1 = exceedance_prob(fc1_np, T)
    exc2 = exceedance_prob(fc2_np, T)

    # PIT
    pit1_all  = rank_obs_in_ensemble(obs, fc1_np)
    pit2_all  = rank_obs_in_ensemble(obs, fc2_np)
    pit1_ext  = rank_obs_in_ensemble(obs[extreme_mask], fc1_np[extreme_mask])
    pit2_ext  = rank_obs_in_ensemble(obs[extreme_mask], fc2_np[extreme_mask])

    # Spread
    spr1 = ensemble_spread(fc1_np);  spr2 = ensemble_spread(fc2_np)
    spr1_ext = spr1[extreme_mask];   spr2_ext = spr2[extreme_mask]
    spr1_ne  = spr1[~extreme_mask];  spr2_ne  = spr2[~extreme_mask]

    # twCRPS terms
    t1_fc1, t2_fc1 = twcrps_terms(fc1_np, obs, T, event_type)
    t1_fc2, t2_fc2 = twcrps_terms(fc2_np, obs, T, event_type)
    twcrps1 = float(np.mean(t1_fc1 - t2_fc1))
    twcrps2 = float(np.mean(t1_fc2 - t2_fc2))

    # twQS_q099 contributions
    alpha = 0.99 if event_type == "above" else 0.01
    qm1, qh1, qf1, qc1 = twqs_q099_contributions(fc1_np, obs, T, event_type, alpha)
    qm2, qh2, qf2, qc2 = twqs_q099_contributions(fc2_np, obs, T, event_type, alpha)
    twqs1 = float(np.mean(qm1 + qh1 + qf1))
    twqs2 = float(np.mean(qm2 + qh2 + qf2))

    # Conditional quantile fan for extreme events
    _qlevels = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    _qlabels = ["q10", "q25", "q50", "q75", "q90", "q95", "q99"]
    if n_ext >= 10:
        fc1_ext_np = fc1_np[extreme_mask]
        fc2_ext_np = fc2_np[extreme_mask]
        qs1 = np.quantile(fc1_ext_np, _qlevels, axis=1)  # (7, n_ext)
        qs2 = np.quantile(fc2_ext_np, _qlevels, axis=1)
        obs_ext_vals = obs[extreme_mask]
    else:
        qs1 = qs2 = obs_ext_vals = None

    # ────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(
        f"twCRPS vs twQS_q099 discrepancy diagnostic  –  Day {day}  "
        f"[season={season or 'all'}, orog={orog or 'all'}]\n"
        f"{m1_name}  (red, fc1)  vs  {m2_name}  (blue, fc2)     "
        f"N={n_all:,}  ({n_ext:,} extreme,  obs freq={obs_freq*100:.1f}%)",
        fontsize=12, y=0.995,
    )
    gs = mgridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.36,
                            left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ── Panel 1: Reliability diagram ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _n_bins = 20
    bin_edges = np.linspace(0, 1, _n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for exc, c, lbl in [(exc1, C1, m1_name), (exc2, C2, m2_name)]:
        obs_freq_bin = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (exc >= lo) & (exc < hi)
            obs_freq_bin.append(obs[mask] > T[mask] if event_type == "above"
                                else obs[mask] < T[mask])
        obs_freq_bin = [m.mean() if len(m) > 0 else np.nan for m in obs_freq_bin]
        ax1.plot(bin_centres, obs_freq_bin, "o-", color=c, ms=4, lw=1.5, label=lbl)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax1.axhline(obs_freq, color="gray", lw=0.8, ls=":", label=f"Clim ({obs_freq*100:.1f}%)")
    ax1.set_xlabel(f"Forecast P(obs {direction} T)", fontsize=9)
    ax1.set_ylabel(f"Observed frequency (obs {direction} T)", fontsize=9)
    ax1.set_title("1 · Reliability diagram", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Rank / PIT histogram ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _nb = 25
    _pit_bins = np.linspace(0, 1, _nb + 1)
    clim_count_all = n_all / _nb
    clim_count_ext = n_ext / _nb if n_ext > 0 else 1
    for (pit, c, lbl, clim) in [
        (pit1_all, C1, f"{m1_name} (all)", clim_count_all),
        (pit2_all, C2, f"{m2_name} (all)", clim_count_all),
    ]:
        counts, _ = np.histogram(pit, bins=_pit_bins)
        ax2.step(_pit_bins[:-1], counts / clim, where="post",
                 color=c, lw=1.5, label=lbl)
    for (pit, c, lbl) in [
        (pit1_ext, C1, f"{m1_name} (extreme)"),
        (pit2_ext, C2, f"{m2_name} (extreme)"),
    ]:
        counts, _ = np.histogram(pit, bins=_pit_bins)
        ax2.step(_pit_bins[:-1], counts / clim_count_ext, where="post",
                 color=c, lw=1.0, ls="--", label=lbl)
    ax2.axhline(1.0, color="gray", lw=0.8, ls=":", label="Uniform (calibrated)")
    ax2.set_xlabel("PIT / rank (0=below all members)", fontsize=9)
    ax2.set_ylabel("Relative frequency (÷ uniform)", fontsize=9)
    ax2.set_title("2 · Rank histogram (PIT)\nAll events (solid)  ·  Extreme-only (dashed)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Spread distribution (violin) ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _vdata = {
        f"{m1_name}\n(all)":      spr1,
        f"{m2_name}\n(all)":      spr2,
        f"{m1_name}\n(extreme)":  spr1_ext if n_ext >= 5 else np.array([np.nan]),
        f"{m2_name}\n(extreme)":  spr2_ext if n_ext >= 5 else np.array([np.nan]),
    }
    _vcolors = [C1, C2, C1, C2]
    _vpositions = [1, 2, 3.5, 4.5]
    for (lbl, data), col, pos in zip(_vdata.items(), _vcolors, _vpositions):
        clean = data[~np.isnan(data)]
        if len(clean) < 3:
            continue
        vp = ax3.violinplot([clean], positions=[pos], widths=0.7,
                            showmedians=True, showextrema=False)
        for part in vp["bodies"]:
            part.set_facecolor(col); part.set_alpha(0.5)
        vp["cmedians"].set_color(col); vp["cmedians"].set_linewidth(2)
    ax3.set_xticks(_vpositions)
    ax3.set_xticklabels([f"{m1_name}\n(all)", f"{m2_name}\n(all)",
                         f"{m1_name}\n(ext)", f"{m2_name}\n(ext)"],
                        fontsize=8)
    ax3.set_ylabel("Ensemble spread  σ  (°C)", fontsize=9)
    ax3.set_title("3 · Ensemble spread distribution\nAll events vs extreme events (obs {})".format(direction + " T"),
                  fontsize=10, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # ── Panel 4: Conditional quantile fan for extreme events ──────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if qs1 is not None:
        _x = np.arange(len(_qlevels))
        _w = 0.35
        for i, (qs, c, lbl) in enumerate([(qs1, C1, m1_name), (qs2, C2, m2_name)]):
            med = np.median(qs, axis=1)  # median over extreme cases, per quantile level
            q25 = np.percentile(qs, 25, axis=1)
            q75 = np.percentile(qs, 75, axis=1)
            offset = -_w/2 + i * _w
            ax4.bar(_x + offset, med, width=_w, color=c, alpha=0.6, label=lbl, zorder=3)
            ax4.errorbar(_x + offset, med, yerr=[med - q25, q75 - med],
                         fmt="none", color=c, capsize=3, lw=1.5, zorder=4)
        # Median observed value (horizontal line, using the median of extreme obs)
        obs_med_ext = np.median(obs_ext_vals)
        obs_p25_ext = np.percentile(obs_ext_vals, 25)
        obs_p75_ext = np.percentile(obs_ext_vals, 75)
        ax4.axhline(obs_med_ext, color=C_OBS, lw=2, ls="-", label=f"Obs median ({obs_med_ext:.1f}°C)")
        ax4.axhspan(obs_p25_ext, obs_p75_ext, color=C_OBS, alpha=0.12,
                    label="Obs IQR")
        ax4.axhline(np.mean(T[extreme_mask]), color="k", lw=1.2, ls="--",
                    label=f"Mean T ({np.mean(T[extreme_mask]):.1f}°C)")
        ax4.set_xticks(_x)
        ax4.set_xticklabels(_qlabels, fontsize=9)
        ax4.set_ylabel("Temperature (°C)", fontsize=9)
        ax4.legend(fontsize=8, loc="upper left" if event_type == "above" else "lower right")
    else:
        ax4.text(0.5, 0.5, f"Too few extreme events\n(n={n_ext})",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=11)
    ax4.set_title(f"4 · Conditional quantile fan  (obs {direction} T only)\n"
                  f"Bar = median q-level across cases  ·  error bar = IQR  "
                  f"[n={n_ext:,}]", fontsize=10, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # ── Panel 5: twCRPS decomposition ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _models = [m1_name, m2_name]
    _colors  = [C1, C2]
    _t1 = [float(np.mean(t1_fc1)), float(np.mean(t1_fc2))]
    _t2 = [float(np.mean(t2_fc1)), float(np.mean(t2_fc2))]
    _net = [_t1[i] - _t2[i] for i in range(2)]
    _x = np.array([0.0, 1.0])
    _bw = 0.35
    for i, (c, label) in enumerate(zip(_colors, _models)):
        ax5.bar(_x[i] - _bw/4, _t1[i], width=_bw, color=c, alpha=0.85,
                label=f"{label}\nT1={_t1[i]:.4f}, T2={_t2[i]:.4f}, net={_net[i]:.4f}",
                zorder=3)
        ax5.bar(_x[i] + _bw/4, _t2[i], width=_bw, color=c, alpha=0.4,
                hatch="//", edgecolor=c, zorder=3)
    # Mark net twCRPS
    for i, c in enumerate(_colors):
        ax5.plot(_x[i], _net[i], marker="D", ms=9, color=c, zorder=5,
                 markeredgecolor="k", markeredgewidth=0.8)
    ax5.axhline(0, color="k", lw=0.8)
    _diff_sign = "↓ better" if twcrps1 > twcrps2 else ("↑ better" if twcrps1 < twcrps2 else "=")
    ax5.set_title(
        f"5 · twCRPS decomposition\n"
        f"Solid bar=accuracy T1  ·  Hatched bar=spread bonus T2  ·  ◆=net twCRPS\n"
        f"{m1_name}={twcrps1:.5f}  {m2_name}={twcrps2:.5f}  ({_diff_sign})",
        fontsize=9, fontweight="bold",
    )
    ax5.set_xticks(_x)
    ax5.set_xticklabels(_models, fontsize=9)
    ax5.set_ylabel("Score contribution (°C)", fontsize=9)
    ax5.legend(fontsize=7, loc="upper right")
    ax5.grid(axis="y", alpha=0.3)

    # ── Panel 6: twQS_q099 contribution breakdown ─────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _categories = ["Miss\n(obs ext, q99 not)", "Hit\n(both ext)",
                   "False alarm\n(q99 ext, obs not)", "Correct neg\n(neither ext)"]
    _cat_colors = ["#e31a1c", "#33a02c", "#ff7f00", "#aaaaaa"]
    _model_contribs = [
        [np.mean(qm1), np.mean(qh1), np.mean(qf1), np.mean(qc1)],
        [np.mean(qm2), np.mean(qh2), np.mean(qf2), np.mean(qc2)],
    ]
    _xpos = np.array([0.0, 0.6])
    _bw6 = 0.5
    bottom1 = np.zeros(2); bottom2 = np.zeros(2)
    bar_handles = []
    for j, (cat, ccol) in enumerate(_zip_cats(_categories, _cat_colors)):
        vals = [_model_contribs[0][j], _model_contribs[1][j]]
        bars = ax6.bar(_xpos, vals, width=_bw6, bottom=bottom1,
                       color=ccol, alpha=0.8, label=cat, zorder=3)
        bar_handles.append(bars)
        bottom1 += vals
    # Label total twQS on top
    for i, (xp, total) in enumerate(zip(_xpos, [twqs1, twqs2])):
        ax6.text(xp, float(bottom1[i]) + 0.0002, f"{total:.5f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    _diff_sign2 = "↓ better" if twqs1 > twqs2 else ("↑ better" if twqs1 < twqs2 else "=")
    ax6.set_title(
        f"6 · twQS_q099 contribution breakdown\n"
        f"alpha={alpha}  |  {m1_name}={twqs1:.5f}  {m2_name}={twqs2:.5f}  ({_diff_sign2})",
        fontsize=10, fontweight="bold",
    )
    ax6.set_xticks(_xpos)
    ax6.set_xticklabels(_models, fontsize=9)
    ax6.set_ylabel(f"Mean twQS contribution (°C)", fontsize=9)
    ax6.legend(fontsize=8, loc="upper right")
    ax6.grid(axis="y", alpha=0.3)

    # ── Score summary annotation ──────────────────────────────────────────────
    _winner_crps = m2_name if twcrps2 < twcrps1 else m1_name
    _winner_twqs = m2_name if twqs2  < twqs1  else m1_name
    _same = _winner_crps == _winner_twqs
    _summary = (
        f"twCRPS winner: {_winner_crps}  |  twQS_q099 winner: {_winner_twqs}  "
        + ("  → CONSISTENT" if _same else "  → DISCREPANT  (check panels 3 & 5 & 6)")
    )
    fig.text(0.5, 0.002, _summary, ha="center", fontsize=10,
             color="black" if _same else "#c0392b", fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────────────
    tag = f"day{day}_{season or 'all'}_{orog or 'all'}"
    out_path = Path(out_dir) / f"ens_tw_discrepancy_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")
    return out_path


def _zip_cats(categories, colors):
    return list(zip(categories, colors))


def _twqs_mean(ens_np, obs, T_arr, alpha, event_type):
    """Mean twQS at a single alpha level, matching the scores library formula.

    Uses Taggart (2022) / Gneiting (2011) consistent scoring with a nondecreasing
    auxiliary function g:
      - 'above': g(x) = max(x-T, 0)  →  eff_alpha = alpha  (miss heavy at q99)
      - 'below': g(x) = min(x, T) - T + const  →  eff_alpha = 1-alpha  (miss heavy at q01)

    For both event types at the extreme α level:
      miss (obs exceeds q-level) → (1-α)·|err|  (expensive)
      false alarm (q-level exceeds obs) → α·|err|  (cheap)
    """
    q = np.quantile(ens_np, alpha, axis=1)
    if event_type == "above":
        obs_v    = np.maximum(obs - T_arr, 0.0)
        q_v      = np.maximum(q   - T_arr, 0.0)
        eff_alpha = alpha
    else:
        obs_v    = np.maximum(T_arr - obs, 0.0)
        q_v      = np.maximum(T_arr - q,   0.0)
        # For lower tail, the consistent scoring function (Taggart 2022) uses
        # g = min(x, T), which is nondecreasing and flips the effective alpha:
        # eff_alpha = 1 - alpha so that misses (obs more extreme than q-level)
        # still carry the heavy (1-alpha) penalty.
        eff_alpha = 1.0 - alpha
    err = obs_v - q_v
    return float(np.mean(np.where(err >= 0, eff_alpha * err, (eff_alpha - 1.0) * err)))


# ─── Alpha comparison figure (QQ + score bar + miss/FA + penalty asymmetry) ───

def make_alpha_comparison_figure(df, T_arr, fc1_cols, fc2_cols, event_type,
                                  m1_name, m2_name, season, orog, day,
                                  twcrps1, twcrps2, out_dir):
    """Second focused figure: why q90/q95 agree with twCRPS but q99 disagrees.

    Six panels:
      Row 1  (QQ plots): sorted ensemble q90/q95/q99 vs sorted extreme obs.
             Points above 1:1 = model quantile too high (warm bias).
      Row 2a (score bar chart): % difference for twCRPS and twQS q90/q95/q99.
             Reveals where the sign flip occurs.
      Row 2b (miss & FA rates): fraction of events that are misses or FAs for
             each alpha level.  Shows AIFS converts misses→hits via warm bias.
      Row 2c (penalty asymmetry): per-event mean miss-penalty vs FA-penalty for
             each alpha level.  At α=0.99 the FA penalty is 100× smaller than
             the miss penalty, so AIFS's warm-bias FAs are nearly free.
    """
    obs    = df["obs_value"].values.astype(np.float64)
    fc1_np = df[fc1_cols].values.astype(np.float64)
    fc2_np = df[fc2_cols].values.astype(np.float64)
    T      = T_arr.astype(np.float64)

    ALPHAS = [0.90, 0.95, 0.99]
    ALPHA_LABELS = ["q90 (α=0.90)", "q95 (α=0.95)", "q99 (α=0.99)"]
    ALPHA_SHORT  = ["q90", "q95", "q99"]

    if event_type == "above":
        ext_mask = obs > T
    else:
        ext_mask = obs < T
    n_ext = ext_mask.sum()
    obs_ext = obs[ext_mask]
    T_mean  = float(np.mean(T))

    # ── Pre-compute per-alpha quantities ──────────────────────────────────────
    twqs_fc1 = {}; twqs_fc2 = {}
    miss_rate1 = {}; miss_rate2 = {}
    fa_rate1   = {}; fa_rate2   = {}
    miss_pen1  = {}; miss_pen2  = {}   # mean penalty per MISS event
    fa_pen1    = {}; fa_pen2    = {}   # mean penalty per FA   event
    q_fc1_ext  = {}; q_fc2_ext  = {}   # quantile values for extreme cases

    for alpha in ALPHAS:
        q1 = np.quantile(fc1_np, alpha, axis=1)
        q2 = np.quantile(fc2_np, alpha, axis=1)

        if event_type == "above":
            miss1_m = ext_mask & (q1 < T);  miss2_m = ext_mask & (q2 < T)
            fa1_m   = (~ext_mask) & (q1 > T); fa2_m = (~ext_mask) & (q2 > T)
            # penalty = alpha*(obs-T) for miss, (1-alpha)*(q-T) for FA
            miss_pen1[alpha] = float(np.mean(alpha * (obs[miss1_m] - T[miss1_m]))) if miss1_m.any() else 0.0
            miss_pen2[alpha] = float(np.mean(alpha * (obs[miss2_m] - T[miss2_m]))) if miss2_m.any() else 0.0
            fa_pen1[alpha]   = float(np.mean((1-alpha) * (q1[fa1_m] - T[fa1_m]))) if fa1_m.any() else 0.0
            fa_pen2[alpha]   = float(np.mean((1-alpha) * (q2[fa2_m] - T[fa2_m]))) if fa2_m.any() else 0.0
        else:
            miss1_m = ext_mask & (q1 > T);  miss2_m = ext_mask & (q2 > T)
            fa1_m   = (~ext_mask) & (q1 < T); fa2_m = (~ext_mask) & (q2 < T)
            miss_pen1[alpha] = float(np.mean((1-alpha) * (T[miss1_m] - obs[miss1_m]))) if miss1_m.any() else 0.0
            miss_pen2[alpha] = float(np.mean((1-alpha) * (T[miss2_m] - obs[miss2_m]))) if miss2_m.any() else 0.0
            fa_pen1[alpha]   = float(np.mean(alpha * (T[fa1_m] - q1[fa1_m]))) if fa1_m.any() else 0.0
            fa_pen2[alpha]   = float(np.mean(alpha * (T[fa2_m] - q2[fa2_m]))) if fa2_m.any() else 0.0

        miss_rate1[alpha] = float(miss1_m.mean())
        miss_rate2[alpha] = float(miss2_m.mean())
        fa_rate1[alpha]   = float(fa1_m.mean())
        fa_rate2[alpha]   = float(fa2_m.mean())
        twqs_fc1[alpha]   = _twqs_mean(fc1_np, obs, T, alpha, event_type)
        twqs_fc2[alpha]   = _twqs_mean(fc2_np, obs, T, alpha, event_type)
        q_fc1_ext[alpha]  = np.sort(q1[ext_mask]) if n_ext >= 5 else None
        q_fc2_ext[alpha]  = np.sort(q2[ext_mask]) if n_ext >= 5 else None

    obs_ext_sorted = np.sort(obs_ext)
    n_all = len(obs)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Why q90/q95 agree with twCRPS but q99 disagrees  –  Day {day}  "
        f"[season={season or 'all'}, orog={orog or 'all'}]\n"
        f"{m1_name} (red)  vs  {m2_name} (blue)     "
        f"N={n_all:,}   extreme events: {n_ext:,} ({n_ext/n_all*100:.1f}%)",
        fontsize=11, y=0.998,
    )
    gs = mgridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.42, wspace=0.35,
                            left=0.07, right=0.97, top=0.92, bottom=0.07)

    direction = "above" if event_type == "above" else "below"

    # ── Row 1: QQ plots ───────────────────────────────────────────────────────
    for col, (alpha, albl, ashort) in enumerate(zip(ALPHAS, ALPHA_LABELS, ALPHA_SHORT)):
        ax = fig.add_subplot(gs[0, col])

        if q_fc1_ext[alpha] is not None:
            # 1:1 line spanning obs_ext range
            lo = float(min(obs_ext_sorted.min(),
                           q_fc1_ext[alpha].min(), q_fc2_ext[alpha].min())) * 0.99
            hi = float(max(obs_ext_sorted.max(),
                           q_fc1_ext[alpha].max(), q_fc2_ext[alpha].max())) * 1.01
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1", zorder=1)

            ax.plot(obs_ext_sorted, q_fc1_ext[alpha], color=C1,  lw=1.8,
                    label=f"{m1_name}")
            ax.plot(obs_ext_sorted, q_fc2_ext[alpha], color=C2,  lw=1.8,
                    label=f"{m2_name}")

            # Shade region above 1:1 (warm bias zone)
            ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                            color="orange", alpha=0.07, label="Model too warm")
            ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                            color="steelblue", alpha=0.07, label="Model too cold")

            # Mark threshold T (average)
            ax.axvline(T_mean, color="gray", lw=0.9, ls=":", alpha=0.7)
            ax.axhline(T_mean, color="gray", lw=0.9, ls=":", alpha=0.7)
            ax.text(T_mean, lo + (hi-lo)*0.02, f" T≈{T_mean:.1f}°C",
                    fontsize=7, color="gray", va="bottom")

            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Sorted obs (extreme events, obs > T)", fontsize=8)
            ax.set_ylabel(f"Sorted model {ashort} (extreme events)", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"n_extreme={n_ext}\n(too few)", ha="center",
                    va="center", transform=ax.transAxes)

        # Annotation: warm-bias fraction (fraction of extreme events where q > obs)
        if q_fc1_ext[alpha] is not None:
            wb1 = float((q_fc1_ext[alpha] > obs_ext_sorted).mean())
            wb2 = float((q_fc2_ext[alpha] > obs_ext_sorted).mean())
            ax.text(0.03, 0.97,
                    f"% q > obs (extreme):\n"
                    f"{m1_name}: {wb1*100:.0f}%\n"
                    f"{m2_name}: {wb2*100:.0f}%",
                    transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(f"{col+1}. QQ — {albl}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.25)

    # ── Row 2, Panel 4: Score % differences bar chart ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    score_labels = ["twCRPS"] + [f"twQS_{s}" for s in ALPHA_SHORT]
    fc1_scores   = [twcrps1] + [twqs_fc1[a] for a in ALPHAS]
    fc2_scores   = [twcrps2] + [twqs_fc2[a] for a in ALPHAS]
    pct_diffs    = [(s2 - s1) / s1 * 100 if s1 != 0 else 0.0
                    for s1, s2 in zip(fc1_scores, fc2_scores)]

    bar_colors = [C2 if p < 0 else C1 for p in pct_diffs]  # blue=fc2 better, red=fc1 better
    bars = ax4.barh(score_labels, pct_diffs, color=bar_colors, alpha=0.75, edgecolor="k", lw=0.5)
    ax4.axvline(0, color="k", lw=1.0)
    for bar, pct in zip(bars, pct_diffs):
        xoff = -0.3 if pct < 0 else 0.3
        ax4.text(pct + xoff, bar.get_y() + bar.get_height()/2,
                 f"{pct:+.1f}%", va="center", ha="left" if pct >= 0 else "right",
                 fontsize=8)
    ax4.set_xlabel(f"% diff  (fc2−fc1)/fc1×100\nNeg = {m2_name} better  ·  Pos = {m1_name} better",
                   fontsize=8)
    ax4.set_title(f"4. Score % differences\n(consistent = same colour)", fontsize=10, fontweight="bold")
    ax4.grid(axis="x", alpha=0.3)
    # Add legend proxy
    from matplotlib.patches import Patch
    ax4.legend(handles=[Patch(color=C2, alpha=0.75, label=f"{m2_name} better"),
                         Patch(color=C1, alpha=0.75, label=f"{m1_name} better")],
               fontsize=7.5, loc="lower right")

    # ── Row 2, Panel 5: Miss & FA rates per alpha ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(ALPHAS))
    bw = 0.2
    ax5.bar(x_pos - 1.5*bw, [miss_rate1[a]*100 for a in ALPHAS], width=bw,
            color=C1, alpha=0.85, label=f"{m1_name} miss", hatch="")
    ax5.bar(x_pos - 0.5*bw, [miss_rate2[a]*100 for a in ALPHAS], width=bw,
            color=C2, alpha=0.85, label=f"{m2_name} miss")
    ax5.bar(x_pos + 0.5*bw, [fa_rate1[a]*100 for a in ALPHAS], width=bw,
            color=C1, alpha=0.4, hatch="//", edgecolor=C1, label=f"{m1_name} FA")
    ax5.bar(x_pos + 1.5*bw, [fa_rate2[a]*100 for a in ALPHAS], width=bw,
            color=C2, alpha=0.4, hatch="//", edgecolor=C2, label=f"{m2_name} FA")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(ALPHA_SHORT, fontsize=9)
    ax5.set_ylabel("Event rate (% of all cases)", fontsize=8)
    ax5.set_title(f"5. Miss rate (solid)  &  FA rate (hatched)\n"
                  f"obs {direction} T, q < T = miss  ·  q > T, obs not = FA",
                  fontsize=9, fontweight="bold")
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(axis="y", alpha=0.3)

    # ── Row 2, Panel 6: Per-event penalty asymmetry ───────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    # Show: for each alpha, the FA penalty weight is (1-alpha) vs miss penalty weight alpha
    # This is the fundamental reason q99 behaves differently
    fa_weights   = [(1 - a) * 100 for a in ALPHAS]   # in % of miss weight
    miss_weights = [a * 100       for a in ALPHAS]

    ax6b = ax6.twinx()
    ax6.bar(x_pos - bw/2, miss_pen1.values(), width=bw, color=C1, alpha=0.85,
            label=f"{m1_name} mean miss penalty")
    ax6.bar(x_pos + bw/2, miss_pen2.values(), width=bw, color=C2, alpha=0.85,
            label=f"{m2_name} mean miss penalty")
    ax6.bar(x_pos - bw/2, [-fa_pen1[a] for a in ALPHAS], width=bw, color=C1,
            alpha=0.4, hatch="//", edgecolor=C1, label=f"{m1_name} mean FA penalty (neg)")
    ax6.bar(x_pos + bw/2, [-fa_pen2[a] for a in ALPHAS], width=bw, color=C2,
            alpha=0.4, hatch="//", edgecolor=C2, label=f"{m2_name} mean FA penalty (neg)")

    # Overlay FA weight factor as a line
    ax6b.plot(x_pos, fa_weights, "k^--", ms=8, lw=1.5, label="FA penalty weight (1-α)×100")
    ax6b.set_ylabel("FA penalty weight  (1−α)×100", fontsize=8, color="k")
    ax6b.set_ylim(0, max(fa_weights) * 3)
    ax6b.tick_params(axis="y", labelsize=8)

    ax6.axhline(0, color="k", lw=0.8)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(ALPHA_SHORT, fontsize=9)
    ax6.set_ylabel("Mean per-event penalty (°C)", fontsize=8)
    ax6.set_title(
        f"6. Per-event penalties: miss (pos) vs FA (neg)\n"
        f"▲ = FA weight (1−α): {fa_weights[0]:.0f}% → {fa_weights[1]:.0f}% → {fa_weights[2]:.0f}%  "
        f"← 10× drop at q99!",
        fontsize=9, fontweight="bold",
    )
    lines1, labs1 = ax6.get_legend_handles_labels()
    lines2, labs2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labs1 + labs2, fontsize=6.5, loc="lower left", ncol=2)
    ax6.grid(axis="y", alpha=0.3)

    # ── Footer explanation ────────────────────────────────────────────────────
    fig.text(
        0.5, 0.001,
        "At α=0.99: FA penalty weight = (1−0.99) = 0.01  →  a warm-biased q99 converts misses to hits "
        "with virtually NO penalty for extra false alarms.  At α=0.90: FA weight = 0.10 (10× larger), "
        "so warm-bias FAs actually cost enough to keep the score consistent with twCRPS.",
        ha="center", fontsize=8.5, style="italic",
        wrap=True,
    )

    tag = f"day{day}_{season or 'all'}_{orog or 'all'}"
    out_path = Path(out_dir) / f"ens_alpha_comparison_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")
    return out_path


# ─── Simple 3-panel explanation figure ───────────────────────────────────────

def make_simple_explanation_figure(df, T_arr, fc1_cols, fc2_cols, event_type,
                                   m1_name, m2_name, season, orog, day,
                                   twcrps1, twcrps2, out_dir):
    """Three clear panels explaining why twCRPS and twQS_q99 disagree.

    Panel 1 — Score % differences: shows sign flip at q99.
    Panel 2 — Warm bias: % of extreme events where model q-level overshoots obs.
    Panel 3 — Penalty weights: miss weight (α) vs FA weight (1−α) per level.
    """
    obs    = df["obs_value"].values.astype(np.float64)
    fc1_np = df[fc1_cols].values.astype(np.float64)
    fc2_np = df[fc2_cols].values.astype(np.float64)
    T      = T_arr.astype(np.float64)

    if event_type == "above":
        ALPHAS = [0.90, 0.95, 0.99]
        LABELS = ["q90", "q95", "q99"]
    else:
        ALPHAS = [0.01, 0.05, 0.10]
        LABELS = ["q01", "q05", "q10"]

    if event_type == "above":
        ext_mask = obs > T
    else:
        ext_mask = obs < T
    non_ext_mask = ~ext_mask
    n_ext = int(ext_mask.sum())
    n_all = len(obs)

    twqs_fc1 = {}; twqs_fc2 = {}
    bias_rate1 = {}; bias_rate2 = {}   # % of relevant events showing model bias

    for alpha in ALPHAS:
        q1 = np.quantile(fc1_np, alpha, axis=1)
        q2 = np.quantile(fc2_np, alpha, axis=1)
        twqs_fc1[alpha] = _twqs_mean(fc1_np, obs, T, alpha, event_type)
        twqs_fc2[alpha] = _twqs_mean(fc2_np, obs, T, alpha, event_type)
        if event_type == "above":
            # % extreme events where q overshoots obs (warm bias during extremes)
            if n_ext > 0:
                bias_rate1[alpha] = float((q1[ext_mask] > obs[ext_mask]).mean()) * 100
                bias_rate2[alpha] = float((q2[ext_mask] > obs[ext_mask]).mean()) * 100
            else:
                bias_rate1[alpha] = bias_rate2[alpha] = 0.0
        else:
            # % extreme events where q-level > obs (model too warm = MISS for cold extreme)
            # Misses are costly (1-α penalty), so this shows what drives twQS for cold events
            if n_ext > 0:
                bias_rate1[alpha] = float((q1[ext_mask] > obs[ext_mask]).mean()) * 100
                bias_rate2[alpha] = float((q2[ext_mask] > obs[ext_mask]).mean()) * 100
            else:
                bias_rate1[alpha] = bias_rate2[alpha] = 0.0

    def pct_diff(s1, s2):
        return 0.0 if abs(s1) < 1e-12 else (s2 - s1) / abs(s1) * 100

    score_names = ["twCRPS"] + [f"twQS  {lbl}" for lbl in LABELS]
    score_vals  = [pct_diff(twcrps1, twcrps2)] + [
        pct_diff(twqs_fc1[a], twqs_fc2[a]) for a in ALPHAS
    ]
    bar_colors = [C1 if v > 0 else C2 for v in score_vals]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Day {day}  ·  {season or 'all'}, {orog or 'all'}  ·  "
        f"{m1_name} (red)  vs  {m2_name} (blue)  ·  "
        f"N={n_all:,} cases, {n_ext:,} extreme ({n_ext/n_all*100:.1f}%)",
        fontsize=10,
    )

    # ── Panel 1: Score % differences ─────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Which model scores better?", fontsize=11, fontweight="bold")
    y = np.arange(len(score_names))
    ax.barh(y, score_vals, color=bar_colors, height=0.55, edgecolor="white", zorder=2)
    ax.axvline(0, color="black", lw=1.5, zorder=3)
    xlim = max(abs(v) for v in score_vals) * 1.5 or 5
    ax.set_xlim(-xlim, xlim)
    for i, v in enumerate(score_vals):
        ha = "right" if v > 0 else "left"
        offset = -xlim * 0.04 if v > 0 else xlim * 0.04
        ax.text(v + offset, i, f"{v:+.1f}%", va="center", ha=ha, fontsize=10,
                fontweight="bold",
                color=C1 if v > 0 else C2)
    ax.set_yticks(y)
    ax.set_yticklabels(score_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("% score difference  (blue − red) / |red|  ×  100", fontsize=9)
    ax.axvspan(-xlim, 0, alpha=0.05, color=C2, zorder=1)
    ax.axvspan(0, xlim, alpha=0.05, color=C1, zorder=1)
    ax.text(-xlim * 0.97, len(score_names) - 0.15, f"{m2_name}\nbetter →",
            fontsize=8, color=C2, ha="left", va="bottom")
    ax.text( xlim * 0.97, len(score_names) - 0.15, f"← {m1_name}\nbetter",
            fontsize=8, color=C1, ha="right", va="bottom")
    if event_type == "above":
        # Arrow pointing to q99 row (sign flip)
        ax.annotate("sign flip\nhere!", xy=(score_vals[3], 3),
                    xytext=(xlim * 0.55, 3.0 + 0.9),
                    fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    else:
        # Arrow pointing to twCRPS (discrepancy vs all twQS)
        ax.annotate("twCRPS = integral\nof twQS → rewards spread",
                    xy=(score_vals[0], 0),
                    xytext=(xlim * 0.35, 0 + 1.2),
                    fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # ── Panel 2: Tail bias ────────────────────────────────────────────────────
    ax = axes[1]
    if event_type == "above":
        bias_title = "Model tail bias:\n% extreme events where  q-level > obs"
        bias_subtitle = ("Higher % = model q-level systematically overshoots extreme obs\n"
                         "(warm bias in tail gets worse at higher quantile levels)")
    else:
        bias_title = "Model miss rate:\n% extreme events where  q-level > obs"
        bias_subtitle = ("Higher % = model q-level too warm during cold extremes (misses)\n"
                         "Misses carry (1−α) penalty at q01 = 99% → drives twQS winner")
    ax.set_title(bias_title, fontsize=11, fontweight="bold")
    x = np.arange(len(ALPHAS))
    v1 = [bias_rate1[a] for a in ALPHAS]
    v2 = [bias_rate2[a] for a in ALPHAS]
    ax.plot(x, v1, "o-", color=C1, lw=2.5, ms=10, label=m1_name, zorder=3)
    ax.plot(x, v2, "o-", color=C2, lw=2.5, ms=10, label=m2_name, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", label="Ideal ≈ 0%")
    for xi, (a1, a2) in enumerate(zip(v1, v2)):
        ax.text(xi + 0.06, a1 + 2.5, f"{a1:.0f}%", color=C1, ha="left",
                fontsize=10, fontweight="bold")
        ax.text(xi + 0.06, a2 + 2.5, f"{a2:.0f}%", color=C2, ha="left",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel("% of extreme events", fontsize=9)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.5, -0.16, bias_subtitle,
            transform=ax.transAxes, ha="center", fontsize=8, style="italic")

    # ── Panel 3: Penalty weights ──────────────────────────────────────────────
    ax = axes[2]
    if event_type == "above":
        panel3_title = "Why q99 is fooled:\npenalty weights for misses vs false alarms"
        panel3_note  = (f"At q99: each false alarm costs only 1%, each miss costs 99%.\n"
                        f"{m2_name}'s warm bias converts misses → false alarms, nearly for free.")
        miss_w = [a * 100  for a in ALPHAS]   # α% = miss weight
        fa_w   = [(1-a)*100 for a in ALPHAS]  # (1-α)% = FA weight
    else:
        panel3_title = "Penalty weights: same structure as warm extremes"
        panel3_note  = (f"At q01: miss costs 99%, false alarm costs only 1%.\n"
                        f"Both event types penalise misses heavily — consistent scoring function.")
        miss_w = [(1-a)*100 for a in ALPHAS]  # (1-α)% = miss weight (99% at q01)
        fa_w   = [a * 100  for a in ALPHAS]   # α% = FA weight (1% at q01)
    ax.set_title(panel3_title, fontsize=11, fontweight="bold")
    width  = 0.32
    ax.bar(x - width/2, miss_w, width, color="tomato",    label="Miss weight  (α %)")
    ax.bar(x + width/2, fa_w,   width, color="steelblue", label="FA weight  ((1−α) %)")
    for xi, (m, f) in enumerate(zip(miss_w, fa_w)):
        ax.text(xi - width/2, m + 1, f"{m:.0f}%", ha="center", fontsize=10,
                color="tomato", fontweight="bold")
        ax.text(xi + width/2, f + 1, f"{f:.0f}%", ha="center", fontsize=10,
                color="steelblue", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel("Penalty weight (%)", fontsize=9)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.5, -0.16, panel3_note,
            transform=ax.transAxes, ha="center", fontsize=8, style="italic",
            color=C2)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    stem = season or "all"
    out_path = os.path.join(
        out_dir, f"ens_simple_explanation_day{day}_{stem}_{orog or 'all'}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")
    return out_path


# ─── Alpha sweep figure ───────────────────────────────────────────────────────

def make_alpha_sweep_figure(df, T_arr, fc1_cols, fc2_cols, event_type,
                            m1_name, m2_name, season, orog, day,
                            twcrps1, twcrps2, out_dir):
    """Plot twQS % score difference for every alpha from 0.90 to 0.99 (step 0.01).

    Shows at which quantile level the sign flip first occurs, with twCRPS
    shown as a horizontal reference line.
    """
    obs    = df["obs_value"].values.astype(np.float64)
    fc1_np = df[fc1_cols].values.astype(np.float64)
    fc2_np = df[fc2_cols].values.astype(np.float64)
    T      = T_arr.astype(np.float64)

    if event_type == "above":
        alphas = np.round(np.arange(0.90, 1.00, 0.01), 2)   # 0.90 … 0.99
    else:
        alphas = np.round(np.arange(0.01, 0.11, 0.01), 2)   # 0.01 … 0.10

    def pct(s1, s2):
        return 0.0 if abs(s1) < 1e-12 else (s2 - s1) / abs(s1) * 100

    twcrps_pct = pct(twcrps1, twcrps2)

    pct_diffs = []
    for alpha in alphas:
        s1 = _twqs_mean(fc1_np, obs, T, alpha, event_type)
        s2 = _twqs_mean(fc2_np, obs, T, alpha, event_type)
        pct_diffs.append(pct(s1, s2))

    # Find sign-flip point
    sign_flip_alpha = None
    for i in range(len(pct_diffs) - 1):
        if pct_diffs[i] * pct_diffs[i + 1] < 0:
            sign_flip_alpha = float(alphas[i + 1])
            break

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"twQS % score difference as a function of quantile level  —  Day {day}  "
        f"[{season or 'all'}, {orog or 'all'}]\n"
        f"{m1_name} (red)  vs  {m2_name} (blue)",
        fontsize=10,
    )

    alpha_labels = [f"q{int(a*100):02d}" for a in alphas]

    x = np.arange(len(alphas))
    bar_colors = [C1 if v > 0 else C2 for v in pct_diffs]
    ax.bar(x, pct_diffs, color=bar_colors, width=0.6, edgecolor="white", zorder=2,
           label="twQS at each α level")

    # twCRPS as horizontal reference
    ax.axhline(twcrps_pct, color="black", lw=2, ls="--", zorder=3,
               label=f"twCRPS  ({twcrps_pct:+.1f}%)")

    # Zero line
    ax.axhline(0, color="gray", lw=1.0, zorder=1)

    # Shade regions
    ylim_val = max(abs(twcrps_pct), max(abs(v) for v in pct_diffs)) * 1.4 or 5
    ax.set_ylim(-ylim_val, ylim_val)
    ax.axhspan(-ylim_val, 0, alpha=0.04, color=C2, zorder=0)
    ax.axhspan(0, ylim_val, alpha=0.04, color=C1, zorder=0)

    # Annotate sign-flip
    if sign_flip_alpha is not None:
        flip_idx = list(alphas).index(sign_flip_alpha)
        ax.axvline(flip_idx, color="darkorange", lw=2, ls=":", zorder=4,
                   label=f"Sign flip at α={sign_flip_alpha:.2f}")
        ax.text(flip_idx + 0.1, ylim_val * 0.85,
                f"Sign flip\nα={sign_flip_alpha:.2f}",
                color="darkorange", fontsize=9, fontweight="bold")

    # Value labels on bars
    for i, v in enumerate(pct_diffs):
        va = "bottom" if v >= 0 else "top"
        offset = ylim_val * 0.02 if v >= 0 else -ylim_val * 0.02
        ax.text(i, v + offset, f"{v:+.1f}%", ha="center", fontsize=7.5,
                color=C1 if v > 0 else C2, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels, fontsize=10)
    ax.set_xlabel("Quantile level  α", fontsize=10)
    ax.set_ylabel("% score difference  (blue − red) / |red| × 100", fontsize=9)
    ax.text(-0.5, ylim_val * 0.85, f"← {m1_name} better", color=C1,
            fontsize=8, ha="left")
    ax.text(-0.5, -ylim_val * 0.85, f"← {m2_name} better", color=C2,
            fontsize=8, ha="left")
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    stem = season or "all"
    out_path = os.path.join(
        out_dir, f"ens_alpha_sweep_day{day}_{stem}_{orog or 'all'}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")
    return out_path


# ─── Tail quantile bias figure (complementary to alpha sweep) ─────────────────

def make_tail_quantile_bias_figure(df, T_arr, fc1_cols, fc2_cols, event_type,
                                   m1_name, m2_name, season, orog, day, out_dir):
    """Show WHERE each model's tail quantiles sit relative to extreme observations.

    Complements the alpha-sweep score-difference figure by showing the physical
    bias direction rather than the score difference.  Three panels:

      1. Mean quantile value at each α level (extreme events only):
         Both model lines + horizontal references for mean(T) and mean(obs|extreme).
         Gap between model line and mean(obs) = systematic bias.

      2. Mean quantile bias:  mean( q_α − obs | extreme )
         'above': positive = warm overshoot (FA-prone); negative = cold miss.
         'below': positive = warm undershoot (miss-prone); negative = cold overshoot.

      3. Hit-conversion rate: % extreme events where model's q-level captures
         the extreme direction correctly (q_α ≥ obs for 'above'; q_α ≤ obs for 'below').
         Higher = model's tail quantile reaches the observed extreme.
    """
    obs    = df["obs_value"].values.astype(np.float64)
    fc1_np = df[fc1_cols].values.astype(np.float64)
    fc2_np = df[fc2_cols].values.astype(np.float64)
    T      = T_arr.astype(np.float64)

    if event_type == "above":
        ext_mask = obs > T
        alphas   = np.round(np.arange(0.90, 1.00, 0.01), 2)
        alpha_labels = [f"q{int(a*100):02d}" for a in alphas]
        obs_ext  = obs[ext_mask]
        T_ext    = T[ext_mask]
        bias_sign_label = "q_α − obs  (positive = warm overshoot)"
        hit_label = "% extreme where  q ≥ obs  (overshoot rate)"
        hit_note  = ("Higher % = model pushes q above obs during extremes\n"
                     "(warm-biased tail → converts misses to FAs at q99)")
        bias_note = ("Positive = model quantile warmer than extreme obs (FA-prone)\n"
                     "Negative = model quantile colder than extreme obs (miss)")
    else:
        ext_mask = obs < T
        alphas   = np.round(np.arange(0.01, 0.11, 0.01), 2)
        alpha_labels = [f"q{int(a*100):02d}" for a in alphas]
        obs_ext  = obs[ext_mask]
        T_ext    = T[ext_mask]
        bias_sign_label = "q_α − obs  (negative = cold enough; positive = too warm)"
        hit_label = "% extreme where  q ≤ obs  (model cold enough)"
        hit_note  = ("Higher % = model q-level reaches or exceeds the cold obs\n"
                     "(fewer misses → lower 99%-cost penalty at q01)")
        bias_note = ("Negative = model quantile colder than cold obs (captures extreme)\n"
                     "Positive = model quantile warmer than cold obs (miss)")

    n_ext = int(ext_mask.sum())
    if n_ext == 0:
        print(f"  [tail bias] No extreme events found, skipping figure.")
        return None

    mean_obs_ext = float(np.mean(obs_ext))
    mean_T_ext   = float(np.mean(T_ext))

    # Compute per-alpha stats for extreme events
    mean_q1, mean_q2  = [], []
    bias_q1, bias_q2  = [], []   # mean(q_alpha - obs | extreme)
    hit_q1,  hit_q2   = [], []   # % extreme where q captures direction

    for alpha in alphas:
        q1_all = np.quantile(fc1_np, alpha, axis=1)
        q2_all = np.quantile(fc2_np, alpha, axis=1)
        q1_ext = q1_all[ext_mask]
        q2_ext = q2_all[ext_mask]

        mean_q1.append(float(np.mean(q1_ext)))
        mean_q2.append(float(np.mean(q2_ext)))
        bias_q1.append(float(np.mean(q1_ext - obs_ext)))
        bias_q2.append(float(np.mean(q2_ext - obs_ext)))

        if event_type == "above":
            # hit = model overshoots (q >= obs during warm extreme)
            hit_q1.append(float(np.mean(q1_ext >= obs_ext)) * 100)
            hit_q2.append(float(np.mean(q2_ext >= obs_ext)) * 100)
        else:
            # hit = model reaches cold obs (q <= obs during cold extreme)
            hit_q1.append(float(np.mean(q1_ext <= obs_ext)) * 100)
            hit_q2.append(float(np.mean(q2_ext <= obs_ext)) * 100)

    x = np.arange(len(alphas))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_all = len(obs)
    fig.suptitle(
        f"Tail quantile bias  —  Day {day}  [{season or 'all'}, {orog or 'all'}]  "
        f"{'(cold extremes)' if event_type == 'below' else '(warm extremes)'}\n"
        f"{m1_name} (red)  vs  {m2_name} (blue)  |  "
        f"N={n_all:,} cases, {n_ext:,} extreme ({n_ext/n_all*100:.1f}%)",
        fontsize=10,
    )

    # ── Panel 1: Mean q_alpha for extreme events ──────────────────────────────
    ax = axes[0]
    direction = "cold" if event_type == "below" else "warm"
    ax.set_title(f"Mean quantile value during {direction} extremes", fontsize=11,
                 fontweight="bold")
    ax.plot(x, mean_q1, "o-", color=C1, lw=2.5, ms=9, label=m1_name, zorder=3)
    ax.plot(x, mean_q2, "o-", color=C2, lw=2.5, ms=9, label=m2_name, zorder=3)
    ax.axhline(mean_obs_ext, color="black", lw=2,   ls="-",
               label=f"Mean obs (extreme)\n= {mean_obs_ext:.1f}°C", zorder=4)
    ax.axhline(mean_T_ext,   color="gray",  lw=1.5, ls="--",
               label=f"Mean threshold T\n= {mean_T_ext:.1f}°C", zorder=4)
    # Shade gap between obs and threshold
    ax.axhspan(min(mean_obs_ext, mean_T_ext), max(mean_obs_ext, mean_T_ext),
               alpha=0.10, color="gray", label="Obs–T gap")
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels, fontsize=9)
    ax.set_xlabel("Quantile level α", fontsize=9)
    ax.set_ylabel("Temperature (°C)", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.text(0.5, -0.18,
            f"Model lines above mean obs = quantile too {'warm (miss)' if event_type=='below' else 'cold (miss)'}.\n"
            f"Gap between model lines = {'AIFS colder tail' if mean_q2[-1] < mean_q1[-1] else 'IFS colder tail'} "
            f"at most extreme α.",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic")

    # ── Panel 2: Mean bias (q - obs) ─────────────────────────────────────────
    ax = axes[1]
    ax.set_title(f"Mean tail bias\n{bias_sign_label}", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=1.5, zorder=3, label="Unbiased")
    ax.plot(x, bias_q1, "o-", color=C1, lw=2.5, ms=9, label=m1_name, zorder=4)
    ax.plot(x, bias_q2, "o-", color=C2, lw=2.5, ms=9, label=m2_name, zorder=4)
    # Value labels
    for xi, (b1, b2) in enumerate(zip(bias_q1, bias_q2)):
        ax.text(xi + 0.07, b1, f"{b1:+.2f}", color=C1, fontsize=7.5,
                va="center", ha="left", fontweight="bold")
        ax.text(xi - 0.07, b2, f"{b2:+.2f}", color=C2, fontsize=7.5,
                va="center", ha="right", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels, fontsize=9)
    ax.set_xlabel("Quantile level α", fontsize=9)
    ax.set_ylabel("Mean (q_α − obs) during extremes (°C)", fontsize=9)
    ax.legend(fontsize=9)
    ax.text(0.5, -0.18, bias_note,
            transform=ax.transAxes, ha="center", fontsize=8, style="italic")

    # ── Panel 3: Hit-conversion rate ─────────────────────────────────────────
    ax = axes[2]
    ax.set_title(f"{hit_label}", fontsize=11, fontweight="bold")
    ax.plot(x, hit_q1, "o-", color=C1, lw=2.5, ms=9, label=m1_name, zorder=3)
    ax.plot(x, hit_q2, "o-", color=C2, lw=2.5, ms=9, label=m2_name, zorder=3)
    for xi, (h1, h2) in enumerate(zip(hit_q1, hit_q2)):
        offset = 2
        ax.text(xi + 0.07, h1 + offset, f"{h1:.0f}%", color=C1, fontsize=8,
                va="bottom", ha="left", fontweight="bold")
        ax.text(xi - 0.07, h2 - offset, f"{h2:.0f}%", color=C2, fontsize=8,
                va="top", ha="right", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels, fontsize=9)
    ax.set_xlabel("Quantile level α", fontsize=9)
    ax.set_ylabel("% of extreme events", fontsize=9)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.5, -0.18, hit_note,
            transform=ax.transAxes, ha="center", fontsize=8, style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    stem = season or "all"
    out_path = os.path.join(
        out_dir, f"ens_tail_bias_day{day}_{stem}_{orog or 'all'}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")
    return out_path

def make_summary_figure(results_by_day, m1_name, m2_name, season, orog, out_dir):
    """Line plots of twCRPS and twQS_q099 for both models vs lead day."""
    days  = sorted(results_by_day.keys())
    tc1   = [results_by_day[d]["twcrps1"]  for d in days]
    tc2   = [results_by_day[d]["twcrps2"]  for d in days]
    tq1   = [results_by_day[d]["twqs1"]    for d in days]
    tq2   = [results_by_day[d]["twqs2"]    for d in days]
    spr1m = [results_by_day[d]["spr1_ext"] for d in days]
    spr2m = [results_by_day[d]["spr2_ext"] for d in days]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"twCRPS vs twQS_q099  –  summary by lead day\n"
        f"[season={season or 'all'}, orog={orog or 'all'}]",
        fontsize=11,
    )

    for ax, (y1, y2, title, ylabel) in zip(axes, [
        (tc1, tc2, "twCRPS", "twCRPS (lower = better)"),
        (tq1, tq2, "twQS_q099", "twQS_q099 (lower = better)"),
        (spr1m, spr2m, "Extreme-event ensemble spread", "σ of members (°C)  [obs>T cases]"),
    ]):
        ax.plot(days, y1, "o-", color=C1, lw=2, ms=6, label=m1_name)
        ax.plot(days, y2, "s-", color=C2, lw=2, ms=6, label=m2_name)
        ax.set_xlabel("Lead day", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(days)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / f"ens_tw_summary_{season or 'all'}_{orog or 'all'}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Summary saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config(args.config)
    m1_name, m2_name = get_names(config)
    event_type = get_event_type(config)
    orog_ranges = get_orog_ranges(config)

    parquet_dir = config.get("extract_points", {}).get("output_path")
    if not parquet_dir or not Path(parquet_dir).exists():
        sys.exit(f"ERROR: extract_points.output_path not found: {parquet_dir}")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        cfg_stem = Path(args.config).stem
        out_dir = Path("case_study_output") / f"tw_discrepancy_{cfg_stem}_{args.season or 'all'}_{args.orog or 'all'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Models: {m1_name}  vs  {m2_name}")
    print(f"Event type: {event_type},  Season: {args.season or 'all'},  Orog: {args.orog or 'all'}")

    results_by_day = {}

    for day in args.days:
        print(f"\nProcessing Day {day}...")
        df, T_arr, fc1_cols, fc2_cols = load_day_ensemble(
            parquet_dir, day, args.season, args.orog, orog_ranges, config,
            max_samples=args.max_samples, seed=args.seed,
        )
        if df is None:
            continue

        obs    = df["obs_value"].values.astype(np.float64)
        fc1_np = df[fc1_cols].values.astype(np.float64)
        fc2_np = df[fc2_cols].values.astype(np.float64)
        T      = T_arr.astype(np.float64)

        # Compute summary stats needed for multi-day figure
        if event_type == "above":
            ext_mask = obs > T
        else:
            ext_mask = obs < T

        t1f1, t2f1 = twcrps_terms(fc1_np, obs, T, event_type)
        t1f2, t2f2 = twcrps_terms(fc2_np, obs, T, event_type)
        alpha = 0.99 if event_type == "above" else 0.01
        qm1, qh1, qf1, _ = twqs_q099_contributions(fc1_np, obs, T, event_type, alpha)
        qm2, qh2, qf2, _ = twqs_q099_contributions(fc2_np, obs, T, event_type, alpha)

        results_by_day[day] = {
            "twcrps1":  float(np.mean(t1f1 - t2f1)),
            "twcrps2":  float(np.mean(t1f2 - t2f2)),
            "twqs1":    float(np.mean(qm1 + qh1 + qf1)),
            "twqs2":    float(np.mean(qm2 + qh2 + qf2)),
            "spr1_ext": float(np.mean(ensemble_spread(fc1_np)[ext_mask])) if ext_mask.any() else np.nan,
            "spr2_ext": float(np.mean(ensemble_spread(fc2_np)[ext_mask])) if ext_mask.any() else np.nan,
        }

        make_figure(df, T_arr, fc1_cols, fc2_cols, event_type,
                    m1_name, m2_name, args.season, args.orog, day, out_dir)
        make_simple_explanation_figure(
            df, T_arr, fc1_cols, fc2_cols, event_type,
            m1_name, m2_name, args.season, args.orog, day,
            results_by_day[day]["twcrps1"], results_by_day[day]["twcrps2"],
            out_dir,
        )
        make_alpha_sweep_figure(
            df, T_arr, fc1_cols, fc2_cols, event_type,
            m1_name, m2_name, args.season, args.orog, day,
            results_by_day[day]["twcrps1"], results_by_day[day]["twcrps2"],
            out_dir,
        )
        make_tail_quantile_bias_figure(
            df, T_arr, fc1_cols, fc2_cols, event_type,
            m1_name, m2_name, args.season, args.orog, day, out_dir,
        )

    if len(results_by_day) >= 2:
        make_summary_figure(results_by_day, m1_name, m2_name,
                            args.season, args.orog, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
