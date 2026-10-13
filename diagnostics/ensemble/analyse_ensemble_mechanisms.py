#!/usr/bin/env python3
"""
DECISION TOOL: Ensemble Mechanism Decomposition
================================================
Decomposes whether ensemble Model 2 outperforms Model 1 on extreme events
due to:
  (A) A better CONTROL member (mean state / default prediction is closer to obs)
  (B) Larger ENSEMBLE SPREAD (wider distribution whose tail covers more extremes)
  (C) Both, or neither

Produces a multi-panel decision figure and prints a text scorecard.
Flexible for any variable, threshold, season, terrain type, and lead time.

Usage examples
--------------
  # List all available datasets in extracted_points/
  python3 analyse_ensemble_mechanisms.py --list-datasets

  # 2m temperature: cold extremes (DJF, 1st percentile)
  python3 analyse_ensemble_mechanisms.py --variable 2t --event-type cold --percentile 1

  # 2m temperature: warm extremes (JJA, 99th percentile)
  python3 analyse_ensemble_mechanisms.py --variable 2t --event-type warm --percentile 99 --season JJA

  # 10m wind speed: high-wind extremes (all terrain)
  python3 analyse_ensemble_mechanisms.py --variable 10ff --event-type warm --percentile 99

  # 24h precipitation: heavy rain (fixed 20 mm/day threshold)
  python3 analyse_ensemble_mechanisms.py --variable tp24 --event-type warm --threshold-value 20

  # Only high orography, days 1 and 5
  python3 analyse_ensemble_mechanisms.py --variable 2t --orog-types high --lead-days 1,5

Options
-------
  --data-dir DIR         Root directory containing extracted_points/ subdirs
                         [default: ./extracted_points]
  --results-dir DIR      Root directory containing results/ subdirs with proper-score CSVs
                         [default: ./results]
  --variable VAR         Variable code: 2t, 10ff, tp24 (auto-selects first found if omitted)
  --event-type TYPE      'cold' (obs ≤ threshold) or 'warm' (obs ≥ threshold) [default: cold]
  --percentile N         Dataset percentile for threshold, e.g. 1 for cold, 99 for warm
                         [default: 1 for cold, 99 for warm]
  --threshold-value V    Override: use a fixed threshold value instead of a percentile
  --season S             DJF | MAM | JJA | SON | ALL  [default: ALL]
  --orog-types LIST      Comma-separated terrain bins: low, mid, high [default: low,mid,high]
  --lead-days LIST       Comma-separated forecast days e.g. 1,3,5  [default: all available]
  --results-tag TAG      Sub-folder in results/ to read proper-score CSVs from
                         (auto-detected from variable name if omitted)
  --output FILE          Output figure path  [default: auto-generated in plots/]
  --dpi N                Figure resolution  [default: 150]
  --list-datasets        Print all available datasets and exit
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ============================================================
# Constants
# ============================================================
OROG_BINS = {
    "low":  (0,   40),
    "mid":  (40,  120),
    "high": (120, 9999),
}
SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
    "ALL": set(range(1, 13)),
}

VARIABLE_UNITS  = {"2t": "K",   "10ff": "m/s", "tp24": "mm"}
VARIABLE_LABELS = {"2t": "2m Temperature", "10ff": "10m Wind Speed", "tp24": "24h Precipitation"}

TERRAIN_COLORS  = {"low": "#1f77b4", "mid": "#2ca02c", "high": "#9467bd"}
TERRAIN_MARKERS = {"low": "o",       "mid": "s",       "high": "^"}

QUADRANT_INFO = {
    "both":    {"color": "#2ca02c", "label": "Both drivers",  "alpha": 0.07},
    "spread":  {"color": "#ff7f0e", "label": "Spread only",   "alpha": 0.07},
    "control": {"color": "#1f77b4", "label": "Control only",  "alpha": 0.07},
    "neither": {"color": "#d62728", "label": "Neither",       "alpha": 0.07},
}

LD_LABELS = {1: "D1", 3: "D3", 5: "D5", 8: "D8", 10: "D10"}


# ============================================================
# Dataset discovery
# ============================================================
def discover_datasets(data_dir: Path) -> dict:
    """Scan extracted_points/ subdirectories.

    Returns
    -------
    dict: variable -> {"models": (m1, m2), "days": [int,...], "dir": Path}
    """
    datasets = {}
    if not data_dir.exists():
        return datasets
    for ens_dir in sorted(data_dir.iterdir()):
        if not (ens_dir.is_dir() and ens_dir.name.endswith("_ens")):
            continue
        parquets = sorted(ens_dir.glob("*.parquet"))
        if not parquets:
            continue
        m = re.match(r"^(.+?)_(.+?)_vs_(.+?)_ens_day(\d+)\.parquet$", parquets[0].name)
        if not m:
            continue
        variable, m1, m2 = m.group(1), m.group(2), m.group(3)
        days = []
        for f in parquets:
            dm = re.search(r"_day(\d+)\.parquet$", f.name)
            if dm:
                days.append(int(dm.group(1)))
        datasets[variable] = {"models": (m1, m2), "days": sorted(days), "dir": ens_dir}
    return datasets


def print_datasets(data_dir: Path):
    ds = discover_datasets(data_dir)
    if not ds:
        print(f"No ensemble datasets found in {data_dir}")
        return
    print(f"\nAvailable ensemble datasets in {data_dir}:\n")
    for var, info in ds.items():
        m1, m2 = info["models"]
        print(f"  Variable : {var}  ({VARIABLE_LABELS.get(var, var)})")
        print(f"  Units    : {VARIABLE_UNITS.get(var, '?')}")
        print(f"  Models   : {m1}  (fc1)  vs  {m2}  (fc2)")
        print(f"  Lead days: {info['days']}")
        print(f"  Directory: {info['dir']}")
        print()


# ============================================================
# Data loading
# ============================================================
def load_parquets(ens_dir: Path, lead_days: list) -> pd.DataFrame:
    dfs = []
    for f in sorted(ens_dir.glob("*.parquet")):
        m = re.search(r"_day(\d+)\.parquet$", f.name)
        if not m:
            continue
        day = int(m.group(1))
        if lead_days and day not in lead_days:
            continue
        df = pd.read_parquet(f)
        df["_lead_day"] = day
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No parquet files loaded from {ens_dir}")
    df = pd.concat(dfs, ignore_index=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], format="%Y%m%d")
    df["_month"] = df["valid_time"].dt.month
    return df.dropna(subset=["obs_value"])


# ============================================================
# Member statistics
# ============================================================
def _member_cols(df: pd.DataFrame, prefix: str) -> list:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_member_")]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _row_stats(df: pd.DataFrame, prefix: str, tail_pct: float) -> pd.DataFrame:
    """Compute control value, ensemble mean, ensemble tail, and perturbed spread."""
    all_cols  = _member_cols(df, prefix)
    ctrl_col  = f"{prefix}_member_0"
    pert_cols = [c for c in all_cols if c != ctrl_col]

    vals = df[all_cols].values   # (n, 51)
    ctrl = df[ctrl_col].values   # (n,)
    pert = df[pert_cols].values  # (n, 50)

    return pd.DataFrame({
        f"{prefix}_ctrl":     ctrl,
        f"{prefix}_ens_mean": np.nanmean(vals, axis=1),
        f"{prefix}_tail":     np.nanpercentile(vals, tail_pct, axis=1),
        f"{prefix}_spread":   np.nanstd(pert, axis=1, ddof=1),
    }, index=df.index)


# ============================================================
# Core analysis per group (terrain × lead_day)
# ============================================================
def analyse_group(
    df: pd.DataFrame,
    prefix1: str, prefix2: str,
    threshold: float,
    event_type: str,
    tail_pct: float,
) -> dict | None:
    """Return a dict of scalar statistics for one terrain×lead_day group."""
    obs = df["obs_value"].values
    s1 = _row_stats(df, prefix1, tail_pct)
    s2 = _row_stats(df, prefix2, tail_pct)
    combined = pd.concat([df[["obs_value"]], s1, s2], axis=1)

    # --- Extreme mask ---
    if event_type == "cold":
        ext_mask = obs <= threshold
    else:
        ext_mask = obs >= threshold

    df_ext = combined[ext_mask].copy()
    if len(df_ext) < 5:
        return None

    obs_ext = df_ext["obs_value"].values

    def _bias(col):   return float(np.mean(df_ext[col].values - obs_ext))
    def _mae(col):    return float(np.mean(np.abs(df_ext[col].values - obs_ext)))
    def _mean(series): return float(np.mean(series))

    # Tail miss: obs is MORE extreme than the ensemble tail
    if event_type == "cold":
        miss1 = float(np.mean(obs_ext < df_ext[f"{prefix1}_tail"].values)) * 100
        miss2 = float(np.mean(obs_ext < df_ext[f"{prefix2}_tail"].values)) * 100
    else:
        miss1 = float(np.mean(obs_ext > df_ext[f"{prefix1}_tail"].values)) * 100
        miss2 = float(np.mean(obs_ext > df_ext[f"{prefix2}_tail"].values)) * 100

    # Tail offset (signed): ensemble_tail − obs
    # cold: negative = ensemble cold tail reaches beyond obs (good)
    # warm: positive = ensemble warm tail reaches beyond obs (good)
    tail_ofs1 = _mean(df_ext[f"{prefix1}_tail"].values - obs_ext)
    tail_ofs2 = _mean(df_ext[f"{prefix2}_tail"].values - obs_ext)

    # Ensemble mean RMSE on extreme events (used for spread-skill ratio)
    mean_rmse1 = float(np.sqrt(np.mean((df_ext[f"{prefix1}_ens_mean"].values - obs_ext) ** 2)))
    mean_rmse2 = float(np.sqrt(np.mean((df_ext[f"{prefix2}_ens_mean"].values - obs_ext) ** 2)))

    # False alarm rate: % of NON-extreme days where ensemble tail still reaches the threshold
    # (inflated spread → high false alarm → penalised by Brier/twCRPS on the many non-extreme days)
    df_non_ext = combined[~ext_mask]
    if len(df_non_ext) < 5:
        far1 = far2 = float("nan")
    else:
        if event_type == "cold":
            far1 = float(np.mean(df_non_ext[f"{prefix1}_tail"].values <= threshold)) * 100
            far2 = float(np.mean(df_non_ext[f"{prefix2}_tail"].values <= threshold)) * 100
        else:
            far1 = float(np.mean(df_non_ext[f"{prefix1}_tail"].values >= threshold)) * 100
            far2 = float(np.mean(df_non_ext[f"{prefix2}_tail"].values >= threshold)) * 100

    return {
        "n_all":       len(combined),
        "n_ext":       len(df_ext),
        "threshold":   threshold,
        # --- Control member on extreme events ---
        "ctrl_bias1":  _bias(f"{prefix1}_ctrl"),
        "ctrl_bias2":  _bias(f"{prefix2}_ctrl"),
        "ctrl_mae1":   _mae(f"{prefix1}_ctrl"),
        "ctrl_mae2":   _mae(f"{prefix2}_ctrl"),
        # --- Ensemble mean on extreme events ---
        "mean_bias1":  _bias(f"{prefix1}_ens_mean"),
        "mean_bias2":  _bias(f"{prefix2}_ens_mean"),
        "mean_mae1":   _mae(f"{prefix1}_ens_mean"),
        "mean_mae2":   _mae(f"{prefix2}_ens_mean"),
        "mean_rmse1":  mean_rmse1,
        "mean_rmse2":  mean_rmse2,
        # --- Spread (perturbed members, std) ---
        "spread_all1": _mean(combined[f"{prefix1}_spread"].values),
        "spread_all2": _mean(combined[f"{prefix2}_spread"].values),
        "spread_ext1": _mean(df_ext[f"{prefix1}_spread"].values),
        "spread_ext2": _mean(df_ext[f"{prefix2}_spread"].values),
        # --- Tail coverage ---
        "miss1":       miss1,
        "miss2":       miss2,
        "far1":        far1,
        "far2":        far2,
        "tail_ofs1":   tail_ofs1,
        "tail_ofs2":   tail_ofs2,
    }


def classify_quadrant(ctrl_adv: float, spread_adv: float,
                       ctrl_thr: float = 0.05, spread_thr: float = 0.02) -> str:
    """Point falls in one of four quadrants on the decision map."""
    c = ctrl_adv   > ctrl_thr
    s = spread_adv > spread_thr
    if c and s:     return "both"
    if (not c) and s: return "spread"
    if c and (not s): return "control"
    return "neither"


# ============================================================
# Proper-score CSV loader
# ============================================================
def load_proper_scores(results_dir: Path, results_tag: str | None, variable: str) -> dict:
    """Load scores_by_leadtime CSVs.  Returns {terrain: DataFrame}."""
    if results_tag:
        score_dir = results_dir / results_tag
    else:
        # Auto-detect: first dir containing both the variable name and "ens"
        candidates = [
            d for d in results_dir.iterdir()
            if d.is_dir() and variable in d.name and "ens" in d.name
        ]
        if not candidates:
            return {}
        score_dir = candidates[0]

    terrain_dfs = {}
    for f in sorted(score_dir.glob("scores_by_leadtime*.csv")):
        for terrain in OROG_BINS:
            if terrain in f.name:
                try:
                    terrain_dfs[terrain] = pd.read_csv(f)
                except Exception:
                    pass
                break
    return terrain_dfs


# ============================================================
# Figure
# ============================================================
def _axhspan_fill(ax, y_lo, y_hi, color, alpha):
    ax.axhspan(y_lo, y_hi, color=color, alpha=alpha, zorder=0)


def build_figure(
    rows: list,               # [(terrain, lead_day, result_dict), ...]
    m1_name: str,
    m2_name: str,
    variable: str,
    event_type: str,
    season: str,
    proper_scores: dict,
    units: str,
    out_path: Path,
    dpi: int,
):
    # Build flat analysis DataFrame
    records = []
    for terrain, lead_day, r in rows:
        if r is None:
            continue
        ctrl_adv   = r["ctrl_mae1"]   - r["ctrl_mae2"]    # +ve = m2 ctrl more accurate
        spread_adv = r["spread_ext2"] - r["spread_ext1"]  # +ve = m2 wider (on extreme events)
        miss_impr  = r["miss1"]       - r["miss2"]        # +ve = m2 fewer misses
        _rm1 = max(r["mean_rmse1"], 1e-9)
        _rm2 = max(r["mean_rmse2"], 1e-9)
        records.append({
            "terrain":    terrain,
            "lead_day":   lead_day,
            "ctrl_adv":   ctrl_adv,
            "spread_adv": spread_adv,
            "miss_impr":  miss_impr,
            "miss1":      r["miss1"],   "miss2":      r["miss2"],
            "far1":       r["far1"],    "far2":       r["far2"],
            "spread1":    r["spread_ext1"], "spread2":    r["spread_ext2"],
            "ctrl_mae1":  r["ctrl_mae1"],   "ctrl_mae2":  r["ctrl_mae2"],
            "ctrl_bias1": r["ctrl_bias1"],  "ctrl_bias2": r["ctrl_bias2"],
            "mean_bias1": r["mean_bias1"],  "mean_bias2": r["mean_bias2"],
            "mean_rmse1": r["mean_rmse1"],  "mean_rmse2": r["mean_rmse2"],
            "ssr1":       r["spread_ext1"] / _rm1,
            "ssr2":       r["spread_ext2"] / _rm2,
            "tail_ofs1":  r["tail_ofs1"],   "tail_ofs2":  r["tail_ofs2"],
            "n_ext":      r["n_ext"],
            "quadrant":   classify_quadrant(ctrl_adv, spread_adv),
        })

    if not records:
        print("ERROR: no usable groups — nothing to plot.")
        return
    df = pd.DataFrame(records)

    has_proper = bool(proper_scores)
    tail_pct_str = "1st" if event_type == "cold" else "99th"
    event_str = event_type

    # Fixed 3-row × 3-col layout
    # Row 0: A(Decision Map) | B(twCRPS)    | C(Brier)
    # Row 1: D(Spread)       | E(Miss rate) | F(False alarm rate)
    # Row 2: G(SSR)          | H(Scorecard, spans cols 1-2)
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        f"Ensemble Mechanism Decomposition:  {m2_name}  vs  {m1_name}\n"
        f"Variable: {VARIABLE_LABELS.get(variable, variable)}  ({units})   |   "
        f"Extremes: {event_str}   |   Season: {season}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.58, wspace=0.38,
        top=0.95, bottom=0.04, left=0.07, right=0.97,
    )

    # ── Axes ────────────────────────────────────────────────────────────
    ax_map    = fig.add_subplot(gs[0, 0])
    ax_twcrps = fig.add_subplot(gs[0, 1])
    ax_brier  = fig.add_subplot(gs[0, 2])
    ax_sprd   = fig.add_subplot(gs[1, 0])
    ax_miss   = fig.add_subplot(gs[1, 1])
    ax_far    = fig.add_subplot(gs[1, 2])
    ax_ssr    = fig.add_subplot(gs[2, 0])
    ax_score  = fig.add_subplot(gs[2, 1:])

    lead_days_sorted = sorted(df["lead_day"].unique())

    # ── Panel A: Decision Map ────────────────────────────────────────────
    ax = ax_map
    ax.axhline(0, color="gray", lw=0.9, ls="--", zorder=1)
    ax.axvline(0, color="gray", lw=0.9, ls="--", zorder=1)

    # Quadrant shading
    xpad = max(abs(df["ctrl_adv"]).max(), 0.3) * 1.35
    ypad = max(abs(df["spread_adv"]).max(), 0.05) * 1.35
    for (xlo, xhi), (ylo, yhi), key in [
        ((-xpad, 0), (0, ypad),  "spread"),
        ((0, xpad),  (0, ypad),  "both"),
        ((-xpad, 0), (-ypad, 0), "neither"),
        ((0, xpad),  (-ypad, 0), "control"),
    ]:
        ax.fill_between([xlo, xhi], [ylo]*2, [yhi]*2,
                        color=QUADRANT_INFO[key]["color"],
                        alpha=QUADRANT_INFO[key]["alpha"], zorder=0)

    ax.text(-xpad * 0.97,  ypad * 0.93, "Spread only",  color=QUADRANT_INFO["spread"]["color"],
            fontsize=7.5, ha="left",  style="italic")
    ax.text( xpad * 0.97,  ypad * 0.93, "Both drivers", color=QUADRANT_INFO["both"]["color"],
            fontsize=7.5, ha="right", style="italic")
    ax.text(-xpad * 0.97, -ypad * 0.93, "Neither",      color=QUADRANT_INFO["neither"]["color"],
            fontsize=7.5, ha="left",  style="italic")
    ax.text( xpad * 0.97, -ypad * 0.93, "Control only", color=QUADRANT_INFO["control"]["color"],
            fontsize=7.5, ha="right", style="italic")

    # Points: color = miss improvement; size = ext sample size
    miss_vals = df["miss_impr"].values
    try:
        half = max(abs(miss_vals).max(), 0.1)
        norm_miss = TwoSlopeNorm(vcenter=0, vmin=-half, vmax=half)
    except Exception:
        norm_miss = Normalize(vmin=miss_vals.min(), vmax=miss_vals.max())

    cmap = plt.cm.RdYlGn
    sm = ScalarMappable(cmap=cmap, norm=norm_miss)
    sm.set_array([])

    for _, row in df.iterrows():
        ax.scatter(
            row["ctrl_adv"], row["spread_adv"],
            marker=TERRAIN_MARKERS[row["terrain"]],
            s=55 + np.sqrt(row["n_ext"]) * 1.5,
            color=sm.to_rgba(row["miss_impr"]),
            edgecolors="k", linewidths=0.5, zorder=5,
        )
        label = LD_LABELS.get(int(row["lead_day"]), f"D{row['lead_day']}")
        ax.annotate(label, (row["ctrl_adv"], row["spread_adv"]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=6.5, color="#333")

    ax.set_xlim(-xpad, xpad)
    ax.set_ylim(-ypad, ypad)
    ax.set_xlabel(
        f"Control advantage: {m1_name} ctrl MAE − {m2_name} ctrl MAE  ({units})\n"
        f"( +ve → {m2_name} default prediction more accurate on extremes )",
        fontsize=7.5,
    )
    ax.set_ylabel(
        f"Spread advantage on extremes: {m2_name} σ − {m1_name} σ  ({units})\n"
        f"( +ve → {m2_name} wider spread when obs is extreme )",
        fontsize=7.5,
    )
    ax.set_title("(A) Decision Map  [all metrics on extreme events only]\nterrain marker, colour = miss-rate improvement",
                 fontweight="bold")

    cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.01, shrink=0.82)
    cb.set_label(f"Miss rate improvement (%)\n{m1_name}% − {m2_name}%  (+ve → {m2_name} better)",
                 fontsize=6.0)

    terrain_handles = [
        plt.scatter([], [], marker=TERRAIN_MARKERS[t], color="gray", s=45, label=t)
        for t in TERRAIN_MARKERS
    ]
    ax.legend(handles=terrain_handles, fontsize=7, title="Terrain",
              title_fontsize=7, loc="lower left", framealpha=0.7)

    # ── Panel B: twCRPS difference ───────────────────────────────────────
    ax = ax_twcrps
    if has_proper:
        for terrain, tdf in proper_scores.items():
            if "twCRPS_diff" not in tdf.columns:
                continue
            ld   = tdf["forecast_day"].values
            diff = tdf["twCRPS_diff"].values
            sig_col = "twCRPS_is_significant"
            sig = tdf[sig_col].values if sig_col in tdf.columns else np.zeros(len(ld), bool)
            col_t = TERRAIN_COLORS.get(terrain, "k")
            ax.plot(ld, diff, "s-", color=col_t, lw=1.8, label=terrain)
            ax.scatter(ld[sig], diff[sig], marker="*", s=90, color=col_t, zorder=6)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        ylo, yhi = ax.get_ylim()
        if ylo < 0:
            ax.axhspan(ylo, 0, color="#2ca02c", alpha=0.08)
        if yhi > 0:
            ax.axhspan(0, yhi, color="#d62728", alpha=0.08)
        _span = yhi - ylo
        if _span > 0:
            if ylo < 0 and yhi > 0:
                ax.text(0.03, ((ylo + 0) / 2 - ylo) / _span,
                        f"↓ {m2_name} better", fontsize=7.5, color="#2ca02c",
                        style="italic", va="center", transform=ax.transAxes)
                ax.text(0.03, ((0 + yhi) / 2 - ylo) / _span,
                        f"↑ {m1_name} better", fontsize=7.5, color="#d62728",
                        style="italic", va="center", transform=ax.transAxes)
            elif yhi <= 0:
                ax.text(0.03, 0.5, f"↓ {m2_name} better overall", fontsize=7.5,
                        color="#2ca02c", style="italic", va="center", transform=ax.transAxes)
            else:
                ax.text(0.03, 0.5, f"↑ {m1_name} better overall", fontsize=7.5,
                        color="#d62728", style="italic", va="center", transform=ax.transAxes)
        ax.set_xticks(lead_days_sorted)
        ax.set_xlabel("Forecast lead day")
        ax.set_ylabel(f"twCRPS:  {m2_name} − {m1_name}  ({units})\n"
                      f"( −ve = {m2_name} better,  +ve = {m1_name} better )")
        ax.set_title("(B) Threshold-weighted CRPS difference\n(★ = statistically significant)",
                     fontweight="bold")
        ax.legend(fontsize=8, title="Terrain", title_fontsize=8, framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "No proper-score CSVs\nfound in results/",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("(B) twCRPS difference", fontweight="bold")

    # ── Panel C: Brier score difference ─────────────────────────────────
    ax = ax_brier
    if has_proper:
        for terrain, tdf in proper_scores.items():
            if "Brier_diff" not in tdf.columns:
                continue
            ld   = tdf["forecast_day"].values
            diff = tdf["Brier_diff"].values
            sig_col = "Brier_is_significant"
            sig = tdf[sig_col].values if sig_col in tdf.columns else np.zeros(len(ld), bool)
            col_t = TERRAIN_COLORS.get(terrain, "k")
            ax.plot(ld, diff, "s-", color=col_t, lw=1.8, label=terrain)
            ax.scatter(ld[sig], diff[sig], marker="*", s=90, color=col_t, zorder=6)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        ylo, yhi = ax.get_ylim()
        if ylo < 0:
            ax.axhspan(ylo, 0, color="#2ca02c", alpha=0.08)
        if yhi > 0:
            ax.axhspan(0, yhi, color="#d62728", alpha=0.08)
        _span = yhi - ylo
        if _span > 0:
            if ylo < 0 and yhi > 0:
                ax.text(0.03, ((ylo + 0) / 2 - ylo) / _span,
                        f"↓ {m2_name} better", fontsize=7.5, color="#2ca02c",
                        style="italic", va="center", transform=ax.transAxes)
                ax.text(0.03, ((0 + yhi) / 2 - ylo) / _span,
                        f"↑ {m1_name} better", fontsize=7.5, color="#d62728",
                        style="italic", va="center", transform=ax.transAxes)
            elif yhi <= 0:
                ax.text(0.03, 0.5, f"↓ {m2_name} better overall", fontsize=7.5,
                        color="#2ca02c", style="italic", va="center", transform=ax.transAxes)
            else:
                ax.text(0.03, 0.5, f"↑ {m1_name} better overall", fontsize=7.5,
                        color="#d62728", style="italic", va="center", transform=ax.transAxes)
        ax.set_xticks(lead_days_sorted)
        ax.set_xlabel("Forecast lead day")
        ax.set_ylabel(f"Brier Score:  {m2_name} − {m1_name}\n"
                      f"( −ve = {m2_name} better,  +ve = {m1_name} better )")
        ax.set_title("(C) Brier Score difference\n(★ = statistically significant)",
                     fontweight="bold")
        ax.legend(fontsize=8, title="Terrain", title_fontsize=8, framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "No proper-score CSVs\nfound in results/",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("(C) Brier Score difference", fontweight="bold")

    # ── Panel D: Ensemble spread (conditioned on extreme events) ────────
    # Always use the parquet-computed spread_ext (std of perturbed members
    # computed only over rows where obs exceeds the extreme threshold).
    # The proper-score CSV spread is NOT used here because it is computed on
    # all events by default (extreme_only_basic=False in ens_scores.py).
    ax = ax_sprd
    for terrain in ["low", "mid", "high"]:
        sub = df[df["terrain"] == terrain].sort_values("lead_day")
        if sub.empty:
            continue
        ld  = sub["lead_day"].values
        col = TERRAIN_COLORS[terrain]
        ax.plot(ld, sub["spread1"].values, "o--", color=col, lw=1.3,
                alpha=0.65, label=f"{m1_name} {terrain}")
        ax.plot(ld, sub["spread2"].values, "s-",  color=col, lw=2.0,
                label=f"{m2_name} {terrain}")
    ax.set_xticks(lead_days_sorted)
    ax.set_xlabel("Forecast lead day")
    ax.set_ylabel(f"Ensemble spread σ  ({units})\n(computed only on days when obs is extreme)")
    ax.set_title(f"(D) Ensemble spread by lead day\n(std of members 1–50, rows where obs exceeds threshold)\n(higher → wider member diversity when obs is extreme)",
                 fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.7)

    # ── Panel E: Tail miss rate by lead day ─────────────────────────────
    # % of EXTREME days where obs falls beyond ensemble tail
    # Low = ensemble covers extremes well
    ax = ax_miss
    for terrain in ["low", "mid", "high"]:
        sub = df[df["terrain"] == terrain].sort_values("lead_day")
        if sub.empty:
            continue
        ld  = sub["lead_day"].values
        col = TERRAIN_COLORS[terrain]
        ax.plot(ld, sub["miss1"].values, "o--", color=col, lw=1.5, alpha=0.65,
                label=f"{m1_name} {terrain}")
        ax.plot(ld, sub["miss2"].values, "s-",  color=col, lw=2.2,
                label=f"{m2_name} {terrain}")
    ax.set_xticks(lead_days_sorted)
    ax.set_xlabel("Forecast lead day")
    ax.set_ylabel(f"Miss rate (%)\n(% of extreme obs beyond ens. {tail_pct_str} pct)")
    ax.set_title(
        f"(E) Tail miss rate by lead day\n"
        f"(% of extreme obs beyond the ensemble {tail_pct_str} percentile)\n"
        "(lower = ensemble better covers the extreme tail)",
        fontweight="bold",
    )
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.7)

    # ── Panel F: False alarm rate by lead day ────────────────────────
    # % of NON-EXTREME days where the ensemble tail still reaches the threshold.
    # High FAR means the ensemble is crying wolf: it predicts ‘extreme likely’
    # on ordinary days, which is what penalises Brier/twCRPS on the 99% of
    # non-extreme days even when miss rate looks good.
    ax = ax_far
    _far_has_data = df["far1"].notna().any()
    if _far_has_data:
        for terrain in ["low", "mid", "high"]:
            sub = df[df["terrain"] == terrain].sort_values("lead_day")
            if sub.empty:
                continue
            ld  = sub["lead_day"].values
            col = TERRAIN_COLORS[terrain]
            ax.plot(ld, sub["far1"].values, "o--", color=col, lw=1.5, alpha=0.65,
                    label=f"{m1_name} {terrain}")
            ax.plot(ld, sub["far2"].values, "s-",  color=col, lw=2.2,
                    label=f"{m2_name} {terrain}")
        ax.set_xticks(lead_days_sorted)
        ax.set_xlabel("Forecast lead day")
        ax.set_ylabel(f"False alarm rate (%)\n(% of non-extreme days where ens. tail hits threshold)")
        ax.set_title(
            f"(F) False alarm rate by lead day\n"
            f"(% of NORMAL days where ens. {tail_pct_str} pct still reaches threshold)\n"
            "(lower = ensemble not crying wolf on ordinary days)",
            fontweight="bold",
        )
        ax.legend(fontsize=6.5, ncol=2, framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "No non-extreme events\nfound for FAR computation",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("(F) False alarm rate", fontweight="bold")

    # ── Panel G: Spread-Skill Ratio on extreme events ───────────────────
    # SSR = spread_ext / RMSE(ens_mean).  = 1 → well-calibrated, <1 under-dispersive
    ax = ax_ssr
    ax.axhline(1.0, color="k", lw=1.2, ls="--", label="Perfect calibration (SSR=1)", zorder=3)
    for terrain in ["low", "mid", "high"]:
        sub = df[df["terrain"] == terrain].sort_values("lead_day")
        if sub.empty:
            continue
        ld  = sub["lead_day"].values
        col = TERRAIN_COLORS[terrain]
        ax.plot(ld, sub["ssr1"].values, "o--", color=col,
                lw=1.3, alpha=0.65, label=f"{m1_name} {terrain}")
        ax.plot(ld, sub["ssr2"].values, "s-",  color=col,
                lw=2.0, label=f"{m2_name} {terrain}")
    # Shade under-dispersive region after plotting (so ylim is set by data)
    _ylo_ssr, _yhi_ssr = ax.get_ylim()
    if _ylo_ssr < 1.0:
        ax.axhspan(_ylo_ssr, min(1.0, _yhi_ssr), color="#d62728", alpha=0.06, zorder=0)
    ax.text(0.03, 0.06, "← under-dispersive  (spread < RMSE)", fontsize=7,
            color="#d62728", style="italic", va="center", transform=ax.transAxes)
    ax.set_xticks(lead_days_sorted)
    ax.set_xlabel("Forecast lead day")
    ax.set_ylabel("Spread-Skill Ratio  (σ_ext / RMSE)\n(both computed only on days when obs is extreme)")
    ax.set_title(
        "(G) Spread-Skill Ratio by lead day\n"
        "(spread ÷ RMSE of ensemble mean, conditioned on extreme events)\n"
        "( =1 well-calibrated,  <1 under-dispersive )",
        fontweight="bold",
    )
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.7)

    # ── Panel H: Decision Scorecard ───────────────────────────────
    ax = ax_score
    ax.axis("off")

    q_counts = df["quadrant"].value_counts().to_dict()
    total    = len(df)
    mean_ctrl_adv   = df["ctrl_adv"].mean()
    mean_spread_adv = df["spread_adv"].mean()
    mean_ctrl_delta = df["ctrl_mae2"].mean() - df["ctrl_mae1"].mean()
    mean_sprd_delta = df["spread2"].mean()   - df["spread1"].mean()
    mean_miss_delta = df["miss2"].mean()     - df["miss1"].mean()

    if mean_ctrl_adv > 0.1 and mean_spread_adv > 0.02:
        mechanism  = f"BOTH: control accuracy AND spread"
        mech_color = QUADRANT_INFO["both"]["color"]
    elif mean_spread_adv > 0.02 and mean_ctrl_adv <= 0.1:
        mechanism  = f"SPREAD: larger {m2_name} variability"
        mech_color = QUADRANT_INFO["spread"]["color"]
    elif mean_ctrl_adv > 0.1 and mean_spread_adv <= 0.02:
        mechanism  = f"CONTROL: {m2_name} ctrl more accurate"
        mech_color = QUADRANT_INFO["control"]["color"]
    else:
        mechanism  = f"NEITHER clearly dominant"
        mech_color = QUADRANT_INFO["neither"]["color"]

    lines = [
        ("DECISION SCORECARD",          {"weight": "bold", "fontsize": 9.5}),
        ("─" * 52,                       {}),
        (f"{m2_name}  vs  {m1_name}",   {"weight": "bold"}),
        (f"Variable  : {VARIABLE_LABELS.get(variable, variable)}", {}),
        (f"Extremes  : {event_str}   Season: {season}",           {}),
        ("",                             {}),
        (f"Groups analysed: {total}  (terrain × lead day)", {}),
        (f"  ✓ Both drivers : {q_counts.get('both',    0):2d}  [ctrl better AND more spread]",
                                         {"color": QUADRANT_INFO["both"]["color"]}),
        (f"  ~ Spread only  : {q_counts.get('spread',  0):2d}  [spread drives improvement]",
                                         {"color": QUADRANT_INFO["spread"]["color"]}),
        (f"  ~ Control only : {q_counts.get('control', 0):2d}  [default prediction drives improvement]",
                                         {"color": QUADRANT_INFO["control"]["color"]}),
        (f"  ✗ Neither      : {q_counts.get('neither', 0):2d}  [{m2_name} no clear advantage]",
                                         {"color": QUADRANT_INFO["neither"]["color"]}),
        ("",                             {}),
        ("Mean differences  ({m2} − {m1}):".format(m2=m2_name, m1=m1_name), {"style": "italic"}),
        (f"  Ctrl member MAE : {mean_ctrl_delta:+.3f} {units}  (<0 → {m2_name} better on extremes)", {}),
        (f"  Ensemble spread : {mean_sprd_delta:+.3f} {units}  (>0 → {m2_name} wider on extremes)", {}),
        (f"  Tail miss rate  : {mean_miss_delta:+.1f}%         (<0 → {m2_name} fewer extremes missed)", {}),
        ("",                             {}),
        ("Primary mechanism:",           {"style": "italic"}),
        (f"  → {mechanism}",             {"weight": "bold", "color": mech_color}),
    ]

    y = 0.97
    for text, kwargs in lines:
        kw = {"transform": ax.transAxes, "va": "top",
              "fontfamily": "monospace", "fontsize": 8.5}
        kw.update(kwargs)
        if "fontsize" in kwargs:
            kw["fontsize"] = kwargs["fontsize"]
        ax.text(0.03, y, text, **kw)
        y -= 0.049

    ax.set_title("(H) Decision Scorecard", fontweight="bold")
    rect = mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.02",
        linewidth=0.8, edgecolor="#999", facecolor="#f9f9f9",
        transform=ax.transAxes, zorder=0,
    )
    ax.add_patch(rect)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        prog="analyse_ensemble_mechanisms.py",
        description="Decompose ensemble extreme performance: control accuracy vs spread.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir",        default="./extracted_points")
    parser.add_argument("--results-dir",     default="./results")
    parser.add_argument("--variable",        default=None)
    parser.add_argument("--event-type",      default="cold", choices=["cold", "warm"])
    parser.add_argument("--percentile",      type=float, default=None,
                        help="Percentile for threshold (auto = 1 for cold, 99 for warm)")
    parser.add_argument("--threshold-value", type=float, default=None)
    parser.add_argument("--season",          default="ALL",
                        choices=["DJF", "MAM", "JJA", "SON", "ALL"])
    parser.add_argument("--orog-types",      default="low,mid,high")
    parser.add_argument("--lead-days",       default=None)
    parser.add_argument("--results-tag",     default=None)
    parser.add_argument("--output",          default=None)
    parser.add_argument("--dpi",             type=int, default=150)
    parser.add_argument("--list-datasets",   action="store_true")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    if args.list_datasets:
        print_datasets(data_dir)
        sys.exit(0)

    datasets = discover_datasets(data_dir)
    if not datasets:
        print(f"ERROR: No ensemble datasets found in {data_dir}")
        sys.exit(1)

    # Variable selection
    variable = args.variable or next(iter(datasets))
    if args.variable is None:
        print(f"Auto-selected variable: {variable}")
    if variable not in datasets:
        print(f"ERROR: '{variable}' not found. Available: {sorted(datasets)}")
        sys.exit(1)

    ds      = datasets[variable]
    m1, m2  = ds["models"]
    units   = VARIABLE_UNITS.get(variable, "?")

    # Percentile default
    percentile = args.percentile
    if percentile is None:
        percentile = 1.0 if args.event_type == "cold" else 99.0

    # Lead days
    lead_days = (
        [int(d) for d in args.lead_days.split(",")]
        if args.lead_days else ds["days"]
    )

    # Terrain types
    orog_types = [o.strip() for o in args.orog_types.split(",")
                  if o.strip() in OROG_BINS]

    # Season month set
    months = SEASON_MONTHS.get(args.season, set(range(1, 13)))

    print(f"\n{'='*70}")
    print(f"Ensemble Mechanism Decomposition")
    print(f"  Variable   : {variable}  ({VARIABLE_LABELS.get(variable, variable)},  units: {units})")
    print(f"  Models     : {m1}  (fc1)  vs  {m2}  (fc2)")
    print(f"  Event type : {args.event_type}")
    if args.threshold_value is not None:
        print(f"  Threshold  : fixed = {args.threshold_value} {units}")
    else:
        print(f"  Threshold  : {percentile}th percentile of obs (per terrain)")
    print(f"  Season     : {args.season}  (months: {sorted(months)})")
    print(f"  Terrain    : {orog_types}")
    print(f"  Lead days  : {lead_days}")
    print(f"{'='*70}")

    # Load data
    print(f"\nLoading parquet files from {ds['dir']}...")
    df_all = load_parquets(ds["dir"], lead_days)
    df_all = df_all[df_all["_month"].isin(months)].copy()
    print(f"  Rows after season filter: {len(df_all):,}")

    # Detect prefixes (fc1, fc2)
    prefixes = sorted({
        c.split("_member_")[0]
        for c in df_all.columns if "_member_" in c
    })
    if len(prefixes) < 2:
        print(f"ERROR: expected 2 model prefixes; found: {prefixes}")
        sys.exit(1)
    p1, p2 = prefixes[0], prefixes[1]

    tail_pct = 1.0 if args.event_type == "cold" else 99.0

    # Analyse each terrain × lead_day combination
    all_rows = []
    for terrain in orog_types:
        lo, hi = OROG_BINS[terrain]
        df_t = df_all[(df_all["sdfor"] >= lo) & (df_all["sdfor"] < hi)]
        if len(df_t) < 50:
            print(f"\n  [{terrain}] Too few rows ({len(df_t)}), skipping.")
            continue

        obs_all = df_t["obs_value"].values
        if args.threshold_value is not None:
            threshold = args.threshold_value
        else:
            threshold = float(np.nanpercentile(obs_all, percentile))

        print(f"\n  [{terrain.upper()} terrain]  sdfor=[{lo},{hi})  n={len(df_t):,}")
        print(f"    Threshold: {threshold:.3f} {units}  ({percentile}th pct, {args.event_type})")
        print(f"    {'D':>4}  {'n_ext':>6}  {'ctrl_adv':>9}  {'spread_adv':>10}  "
              f"{'miss_impr':>10}  quadrant")

        for ld in lead_days:
            df_ld = df_t[df_t["_lead_day"] == ld]
            if len(df_ld) < 20:
                continue
            r = analyse_group(df_ld, p1, p2, threshold, args.event_type, tail_pct)
            all_rows.append((terrain, ld, r))
            if r:
                c_adv = r["ctrl_mae1"]   - r["ctrl_mae2"]
                s_adv = r["spread_all2"] - r["spread_all1"]
                m_imp = r["miss1"]       - r["miss2"]
                q     = classify_quadrant(c_adv, s_adv)
                print(f"    D{ld:>2}  {r['n_ext']:>6,}  "
                      f"{c_adv:>+9.3f}  {s_adv:>+10.3f}  {m_imp:>+10.1f}%  {q}")

    if not all_rows:
        print("\nERROR: no groups analysed – check filters and data availability.")
        sys.exit(1)

    # Load proper scores from existing CSVs
    proper = load_proper_scores(results_dir, args.results_tag, variable)
    if proper:
        print(f"\nLoaded proper-score CSVs for terrains: {sorted(proper)}")
    else:
        print("\n(No proper-score CSVs found – panels G/H/I will be omitted.)")

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        tag = (f"fixed{args.threshold_value}" if args.threshold_value
               else f"pct{int(percentile)}")
        out_path = Path(f"plots/decomp_{variable}_{args.event_type}_{args.season}_{tag}.png")

    print(f"\nBuilding figure → {out_path}  (dpi={args.dpi})...")
    build_figure(
        all_rows, m1, m2, variable, args.event_type, args.season,
        proper, units, out_path, args.dpi,
    )


if __name__ == "__main__":
    main()
