#!/usr/bin/env python3
"""
Extreme Events Diagnostic Tool — Scorecards4Extremes
=====================================================
Performs detailed diagnostic analysis of a specific condition from a
scorecards4extremes config file.  All data is read from the extracted
parquet files produced by the main pipeline (run.py).

Generates 21 diagnostic plots:
  1. Skill Score Comparison (POD, FAR, CSI)
  2. ETS and PSS Comparison
  3. Skill Score Evolution across threshold sweep
  4. Error Distribution (histogram, box plot, scatter, stats table)
  5. Frequency Bias Evolution
  6. Empirical Distributions (extreme region + full)
  7. Q-Q Plots (extreme region + full)
  8. Contingency Table Comparison
  9. Conditional Error Analysis (MAE/MSE for extreme events only)
 10. twMAE Decomposition (hits / misses / false alarm contributions)
 11. twMAE Percentile Decomposition (decomposition across threshold sweep)
 12. twMAE Component Fractions (100 % stacked — which failure mode dominates?)
 13. Extreme Intensity Scatter (fc vs obs for hits only — intensity bias)
 14. Miss & FA Severity (how extreme were the missed events / false alarms?)
 15. Conditional Bias & Noise (systematic vs random error on extreme cases)
 16. twMAE Skill Score (vs obs-based reference — analogue of BSS)
 17. Error Depth Profile (error binned by exceedance magnitude — hits only)
 18. Summary Scorecard Table (all components + auto-generated narrative)
 19. Count Evolution (absolute hits / misses / FAs across threshold sweep)
 20. Count Difference (Δhits / Δmisses / ΔFAs between models across sweep)
 21. Detection Profile (normalised 100% stacked bars — fraction of total sample)
 22. Conditional Bias Decomposed (real events vs false alarms — see Plot 15's docstring)

Usage
-----
  # 99th-percentile warm extremes, day 3, all seasons/terrain:
  python diagnose_extremes.py --config config_2t_local_p99obsclim_aifs_ifs_single.yaml \\
      --day 3 --threshold-pct 99

  # 1st-percentile cold extremes, day 5, DJF, flat terrain:
  python diagnose_extremes.py --config config_2t_local_p99obsclim_aifs_ifs_single.yaml \\
      --day 5 --threshold-pct 1 --season DJF --orog flat

  # Fixed threshold (e.g. 30mm precip), day 3:
  python diagnose_extremes.py --config config_tp24_local_p99obs.yaml \\
      --day 3 --threshold-value 30.0
"""

import argparse
import gc
import re
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — never allocates a display buffer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling _style module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # src/, for `import threshold`
import _style

warnings.filterwarnings("ignore")

# ============================================================================
# MODULE-LEVEL SETTINGS (set from CLI after parsing)
# ============================================================================

SAVE_FIGURES = True
OUTPUT_PATH = None   # set in main()

# ============================================================================
# CONSTANTS
# ============================================================================

VARIABLE_LABELS = {
    "2t":     ("2m Temperature",     "°C"),
    "10ff":   ("10m Wind Speed",     "m/s"),
    "tp24":   ("24h Precipitation",  "mm"),
    "aod500": ("AOD 500 nm",         ""),
    "go3":    ("Ozone (O3)",         "ppb"),
    "pm2p5":  ("PM2.5",              "µg/m³"),
}

SEASON_MONTHS = {
    "DJF": [12, 1, 2], "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],  "SON": [9, 10, 11],
    "ASO": [8, 9, 10],
}

OROGRAPHY_RANGES = {
    "flat":    (0,   40),
    "low":     (0,   40),
    "hilly":   (40,  120),
    "mid":     (40,  120),
    "complex": (120, 3000),
    "high":    (120, 3000),
}

# Predefined geographic areas [North, West, South, East] — MUST mirror filter.py
AREAS = {
    "europe":          [68, -15, 27, 50],
    "nh_extratropics": [90, -180, 20, 180],
    "tropics":         [20, -180, -20, 180],
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_model_names(config):
    fc1 = config["read_data"]["forecast_model1"]["name"]
    fc2 = config["read_data"]["forecast_model2"]["name"]
    return fc1, fc2


def _significance_from_csv(config, condition):
    """Best-effort lookup of the pipeline's bootstrap significance flags.

    The main pipeline (run.py) writes `<metric>_is_significant` columns into the
    per-condition score CSVs.  We read those so the diagnostics can flag whether a
    model difference is robust.  Returns {metric_lower: bool} for the current
    day/season/orography, or {} if no matching CSV is found (the markers are then
    simply omitted — this is an optional overlay, never a hard dependency).
    """
    results_dir = config.get("save", {}).get("output_directory")
    if not results_dir:
        return {}
    var = condition["var_short"]
    season = condition.get("season")
    orog = condition.get("terrain")
    day = condition.get("forecast_day")
    # Filename conventions vary by config (season-split vs pooled); try both.
    candidates = []
    if season and orog and orog != "all":
        candidates.append(f"scores_by_leadtime_{var}_{season}_{orog}.csv")
    if orog and orog != "all":
        candidates.append(f"scores_by_leadtime_{var}_{orog}.csv")
    if season:
        candidates.append(f"scores_by_leadtime_{var}_{season}.csv")
    for name in candidates:
        path = Path(results_dir) / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if day is not None and "forecast_day" in df.columns:
                df = df[df["forecast_day"] == day]
            if df.empty:
                continue
            out = {}
            for col in df.columns:
                if col.endswith("_is_significant"):
                    metric = col[: -len("_is_significant")].lower()
                    out[metric] = bool(df[col].all())
            if out:
                return out
        except Exception as exc:
            print(f"  Note: could not read significance from {path.name}: {exc}")
    return {}


def load_day(config, day):
    """Load a single forecast-day parquet, collapsing ensemble members if present."""
    var  = config["variable"]
    fc1  = config["read_data"]["forecast_model1"]["name"]
    fc2  = config["read_data"]["forecast_model2"]["name"]
    base = Path(config["extract_points"]["output_path"])

    for pat in [
        f"{var}_{fc1}_vs_{fc2}_day{day}.parquet",
        f"{var}_{fc1}_vs_{fc2}_99th_day{day}.parquet",
        f"{var}_{fc1}_vs_{fc2}_1st_day{day}.parquet",
        f"{var}_{fc1}_vs_{fc2}_ens_day{day}.parquet",
    ]:
        fp = base / pat
        if fp.exists():
            print(f"  Loading {fp.name} ...")
            df = pd.read_parquet(fp)
            for prefix, target in [("fc1_member_", "fc1_value"),
                                    ("fc2_member_", "fc2_value")]:
                members = [c for c in df.columns if c.startswith(prefix)]
                if members and target not in df.columns:
                    df[target] = df[members].mean(axis=1)
                    df = df.drop(columns=members)
            return df, fc1, fc2

    raise FileNotFoundError(
        f"No day-{day} parquet found in {base}\n"
        f"Run the main pipeline first: python run.py <config>"
    )


def filter_data(df, config, season=None, orog=None):
    """Apply all filters exactly as filter.py / run.py does, so twMAE matches the heatmap CSV."""
    from datetime import datetime

    cfg = config.get("filter", {})
    var = config.get("variable", "")

    # ── Geographic area ───────────────────────────────────────────────────────
    area_name = cfg.get("area")
    if area_name:
        if area_name in AREAS:
            lat_north, lon_west, lat_south, lon_east = AREAS[area_name]
            before = len(df)
            df = df[(df["lat"] >= lat_south) & (df["lat"] <= lat_north) &
                    (df["lon"] >= lon_west) & (df["lon"] <= lon_east)]
            print(f"  Area filter ({area_name}): {len(df):,} rows (removed {before - len(df):,})")
        else:
            print(f"  Warning: Unknown area '{area_name}', skipping area filter")

    # ── Date range ────────────────────────────────────────────────────────────
    sd = config.get("start_date", "")
    ed = config.get("end_date",   "")
    if sd and ed:
        sd_str = datetime.strptime(sd, "%Y-%m-%d").strftime("%Y%m%d")
        ed_str = datetime.strptime(ed, "%Y-%m-%d").strftime("%Y%m%d")
        df = df[(df["date"].astype(str) >= sd_str) &
                (df["date"].astype(str) <= ed_str)]

    # ── Season ────────────────────────────────────────────────────────────────
    if season:
        months = SEASON_MONTHS[season.upper()]
        df = df[df["date"].astype(str).str[4:6].astype(int).isin(months)]
        print(f"  Season filter ({season}): {len(df):,} rows")

    # ── Orography ─────────────────────────────────────────────────────────────
    if orog and "sdfor" in df.columns:
        orog_ranges = cfg.get("orography_ranges",
                              {"flat": [0, 40], "hilly": [40, 120], "complex": [120, 3000]})
        if orog.lower() in orog_ranges:
            lo, hi = orog_ranges[orog.lower()]
        else:
            lo, hi = OROGRAPHY_RANGES[orog.lower()]
        df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]
        print(f"  Orography filter ({orog}): {len(df):,} rows")

    # ── Coastal station removal ───────────────────────────────────────────────
    remove_coastal = cfg.get("remove_coastal_stations", True)
    lsm_threshold  = cfg.get("coastal_lsm_threshold", 0.9)
    if remove_coastal and "lsm" in df.columns:
        before = len(df)
        df = df[df["lsm"] > lsm_threshold]
        print(f"  Coastal filter (lsm > {lsm_threshold}): {len(df):,} rows "
              f"(removed {before - len(df)})")
    elif remove_coastal:
        print(f"  Coastal filter: lsm column not found, skipping")

    # ── Drop NaN in core columns ──────────────────────────────────────────────
    df = df.dropna(subset=["obs_value", "fc1_value", "fc2_value"])

    # ── Outlier removal ───────────────────────────────────────────────────────
    if cfg.get("remove_outliers", False) and var != "tp24":
        thr_std = cfg.get("outlier_threshold_std", 5.0)
        before  = len(df)
        for col in ("fc1_value", "fc2_value"):
            m, s = df[col].mean(), df[col].std()
            df = df[np.abs(df[col] - m) < thr_std * s]
        print(f"  Outlier removal: {len(df):,} rows (removed {before - len(df)})")

    # ── Variable-specific QC ─────────────────────────────────────────────────
    if var == "2t":
        lo = cfg.get("min_valid_temperature", -60.0)
        hi = cfg.get("max_valid_temperature",  60.0)
        before = len(df)
        valid = (df["obs_value"] >= lo) & (df["obs_value"] <= hi)
        for col in ("fc1_value", "fc2_value"):
            if col in df.columns:
                valid &= (df[col] >= lo) & (df[col] <= hi)
        df = df[valid]
        if len(df) < before:
            print(f"  Temperature QC ({lo}°C–{hi}°C): {len(df):,} rows "
                  f"(removed {before - len(df)})")
    elif var == "tp24":
        mx = cfg.get("max_valid_precipitation")
        if mx is not None:
            before = len(df)
            df = df[df["obs_value"] <= mx]
            if len(df) < before:
                print(f"  Precipitation QC (max={mx}mm): {len(df):,} rows "
                      f"(removed {before - len(df)})")

    print(f"  After filtering: {len(df):,} rows")
    return df


# ============================================================================
# EXTREME EVENT HELPERS  (ported unchanged from original script)
# ============================================================================

def is_extreme_event(data_values, threshold, var_short, threshold_percentile):
    if var_short == "2t":
        if threshold_percentile is None:
            return data_values < threshold if threshold < 0 else data_values > threshold
        elif threshold_percentile <= 50:
            return data_values < threshold
        else:
            return data_values > threshold
    else:
        return data_values > threshold


def get_threshold_range_and_labels(condition, obs_data):
    threshold_mode = condition.get("threshold_mode", "percentile")
    var_short = condition["var_short"]

    if threshold_mode == "percentile":
        threshold_percentile = condition.get("threshold_percentile", 95)
        if var_short == "2t" and threshold_percentile <= 50:
            percentiles = np.arange(1, 41, 2)
        elif threshold_percentile >= 60:
            percentiles = np.arange(60, 100, 1)
        else:
            percentiles = np.arange(
                max(1,   threshold_percentile - 20),
                min(100, threshold_percentile + 20), 2)

        thresholds = np.array([np.percentile(obs_data, p) for p in percentiles])
        labels = [f"{int(p)}" for p in percentiles]
        xlabel = "Percentile"
        main_threshold_label = f"Main threshold ({threshold_percentile}th)"
        return thresholds, labels, xlabel, main_threshold_label, percentiles

    else:  # fixed
        fixed_threshold = condition.get("threshold_value", 100.0)
        if var_short == "2t":
            thresholds = (np.linspace(fixed_threshold - 15, fixed_threshold + 5, 15)
                          if fixed_threshold < 0
                          else np.linspace(fixed_threshold - 5, fixed_threshold + 15, 15))
            labels = [f"{t:.1f}°C" for t in thresholds]; units = "°C"
        elif var_short == "tp24":
            thresholds = np.linspace(max(0.1, fixed_threshold - 10), fixed_threshold + 20, 15)
            labels = [f"{t:.1f}mm" for t in thresholds]; units = "mm"
        elif var_short == "10ff":
            thresholds = np.linspace(max(0.1, fixed_threshold - 5), fixed_threshold + 10, 15)
            labels = [f"{t:.1f}m/s" for t in thresholds]; units = "m/s"
        elif var_short == "aod500":
            thresholds = np.linspace(max(0.0, fixed_threshold - 0.3), fixed_threshold + 0.5, 15)
            labels = [f"{t:.2f}" for t in thresholds]; units = ""
        elif var_short == "go3":
            thresholds = np.linspace(max(0.1, fixed_threshold - 20), fixed_threshold + 30, 15)
            labels = [f"{t:.1f}ppb" for t in thresholds]; units = "ppb"
        elif var_short == "pm2p5":
            thresholds = np.linspace(max(0.1, fixed_threshold - 20), fixed_threshold + 30, 15)
            labels = [f"{t:.1f}µg/m³" for t in thresholds]; units = "µg/m³"
        else:
            thresholds = np.linspace(fixed_threshold - 10, fixed_threshold + 10, 15)
            labels = [f"{t:.1f}" for t in thresholds]; units = ""

        xlabel = f"Threshold Value ({units})" if units else "Threshold Value"
        main_threshold_label = f"Main threshold ({fixed_threshold}{units})"
        return thresholds, labels, xlabel, main_threshold_label, None


def get_extreme_description(var_short, threshold_percentile=None,
                             threshold_value=None, threshold_mode="percentile"):
    if threshold_mode == "fixed":
        units = {"2t": "°C", "10ff": "m/s", "tp24": "mm",
                 "go3": "ppb", "pm2p5": "µg/m³", "aod500": ""}.get(var_short, "")
        return f"Fixed threshold (= {threshold_value}{units})"
    if var_short == "2t":
        if threshold_percentile <= 50:
            return f"Cold extremes (< {threshold_percentile}th percentile)"
        return f"Heat extremes (> {threshold_percentile}th percentile)"
    return f"Extremes (> {threshold_percentile}th percentile)"


def threshold_conditional_mae(forecast, obs, threshold, threshold_percentile=None,
                               var_short=None):
    if var_short == "2t":
        if threshold_percentile is None:
            mask = obs <= threshold if threshold < 0 else obs >= threshold
        elif threshold_percentile < 50:
            mask = obs <= threshold
        else:
            mask = obs >= threshold
    else:
        mask = obs >= threshold
    return np.nan if np.sum(mask) == 0 else np.mean(np.abs(forecast[mask] - obs[mask]))


def threshold_conditional_mse(forecast, obs, threshold, threshold_percentile=None,
                               var_short=None):
    if var_short == "2t":
        if threshold_percentile is None:
            mask = obs <= threshold if threshold < 0 else obs >= threshold
        elif threshold_percentile < 50:
            mask = obs <= threshold
        else:
            mask = obs >= threshold
    else:
        mask = obs >= threshold
    return np.nan if np.sum(mask) == 0 else np.mean((forecast[mask] - obs[mask]) ** 2)


def calculate_skill_scores(fc_extreme, obs_extreme):
    hits = np.sum(fc_extreme & obs_extreme)
    misses = np.sum(~fc_extreme & obs_extreme)
    false_alarms = np.sum(fc_extreme & ~obs_extreme)
    correct_neg = np.sum(~fc_extreme & ~obs_extreme)

    def sd(a, b):
        return a / b if b != 0 else 0.0

    pod  = sd(hits, hits + misses)
    far  = sd(false_alarms, hits + false_alarms)
    csi  = sd(hits, hits + misses + false_alarms)
    pofd = sd(false_alarms, false_alarms + correct_neg)
    pss  = pod - pofd
    hits_random = sd((hits + misses) * (hits + false_alarms), len(obs_extreme))
    ets  = sd(hits - hits_random, hits + misses + false_alarms - hits_random)

    return {"hits": hits, "misses": misses, "false_alarms": false_alarms,
            "pod": pod, "far": far, "csi": csi, "ets": ets, "pss": pss}


def get_threshold_description_from_condition(condition):
    mode = condition.get("threshold_mode", "percentile")
    var  = condition["var_short"]
    if mode == "percentile":
        return get_extreme_description(var, threshold_percentile=condition.get("threshold_percentile", 95),
                                        threshold_mode="percentile")
    return get_extreme_description(var, threshold_value=condition.get("threshold_value"),
                                    threshold_mode="fixed")


def verify_extreme_detection_logic(obs_data, condition):
    print("\n" + "=" * 60)
    print("VERIFYING EXTREME EVENT DETECTION LOGIC")
    print("=" * 60)
    var_short = condition["var_short"]
    mode = condition.get("threshold_mode", "percentile")

    if mode == "percentile":
        pct = condition.get("threshold_percentile", 95)
        threshold = np.percentile(obs_data, pct)
        print(f"Variable: {var_short}  |  Mode: percentile  |  pct={pct}  |  threshold={threshold:.4f}")
    else:
        threshold = condition.get("threshold_value", 0.0)
        print(f"Variable: {var_short}  |  Mode: fixed  |  threshold={threshold}")

    for p in [1, 5, 10, 50, 90, 95, 99, 99.9]:
        print(f"  p{p}: {np.nanpercentile(obs_data, p):.4f}")

    pct_for_detect = condition.get("threshold_percentile") if mode == "percentile" else None
    obs_extreme = is_extreme_event(obs_data, threshold, var_short, pct_for_detect)
    actual_pct  = 100 * np.mean(obs_extreme)
    print(f"  Extreme events: {np.sum(obs_extreme)} ({actual_pct:.3f}%)")
    print("=" * 60)


# ============================================================================
# 9 DIAGNOSTIC PLOTS (ported with only figure-saving / labels adapted)
# ============================================================================

def _savefig(fig, filename):
    if SAVE_FIGURES and OUTPUT_PATH:
        out = Path(OUTPUT_PATH) / filename
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {out}")
    plt.close(fig)


def plot_skill_scores_comparison(fc1_data, fc2_data, obs_data, threshold, condition):
    """Detection skill: POD, FAR, CSI, ETS, PSS in one panel.

    (Merges the former separate 'skill scores' and 'ETS/PSS' figures — they were
    the same grouped-bar form.)  FAR is lower-is-better; it is marked so the
    reader is not misled by sharing an axis with higher-is-better metrics.
    Bootstrap significance of the difference is annotated where available.
    """
    print("\n[skill] Detection skill (POD, FAR, CSI, ETS, PSS)...")
    var_short = condition["var_short"]
    pct = condition.get("threshold_percentile") if condition.get("threshold_mode") == "percentile" else None
    obs_e = is_extreme_event(obs_data,  threshold, var_short, pct)
    fc1_e = is_extreme_event(fc1_data,  threshold, var_short, pct)
    fc2_e = is_extreme_event(fc2_data,  threshold, var_short, pct)
    scores1 = calculate_skill_scores(fc1_e, obs_e)
    scores2 = calculate_skill_scores(fc2_e, obs_e)
    sig = _significance_from_csv(config=condition["_config"], condition=condition) \
        if condition.get("_config") is not None else {}

    keys = ["pod", "far", "csi", "ets", "pss"]
    labels = ["POD ↑", "FAR ↓", "CSI ↑", "ETS ↑", "PSS ↑"]  # ↑/↓ = better direction
    lower_better = {"far"}
    v1 = [scores1[k] for k in keys]
    v2 = [scores2[k] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(keys)); w = 0.38
    bars1 = ax.bar(x - w/2, v1, w, label=condition["expver1"], color=_style.C_FC1,
                   alpha=0.9, edgecolor="white", lw=0.8)
    bars2 = ax.bar(x + w/2, v2, w, label=condition["expver2"], color=_style.C_FC2,
                   alpha=0.9, edgecolor="white", lw=0.8)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    # Winner star + significance per metric
    y_top = max(max(v1), max(v2), 1.0)
    for i, k in enumerate(keys):
        model2_wins = (v2[i] < v1[i]) if k in lower_better else (v2[i] > v1[i])
        ax.annotate("★", xy=(x[i] + (w/2 if model2_wins else -w/2),
                              max(v1[i], v2[i]) + 0.05),
                    ha="center", fontsize=12, color=_style.winner_color(model2_wins))
        marker = _style.significance_marker(sig.get(k)) if k in sig else ""
        if marker:
            ax.annotate(marker, xy=(x[i], y_top * 1.10), ha="center", fontsize=8,
                        style="italic",
                        color="#333333" if marker.startswith("✓") else "#999999")

    ax.axhline(0, color=_style.C_REF, ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Metric  (↑ = higher better · ↓ = lower better)")
    ax.set_ylabel("Score")
    ax.set_title(f"Detection skill — {get_threshold_description_from_condition(condition)}\n"
                 f"★ = winner   ·   ✓ sig. / n.s. = bootstrap significance of the difference",
                 fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(); ax.set_ylim(0, 1)
    _savefig(fig, f"1_skill_scores_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")
    return scores1, scores2


def plot_threshold_evolution(fc1_data, fc2_data, obs_data, condition):
    print("[3/9] Skill Score Threshold Evolution...")
    var_short = condition["var_short"]
    mode = condition.get("threshold_mode", "percentile")
    thresholds, _, xlabel, main_label, percentiles = get_threshold_range_and_labels(condition, obs_data)
    main_ref = condition.get("threshold_percentile", 95) if mode == "percentile" else condition.get("threshold_value", 0.0)

    results = []
    for i, thresh in enumerate(thresholds):
        pct_det = percentiles[i] if percentiles is not None else None
        obs_e = is_extreme_event(obs_data, thresh, var_short, pct_det)
        fc1_e = is_extreme_event(fc1_data, thresh, var_short, pct_det)
        fc2_e = is_extreme_event(fc2_data, thresh, var_short, pct_det)
        if np.sum(obs_e) < 5:
            continue
        x_val = percentiles[i] if mode == "percentile" else thresh
        results.extend([
            {"X": x_val, "Model": condition["expver1"], **calculate_skill_scores(fc1_e, obs_e)},
            {"X": x_val, "Model": condition["expver2"], **calculate_skill_scores(fc2_e, obs_e)},
        ])

    if not results:
        print("  [SKIP] No valid results for threshold evolution.")
        return

    rdf = pd.DataFrame(results)
    m1  = rdf[rdf["Model"] == condition["expver1"]]
    m2  = rdf[rdf["Model"] == condition["expver2"]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, (metric, name) in enumerate(
        [("pod","POD"), ("far","FAR"), ("csi","CSI"), ("ets","ETS"), ("pss","PSS")]
    ):
        ax = axes[i]
        ax.plot(m1["X"], m1[metric], "o-", label=condition["expver1"], lw=2, ms=6, color=_style.C_FC1)
        ax.plot(m2["X"], m2[metric], "s-", label=condition["expver2"], lw=2, ms=6, color=_style.C_FC2)
        ax.axvline(main_ref, color=_style.C_THRESHOLD, ls="--", alpha=0.7, label=main_label)
        if metric in ("ets", "pss"):
            ax.axhline(0, color="k", ls="--", alpha=0.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(name); ax.set_title(f"{name} vs {xlabel}")
        ax.legend(); ax.grid(True, alpha=0.3)
    axes[5].remove()
    fig.suptitle(f"Skill Score Evolution — {get_threshold_description_from_condition(condition)}", fontsize=14)
    plt.tight_layout()
    _savefig(fig, f"3_threshold_evolution_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_error_distribution(fc1_data, fc2_data, obs_data, condition):
    print("[4/9] Error Distribution...")
    e1 = fc1_data - obs_data
    e2 = fc2_data - obs_data

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].hist(e1, bins=50, alpha=0.7, density=True, label=condition["expver1"])
    axes[0, 0].hist(e2, bins=50, alpha=0.7, density=True, label=condition["expver2"])
    axes[0, 0].axvline(0, color="k", ls="--", alpha=0.5)
    axes[0, 0].set_xlabel("Forecast Error"); axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Error Distribution"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    bp = axes[0, 1].boxplot([e1, e2], labels=[condition["expver1"], condition["expver2"]],
                            patch_artist=True)
    for patch, col in zip(bp["boxes"], ["lightblue", "lightcoral"]):
        patch.set_facecolor(col)
    axes[0, 1].axhline(0, color="k", ls="--", alpha=0.5)
    axes[0, 1].set_ylabel("Forecast Error"); axes[0, 1].set_title("Error Box Plot")
    axes[0, 1].grid(True, alpha=0.3)

    idx = np.random.choice(len(obs_data), min(1000, len(obs_data)), replace=False)
    axes[1, 0].scatter(obs_data[idx], e1[idx], alpha=0.5, s=20, label=condition["expver1"])
    axes[1, 0].scatter(obs_data[idx], e2[idx], alpha=0.5, s=20, label=condition["expver2"])
    axes[1, 0].axhline(0, color="k", ls="--", alpha=0.5)
    axes[1, 0].set_xlabel(f"Observed {condition['var_short']}")
    axes[1, 0].set_ylabel("Error"); axes[1, 0].set_title("Error vs Observation")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    stats = pd.DataFrame({
        "Model":      [condition["expver1"], condition["expver2"]],
        "Mean Error": [np.mean(e1), np.mean(e2)],
        "MAE":        [np.mean(np.abs(e1)), np.mean(np.abs(e2))],
        "RMSE":       [np.sqrt(np.mean(e1**2)), np.sqrt(np.mean(e2**2))],
        "Std Error":  [np.std(e1), np.std(e2)],
    })
    axes[1, 1].axis("off")
    tbl = axes[1, 1].table(cellText=stats.round(4).values, colLabels=stats.columns,
                           cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    axes[1, 1].set_title("Error Statistics")

    plt.tight_layout()
    _savefig(fig, f"4_error_distribution_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_frequency_bias_evolution(fc1_data, fc2_data, obs_data, condition):
    print("[5/9] Frequency Bias Evolution...")
    var_short = condition["var_short"]
    mode = condition.get("threshold_mode", "percentile")
    thresholds, _, xlabel, main_label, percentiles = get_threshold_range_and_labels(condition, obs_data)
    main_ref = condition.get("threshold_percentile", 95) if mode == "percentile" else condition.get("threshold_value", 0.0)

    rows = []
    for i, thresh in enumerate(thresholds):
        pct_det = percentiles[i] if percentiles is not None else None
        obs_e = is_extreme_event(obs_data, thresh, var_short, pct_det)
        fc1_e = is_extreme_event(fc1_data, thresh, var_short, pct_det)
        fc2_e = is_extreme_event(fc2_data, thresh, var_short, pct_det)
        obs_f = np.mean(obs_e)
        x_val = percentiles[i] if mode == "percentile" else thresh
        rows.extend([
            {"X": x_val, "Model": condition["expver1"],
             "Bias": np.mean(fc1_e)/obs_f if obs_f > 0 else np.nan},
            {"X": x_val, "Model": condition["expver2"],
             "Bias": np.mean(fc2_e)/obs_f if obs_f > 0 else np.nan},
        ])

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, style in [(condition["expver1"], "o-"), (condition["expver2"], "s-")]:
        sub = df[df["Model"] == name]
        ax.plot(sub["X"], sub["Bias"], style, label=name, lw=2, ms=6)
    ax.axhline(1, color=_style.C_FC2, ls="--", alpha=0.7, label="Perfect bias")
    ax.axvline(main_ref, color=_style.C_THRESHOLD, ls="--", alpha=0.7, label=main_label)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("Frequency Bias (fc/obs)")
    ax.set_title(f"Frequency Bias Evolution — {get_threshold_description_from_condition(condition)}")
    ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, f"5_frequency_bias_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_empirical_distributions(fc1_data, fc2_data, obs_data, threshold, condition):
    print("[6/9] Empirical Distributions...")
    var_short = condition["var_short"]
    pct = condition.get("threshold_percentile") if condition.get("threshold_mode") == "percentile" else None
    obs_e = is_extreme_event(obs_data, threshold, var_short, pct)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if np.sum(obs_e) > 0:
        obs_ex = obs_data[obs_e]; fc1_ex = fc1_data[obs_e]; fc2_ex = fc2_data[obs_e]
        all_ex = np.concatenate([obs_ex, fc1_ex, fc2_ex])
        p1, p99 = np.nanpercentile(all_ex, [1, 99])
        pad = (p99 - p1) * 0.05
        axes[0].hist(obs_ex, bins=30, alpha=0.7, density=True, color=_style.C_OBS,
                     label=f"Obs (n={len(obs_ex)})")
        axes[0].hist(fc1_ex, bins=30, alpha=0.7, density=True, color=_style.C_FC1,
                     label=condition["expver1"])
        axes[0].hist(fc2_ex, bins=30, alpha=0.7, density=True, color=_style.C_FC2,
                     label=condition["expver2"])
        axes[0].axvline(threshold, color=_style.C_THRESHOLD, ls="--", lw=2,
                        label=f"Threshold ({threshold:.2f})")
        axes[0].set_xlim(p1 - pad, p99 + pad)
        axes[0].set_title("Forecasts When Obs Were Extreme")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

    all_full = np.concatenate([obs_data, fc1_data, fc2_data])
    p1f, p99f = np.nanpercentile(all_full, [0.5, 99.5])
    padf = (p99f - p1f) * 0.05
    axes[1].hist(obs_data, bins=50, alpha=0.7, density=True, color=_style.C_OBS, label="Obs")
    axes[1].hist(fc1_data, bins=50, alpha=0.7, density=True, color=_style.C_FC1,
                 label=condition["expver1"])
    axes[1].hist(fc2_data, bins=50, alpha=0.7, density=True, color=_style.C_FC2,
                 label=condition["expver2"])
    axes[1].axvline(threshold, color=_style.C_THRESHOLD, ls="--", lw=2,
                    label=f"Threshold ({threshold:.2f})")
    axes[1].set_xlim(p1f - padf, p99f + padf)
    axes[1].set_title("Full Database Distribution")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel(f"{condition['var_short']} value"); ax.set_ylabel("Density")
    plt.tight_layout()
    _savefig(fig, f"6_empirical_dist_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_qq_plots(fc1_data, fc2_data, obs_data, threshold, condition):
    print("[7/9] Q-Q Plots...")
    var_short = condition["var_short"]
    pct = condition.get("threshold_percentile") if condition.get("threshold_mode") == "percentile" else None
    obs_e = is_extreme_event(obs_data, threshold, var_short, pct)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if np.sum(obs_e) > 10:
        if var_short == "2t" and pct is not None and pct <= 50:
            q_extreme = np.linspace(0.1, 25, 200)
        else:
            q_extreme = np.linspace(75, 99.9, 200)
        obs_q = np.percentile(obs_data, q_extreme)
        fc1_q = np.percentile(fc1_data, q_extreme)
        fc2_q = np.percentile(fc2_data, q_extreme)
        if var_short == "2t" and pct is not None and pct <= 50:
            mask = obs_q <= threshold + 2
        else:
            mask = obs_q >= threshold - 2
        if np.sum(mask) > 5:
            axes[0].scatter(obs_q[mask], fc1_q[mask], alpha=0.7, s=20, color=_style.C_FC1,
                            label=condition["expver1"])
            axes[0].scatter(obs_q[mask], fc2_q[mask], alpha=0.7, s=20, color=_style.C_FC2,
                            label=condition["expver2"])
            lims = [min(obs_q[mask].min(), fc1_q[mask].min(), fc2_q[mask].min()),
                    max(obs_q[mask].max(), fc1_q[mask].max(), fc2_q[mask].max())]
            axes[0].plot(lims, lims, "k--", alpha=0.5, lw=2, label="Perfect")
            axes[0].axvline(threshold, color=_style.C_THRESHOLD, ls="--", alpha=0.7)
            axes[0].axhline(threshold, color=_style.C_THRESHOLD, ls="--", alpha=0.7)
            axes[0].set_title("Q-Q: Extreme Region (Zoomed)")
            axes[0].legend(); axes[0].grid(True, alpha=0.3)

    q_full = np.linspace(0.1, 99.9, 150)
    obs_qf = np.percentile(obs_data, q_full)
    fc1_qf = np.percentile(fc1_data, q_full)
    fc2_qf = np.percentile(fc2_data, q_full)
    axes[1].scatter(obs_qf, fc1_qf, alpha=0.7, s=15, color=_style.C_FC1, label=condition["expver1"])
    axes[1].scatter(obs_qf, fc2_qf, alpha=0.7, s=15, color=_style.C_FC2,  label=condition["expver2"])
    lims_f = [min(obs_qf.min(), fc1_qf.min(), fc2_qf.min()),
              max(obs_qf.max(), fc1_qf.max(), fc2_qf.max())]
    axes[1].plot(lims_f, lims_f, "k--", alpha=0.5, lw=2, label="Perfect")
    axes[1].axvline(threshold, color=_style.C_THRESHOLD, ls="--", alpha=0.7,
                    label=f"Threshold ({threshold:.2f})")
    axes[1].axhline(threshold, color=_style.C_THRESHOLD, ls="--", alpha=0.7)
    axes[1].set_title("Q-Q: Full Database")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Observed Quantiles"); ax.set_ylabel("Forecast Quantiles")
    plt.tight_layout()
    _savefig(fig, f"7_qq_plots_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_contingency_table_comparison(fc1_data, fc2_data, obs_data, threshold, condition):
    print("[8/9] Contingency Table Comparison...")
    var_short = condition["var_short"]
    pct = condition.get("threshold_percentile") if condition.get("threshold_mode") == "percentile" else None
    obs_e = is_extreme_event(obs_data, threshold, var_short, pct)
    fc1_e = is_extreme_event(fc1_data, threshold, var_short, pct)
    fc2_e = is_extreme_event(fc2_data, threshold, var_short, pct)

    def _ct(fc_e, obs_e):
        return (np.sum(fc_e & obs_e), np.sum(~fc_e & obs_e), np.sum(fc_e & ~obs_e))

    h1, m1, fa1 = _ct(fc1_e, obs_e)
    h2, m2, fa2 = _ct(fc2_e, obs_e)
    labels = ["Hits", "Misses", "False Alarms"]
    colors = [_style.C_HIT, _style.C_MISS, _style.C_FA]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, vals, title in [
        (axes[0], [h1, m1, fa1], condition["expver1"]),
        (axes[1], [h2, m2, fa2], condition["expver2"]),
    ]:
        bars = ax.bar(labels, vals, color=colors, alpha=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + v*0.01, str(v),
                    ha="center", va="bottom", fontsize=12, weight="bold")
        ax.set_title(f"{title} — Contingency"); ax.grid(True, alpha=0.3)

    x = np.arange(len(labels)); w = 0.35
    bars1 = axes[2].bar(x - w/2, [h1, m1, fa1], w, label=condition["expver1"], alpha=0.7, color=_style.C_FC1)
    bars2 = axes[2].bar(x + w/2, [h2, m2, fa2], w, label=condition["expver2"], alpha=0.7, color=_style.C_FC2)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2, h + h*0.01, str(int(h)),
                         ha="center", va="bottom", fontsize=10)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_title("Comparison"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, f"8_contingency_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_twmae_decomposition(fc1_data, fc2_data, obs_data, threshold, condition):
    """Plot 10: twMAE decomposition — four focused panels.

    For threshold T, twMAE = mean(|max(fc,T) − max(obs,T)|) decomposes as:
      Hits  (obs≥T, fc≥T)  cost = |fc − obs|          → captures forecast accuracy on events
      Misses(obs≥T, fc<T)  cost = obs − T              → captures event under-prediction
      FAs   (obs<T, fc≥T)  cost = fc  − T              → captures false alarm severity
      CNs   (obs<T, fc<T)  cost = 0                    → zero contribution

    When ``condition['threshold_arr']`` is a per-row NumPy array (e.g. from
    local_obs_climatology), all four case masks and per-case costs are computed
    element-wise so the decomposition exactly mirrors the main pipeline (verified
    against det_scores.calculate_twmae_components / calculate_twmae: hit_cost,
    miss_cost, fa_cost and the hit/miss/FA contribution totals match the
    scores_by_leadtime CSV to within floating-point / minor dataset-snapshot
    tolerance for both scalar and per-station thresholds).

    Panel 4 is a waterfall/bridge chart of twMAE(fc2) − twMAE(fc1), decomposed
    into Δhit / Δmiss / ΔFA contribution steps (in that order). Because the
    three contributions are purely additive (they sum exactly to twMAE with no
    cross terms), the bridge is exact and its three steps are order-independent.
    """
    print("[10/11] twMAE Decomposition (hits / misses / false alarms)...")

    var_short = condition["var_short"]
    mode = condition.get("threshold_mode", "percentile")

    # --- Resolve threshold: per-station array or scalar ----------------------
    threshold_arr = condition.get("threshold_arr", None)
    use_per_station = (threshold_arr is not None and
                       isinstance(threshold_arr, np.ndarray) and
                       len(threshold_arr) == len(obs_data))
    if use_per_station:
        valid_thr = threshold_arr[~np.isnan(threshold_arr)]
        T_display = float(np.mean(valid_thr)) if len(valid_thr) else float(threshold)
        T_label   = f"per-station (mean={T_display:.3f})"
    else:
        T_display = float(threshold)
        T_label   = f"{T_display:.4f}"

    names = [condition["expver1"], condition["expver2"]]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))

    # ---- Shared decomposition helper ----------------------------------------
    event_type = condition.get('event_type', 'above')
    def _decompose(fc, obs, T_in):
        """Decompose twMAE into hit / miss / FA contributions.

        Works for both upper tail (event_type='above') and lower tail ('below').
        T_in can be a scalar or a 1-D numpy array of the same length as fc/obs.
        NaN entries in an array T_in are excluded from all counts and means.
        """
        if isinstance(T_in, np.ndarray):
            valid = ~np.isnan(T_in)
            fc_v  = fc[valid];  obs_v = obs[valid];  T_v = T_in[valid]
        else:
            fc_v  = fc;  obs_v = obs;  T_v = T_in
        N = len(obs_v)
        if N == 0:
            return {k: 0 for k in
                    ("n_hit", "n_miss", "n_fa", "N",
                     "hit_cost", "miss_cost", "fa_cost",
                     "hit_contrib", "miss_contrib", "fa_contrib", "total")}
        scalar_T = not isinstance(T_v, np.ndarray)
        if event_type == 'below':
            hit_m  = (obs_v <= T_v) & (fc_v <= T_v)
            miss_m = (obs_v <= T_v) & (fc_v >  T_v)
            fa_m   = (obs_v >  T_v) & (fc_v <= T_v)
            T_miss = T_v if scalar_T else T_v[miss_m]
            T_fa   = T_v if scalar_T else T_v[fa_m]
            hit_cost  = float(np.mean(np.abs(fc_v[hit_m]  - obs_v[hit_m]))) if hit_m.sum()  > 0 else 0.0
            miss_cost = float(np.mean(T_miss - obs_v[miss_m]))               if miss_m.sum() > 0 else 0.0
            fa_cost   = float(np.mean(obs_v[fa_m]   - T_fa))                 if fa_m.sum()   > 0 else 0.0
        else:
            hit_m  = (obs_v >= T_v) & (fc_v >= T_v)
            miss_m = (obs_v >= T_v) & (fc_v <  T_v)
            fa_m   = (obs_v <  T_v) & (fc_v >= T_v)
            T_miss = T_v if scalar_T else T_v[miss_m]
            T_fa   = T_v if scalar_T else T_v[fa_m]
            hit_cost  = float(np.mean(np.abs(fc_v[hit_m]  - obs_v[hit_m]))) if hit_m.sum()  > 0 else 0.0
            miss_cost = float(np.mean(obs_v[miss_m] - T_miss))              if miss_m.sum() > 0 else 0.0
            fa_cost   = float(np.mean(fc_v[fa_m]   - T_fa))                 if fa_m.sum()   > 0 else 0.0

        hit_contrib  = hit_cost  * hit_m.sum()  / N
        miss_contrib = miss_cost * miss_m.sum() / N
        fa_contrib   = fa_cost   * fa_m.sum()   / N
        return {
            "n_hit": int(hit_m.sum()), "n_miss": int(miss_m.sum()),
            "n_fa":  int(fa_m.sum()),  "N": N,
            "hit_cost": hit_cost,  "miss_cost": miss_cost,  "fa_cost": fa_cost,
            "hit_contrib": hit_contrib, "miss_contrib": miss_contrib,
            "fa_contrib":  fa_contrib,
            "total": hit_contrib + miss_contrib + fa_contrib,
        }

    T_for_decomp = threshold_arr if use_per_station else float(threshold)
    d1 = _decompose(fc1_data, obs_data, T_for_decomp)
    d2 = _decompose(fc2_data, obs_data, T_for_decomp)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    c_hit = _style.C_HIT; c_miss = _style.C_MISS; c_fa = _style.C_FA
    c1 = _style.C_FC1; c2 = _style.C_FC2   # fc1=dark blue, fc2=dark red

    # =========================================================================
    # Panel 1 (top-left): twMAE budget — stacked bar
    # Each bar = fraction_of_samples × mean_cost_per_case, stacked hit/miss/FA
    # =========================================================================
    ax = axes[0, 0]
    x = np.arange(2); w = 0.45
    bot = np.zeros(2)
    for label, col, vals in [
        ("Hits",         c_hit,  [d1["hit_contrib"],  d2["hit_contrib"]]),
        ("Misses",       c_miss, [d1["miss_contrib"], d2["miss_contrib"]]),
        ("False Alarms", c_fa,   [d1["fa_contrib"],   d2["fa_contrib"]]),
    ]:
        bars = ax.bar(x, vals, w, bottom=bot, label=label, color=col, alpha=0.88)
        for bar, v, b in zip(bars, vals, bot):
            if v > 0.003 * max(d1["total"], d2["total"], 1e-9):
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                        f"{v:.5f}", ha="center", va="center", fontsize=9,
                        color="white", weight="bold")
        bot += np.array(vals)
    for i, d in enumerate([d1, d2]):
        ax.text(i, d["total"] * 1.04, f"Total\n{d['total']:.5f}",
                ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(f"Contribution to twMAE ({unit})")
    ax.set_title("Panel 1 — twMAE Budget\n"
                 r"Each segment = (n_case / N) × mean cost per case",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # =========================================================================
    # Panel 2 (top-right): Case counts — grouped bar
    # =========================================================================
    ax = axes[0, 1]
    if event_type == 'below':
        cases = ["Hits\n(obs≤T, fc≤T)", "Misses\n(obs≤T, fc>T)", "False Alarms\n(obs>T, fc≤T)"]
    else:
        cases = ["Hits\n(obs≥T, fc≥T)", "Misses\n(obs≥T, fc<T)", "False Alarms\n(obs<T, fc≥T)"]
    counts1 = [d1["n_hit"], d1["n_miss"], d1["n_fa"]]
    counts2 = [d2["n_hit"], d2["n_miss"], d2["n_fa"]]
    xc = np.arange(3); wc = 0.35
    bars1 = ax.bar(xc - wc/2, counts1, wc, label=names[0], color=c1, alpha=0.82)
    bars2 = ax.bar(xc + wc/2, counts2, wc, label=names[1], color=c2, alpha=0.82)
    for bars, counts in [(bars1, counts1), (bars2, counts2)]:
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"{v:,}",
                    ha="center", va="bottom", fontsize=8, weight="bold",
                    color=bar.get_facecolor())
    for xi, col in zip(xc, [c_hit, c_miss, c_fa]):
        ax.axvspan(xi - 0.5, xi + 0.5, color=col, alpha=0.07)
    ax.set_xticks(xc); ax.set_xticklabels(cases, fontsize=9)
    ax.set_ylabel("Number of cases  (N = {:,})".format(d1["N"]))
    ax.set_title("Panel 2 — Case Counts\n"
                 "How many of each outcome does each model produce?", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # =========================================================================
    # Panel 3 (bottom-left): Mean cost per case — grouped bar
    # Bars show cost-per-case (intensive); bracketed annotation below each
    # bar pair shows the Panel-1 contribution (n/N × cost) so the user can
    # see why a lower per-case cost can still produce a worse twMAE total.
    # =========================================================================
    ax = axes[1, 0]
    case_labels = ["Hit error\n|fc − obs|",
                   "Miss penalty\nT − obs" if event_type == 'below' else "Miss penalty\nobs − T",
                   "FA penalty\nobs − T" if event_type == 'below' else "FA penalty\nfc − T"]
    costs1 = [d1["hit_cost"], d1["miss_cost"], d1["fa_cost"]]
    costs2 = [d2["hit_cost"], d2["miss_cost"], d2["fa_cost"]]
    contribs1 = [d1["hit_contrib"], d1["miss_contrib"], d1["fa_contrib"]]
    contribs2 = [d2["hit_contrib"], d2["miss_contrib"], d2["fa_contrib"]]
    xp = np.arange(3); wp = 0.35
    bars1 = ax.bar(xp - wp/2, costs1, wp, label=names[0], color=c1, alpha=0.82)
    bars2 = ax.bar(xp + wp/2, costs2, wp, label=names[1], color=c2, alpha=0.82)
    # Per-case cost labels above each bar
    for bars, costs in [(bars1, costs1), (bars2, costs2)]:
        for bar, v in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9, weight="bold",
                    color=bar.get_facecolor())
    # Contribution annotations below the x-axis labels (twMAE weight = n/N × cost)
    ax2_ylim = ax.get_ylim()
    for xi, c1v, c2v in zip(xp, contribs1, contribs2):
        ax.annotate(
            f"contrib: {c1v:.5f} / {c2v:.5f}\n({names[0]} / {names[1]})",
            xy=(xi, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -46), textcoords="offset points",
            ha="center", va="top", fontsize=7.5, color="#444444",
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#cccccc", lw=0.7),
        )
    for xi, col in zip(xp, [c_hit, c_miss, c_fa]):
        ax.axvspan(xi - 0.5, xi + 0.5, color=col, alpha=0.07)
    ax.set_xticks(xp); ax.set_xticklabels(case_labels, fontsize=9)
    ax.set_ylabel(f"Mean cost per case ({unit})")
    ax.set_title(
        "Panel 3 — Severity per case type\n"
        "Bar height = average error/penalty for ONE case of that type\n"
        "↳ annotation below = bar × (n/N) = its actual share of twMAE (Panel 1)",
        fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.subplots_adjust(bottom=0.15)  # make room for contribution annotations

    # =========================================================================
    # Panel 4 (bottom-right): Waterfall — twMAE difference decomposed
    #
    # Shows twMAE(fc2) − twMAE(fc1) split into hit / miss / FA components as a
    # bridge/waterfall chart.  Red floating bars = fc2 worse, green = fc2 better.
    # This makes it immediately obvious WHICH component drives the net result.
    # =========================================================================
    ax = axes[1, 1]

    delta_hit   = d2["hit_contrib"]  - d1["hit_contrib"]
    delta_miss  = d2["miss_contrib"] - d1["miss_contrib"]
    delta_fa    = d2["fa_contrib"]   - d1["fa_contrib"]
    delta_total = d2["total"] - d1["total"]

    ww = 0.55
    positions = [0, 1, 2, 3, 4]
    xlabels = [names[0], "Δ Hits", "Δ Misses", "Δ FAs", names[1]]

    # Baseline bar: fc1 total
    ax.bar(0, d1["total"], ww, bottom=0, color=c1, alpha=0.85, zorder=3)
    ax.text(0, d1["total"] * 1.015, f"{d1['total']:.5f}",
            ha="center", va="bottom", fontsize=9, weight="bold", color=c1)

    # Three floating delta bars
    running = d1["total"]
    delta_vals  = [delta_hit,   delta_miss,   delta_fa]
    delta_hatch = [c_hit,       c_miss,       c_fa]
    for i, (dv, base_col) in enumerate(zip(delta_vals, delta_hatch), start=1):
        bar_col = "#c62828" if dv > 0 else "#2e7d32"   # red=fc2 worse, green=fc2 better
        bottom  = running if dv >= 0 else running + dv
        height  = abs(dv)
        ax.bar(i, height, ww, bottom=bottom, color=bar_col, alpha=0.82, zorder=3,
               edgecolor=base_col, linewidth=1.5)
        # Value label inside/near the bar
        label_y = bottom + height / 2 if height > 0 else bottom
        ax.text(i, label_y, f"{dv:+.5f}",
                ha="center", va="center", fontsize=9, weight="bold",
                color="white" if height > 1e-7 else bar_col)
        # Connector line at the running total
        running += dv
        ax.plot([i - ww / 2, i + ww / 2 + (1 - ww)], [running, running],
                color="grey", ls="--", lw=0.9, alpha=0.6, zorder=2)

    # Result bar: fc2 total
    ax.bar(4, d2["total"], ww, bottom=0, color=c2, alpha=0.85, zorder=3)
    ax.text(4, d2["total"] * 1.015, f"{d2['total']:.5f}",
            ha="center", va="bottom", fontsize=9, weight="bold", color=c2)

    # Horizontal reference line at fc1 baseline
    ax.axhline(d1["total"], color=c1, ls=":", lw=1.2, alpha=0.45)

    # Net-change summary box
    net_sign  = f"▲ {names[1]} worse" if delta_total > 0 else f"▼ {names[1]} better"
    net_color = "#c62828" if delta_total > 0 else "#2e7d32"
    ax.text(2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else max(d1["total"], d2["total"]) * 1.1,
            f"Net Δ = {delta_total:+.5f}  ({net_sign})",
            ha="center", va="top", fontsize=10, weight="bold", color=net_color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=net_color, lw=1.5),
            zorder=5)

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=c1, label=f"{names[0]} baseline"),
        Patch(facecolor=c2, label=f"{names[1]} result"),
        Patch(facecolor="#c62828", label=f"{names[1]} worse (+)"),
        Patch(facecolor="#2e7d32", label=f"{names[1]} better (−)"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")

    ax.set_xticks(positions); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel(f"twMAE contribution ({unit})")
    ax.set_title(f"Panel 4 — twMAE Difference Waterfall\n"
                 f"{names[1]} − {names[0]}  decomposed into hit / miss / FA",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"twMAE Decomposition — {var_lbl} | Day {condition['forecast_day']} "
        f"| T = {T_label} {unit}  ({names[0]}  vs  {names[1]})",
        fontsize=13, weight="bold", y=1.01
    )
    plt.tight_layout()
    _savefig(fig, f"10_twmae_decomposition_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")

    # Console summary
    print(f"\n  twMAE Decomposition Summary (T = {T_label} {unit}):")
    print(f"  {'':30s} {'hit contrib':>12}  {'miss contrib':>12}  {'FA contrib':>12}  {'total':>10}")
    print(f"  {'-'*70}")
    for lbl, d in [(names[0], d1), (names[1], d2)]:
        print(f"  {lbl:30s} {d['hit_contrib']:>12.5f}  {d['miss_contrib']:>12.5f}  "
              f"{d['fa_contrib']:>12.5f}  {d['total']:>10.5f}")
        print(f"  {'  (n, mean cost)':30s} "
              f"  ({d['n_hit']:>8,}, {d['hit_cost']:.4f})"
              f"  ({d['n_miss']:>8,}, {d['miss_cost']:.4f})"
              f"  ({d['n_fa']:>8,}, {d['fa_cost']:.4f})")
    print(f"\n  Net Δ twMAE ({names[1]} − {names[0]}):  {delta_total:+.5f}")
    print(f"    Δ hit contrib  = {delta_hit:+.5f}")
    print(f"    Δ miss contrib = {delta_miss:+.5f}")
    print(f"    Δ FA contrib   = {delta_fa:+.5f}")




def plot_twmae_percentile_decomposition(fc1_data, fc2_data, obs_data, condition):
    """Plot 11: twMAE stacked-bar decomposition across key percentile thresholds.

    Warm (event_type='above'): p90, p93, p96, p99
    Cold (event_type='below'): p1,  p3,  p6,  p8,  p9

    Sub-plot 1 — Grouped stacked bars (left=fc1, right=fc2) at each percentile,
    stacked hit/miss/FA → bar height equals twMAE.  Immediately shows both the
    total error and how it is composed for each model.

    Sub-plot 2 — Δ twMAE (fc2 − fc1) decomposed into Δhit / Δmiss / ΔFA
    stacked bars (red = fc2 worse, green = fc2 better) + net-change line.

    Per-station obs-climatology thresholds are used for every integer percentile
    when the config uses local_obs_climatology (matches the main pipeline exactly).
    """
    print("[11/11] twMAE percentile decomposition (stacked bars)...")

    import copy
    var_short  = condition["var_short"]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    names  = [condition["expver1"], condition["expver2"]]
    c1 = _style.C_FC1; c2 = _style.C_FC2
    c_hit = _style.C_HIT; c_miss = _style.C_MISS; c_fa = _style.C_FA

    # Percentile set depends on tail direction
    cfg_raw    = condition.get('_config', None)
    # event_type already resolved in condition by main(); _config is a fallback
    _thr_cfg   = (cfg_raw or {}).get('threshold', {})
    if _thr_cfg.get('method') == 'fixed':
        _et_from_cfg = _thr_cfg.get('fixed', {}).get('event_type', 'above')
    else:
        _et_from_cfg = _thr_cfg.get('event_type', 'above')
    event_type = condition.get('event_type', _et_from_cfg)
    # For warm (above): sweep from moderate to most extreme (left→right)
    # For cold (below): sweep from moderate to most extreme too (10→1, left→right)
    if event_type == 'below':
        pct_values = [10, 8, 6, 3, 1]
    else:
        pct_values = [90, 93, 96, 99]
    pct_labels = [f"p{p}" for p in pct_values]

    df_raw     = condition.get('_df', None)
    thr_method = (cfg_raw or {}).get('threshold', {}).get('method', '')
    use_perstation = (df_raw is not None and cfg_raw is not None
                      and thr_method == 'local_obs_climatology')

    if use_perstation:
        try:
            import threshold as _thr_module
        except ImportError:
            use_perstation = False

    # ---- decompose twMAE into hit / miss / FA for a per-station array --------
    def _decompose(fc, obs, T_in, etype):
        if isinstance(T_in, np.ndarray):
            valid = ~np.isnan(T_in)
            fc_v  = fc[valid]; obs_v = obs[valid]; T_v = T_in[valid]
        else:
            fc_v  = fc; obs_v = obs; T_v = T_in
        N = len(obs_v)
        if N == 0:
            return dict(hit_contrib=0., miss_contrib=0., fa_contrib=0., total=0., T_repr=float(T_in if np.isscalar(T_in) else np.nan))
        scalar_T = np.isscalar(T_v)
        if etype == 'below':
            hit_m  = (obs_v <= T_v) & (fc_v <= T_v)
            miss_m = (obs_v <= T_v) & (fc_v >  T_v)
            fa_m   = (obs_v >  T_v) & (fc_v <= T_v)
            T_miss = T_v if scalar_T else T_v[miss_m]
            T_fa   = T_v if scalar_T else T_v[fa_m]
            hit_cost  = float(np.mean(np.abs(fc_v[hit_m] - obs_v[hit_m]))) if hit_m.sum() > 0 else 0.
            miss_cost = float(np.mean(T_miss - obs_v[miss_m]))              if miss_m.sum() > 0 else 0.
            fa_cost   = float(np.mean(obs_v[fa_m] - T_fa))                  if fa_m.sum()  > 0 else 0.
        else:
            hit_m  = (obs_v >= T_v) & (fc_v >= T_v)
            miss_m = (obs_v >= T_v) & (fc_v <  T_v)
            fa_m   = (obs_v <  T_v) & (fc_v >= T_v)
            T_miss = T_v if scalar_T else T_v[miss_m]
            T_fa   = T_v if scalar_T else T_v[fa_m]
            hit_cost  = float(np.mean(np.abs(fc_v[hit_m] - obs_v[hit_m]))) if hit_m.sum() > 0 else 0.
            miss_cost = float(np.mean(obs_v[miss_m] - T_miss))             if miss_m.sum() > 0 else 0.
            fa_cost   = float(np.mean(fc_v[fa_m]   - T_fa))                if fa_m.sum()  > 0 else 0.
        return dict(
            hit_contrib  = hit_cost  * hit_m.sum()  / N,
            miss_contrib = miss_cost * miss_m.sum() / N,
            fa_contrib   = fa_cost   * fa_m.sum()   / N,
            total        = (hit_cost*hit_m.sum() + miss_cost*miss_m.sum() + fa_cost*fa_m.sum()) / N,
            T_repr       = float(np.nanmean(T_v)),
        )

    # ---- compute results at each percentile --------------------------------
    results1, results2, thresholds, is_perstation = [], [], [], []
    for pct in pct_values:
        T_arr = None
        if use_perstation:
            cfg_copy = copy.deepcopy(cfg_raw)
            cfg_copy['threshold']['local_obs_climatology']['percentile'] = int(pct)
            try:
                _ser  = _thr_module._compute_local_obs_climatology_threshold(cfg_copy, df_raw)
                T_arr = _ser.values.astype(np.float32)
                print(f"    p{pct}: per-station  mean T={float(np.nanmean(T_arr)):.3f}")
            except Exception as e:
                print(f"    p{pct}: WARNING per-station failed ({e}), using pooled")
                T_arr = None

        if T_arr is not None:
            r1 = _decompose(fc1_data, obs_data, T_arr, event_type)
            r2 = _decompose(fc2_data, obs_data, T_arr, event_type)
            is_perstation.append(True)
        else:
            T_sc = float(np.percentile(obs_data, pct))
            r1 = _decompose(fc1_data, obs_data, T_sc, event_type)
            r2 = _decompose(fc2_data, obs_data, T_sc, event_type)
            is_perstation.append(False)
            print(f"    p{pct}: pooled  T={T_sc:.3f}")
        results1.append(r1); results2.append(r2)
        thresholds.append(r1['T_repr'])

    # ---- build figure -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    n = len(pct_values)
    x = np.arange(n)
    w = 0.38    # width of each model's bar

    from matplotlib.patches import Patch
    from matplotlib.lines  import Line2D

    # =========================================================================
    # LEFT — Grouped stacked bars: absolute twMAE decomposition for each model
    #
    # Stack order: Misses + FAs at bottom (pure failures → push bar upward),
    # Hits at top with hatching (model DID catch these events, but not perfectly).
    # A dashed separator line + label divide the "failure zone" from the
    # "detected event zone" so the two roles are immediately clear.
    # =========================================================================
    ax = axes[0]
    for offset, results, edge_col, model_name in [
        (-w/2, results1, c1, names[0]),
        ( w/2, results2, c2, names[1]),
    ]:
        for xi, r in zip(x, results):
            miss_v = r["miss_contrib"]
            fa_v   = r["fa_contrib"]
            hit_v  = r["hit_contrib"]
            total  = r["total"]

            # ---- Misses (bottom) ----
            ax.bar(xi + offset, miss_v, w,
                   bottom=0, color=c_miss, alpha=0.90,
                   edgecolor=edge_col, linewidth=1.2)
            # ---- False Alarms (above misses) ----
            ax.bar(xi + offset, fa_v, w,
                   bottom=miss_v, color=c_fa, alpha=0.90,
                   edgecolor=edge_col, linewidth=1.2)
            # ---- Hits (top, hatched) ----
            ax.bar(xi + offset, hit_v, w,
                   bottom=miss_v + fa_v, color=c_hit, alpha=0.75,
                   edgecolor=edge_col, linewidth=1.5,
                   hatch="///", label="_nolegend_")

            # Separator line between failure zone and hit zone
            sep_y = miss_v + fa_v
            ax.plot([xi + offset - w/2, xi + offset + w/2], [sep_y, sep_y],
                    color="white", lw=1.8, zorder=5)

            # Total twMAE label at top
            ax.text(xi + offset, total * 1.015, f"{total:.4f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=edge_col, weight="bold")

    # Bottom x-axis: percentile labels; top x-axis: threshold values
    ax.set_xticks(x)
    ax.set_xticklabels(pct_labels, fontsize=11)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(x)
    ax2.set_xticklabels([f"{T:.2f}" for T in thresholds], fontsize=8, rotation=30)
    ax2.set_xlabel(f"Threshold ({unit})", fontsize=9)

    ax.set_ylabel(f"twMAE ({unit})")
    thr_note = "per-station obs clim" if any(is_perstation) else "pooled obs"
    ax.set_title(
        f"twMAE by percentile — {var_lbl}\n"
        f"Day {condition['forecast_day']} | {condition.get('season','all')} | "
        f"{condition.get('terrain','all')} | {thr_note}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Legend: model identity (border colour) + stacking role
    leg_els = [
        Patch(facecolor="white", edgecolor=c1, linewidth=2, label=f"← {names[0]}"),
        Patch(facecolor="white", edgecolor=c2, linewidth=2, label=f"→ {names[1]}"),
        Patch(facecolor=c_miss, alpha=0.90, label="Misses  (events not caught)"),
        Patch(facecolor=c_fa,   alpha=0.90, label="False Alarms  (wrong predictions)"),
        Patch(facecolor=c_hit,  alpha=0.75, hatch="///",
              label="Hits  (events caught — residual error)"),
    ]
    ax.legend(handles=leg_els, fontsize=8, loc="upper left")

    # =========================================================================
    # RIGHT — Δ twMAE decomposition (fc2 − fc1) at each percentile
    # =========================================================================
    ax = axes[1]
    dh   = [r2["hit_contrib"]  - r1["hit_contrib"]  for r1, r2 in zip(results1, results2)]
    dm   = [r2["miss_contrib"] - r1["miss_contrib"] for r1, r2 in zip(results1, results2)]
    df_  = [r2["fa_contrib"]   - r1["fa_contrib"]   for r1, r2 in zip(results1, results2)]
    dnet = [h + m + f for h, m, f in zip(dh, dm, df_)]

    ww = 0.55
    for i, (dh_v, dm_v, df_v) in enumerate(zip(dh, dm, df_)):
        running = 0.0
        for dv, edge_col in [(dh_v, c_hit), (dm_v, c_miss), (df_v, c_fa)]:
            if abs(dv) < 1e-10:
                running += dv; continue
            fill_col = "#c62828" if dv > 0 else "#2e7d32"
            bottom   = running if dv >= 0 else running + dv
            ax.bar(i, abs(dv), ww, bottom=bottom,
                   color=fill_col, alpha=0.80, edgecolor=edge_col, linewidth=1.5)
            running += dv

    # Net Δ line
    ax.plot(x, dnet, color="black", lw=2.2, marker="D", ms=8, zorder=5)
    for xi, dn in zip(x, dnet):
        ax.text(xi, dn, f" {dn:+.5f}", fontsize=7.5,
                ha="center", va=("bottom" if dn >= 0 else "top"),
                color="black", weight="bold")

    ax.axhline(0, color="grey", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(pct_labels, fontsize=11)
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim()); ax3.set_xticks(x)
    ax3.set_xticklabels([f"{T:.2f}" for T in thresholds], fontsize=8, rotation=30)
    ax3.set_xlabel(f"Threshold ({unit})", fontsize=9)

    ax.set_ylabel(f"Δ twMAE   ({names[1]} − {names[0]}, {unit})")
    ax.set_title(
        f"Δ twMAE decomposed — {var_lbl}\n"
        f"Red = {names[1]} worse (+)   Green = {names[1]} better (−)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, axis="y")

    leg_els2 = [
        Patch(facecolor="#c62828", label=f"{names[1]} worse (+Δ)"),
        Patch(facecolor="#2e7d32", label=f"{names[1]} better (−Δ)"),
        Patch(facecolor="white", edgecolor=c_hit,  linewidth=2, label="Hit component"),
        Patch(facecolor="white", edgecolor=c_miss, linewidth=2, label="Miss component"),
        Patch(facecolor="white", edgecolor=c_fa,   linewidth=2, label="FA component"),
        Line2D([0], [0], color="black", marker="D", ms=8, lw=2, label="Net Δ twMAE"),
    ]
    ax.legend(handles=leg_els2, fontsize=8, loc="lower left")

    fig.suptitle(
        f"twMAE by percentile threshold — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}",
        fontsize=12, weight="bold",
    )
    plt.tight_layout()
    _savefig(fig, f"11_twmae_pct_decomp_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")

    # Console table
    ps_note = lambda ps: "per-stn" if ps else "pooled "
    print(f"\n  {'Pct':>6}  {'T (repr)':>9}  {'type':>8}  "
          f"{names[0]+' twMAE':>14}  {names[1]+' twMAE':>14}  "
          f"{'Δhit':>10}  {'Δmiss':>10}  {'ΔFA':>10}  {'Net Δ':>10}")
    print(f"  {'-'*97}")
    for pct, T, ps, r1, r2, dh_v, dm_v, df_v, dn_v in zip(
            pct_values, thresholds, is_perstation,
            results1, results2, dh, dm, df_, dnet):
        print(f"  {pct:>6}  {T:>9.3f}  {ps_note(ps):>8}  "
              f"{r1['total']:>14.5f}  {r2['total']:>14.5f}  "
              f"{dh_v:>+10.5f}  {dm_v:>+10.5f}  {df_v:>+10.5f}  {dn_v:>+10.5f}")


def _perstation_sweep(condition, pct_values, obs_data, fc1_data, fc2_data):
    """Shared helper: compute per-station T and (obs,fc1,fc2 masked arrays) at
    each percentile level, mirroring the approach in plot_twmae_percentile_decomposition.

    Returns a list of dicts, one per pct level:
        {'pct': int, 'T_arr': ndarray|float, 'per_station': bool,
         'obs_v': ndarray, 'fc1_v': ndarray, 'fc2_v': ndarray, 'T_v': ndarray}
    T_arr is always aligned with obs_data (possibly NaN-masked).
    """
    import copy
    cfg_raw    = condition.get('_config', None)
    df_raw     = condition.get('_df', None)
    thr_method = (cfg_raw or {}).get('threshold', {}).get('method', '')
    can_ps = (df_raw is not None and cfg_raw is not None
              and thr_method == 'local_obs_climatology')
    if can_ps:
        try:
            import threshold as _thr_module
        except ImportError:
            can_ps = False

    results = []
    for pct in pct_values:
        T_arr = None
        if can_ps:
            cfg_copy = copy.deepcopy(cfg_raw)
            cfg_copy['threshold']['local_obs_climatology']['percentile'] = int(pct)
            try:
                _ser  = _thr_module._compute_local_obs_climatology_threshold(cfg_copy, df_raw)
                T_arr = _ser.values.astype(np.float32)
            except Exception:
                T_arr = None

        per_station = T_arr is not None
        if not per_station:
            T_arr = np.full(len(obs_data), float(np.percentile(obs_data, pct)),
                            dtype=np.float32)

        valid  = ~np.isnan(T_arr)
        obs_v  = obs_data[valid]
        fc1_v  = fc1_data[valid]
        fc2_v  = fc2_data[valid]
        T_v    = T_arr[valid]
        results.append(dict(pct=pct, T_arr=T_arr, per_station=per_station,
                            obs_v=obs_v, fc1_v=fc1_v, fc2_v=fc2_v, T_v=T_v))
    return results


def plot_conditional_bias_noise(fc1_data, fc2_data, obs_data, condition):
    """Plot 15: Conditional bias (mean fc−obs) and noise (std of fc−obs) for extreme
    cases, sweeping across percentile threshold levels with per-station T.

    twMAE on hits = mean|fc−obs| ≈ |bias| + noise (roughly).
    Splitting them answers: is model 1 better because it is less systematically
    biased in extreme event intensity, or because it is less noisy (more consistent)?
    These require fundamentally different model fixes.

    Reference: Murphy (1987) — 'A General Framework for Forecast Verification',
    MWR 115:1330–1338 (bias²+conditional-bias+correlation decomposition of MSE,
    applied here to the extreme-case subset).
    """
    print("[15/17] Conditional Bias & Noise (extreme cases, per-station T)...")

    import copy
    var_short  = condition["var_short"]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    names  = [condition["expver1"], condition["expver2"]]
    c1 = _style.C_FC1; c2 = _style.C_FC2
    event_type = condition.get("event_type", "above")

    cfg_raw = condition.get('_config', None)
    _thr_cfg = (cfg_raw or {}).get('threshold', {})
    if event_type == 'below':
        pct_values = [10, 8, 6, 3, 1]
    else:
        pct_values = [90, 93, 96, 99]
    pct_labels = [f"p{p}" for p in pct_values]

    sweep = _perstation_sweep(condition, pct_values, obs_data, fc1_data, fc2_data)

    # For each level, compute mean bias and noise for extreme cases (obs in tail)
    bias1, bias2, noise1, noise2, T_reprs = [], [], [], [], []
    for s in sweep:
        obs_v, fc1_v, fc2_v, T_v = s['obs_v'], s['fc1_v'], s['fc2_v'], s['T_v']
        if event_type == 'below':
            ext_m1 = (obs_v <= T_v) | (fc1_v <= T_v)   # any party in tail
            ext_m2 = (obs_v <= T_v) | (fc2_v <= T_v)
        else:
            ext_m1 = (obs_v >= T_v) | (fc1_v >= T_v)
            ext_m2 = (obs_v >= T_v) | (fc2_v >= T_v)
        err1 = fc1_v[ext_m1] - obs_v[ext_m1]
        err2 = fc2_v[ext_m2] - obs_v[ext_m2]
        bias1.append(float(np.mean(err1)) if len(err1) > 0 else np.nan)
        bias2.append(float(np.mean(err2)) if len(err2) > 0 else np.nan)
        noise1.append(float(np.std(err1))  if len(err1) > 0 else np.nan)
        noise2.append(float(np.std(err2))  if len(err2) > 0 else np.nan)
        T_reprs.append(float(np.mean(T_v)))
        ps_note = "per-station" if s['per_station'] else "pooled"
        print(f"    p{s['pct']:>3} ({ps_note}): "
              f"bias1={bias1[-1]:+.4f}  noise1={noise1[-1]:.4f}  "
              f"bias2={bias2[-1]:+.4f}  noise2={noise2[-1]:.4f}")

    x = np.arange(len(pct_values))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── LEFT: conditional bias ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, bias1, "o-", color=c1, lw=2.2, ms=8, label=names[0])
    ax.plot(x, bias2, "s-", color=c2, lw=2.2, ms=8, label=names[1])
    ax.axhline(0, color="grey", ls="--", lw=1.2)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(x)
    ax2.set_xticklabels([f"{T:.2f}" for T in T_reprs], fontsize=8, rotation=30)
    ax2.set_xlabel(f"Mean T ({unit})", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=11)
    ax.set_ylabel(f"Mean (fc − obs) on extreme cases ({unit})")
    ax.set_title("Conditional Bias\nE[fc − obs | obs or fc in tail]\n"
                 "Positive = model over-predicts extreme intensity", fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── RIGHT: conditional noise (std) ────────────────────────────────────
    ax = axes[1]
    ax.plot(x, noise1, "o-", color=c1, lw=2.2, ms=8, label=names[0])
    ax.plot(x, noise2, "s-", color=c2, lw=2.2, ms=8, label=names[1])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(x)
    ax2.set_xticklabels([f"{T:.2f}" for T in T_reprs], fontsize=8, rotation=30)
    ax2.set_xlabel(f"Mean T ({unit})", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=11)
    ax.set_ylabel(f"Std(fc − obs) on extreme cases ({unit})")
    ax.set_title("Conditional Noise (Random Error)\nstd(fc − obs | obs or fc in tail)\n"
                 "Lower = model is more consistent on extremes", fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3)

    thr_type = "per-station obs clim" if any(s['per_station'] for s in sweep) else "pooled"
    fig.suptitle(
        f"Conditional Bias & Noise — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}  "
        f"|  T: {thr_type}",
        fontsize=11, weight="bold",
    )
    # ── Auto-interpretation ──────────────────────────────────────────────
    _b1e, _b2e = bias1[-1],  bias2[-1]   # most extreme level
    _n1e, _n2e = noise1[-1], noise2[-1]
    _pe = pct_values[-1]
    if not (np.isnan(_b1e) or np.isnan(_b2e) or np.isnan(_n1e) or np.isnan(_n2e)):
        _bbias = names[0] if abs(_b1e) <= abs(_b2e) else names[1]
        _bnoise = names[0] if _n1e <= _n2e else names[1]
        _bd1 = 'over-predicts' if _b1e > 0 else 'under-predicts'
        _bd2 = 'over-predicts' if _b2e > 0 else 'under-predicts'
        _dom_src = 'systematic bias' if abs(_b1e-_b2e) >= abs(_n1e-_n2e) else 'random noise'
        _interp = (
            f"► At p{_pe} (most extreme threshold): {_bbias} has smaller systematic bias "
            f"({abs(_b1e):.3f} vs {abs(_b2e):.3f} {unit}).  "
            f"{names[0]} {_bd1} by {abs(_b1e):.3f} {unit};  "
            f"{names[1]} {_bd2} by {abs(_b2e):.3f} {unit}.\n"
            f"{_bnoise} has less random error (noise: {min(_n1e,_n2e):.3f} vs "
            f"{max(_n1e,_n2e):.3f} {unit}).  "
            f"Dominant source of model difference: {_dom_src} — "
            f"{'a bias correction could help the weaker model' if _dom_src=='systematic bias' else 'this reflects intrinsic forecast uncertainty, harder to reduce'}."
        )
    else:
        _interp = "► Insufficient data at the most extreme level."
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(0.5, 0.01, _interp, ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.93))
    _savefig(fig, f"15_conditional_bias_noise_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_conditional_bias_decomposed(fc1_data, fc2_data, obs_data, condition):
    """Plot 22: Conditional bias, split into 'real events' vs 'false alarms'.

    Figure 15's conditional bias mixes two opposite-signed error sources into
    one blended average: (obs≥T, fc<T) misses pull it negative (model
    UNDER-predicts real extremes — the classic, expected behaviour), while
    (obs<T, fc≥T) false alarms pull it positive (model invents an extreme on a
    day that wasn't one) — and since FAs are often as numerous as misses (or
    more) with larger magnitude, the blended Fig-15 number can end up positive
    even though the model clearly underestimates on the real events it misses.
    This plot reports the two contributions SEPARATELY so that doesn't happen:

      Left  — Bias on REAL events, mean(fc − obs | obs ≥ T)  (hits ∪ misses).
              Negative = model under-predicts the true extreme (expected/typical
              NWP behaviour — smoothing/damping of extremes).
      Right — Bias on FALSE ALARMS, mean(fc − obs | fc ≥ T, obs < T).
              Always positive by construction (fc crossed T, obs didn't) — shows
              how far the model overshoots on its spurious extreme calls.

    (event_type='below' mirrors this with obs≤T / fc≤T.)
    """
    print("[22] Conditional Bias — Real Events vs False Alarms (per-station T)...")

    var_short  = condition["var_short"]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    names  = [condition["expver1"], condition["expver2"]]
    c1 = _style.C_FC1; c2 = _style.C_FC2
    event_type = condition.get("event_type", "above")

    if event_type == 'below':
        pct_values = [10, 8, 6, 3, 1]
    else:
        pct_values = [90, 93, 96, 99]
    pct_labels = [f"p{p}" for p in pct_values]

    sweep = _perstation_sweep(condition, pct_values, obs_data, fc1_data, fc2_data)

    real1, real2, fa1, fa2, T_reprs = [], [], [], [], []
    for s in sweep:
        obs_v, fc1_v, fc2_v, T_v = s['obs_v'], s['fc1_v'], s['fc2_v'], s['T_v']
        if event_type == 'below':
            real_m1 = obs_v <= T_v;                 real_m2 = obs_v <= T_v
            fa_m1   = (obs_v > T_v) & (fc1_v <= T_v); fa_m2 = (obs_v > T_v) & (fc2_v <= T_v)
        else:
            real_m1 = obs_v >= T_v;                 real_m2 = obs_v >= T_v
            fa_m1   = (obs_v < T_v) & (fc1_v >= T_v); fa_m2 = (obs_v < T_v) & (fc2_v >= T_v)

        err1 = fc1_v - obs_v; err2 = fc2_v - obs_v
        real1.append(float(np.mean(err1[real_m1])) if real_m1.sum() > 0 else np.nan)
        real2.append(float(np.mean(err2[real_m2])) if real_m2.sum() > 0 else np.nan)
        fa1.append(float(np.mean(err1[fa_m1])) if fa_m1.sum() > 0 else np.nan)
        fa2.append(float(np.mean(err2[fa_m2])) if fa_m2.sum() > 0 else np.nan)
        T_reprs.append(float(np.mean(T_v)))
        ps_note = "per-station" if s['per_station'] else "pooled"
        print(f"    p{s['pct']:>3} ({ps_note}): "
              f"real1={real1[-1]:+.4f} (n={int(real_m1.sum())})  "
              f"real2={real2[-1]:+.4f} (n={int(real_m2.sum())})  "
              f"fa1={fa1[-1]:+.4f} (n={int(fa_m1.sum())})  "
              f"fa2={fa2[-1]:+.4f} (n={int(fa_m2.sum())})")

    x = np.arange(len(pct_values))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── LEFT: bias on real events (hits ∪ misses) ─────────────────────────
    ax = axes[0]
    ax.plot(x, real1, "o-", color=c1, lw=2.2, ms=8, label=names[0])
    ax.plot(x, real2, "s-", color=c2, lw=2.2, ms=8, label=names[1])
    ax.axhline(0, color="grey", ls="--", lw=1.2)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(x)
    ax2.set_xticklabels([f"{T:.2f}" for T in T_reprs], fontsize=8, rotation=30)
    ax2.set_xlabel(f"Mean T ({unit})", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=11)
    ax.set_ylabel(f"Mean (fc − obs) on real events ({unit})")
    ax.set_title("Bias on REAL Events\nE[fc − obs | obs in tail]  (hits \u222a misses)\n"
                 "Negative = model under-predicts the true extreme (typical)", fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── RIGHT: bias on false alarms ────────────────────────────────────────
    ax = axes[1]
    ax.plot(x, fa1, "o-", color=c1, lw=2.2, ms=8, label=names[0])
    ax.plot(x, fa2, "s-", color=c2, lw=2.2, ms=8, label=names[1])
    ax.axhline(0, color="grey", ls="--", lw=1.2)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(x)
    ax2.set_xticklabels([f"{T:.2f}" for T in T_reprs], fontsize=8, rotation=30)
    ax2.set_xlabel(f"Mean T ({unit})", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=11)
    ax.set_ylabel(f"Mean (fc − obs) on false alarms ({unit})")
    ax.set_title("Bias on FALSE ALARMS\nE[fc − obs | fc in tail, obs not]\n"
                 "Always positive by construction — lower = less severe overshoot", fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3)

    thr_type = "per-station obs clim" if any(s['per_station'] for s in sweep) else "pooled"
    fig.suptitle(
        f"Conditional Bias Decomposed — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}  "
        f"|  T: {thr_type}",
        fontsize=11, weight="bold",
    )
    # ── Auto-interpretation ──────────────────────────────────────────────
    _r1e, _r2e = real1[-1], real2[-1]
    _f1e, _f2e = fa1[-1],   fa2[-1]
    _pe = pct_values[-1]
    if not any(np.isnan(v) for v in (_r1e, _r2e, _f1e, _f2e)):
        _real_winner = names[0] if abs(_r1e) <= abs(_r2e) else names[1]
        _fa_winner   = names[0] if _f1e <= _f2e else names[1]
        _interp = (
            f"► At p{_pe}: on REAL events, {names[0]} under/over-predicts by {_r1e:+.3f} {unit}, "
            f"{names[1]} by {_r2e:+.3f} {unit} ({_real_winner} closer to zero = truer intensity).  "
            f"On FALSE ALARMS, {names[0]} overshoots by {_f1e:+.3f} {unit}, "
            f"{names[1]} by {_f2e:+.3f} {unit} ({_fa_winner} overshoots less).\n"
            f"Compare to Fig 15's blended bias — if that number's sign looks surprising, "
            f"it's because it mixes these two (typically opposite-signed) contributions "
            f"weighted by how many cases fall in each bucket."
        )
    else:
        _interp = "► Insufficient data at the most extreme level."
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(0.5, 0.01, _interp, ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.93))
    _savefig(fig, f"22_conditional_bias_decomposed_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_twmae_skill_score(fc1_data, fc2_data, obs_data, condition):
    """Plot 16: twMAE skill score vs threshold sweep (per-station T at each level).

    Reference forecast: always predict exactly T_s (the station threshold).
    Using Taggart's chaining function v_T(x) = max(x−T, 0), this gives:
        twMAE_ref = (1/N) * Σ |v_T(T_s) − v_T(obs_i)|
                  = (1/N) * Σ max(obs_i − T_i, 0)   [above-threshold case]
    This is *purely obs-based* — no model data needed.  It equals the mean
    exceedance of the observations above their station threshold.

    The skill score is then:
        twSS = 1 − twMAE_model / twMAE_ref
    Positive = better than "always predict T"; 0 = no skill; negative = worse.

    This is the deterministic analogue of the Brier Skill Score:
        BSS = 1 − BS / BS_clim
    where the climatological reference is also computed from observations alone.
    """
    print("[16/17] twMAE Skill Score vs threshold sweep (per-station T)...")

    var_short  = condition["var_short"]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    names  = [condition["expver1"], condition["expver2"]]
    c1 = _style.C_FC1; c2 = _style.C_FC2
    event_type = condition.get("event_type", "above")

    if event_type == 'below':
        pct_values = [10, 8, 6, 3, 1]
    else:
        pct_values = [90, 93, 96, 99]
    pct_labels = [f"p{p}" for p in pct_values]

    sweep = _perstation_sweep(condition, pct_values, obs_data, fc1_data, fc2_data)

    ss1, ss2, twmae1_vals, twmae2_vals, ref_vals, T_reprs = [], [], [], [], [], []
    for s in sweep:
        obs_v, fc1_v, fc2_v, T_v = s['obs_v'], s['fc1_v'], s['fc2_v'], s['T_v']
        N = len(obs_v)
        if N == 0:
            ss1.append(np.nan); ss2.append(np.nan); ref_vals.append(np.nan)
            T_reprs.append(np.nan); continue

        # Reference: twMAE of forecast that always predicts T_s
        if event_type == 'below':
            # v_T(x) = max(T-x, 0); v_T(T) = 0; ref cost = max(T-obs, 0)
            ref = float(np.mean(np.maximum(T_v - obs_v, 0.0)))
            # twMAE model: mean |v_T(fc) - v_T(obs)|
            tw1 = float(np.mean(np.abs(np.maximum(T_v - fc1_v, 0.0) -
                                       np.maximum(T_v - obs_v, 0.0))))
            tw2 = float(np.mean(np.abs(np.maximum(T_v - fc2_v, 0.0) -
                                       np.maximum(T_v - obs_v, 0.0))))
        else:
            ref = float(np.mean(np.maximum(obs_v - T_v, 0.0)))
            tw1 = float(np.mean(np.abs(np.maximum(fc1_v - T_v, 0.0) -
                                       np.maximum(obs_v - T_v, 0.0))))
            tw2 = float(np.mean(np.abs(np.maximum(fc2_v - T_v, 0.0) -
                                       np.maximum(obs_v - T_v, 0.0))))

        twmae1_vals.append(tw1); twmae2_vals.append(tw2); ref_vals.append(ref)
        ss1.append(1.0 - tw1 / ref if ref > 0 else np.nan)
        ss2.append(1.0 - tw2 / ref if ref > 0 else np.nan)
        T_reprs.append(float(np.mean(T_v)))
        ps_note = "per-station" if s['per_station'] else "pooled"
        print(f"    p{s['pct']:>3} ({ps_note}): "
              f"twMAE_ref={ref:.5f}  twSS1={ss1[-1]:+.3f}  twSS2={ss2[-1]:+.3f}")

    x = np.arange(len(pct_values))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── LEFT: skill scores ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, ss1, "o-", color=c1, lw=2.5, ms=9, label=names[0])
    ax.plot(x, ss2, "s-", color=c2, lw=2.5, ms=9, label=names[1])
    ax.axhline(0, color="black",   ls="--", lw=1.5, label="No skill (= always predict T)")
    ax.axhline(1, color="darkgreen", ls=":", lw=1.2, alpha=0.6, label="Perfect (=1)")
    for xi, v1, v2 in zip(x, ss1, ss2):
        if not (np.isnan(v1) or np.isnan(v2)):
            ax.annotate(f"Δ={v2-v1:+.3f}", xy=(xi, max(v1, v2)),
                        xytext=(xi, max(v1, v2) + 0.015),
                        ha="center", fontsize=8, color="black", weight="bold")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(x)
    ax2.set_xticklabels([f"{T:.2f}" for T in T_reprs], fontsize=8, rotation=30)
    ax2.set_xlabel(f"Mean T ({unit})", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=11)
    ax.set_ylabel("twMAE Skill Score  (1 − twMAE / twMAE_ref)")
    ax.set_title("twMAE Skill Score\nReference = always predict station threshold T\n"
                 "twMAE_ref = mean obs exceedance above T  (obs-based, no model data)",
                 fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── RIGHT: absolute twMAE + reference ─────────────────────────────────
    ax = axes[1]
    ax.plot(x, ref_vals,    "k^--", lw=2.0, ms=8, label="Reference (= mean exceedance)")
    ax.plot(x, twmae1_vals, "o-",  color=c1, lw=2.2, ms=8, label=names[0])
    ax.plot(x, twmae2_vals, "s-",  color=c2, lw=2.2, ms=8, label=names[1])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(x)
    ax2.set_xticklabels([f"{T:.2f}" for T in T_reprs], fontsize=8, rotation=30)
    ax2.set_xlabel(f"Mean T ({unit})", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=11)
    ax.set_ylabel(f"twMAE ({unit})")
    ax.set_title("Absolute twMAE vs Reference\nGap below reference = skill gained", fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    thr_type = "per-station obs clim" if any(s['per_station'] for s in sweep) else "pooled"
    fig.suptitle(
        f"twMAE Skill Score — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}  "
        f"|  T: {thr_type}",
        fontsize=11, weight="bold",
    )
    # ── Auto-interpretation ──────────────────────────────────────────────
    _ss1e, _ss2e = ss1[-1], ss2[-1]   # most extreme level
    _pe16 = pct_values[-1]
    if not (np.isnan(_ss1e) or np.isnan(_ss2e)):
        _sk_better = names[0] if _ss1e >= _ss2e else names[1]
        _sk_worse  = names[1] if _ss1e >= _ss2e else names[0]
        _sk_b, _sk_w = max(_ss1e, _ss2e), min(_ss1e, _ss2e)
        # trend: does skill increase or decrease toward more extreme thresholds?
        _valid_ss1 = [s for s in ss1 if not np.isnan(s)]
        _valid_ss2 = [s for s in ss2 if not np.isnan(s)]
        _tr1 = (_valid_ss1[-1] - _valid_ss1[0]) if len(_valid_ss1) > 1 else 0
        _tr2 = (_valid_ss2[-1] - _valid_ss2[0]) if len(_valid_ss2) > 1 else 0
        _td1 = 'improves' if _tr1 > 0.01 else ('degrades' if _tr1 < -0.01 else 'stays stable')
        _td2 = 'improves' if _tr2 > 0.01 else ('degrades' if _tr2 < -0.01 else 'stays stable')
        _ref_note = "(reference = always predicting T; skill > 0 means better than that)"
        _interp = (
            f"► At p{_pe16}: {_sk_better} has higher skill (twSS={_sk_b:.3f} vs {_sk_w:.3f}) {_ref_note}.\n"
            f"Toward more extreme thresholds: {names[0]}'s skill {_td1};  {names[1]}'s {_td2}.  "
            f"{'Both models improve at more extreme levels — they gain more than climatology as events get rarer.' if _tr1>0 and _tr2>0 else 'At least one model degrades toward the rarest events — it loses skill for the most extreme cases.'}"
        )
    else:
        _interp = "► Insufficient data to compute skill scores."
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(0.5, 0.01, _interp, ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.93))
    _savefig(fig, f"16_twmae_skill_score_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_error_depth_profile(fc1_data, fc2_data, obs_data, threshold, condition):
    """Plot 17: Mean absolute error on hits, binned by exceedance depth (obs − T_s).

    All existing plots treat extreme cases as a single group.  This plot asks:
    does one model outperform specifically on the *most* extreme events, or only
    on the moderate ones that barely exceed the threshold?

    Hit cases (obs > T_s, fc > T_s) are binned by how far the observation
    penetrates the extreme tail: (obs_i − T_i) in physical units.  Within each
    bin the mean absolute error |fc − obs| is computed for both models.

    Reference: Lerch, Thorarinsdottir, Ravazzolo & Gneiting (2017), 'Forecaster's
    Dilemma: Extreme Events and Forecast Evaluation', Statistical Science 32(1):
    106–127.  They show that the depth of penetration into the tail matters more
    than the event count for high-impact weather evaluation.
    """
    print("[17/17] Error Depth Profile (hits binned by exceedance, per-station T)...")

    var_short  = condition["var_short"]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    names  = [condition["expver1"], condition["expver2"]]
    c1 = _style.C_FC1; c2 = _style.C_FC2
    event_type = condition.get("event_type", "above")

    threshold_arr = condition.get("threshold_arr", None)
    use_per_station = (threshold_arr is not None and
                       isinstance(threshold_arr, np.ndarray) and
                       len(threshold_arr) == len(obs_data))
    T_use = threshold_arr if use_per_station else np.full(len(obs_data),
                                                          float(threshold),
                                                          dtype=np.float32)
    valid  = ~np.isnan(T_use)
    fc1_v  = fc1_data[valid]; fc2_v = fc2_data[valid]
    obs_v  = obs_data[valid]; T_v   = T_use[valid]

    if event_type == 'below':
        hit1_m = (obs_v <= T_v) & (fc1_v <= T_v)
        hit2_m = (obs_v <= T_v) & (fc2_v <= T_v)
        exc1 = T_v[hit1_m] - obs_v[hit1_m]   # exceedance = how far below T
        exc2 = T_v[hit2_m] - obs_v[hit2_m]
        err1 = np.abs(fc1_v[hit1_m] - obs_v[hit1_m])
        err2 = np.abs(fc2_v[hit2_m] - obs_v[hit2_m])
        xlabel_exc = f"Obs exceedance below T  (T − obs, {unit})"
    else:
        hit1_m = (obs_v >= T_v) & (fc1_v >= T_v)
        hit2_m = (obs_v >= T_v) & (fc2_v >= T_v)
        exc1 = obs_v[hit1_m] - T_v[hit1_m]
        exc2 = obs_v[hit2_m] - T_v[hit2_m]
        err1 = np.abs(fc1_v[hit1_m] - obs_v[hit1_m])
        err2 = np.abs(fc2_v[hit2_m] - obs_v[hit2_m])
        xlabel_exc = f"Obs exceedance above T  (obs − T, {unit})"

    if len(exc1) < 10 or len(exc2) < 10:
        print("    Too few hit cases — skipping error depth profile")
        return

    # Build bins from the combined exceedance range
    all_exc = np.concatenate([exc1, exc2])
    bin_edges = np.percentile(all_exc, np.linspace(0, 100, 6))   # 5 quantile bins
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(all_exc.min(), all_exc.max(), 6)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    def _bin_mae(exc, err):
        means, counts, stds = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m = (exc >= lo) & (exc < hi)
            means.append(float(np.mean(err[m]))   if m.sum() > 0 else np.nan)
            stds.append( float(np.std(err[m]))    if m.sum() > 0 else np.nan)
            counts.append(int(m.sum()))
        return means, stds, counts

    mae1_bins, std1_bins, cnt1_bins = _bin_mae(exc1, err1)
    mae2_bins, std2_bins, cnt2_bins = _bin_mae(exc2, err2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    xi = np.arange(n_bins)

    # ── LEFT: mean |fc−obs| per exceedance bin ────────────────────────────
    ax = axes[0]
    ax.plot(xi, mae1_bins, "o-", color=c1, lw=2.2, ms=8, label=names[0])
    ax.plot(xi, mae2_bins, "s-", color=c2, lw=2.2, ms=8, label=names[1])
    # shading for ±1 std
    for mae_b, std_b, col in [(mae1_bins, std1_bins, c1), (mae2_bins, std2_bins, c2)]:
        lo = [m - s if not (np.isnan(m) or np.isnan(s)) else np.nan
              for m, s in zip(mae_b, std_b)]
        hi = [m + s if not (np.isnan(m) or np.isnan(s)) else np.nan
              for m, s in zip(mae_b, std_b)]
        ax.fill_between(xi, lo, hi, alpha=0.12, color=col)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{lo:.2f}–{hi:.2f}" for lo, hi in
                        zip(bin_edges[:-1], bin_edges[1:])],
                       rotation=30, fontsize=8)
    ax.set_xlabel(xlabel_exc)
    ax.set_ylabel(f"Mean |fc − obs| on hits ({unit})")
    ax.set_title("Error depth profile (hits only)\n"
                 "Does the model improve on the deepest extremes?", fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── RIGHT: Δ MAE (model2 − model1) per bin + hit counts ──────────────
    ax = axes[1]
    delta_mae = [m2 - m1 if not (np.isnan(m1) or np.isnan(m2)) else np.nan
                 for m1, m2 in zip(mae1_bins, mae2_bins)]
    bar_cols = ["#2e7d32" if (d is not np.nan and not np.isnan(d) and d < 0)
                else "#c62828" for d in delta_mae]
    ax.bar(xi, delta_mae, 0.6, color=bar_cols, alpha=0.85)
    ax.axhline(0, color="grey", ls="--", lw=1.2)
    ax2 = ax.twinx()
    ax2.bar(xi - 0.3, cnt1_bins, 0.28, alpha=0.25, color=c1, label=f"{names[0]} n")
    ax2.bar(xi + 0.0, cnt2_bins, 0.28, alpha=0.25, color=c2, label=f"{names[1]} n")
    ax2.set_ylabel("Hit count per bin")
    ax2.legend(fontsize=8, loc="upper right")
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{lo:.2f}–{hi:.2f}" for lo, hi in
                        zip(bin_edges[:-1], bin_edges[1:])],
                       rotation=30, fontsize=8)
    ax.set_xlabel(xlabel_exc)
    ax.set_ylabel(f"Δ MAE on hits  ({names[1]} − {names[0]}, {unit})\nGreen = {names[1]} better")
    ax.set_title("Δ Error by exceedance depth\n"
                 "Where in the tail does model 1 gain advantage?", fontsize=9)
    ax.grid(True, alpha=0.3)

    T_note = (f"per-station (mean={float(np.nanmean(T_use)):.3f} {unit})"
              if use_per_station else f"{float(threshold):.3f} {unit}")
    fig.suptitle(
        f"Error Depth Profile — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}  "
        f"|  T = {T_note}",
        fontsize=11, weight="bold",
    )
    # ── Auto-interpretation ──────────────────────────────────────────────
    _vd = [(i, d) for i, d in enumerate(delta_mae)
           if d is not None and not np.isnan(d)]
    if _vd:
        _overall_better = names[1] if sum(d for _, d in _vd) <= 0 else names[0]
        _best_i,  _best_d  = min(_vd, key=lambda x: x[1])  # most negative Δ = names[1] best
        _worst_i, _worst_d = max(_vd, key=lambda x: x[1])  # most positive Δ = names[0] best
        _n_bins = len(delta_mae)
        _deep_best = _best_i >= _n_bins // 2   # advantage in deeper half of tail
        _lo17 = bin_edges[_best_i]; _hi17 = bin_edges[_best_i + 1]
        _side = "below" if event_type == 'below' else "above"
        _interp = (
            f"► {_overall_better} is better overall at predicting the intensity of extreme events (hits).\n"
            f"Largest advantage in exceedance bin {_lo17:.2f}–{_hi17:.2f} {unit} {_side} T "
            f"(Δ MAE = {abs(_best_d):.4f} {unit}, {'in the deeper half of the tail' if _deep_best else 'in the shallower half'}).\n"
            f"{'The advantage grows for the most extreme events — ' + _overall_better + ' is especially better at the rarest, most dangerous cases.' if _best_i == _n_bins-1 else 'The advantage is strongest for moderate extremes; both models converge toward the most extreme events.'}"
        )
    else:
        _interp = "► Insufficient hit data to compare errors by exceedance depth."
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(0.5, 0.01, _interp, ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.93))
    _savefig(fig, f"17_error_depth_profile_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")

    print(f"\n  Error depth profile summary:")
    print(f"  {'Bin (exceedance)':>22}  {'n1':>6}  {names[0]+' MAE':>12}  "
          f"{'n2':>6}  {names[1]+' MAE':>12}  {'Δ MAE':>10}")
    print(f"  {'-'*75}")
    for lo, hi, m1, m2, n1, n2 in zip(bin_edges[:-1], bin_edges[1:],
                                        mae1_bins, mae2_bins,
                                        cnt1_bins, cnt2_bins):
        dm = (m2 - m1) if not (np.isnan(m1) or np.isnan(m2)) else float('nan')
        print(f"  {lo:.2f}–{hi:.2f} {unit:>6}  {n1:>6}  {m1:>12.5f}  "
              f"{n2:>6}  {m2:>12.5f}  {dm:>+10.5f}")


def plot_summary_scorecard(fc1_data, fc2_data, obs_data, threshold, condition):
    """Plot 18: Summary scorecard — twMAE components, detection scores, overall errors.

    A single-figure 'newspaper front page' for the diagnostic run.  Displays a
    colour-coded table (green cell = winner) with three sections:

      ① twMAE decomposition  — hits / misses / false alarms / total
      ② Detection scores     — POD, FAR, ETS, PSS
      ③ Overall errors       — bias, MAE

    A narrative paragraph below the table auto-interprets the dominant driver of
    any difference in plain English, mirroring the analysis a scientist would write.
    """
    print("[18/18] Summary Scorecard Table...")

    var_short = condition["var_short"]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    names     = [condition["expver1"], condition["expver2"]]
    event_type = condition.get("event_type", "above")

    # ── Per-station threshold ─────────────────────────────────────────────────
    threshold_arr  = condition.get("threshold_arr", None)
    use_per_station = (threshold_arr is not None and
                       isinstance(threshold_arr, np.ndarray) and
                       len(threshold_arr) == len(obs_data))
    T_use = threshold_arr if use_per_station else np.full(len(obs_data),
                                                          float(threshold),
                                                          dtype=np.float32)
    valid = ~np.isnan(T_use)
    fc1_v = fc1_data[valid]; fc2_v = fc2_data[valid]
    obs_v = obs_data[valid]; T_v   = T_use[valid]
    N = len(obs_v)

    # ── twMAE decomposition ───────────────────────────────────────────────────
    def _decomp(fc):
        if event_type == 'below':
            hit_m  = (obs_v <= T_v) & (fc <= T_v)
            miss_m = (obs_v <= T_v) & (fc >  T_v)
            fa_m   = (obs_v >  T_v) & (fc <= T_v)
            hc = float(np.mean(np.abs(fc[hit_m]  - obs_v[hit_m])))  if hit_m.sum()  > 0 else 0.
            mc = float(np.mean(T_v[miss_m] - obs_v[miss_m]))        if miss_m.sum() > 0 else 0.
            fc_ = float(np.mean(obs_v[fa_m]   - T_v[fa_m]))         if fa_m.sum()   > 0 else 0.
        else:
            hit_m  = (obs_v >= T_v) & (fc >= T_v)
            miss_m = (obs_v >= T_v) & (fc <  T_v)
            fa_m   = (obs_v <  T_v) & (fc >= T_v)
            hc = float(np.mean(np.abs(fc[hit_m]  - obs_v[hit_m])))  if hit_m.sum()  > 0 else 0.
            mc = float(np.mean(obs_v[miss_m] - T_v[miss_m]))        if miss_m.sum() > 0 else 0.
            fc_ = float(np.mean(fc[fa_m]   - T_v[fa_m]))            if fa_m.sum()   > 0 else 0.
        h_contrib  = hc  * hit_m.sum()  / N
        m_contrib  = mc  * miss_m.sum() / N
        fa_contrib = fc_ * fa_m.sum()   / N
        return {"hits": h_contrib, "misses": m_contrib, "fa": fa_contrib,
                "total": h_contrib + m_contrib + fa_contrib,
                "n_hit": int(hit_m.sum()), "n_miss": int(miss_m.sum()),
                "n_fa":  int(fa_m.sum())}

    d1 = _decomp(fc1_v); d2 = _decomp(fc2_v)

    # ── Detection scores ──────────────────────────────────────────────────────
    pct = (condition.get("threshold_percentile")
           if condition.get("threshold_mode") == "percentile" else None)
    obs_e = is_extreme_event(obs_data, threshold, var_short, pct)
    fc1_e = is_extreme_event(fc1_data, threshold, var_short, pct)
    fc2_e = is_extreme_event(fc2_data, threshold, var_short, pct)
    s1 = calculate_skill_scores(fc1_e, obs_e)
    s2 = calculate_skill_scores(fc2_e, obs_e)

    # ── Overall errors ────────────────────────────────────────────────────────
    bias1 = float(np.nanmean(fc1_data - obs_data))
    bias2 = float(np.nanmean(fc2_data - obs_data))
    mae1  = float(np.nanmean(np.abs(fc1_data - obs_data)))
    mae2  = float(np.nanmean(np.abs(fc2_data - obs_data)))

    # ── Helper: winner + % change ─────────────────────────────────────────────
    def _compare(v1, v2, lower_is_better=True):
        """Return (winner_name, delta_pct_str, abs_diff_str)."""
        if lower_is_better:
            winner = names[0] if v1 < v2 else (names[1] if v2 < v1 else "Tie")
        else:
            winner = names[0] if v1 > v2 else (names[1] if v2 > v1 else "Tie")
        denom = max(abs(v1), abs(v2), 1e-12)
        dp = 100 * (v2 - v1) / denom
        return winner, f"{dp:+.1f}%"

    def _bias_compare(b1, b2):
        winner = names[0] if abs(b1) < abs(b2) else (names[1] if abs(b2) < abs(b1) else "Tie")
        dp = 100 * (abs(b2) - abs(b1)) / (max(abs(b1), abs(b2)) + 1e-12)
        return winner, f"{dp:+.1f}%"

    # ── Build table rows ──────────────────────────────────────────────────────
    # Each row: (label, val1_str, val2_str, delta_str, winner_str, row_type)
    # row_type: 'section' | 'total' | 'normal'
    SECTION = 'section'; TOTAL = 'total'; NORMAL = 'normal'

    w_hits,  dp_hits  = _compare(d1["hits"],   d2["hits"],   lower_is_better=True)
    w_miss,  dp_miss  = _compare(d1["misses"], d2["misses"], lower_is_better=True)
    w_fa,    dp_fa    = _compare(d1["fa"],     d2["fa"],     lower_is_better=True)
    w_tot,   dp_tot   = _compare(d1["total"],  d2["total"],  lower_is_better=True)
    w_pod,   dp_pod   = _compare(s1["pod"],    s2["pod"],    lower_is_better=False)
    w_far,   dp_far   = _compare(s1["far"],    s2["far"],    lower_is_better=True)
    w_ets,   dp_ets   = _compare(s1["ets"],    s2["ets"],    lower_is_better=False)
    w_pss,   dp_pss   = _compare(s1["pss"],    s2["pss"],    lower_is_better=False)
    w_bias,  dp_bias  = _bias_compare(bias1, bias2)
    w_mae,   dp_mae   = _compare(mae1, mae2, lower_is_better=True)

    # Bootstrap significance of the difference (from the pipeline CSV, if present).
    # Appended to the Δ column so the winner text stays clean for cell colouring.
    sig = (_significance_from_csv(condition["_config"], condition)
           if condition.get("_config") is not None else {})

    def _sig(dp, key):
        if key in sig:
            return f"{dp}  {'✓' if sig[key] else 'n.s.'}"
        return dp

    dp_tot  = _sig(dp_tot,  "twmae")
    dp_pod  = _sig(dp_pod,  "pod")
    dp_far  = _sig(dp_far,  "far")
    dp_ets  = _sig(dp_ets,  "ets")
    dp_pss  = _sig(dp_pss,  "pss")
    dp_bias = _sig(dp_bias, "bias")
    dp_mae  = _sig(dp_mae,  "mae")

    def f5(v): return f"{v:.5f}"
    def f3(v): return f"{v:.3f}"

    rows = [
        ("── twMAE decomposition ──────────────────", "", "", "", "",           SECTION),
        ("  Hits  (intensity error on caught events)",f5(d1["hits"]),  f5(d2["hits"]),  dp_hits, w_hits,  NORMAL),
        ("  Misses  (undetected extreme events)",     f5(d1["misses"]),f5(d2["misses"]),dp_miss, w_miss,  NORMAL),
        ("  False Alarms  (spurious extremes pred.)", f5(d1["fa"]),    f5(d2["fa"]),    dp_fa,   w_fa,    NORMAL),
        ("  TOTAL twMAE",                             f5(d1["total"]), f5(d2["total"]), dp_tot,  w_tot,   TOTAL),
        ("── Detection scores ─────────────────────", "", "", "", "",           SECTION),
        ("  POD  (probability of detection)",         f3(s1["pod"]),   f3(s2["pod"]),   dp_pod,  w_pod,   NORMAL),
        ("  FAR  (false alarm ratio)",                f3(s1["far"]),   f3(s2["far"]),   dp_far,  w_far,   NORMAL),
        ("  ETS  (equitable threat score)",           f3(s1["ets"]),   f3(s2["ets"]),   dp_ets,  w_ets,   NORMAL),
        ("  PSS  (Peirce skill score)",               f3(s1["pss"]),   f3(s2["pss"]),   dp_pss,  w_pss,   NORMAL),
        ("── Overall errors (all observations) ────", "", "", "", "",           SECTION),
        ("  Bias  (mean fc − obs)",                   f3(bias1),       f3(bias2),       dp_bias, w_bias,  NORMAL),
        ("  MAE   (mean absolute error)",             f3(mae1),        f3(mae2),        dp_mae,  w_mae,   NORMAL),
    ]
    col_headers = ["Component / Metric",
                   names[0], names[1],
                   f"\u0394 (m2\u2212m1)  \u00b7  \u2713/n.s.", "Winner"]

    # ── Cell colours matrix (one row per data row, one per column) ────────────
    CLR_SECTION = "#D5D8DC"   # grey section header
    CLR_TOTAL   = "#D6EAF8"   # light blue total row
    CLR_WIN     = "#D5F5E3"   # light green — winner value cell
    CLR_LOSE    = "#FDEDEC"   # light red   — loser value cell
    CLR_DEFAULT = "#FDFEFE"

    def _row_colours(label, val1, val2, delta, winner, rtype):
        if rtype == SECTION:
            return [CLR_SECTION] * 5
        base = CLR_TOTAL if rtype == TOTAL else CLR_DEFAULT
        c1_cell = CLR_WIN  if winner == names[0] else (CLR_LOSE if winner == names[1] else base)
        c2_cell = CLR_WIN  if winner == names[1] else (CLR_LOSE if winner == names[0] else base)
        w_cell  = "#A9DFBF" if winner == names[0] else ("#A9DFBF" if winner == names[1] else base)
        return [base, c1_cell, c2_cell, base, w_cell]

    cell_text   = [list(r[:5]) for r in rows]
    cell_colour = [_row_colours(*r) for r in rows]

    # ── Draw figure ───────────────────────────────────────────────────────────
    # Height trimmed (was 10) so the table + narrative fill the figure instead of
    # leaving ~45% empty vertical space.
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_axis_off()

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellColours=cell_colour,
        loc='upper center',
        bbox=[0.0, 0.20, 1.0, 0.76],   # [left, bottom, width, height] in axes coords
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    # Style header row and individual cells
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if row_idx == 0:   # header
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color='white', weight='bold', fontsize=10)
        else:
            data_row = rows[row_idx - 1]
            rtype = data_row[5]
            if rtype == SECTION:
                cell.set_text_props(weight='bold', color='#2C3E50', fontsize=9)
            elif rtype == TOTAL:
                cell.set_text_props(weight='bold', fontsize=10)
            # Left-align the label column, centre the rest
            if col_idx == 0:
                cell.set_text_props(ha='left')

    # Set column widths
    col_widths = [0.40, 0.14, 0.14, 0.13, 0.19]
    for col_idx, cw in enumerate(col_widths):
        tbl.auto_set_column_width(col_idx)
        for row_idx in range(len(rows) + 1):
            if (row_idx, col_idx) in tbl.get_celld():
                tbl.get_celld()[(row_idx, col_idx)].set_width(cw)

    # ── Auto-generated narrative ──────────────────────────────────────────────
    overall_winner = names[0] if d1["total"] < d2["total"] else names[1]
    overall_loser  = names[1] if overall_winner == names[0] else names[0]
    total_imp = 100 * abs(d2["total"] - d1["total"]) / (max(d1["total"], d2["total"]) + 1e-12)

    # Dominant twMAE component (where is the biggest absolute difference?)
    deltas = {"Hits": abs(d2["hits"] - d1["hits"]),
              "Misses": abs(d2["misses"] - d1["misses"]),
              "False Alarms": abs(d2["fa"] - d1["fa"])}
    dom_comp = max(deltas, key=deltas.get)
    dom_imp  = 100 * deltas[dom_comp] / (max(d1[dom_comp.lower().replace(" ", "_").replace("_alarms","").replace("false","fa")], d2[dom_comp.lower().replace(" ","_").replace("_alarms","").replace("false","fa")]) + 1e-12)
    # simpler key lookup:
    _key = {"Hits": "hits", "Misses": "misses", "False Alarms": "fa"}[dom_comp]
    dom_imp = 100 * deltas[dom_comp] / (max(d1[_key], d2[_key]) + 1e-12)
    dom_winner = names[0] if d1[_key] < d2[_key] else names[1]

    pod_winner  = names[0] if s1["pod"] > s2["pod"] else names[1]
    far_winner  = names[0] if s1["far"] < s2["far"] else names[1]
    pod_hi = max(s1["pod"], s2["pod"]); pod_lo = min(s1["pod"], s2["pod"])
    far_lo = min(s1["far"], s2["far"]); far_hi = max(s1["far"], s2["far"])

    miss_similar = abs(d1["misses"] - d2["misses"]) < 0.001
    miss_winner  = names[0] if d1["misses"] < d2["misses"] else names[1]

    side = "below" if event_type == "below" else "above"
    event_label = "cold" if event_type == "below" else "warm"

    narrative_parts = [
        f"{overall_winner} wins total twMAE by {total_imp:.0f}%.",
        f"Dominant driver: {dom_comp} component — {dom_winner} reduces this by {dom_imp:.0f}%.",
        (f"{pod_winner} detects more {event_label} events (POD {pod_hi:.3f} vs {pod_lo:.3f}), "
         f"but {far_winner} is more precise (FAR {far_lo:.3f} vs {far_hi:.3f})."),
    ]
    if pod_winner != far_winner:
        narrative_parts.append(
            f"A high POD/high FAR model shouts 'extreme!' more often — catching more real events "
            f"but also generating many spurious ones, which inflates the False Alarm component of twMAE."
        )
    narrative_parts.append(
        f"Misses are {'similar for both models' if miss_similar else f'smaller for {miss_winner}'}. "
        f"Overall MAE {'favours ' + (names[0] if mae1 < mae2 else names[1]) if mae1 != mae2 else 'is equal'}; "
        f"bias is closer to zero for {w_bias} ({f3(bias1 if w_bias==names[0] else bias2)} {unit})."
    )

    narrative = "  ".join(narrative_parts)
    wrapped   = textwrap.fill(narrative, width=130)

    T_note = (f"per-station (mean={float(np.nanmean(T_use)):.3f} {unit})"
              if use_per_station else f"{float(threshold):.3f} {unit}")
    fig.suptitle(
        f"Summary Scorecard — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}  "
        f"|  T = {T_note}",
        fontsize=12, weight="bold", y=0.99,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.97])
    fig.text(0.5, 0.01, wrapped, ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.95))

    _savefig(fig, f"18_summary_scorecard_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_twmae_component_fractions(fc1_data, fc2_data, obs_data, threshold, condition):
    """Plot 12: twMAE component fractions (100 % stacked) + absolute side-by-side.

    Answers immediately: which failure mode (misses / false alarms / residual hit
    error) dominates for each model, and which mode explains the difference?

    Inspired by the Murphy (1987) MSE-decomposition idea applied to the Taggart
    (2022) twMAE structure:
      twMAE = miss_contrib + FA_contrib + hit_contrib
    Normalised to 100 % these fractions reveal whether one model is better because
    it misses fewer events, triggers fewer false alarms, or predicts the intensity
    of caught events more accurately — the three fundamentally different pathways
    to a lower twMAE.
    """
    print("[12/14] twMAE Component Fractions...")

    var_short  = condition["var_short"]
    event_type = condition.get("event_type", "above")
    names      = [condition["expver1"], condition["expver2"]]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    c1 = _style.C_FC1; c2 = _style.C_FC2
    c_hit = _style.C_HIT; c_miss = _style.C_MISS; c_fa = _style.C_FA

    threshold_arr = condition.get("threshold_arr", None)
    use_per_station = (threshold_arr is not None and
                       isinstance(threshold_arr, np.ndarray) and
                       len(threshold_arr) == len(obs_data))

    def _decompose_simple(fc, obs, T_in):
        if isinstance(T_in, np.ndarray):
            valid = ~np.isnan(T_in)
            fc_v, obs_v, T_v = fc[valid], obs[valid], T_in[valid]
        else:
            fc_v, obs_v, T_v = fc, obs, T_in
        N = len(obs_v)
        if N == 0:
            return dict(hit=0., miss=0., fa=0., total=0.)
        scalar_T = not isinstance(T_v, np.ndarray)
        if event_type == "below":
            hit_m  = (obs_v <= T_v) & (fc_v <= T_v)
            miss_m = (obs_v <= T_v) & (fc_v >  T_v)
            fa_m   = (obs_v >  T_v) & (fc_v <= T_v)
            T_miss = T_v if scalar_T else T_v[miss_m]
            T_fa   = T_v if scalar_T else T_v[fa_m]
            h = float(np.mean(np.abs(fc_v[hit_m] - obs_v[hit_m]))) * hit_m.sum() / N  if hit_m.sum() > 0 else 0.
            m = float(np.mean(T_miss - obs_v[miss_m]))              * miss_m.sum() / N if miss_m.sum() > 0 else 0.
            f = float(np.mean(obs_v[fa_m] - T_fa))                  * fa_m.sum()  / N  if fa_m.sum()  > 0 else 0.
        else:
            hit_m  = (obs_v >= T_v) & (fc_v >= T_v)
            miss_m = (obs_v >= T_v) & (fc_v <  T_v)
            fa_m   = (obs_v <  T_v) & (fc_v >= T_v)
            T_miss = T_v if scalar_T else T_v[miss_m]
            T_fa   = T_v if scalar_T else T_v[fa_m]
            h = float(np.mean(np.abs(fc_v[hit_m] - obs_v[hit_m]))) * hit_m.sum() / N  if hit_m.sum() > 0 else 0.
            m = float(np.mean(obs_v[miss_m] - T_miss))              * miss_m.sum() / N if miss_m.sum() > 0 else 0.
            f = float(np.mean(fc_v[fa_m]   - T_fa))                 * fa_m.sum()  / N  if fa_m.sum()  > 0 else 0.
        total = h + m + f
        return dict(hit=h, miss=m, fa=f, total=total)

    T_use = threshold_arr if use_per_station else threshold
    r1 = _decompose_simple(fc1_data, obs_data, T_use)
    r2 = _decompose_simple(fc2_data, obs_data, T_use)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── LEFT: 100 % stacked normalised fractions ──────────────────────────
    ax = axes[0]
    for xi, r, edge_col, name in [(0, r1, c1, names[0]), (1, r2, c2, names[1])]:
        tot = r["total"] if r["total"] > 0 else 1.0
        fracs = [r["miss"] / tot, r["fa"] / tot, r["hit"] / tot]
        labels = ["Misses", "False Alarms", "Hits (residual error)"]
        colours = [c_miss, c_fa, c_hit]
        hatches = ["", "", "///"]
        bottom = 0.0
        for frac, col, hatch, lbl in zip(fracs, colours, hatches, labels):
            bar = ax.bar(xi, frac * 100, 0.55, bottom=bottom * 100,
                         color=col, edgecolor=edge_col, linewidth=2,
                         hatch=hatch, alpha=0.85, label=lbl if xi == 0 else "_")
            if frac > 0.04:
                ax.text(xi, (bottom + frac / 2) * 100,
                        f"{frac*100:.1f}%", ha="center", va="center",
                        fontsize=11, weight="bold", color="white")
            bottom += frac
        ax.text(xi, 102, f"twMAE={r['total']:.4f}", ha="center",
                fontsize=9, color=edge_col, weight="bold")

    ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 110); ax.set_ylabel("% of total twMAE")
    ax.set_title("Component fractions of twMAE\n(which failure mode dominates?)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=c_miss, label="Misses  — obs extreme, fc not"),
        Patch(facecolor=c_fa,   label="False Alarms  — fc extreme, obs not"),
        Patch(facecolor=c_hit, hatch="///", label="Hits  — both extreme, residual |fc−obs|"),
    ], fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2, axis="y")

    # ── RIGHT: absolute side-by-side grouped bars ─────────────────────────
    ax = axes[1]
    x = np.array([0, 1, 2])
    w = 0.35
    abs_labels = ["Misses", "False Alarms", "Hits"]
    for offset, r, edge_col, name in [(-w/2, r1, c1, names[0]),
                                       ( w/2, r2, c2, names[1])]:
        vals = [r["miss"], r["fa"], r["hit"]]
        cols = [c_miss, c_fa, c_hit]
        for xi, v, col in zip(x, vals, cols):
            ax.bar(xi + offset, v, w, color=col, edgecolor=edge_col,
                   linewidth=2, alpha=0.85)
            ax.text(xi + offset, v + 0.0002, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8.5,
                    color=edge_col, weight="bold")
    from matplotlib.patches import Patch
    ax.set_xticks(x); ax.set_xticklabels(abs_labels, fontsize=11)
    ax.set_ylabel(f"Absolute twMAE contribution ({unit})")
    ax.set_title("Absolute twMAE components\n(where is each model losing?)", fontsize=10)
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor=c1, linewidth=2, label=names[0]),
        Patch(facecolor="white", edgecolor=c2, linewidth=2, label=names[1]),
    ], fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Δ arrows between models on each bar
    for xi, v1, v2 in zip(x, [r1["miss"], r1["fa"], r1["hit"]],
                              [r2["miss"], r2["fa"], r2["hit"]]):
        delta = v2 - v1
        col = "#2e7d32" if delta < 0 else "#c62828"
        ymax = max(v1, v2)
        ax.annotate(f"Δ={delta:+.4f}", xy=(xi, ymax + 0.0004),
                    ha="center", fontsize=8, color=col, weight="bold")

    T_note = f"per-station (mean={float(np.nanmean(threshold_arr)):.3f} {unit})" if use_per_station else f"{threshold:.3f} {unit}"
    fig.suptitle(
        f"twMAE Component Fractions — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}  "
        f"|  T = {T_note}",
        fontsize=11, weight="bold",
    )
    # ── Auto-interpretation ──────────────────────────────────────────────
    _t1, _t2 = r1['total'], r2['total']
    if _t1 > 0 or _t2 > 0:
        _better, _worse = (names[0], names[1]) if _t1 <= _t2 else (names[1], names[0])
        _rb, _rw = (r1, r2) if _t1 <= _t2 else (r2, r1)
        _tb, _tw = min(_t1, _t2), max(_t1, _t2)
        _pct_improv = (_tw - _tb) / _tw * 100 if _tw > 0 else 0.
        _sign = 1 if _t1 <= _t2 else -1
        _d = {'misses':       _sign * (r2['miss'] - r1['miss']),
              'false alarms': _sign * (r2['fa']   - r1['fa']),
              'hit residuals': _sign * (r2['hit']  - r1['hit'])}
        _dom = max(_d, key=lambda k: _d[k])
        _interp = (
            f"► {_better} has lower total twMAE: {_tb:.4f} vs {_tw:.4f} {unit}  "
            f"({_pct_improv:.1f}% improvement).\n"
            f"Main reason: {_better} has fewer {_dom}  "
            f"({_d[_dom]:.4f} {unit} advantage = "
            f"{_d[_dom]/_tw*100:.0f}% of {_worse}'s total twMAE).\n"
            f"Error breakdown — {_better}: {_rb['miss']/_tb*100:.0f}% misses / "
            f"{_rb['fa']/_tb*100:.0f}% false alarms / {_rb['hit']/_tb*100:.0f}% hit residuals  |  "
            f"{_worse}: {_rw['miss']/_tw*100:.0f}% / {_rw['fa']/_tw*100:.0f}% / {_rw['hit']/_tw*100:.0f}%."
        )
    else:
        _interp = "► Both models show zero twMAE — no extreme events in this stratum."
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(0.5, 0.01, _interp, ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.93))
    _savefig(fig, f"12_twmae_fractions_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")

    print(f"\n  twMAE component fractions:")
    print(f"  {'Component':<16} {names[0]+' abs':>12} {names[0]+' %':>8}  "
          f"{names[1]+' abs':>12} {names[1]+' %':>8}  {'Δ abs':>10}")
    print(f"  {'-'*72}")
    for comp, v1, v2 in [("Misses",  r1["miss"], r2["miss"]),
                          ("False Alarms", r1["fa"], r2["fa"]),
                          ("Hits", r1["hit"], r2["hit"]),
                          ("TOTAL", r1["total"], r2["total"])]:
        t1 = r1["total"] if r1["total"] > 0 else 1
        t2 = r2["total"] if r2["total"] > 0 else 1
        print(f"  {comp:<16} {v1:>12.5f} {v1/t1*100:>7.1f}%  "
              f"{v2:>12.5f} {v2/t2*100:>7.1f}%  {v2-v1:>+10.5f}")


def plot_extreme_intensity_scatter(fc1_data, fc2_data, obs_data, threshold, condition):
    """Plot 13: Conditional intensity scatter — fc vs obs for hits only.

    When both forecast and observation exceed the threshold (a 'hit'), the
    relevant question is: how accurately does the model predict the *magnitude*
    of the extreme event?  This plot shows that relationship as a scatter for
    each model separately.

    The perfect-forecast diagonal (y = x), a least-squares regression line,
    and the conditional mean bias E[fc − obs | hit] are shown.  A model with a
    smaller mean bias and a regression slope closer to 1 predicts extreme-event
    intensity more accurately, which directly reduces the hit contribution to
    twMAE even if the frequency of hits is identical to the other model.

    Theoretical motivation:
      Murphy (1987) showed that MSE decomposes into a squared-bias term, a
      conditional-bias term (regression slope ≠ 1), and a correlation term.
      The same logic applies here restricted to the extreme tail: a model can
      outperform on twMAE-hits because it is less biased, better correlated,
      or both, even when ETS/POD are identical.
    """
    print("[13/14] Extreme Intensity Scatter (hits only)...")

    var_short  = condition["var_short"]
    event_type = condition.get("event_type", "above")
    names      = [condition["expver1"], condition["expver2"]]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    c1 = _style.C_FC1; c2 = _style.C_FC2

    threshold_arr = condition.get("threshold_arr", None)
    use_per_station = (threshold_arr is not None and
                       isinstance(threshold_arr, np.ndarray) and
                       len(threshold_arr) == len(obs_data))
    T_use = threshold_arr if use_per_station else np.full(len(obs_data), threshold)

    valid = ~np.isnan(T_use)
    fc1_v = fc1_data[valid]; fc2_v = fc2_data[valid]
    obs_v = obs_data[valid];  T_v   = T_use[valid]

    if event_type == "below":
        hit1_m = (obs_v <= T_v) & (fc1_v <= T_v)
        hit2_m = (obs_v <= T_v) & (fc2_v <= T_v)
    else:
        hit1_m = (obs_v >= T_v) & (fc1_v >= T_v)
        hit2_m = (obs_v >= T_v) & (fc2_v >= T_v)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    T_scalar = float(np.nanmean(T_use))
    _scatter_metrics = {}

    for ax, fc_v, hit_m, col, name in [
        (axes[0], fc1_v, hit1_m, c1, names[0]),
        (axes[1], fc2_v, hit2_m, c2, names[1]),
    ]:
        obs_hits = obs_v[hit_m]
        fc_hits  = fc_v[hit_m]
        n_hits   = hit_m.sum()

        if n_hits < 5:
            ax.text(0.5, 0.5, f"Too few hits (n={n_hits})", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
            ax.set_title(name)
            continue

        # scatter (subsample for speed if very large)
        idx = np.random.choice(n_hits, min(n_hits, 3000), replace=False)
        ax.scatter(obs_hits[idx], fc_hits[idx], alpha=0.35, s=12, color=col,
                   label=f"Hits (n={n_hits})")

        # perfect diagonal
        vmin = min(obs_hits.min(), fc_hits.min())
        vmax = max(obs_hits.max(), fc_hits.max())
        pad  = (vmax - vmin) * 0.05
        diag = np.array([vmin - pad, vmax + pad])
        ax.plot(diag, diag, "k--", lw=1.5, alpha=0.7, label="Perfect (fc=obs)")

        # regression line
        coef = np.polyfit(obs_hits, fc_hits, 1)
        ax.plot(diag, np.polyval(coef, diag), color=col, lw=2,
                label=f"Regression  slope={coef[0]:.2f}")

        # threshold marker
        ax.axvline(T_scalar, color="grey", ls=":", lw=1.3, alpha=0.7, label=f"T={T_scalar:.2f}")
        ax.axhline(T_scalar, color="grey", ls=":", lw=1.3, alpha=0.7)

        # conditional mean bias annotation
        cond_bias = float(np.mean(fc_hits - obs_hits))
        cond_mae  = float(np.mean(np.abs(fc_hits - obs_hits)))
        _scatter_metrics[name] = {'bias': cond_bias, 'mae': cond_mae,
                                  'slope': coef[0], 'n': n_hits}
        ax.text(0.04, 0.96,
                f"Mean bias (fc−obs): {cond_bias:+.3f} {unit}\n"
                f"Mean |fc−obs|: {cond_mae:.3f} {unit}\n"
                f"Regression slope: {coef[0]:.3f}  intercept: {coef[1]:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Observed ({unit})")
        ax.set_ylabel(f"Forecast ({unit})")
        ax.set_title(f"{name}\nIntensity error on hits (both fc & obs extreme)",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    T_note = (f"per-station (mean={T_scalar:.3f} {unit})"
              if use_per_station else f"{threshold:.3f} {unit}")
    fig.suptitle(
        f"Extreme Intensity Scatter (hits only) — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}  "
        f"|  T = {T_note}",
        fontsize=11, weight="bold",
    )
    # ── Auto-interpretation ──────────────────────────────────────────────
    if len(_scatter_metrics) == 2 and all(n in _scatter_metrics for n in names):
        _m1, _m2 = _scatter_metrics[names[0]], _scatter_metrics[names[1]]
        _mae_better  = names[0] if _m1['mae']  <= _m2['mae']  else names[1]
        _bias_better = names[0] if abs(_m1['bias']) <= abs(_m2['bias']) else names[1]
        _slope_better = names[0] if abs(_m1['slope']-1) <= abs(_m2['slope']-1) else names[1]
        _bd1 = 'over-predicts' if _m1['bias'] > 0 else 'under-predicts'
        _bd2 = 'over-predicts' if _m2['bias'] > 0 else 'under-predicts'
        _mae_b = _scatter_metrics[_mae_better]['mae']
        _mae_w = _scatter_metrics[names[1] if _mae_better == names[0] else names[0]]['mae']
        _interp = (
            f"► For events caught by both models (hits): {_mae_better} predicts extreme "
            f"{var_lbl} intensity more accurately (MAE: {_mae_b:.3f} vs {_mae_w:.3f} {unit}).\n"
            f"{names[0]} {_bd1} extreme events by {abs(_m1['bias']):.3f} {unit} on average;  "
            f"{names[1]} {_bd2} by {abs(_m2['bias']):.3f} {unit}.\n"
            f"{_slope_better} has regression slope closer to 1.0 "
            f"(slope {_scatter_metrics[_slope_better]['slope']:.3f} vs "
            f"{_scatter_metrics[names[1] if _slope_better==names[0] else names[0]]['slope']:.3f}), "
            f"meaning it scales better with event severity."
        )
    else:
        _interp = "► Too few hit cases in one or both models to compare intensity errors."
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(0.5, 0.01, _interp, ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.93))
    _savefig(fig, f"13_extreme_intensity_scatter_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_miss_fa_severity(fc1_data, fc2_data, obs_data, condition):
    """Plot 14: Miss severity and FA severity across threshold sweep.

    Even when two models have the same number of misses and false alarms, the
    *severity* of those failures can differ greatly.  This plot measures:

      Miss severity  = mean(obs − T  | obs > T, fc < T)
        How far into the extreme tail were the events the model failed to predict?
        A high value means the model systematically misses the most dangerous events.

      FA severity    = mean(fc  − T  | fc > T, obs < T)
        How far above the threshold was the forecast on false-alarm cases?
        A high value means false alarms are not merely marginal but aggressively
        over-predicted.

    Both are computed at each threshold level in a sweep (same range as plots 3/5/11).
    If one model has lower miss severity at high thresholds, it is better at
    capturing the most dangerous events even if its overall miss rate is similar.

    Theoretical motivation:
      Lerch et al. (2017, Statistical Science) argue that the *nature* of failures
      matters more than their count for high-impact weather.  A model with many
      marginal misses may still outperform a model with fewer but catastrophic ones.
    """
    print("[14/14] Miss / FA Severity across threshold sweep...")

    var_short  = condition["var_short"]
    event_type = condition.get("event_type", "above")
    names      = [condition["expver1"], condition["expver2"]]
    var_lbl, unit = VARIABLE_LABELS.get(var_short, (var_short, ""))
    c1 = _style.C_FC1; c2 = _style.C_FC2

    mode = condition.get("threshold_mode", "percentile")
    thresholds, _, xlabel, main_label, percentiles = \
        get_threshold_range_and_labels(condition, obs_data)
    main_ref = (condition.get("threshold_percentile", 95) if mode == "percentile"
                else condition.get("threshold_value", 0.0))
    x_vals = percentiles if (percentiles is not None) else thresholds

    rows = []
    for i, T_sc in enumerate(thresholds):
        if event_type == "below":
            miss1_m = (obs_data <= T_sc) & (fc1_data >  T_sc)
            fa1_m   = (obs_data >  T_sc) & (fc1_data <= T_sc)
            miss2_m = (obs_data <= T_sc) & (fc2_data >  T_sc)
            fa2_m   = (obs_data >  T_sc) & (fc2_data <= T_sc)
            ms1 = float(np.mean(T_sc - obs_data[miss1_m])) if miss1_m.sum() > 0 else np.nan
            fa1 = float(np.mean(obs_data[fa1_m] - T_sc))  if fa1_m.sum()  > 0 else np.nan
            ms2 = float(np.mean(T_sc - obs_data[miss2_m])) if miss2_m.sum() > 0 else np.nan
            fa2 = float(np.mean(obs_data[fa2_m] - T_sc))  if fa2_m.sum()  > 0 else np.nan
        else:
            miss1_m = (obs_data >= T_sc) & (fc1_data <  T_sc)
            fa1_m   = (obs_data <  T_sc) & (fc1_data >= T_sc)
            miss2_m = (obs_data >= T_sc) & (fc2_data <  T_sc)
            fa2_m   = (obs_data <  T_sc) & (fc2_data >= T_sc)
            ms1 = float(np.mean(obs_data[miss1_m] - T_sc)) if miss1_m.sum() > 0 else np.nan
            fa1 = float(np.mean(fc1_data[fa1_m]   - T_sc)) if fa1_m.sum()  > 0 else np.nan
            ms2 = float(np.mean(obs_data[miss2_m] - T_sc)) if miss2_m.sum() > 0 else np.nan
            fa2 = float(np.mean(fc2_data[fa2_m]   - T_sc)) if fa2_m.sum()  > 0 else np.nan
        xi = x_vals[i]
        rows.append({"x": xi, "Model": names[0], "type": "Miss severity",  "val": ms1})
        rows.append({"x": xi, "Model": names[0], "type": "FA severity",    "val": fa1})
        rows.append({"x": xi, "Model": names[1], "type": "Miss severity",  "val": ms2})
        rows.append({"x": xi, "Model": names[1], "type": "FA severity",    "val": fa2})

    df_sev = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, sev_type, ylabel, title in [
        (axes[0], "Miss severity",
         f"Mean depth of missed event above T ({unit})",
         "Miss Severity — how extreme were the missed events?\n"
         "(higher = model missed the most dangerous cases)"),
        (axes[1], "FA severity",
         f"Mean height of false alarm above T ({unit})",
         "False Alarm Severity — how aggressive were the FAs?\n"
         "(higher = model over-predicted intensity on wrong calls)"),
    ]:
        for name, col, marker in [(names[0], c1, "o"), (names[1], c2, "s")]:
            sub = df_sev[(df_sev["Model"] == name) & (df_sev["type"] == sev_type)]
            ax.plot(sub["x"], sub["val"], f"{marker}-", color=col, lw=2.2,
                    ms=7, label=name)
        ax.axvline(main_ref, color=_style.C_THRESHOLD, ls="--", alpha=0.7, label=main_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Miss & FA Severity — {names[0]}  vs  {names[1]}\n"
        f"{var_lbl}  |  Day {condition['forecast_day']}  "
        f"|  {condition.get('season','all')}  |  {condition.get('terrain','all')}",
        fontsize=11, weight="bold",
    )
    # ── Auto-interpretation ──────────────────────────────────────────────
    _main14 = df_sev[np.isclose(df_sev["x"], main_ref, atol=1.5)]
    def _sv(mdl, stype):
        _r = _main14[(_main14["Model"] == mdl) & (_main14["type"] == stype)]["val"]
        return float(_r.iloc[0]) if len(_r) > 0 and not np.isnan(float(_r.iloc[0])) else None
    _ms1, _ms2 = _sv(names[0], "Miss severity"), _sv(names[1], "Miss severity")
    _fa1, _fa2 = _sv(names[0], "FA severity"),   _sv(names[1], "FA severity")
    _parts14 = []
    if _ms1 is not None and _ms2 is not None:
        _msb = names[0] if _ms1 <= _ms2 else names[1]
        _parts14.append(
            f"Miss severity: {_msb} misses shallower events "
            f"({min(_ms1,_ms2):.3f} vs {max(_ms1,_ms2):.3f} {unit} above T) — "
            f"its missed events are on average less dangerous."
        )
    if _fa1 is not None and _fa2 is not None:
        _fab = names[0] if _fa1 <= _fa2 else names[1]
        _parts14.append(
            f"FA severity: {_fab}'s false alarms are less aggressive "
            f"({min(_fa1,_fa2):.3f} vs {max(_fa1,_fa2):.3f} {unit} above T) — "
            f"its wrong predictions overshoot the threshold by less."
        )
    _interp = ("► " + "\n".join(_parts14)) if _parts14 else "► Insufficient data at main threshold."
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(0.5, 0.01, _interp, ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='darkorange', alpha=0.93))
    _savefig(fig, f"14_miss_fa_severity_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")

    print(f"\n  Miss / FA severity at main threshold ({main_label}):")
    main_rows = df_sev[np.isclose(df_sev["x"], main_ref, atol=1.5)]
    if not main_rows.empty:
        for _, row in main_rows.iterrows():
            print(f"    {row['Model']:<25} {row['type']:<18} {row['val']:.4f} {unit}")


def plot_count_evolution(fc1_data, fc2_data, obs_data, condition):
    """Plot 19: Absolute counts of hits / misses / false alarms across threshold sweep.

    Complements plot 8 (counts at one threshold) and plot 1 (ratios at one threshold)
    by showing HOW MANY hits, misses and false alarms each model produces across the
    whole threshold range.  No weighting by error magnitude — pure detection counts.

      Panel 1 — Hits:         number of correctly detected extreme events
      Panel 2 — Misses:       number of undetected extreme events
      Panel 3 — False Alarms: number of falsely predicted extreme events
    """
    print("[19/21] Count evolution (hits / misses / false alarms across threshold sweep)...")

    event_type = condition.get("event_type", "above")
    names      = [condition["expver1"], condition["expver2"]]
    var_lbl, unit = VARIABLE_LABELS.get(condition["var_short"], (condition["var_short"], ""))
    c1 = _style.C_FC1; c2 = _style.C_FC2

    thresholds, _, xlabel, main_label, percentiles = \
        get_threshold_range_and_labels(condition, obs_data)
    x_vals = percentiles if (percentiles is not None) else thresholds

    hits1, miss1, fa1 = [], [], []
    hits2, miss2, fa2 = [], [], []
    for T in thresholds:
        if event_type == "below":
            h1 = int(np.sum((obs_data <= T) & (fc1_data <= T)))
            m1 = int(np.sum((obs_data <= T) & (fc1_data >  T)))
            f1 = int(np.sum((obs_data >  T) & (fc1_data <= T)))
            h2 = int(np.sum((obs_data <= T) & (fc2_data <= T)))
            m2 = int(np.sum((obs_data <= T) & (fc2_data >  T)))
            f2 = int(np.sum((obs_data >  T) & (fc2_data <= T)))
        else:
            h1 = int(np.sum((obs_data >= T) & (fc1_data >= T)))
            m1 = int(np.sum((obs_data >= T) & (fc1_data <  T)))
            f1 = int(np.sum((obs_data <  T) & (fc1_data >= T)))
            h2 = int(np.sum((obs_data >= T) & (fc2_data >= T)))
            m2 = int(np.sum((obs_data >= T) & (fc2_data <  T)))
            f2 = int(np.sum((obs_data <  T) & (fc2_data >= T)))
        hits1.append(h1); miss1.append(m1); fa1.append(f1)
        hits2.append(h2); miss2.append(m2); fa2.append(f2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_data = [
        (axes[0], hits1, hits2, "Hits",         "#2e7d32", "#81c784"),
        (axes[1], miss1, miss2, "Misses",        "#e65100", "#ffb74d"),
        (axes[2], fa1,   fa2,   "False Alarms",  "#6a1b9a", "#ce93d8"),
    ]
    for ax, d1, d2, title, col1, col2 in panel_data:
        ax.plot(x_vals, d1, color=c1, lw=2,   label=names[0], marker='o', ms=3)
        ax.plot(x_vals, d2, color=c2, lw=2,   label=names[1], marker='s', ms=3)
        ax.fill_between(x_vals, d1, d2,
                        where=[a > b for a, b in zip(d1, d2)],
                        alpha=0.12, color=c1, interpolate=True)
        ax.fill_between(x_vals, d1, d2,
                        where=[a < b for a, b in zip(d1, d2)],
                        alpha=0.12, color=c2, interpolate=True)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Count Evolution — {var_lbl} | Day {condition['forecast_day']} | "
        f"{get_threshold_description_from_condition(condition)}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _savefig(fig, f"19_count_evolution_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_count_difference(fc1_data, fc2_data, obs_data, condition):
    """Plot 20: Count difference between models (M1 − M2) across threshold sweep.

    Shows Δhits, Δmisses, Δfalse-alarms = M1 − M2 as a function of threshold.
    Positive Δhits   → M1 detects more events (better)
    Positive Δmisses → M1 misses more events   (worse)
    Positive ΔFAs    → M1 has more false alarms (worse)
    Allows immediate identification of the threshold range where one model
    dominates without confounding from error magnitudes.
    """
    print("[20/21] Count difference between models across threshold sweep...")

    event_type = condition.get("event_type", "above")
    names      = [condition["expver1"], condition["expver2"]]
    var_lbl, unit = VARIABLE_LABELS.get(condition["var_short"], (condition["var_short"], ""))

    thresholds, _, xlabel, main_label, percentiles = \
        get_threshold_range_and_labels(condition, obs_data)
    x_vals = percentiles if (percentiles is not None) else thresholds

    dh, dm, df_ = [], [], []
    for T in thresholds:
        if event_type == "below":
            dh.append(int(np.sum((obs_data <= T) & (fc1_data <= T))) -
                      int(np.sum((obs_data <= T) & (fc2_data <= T))))
            dm.append(int(np.sum((obs_data <= T) & (fc1_data >  T))) -
                      int(np.sum((obs_data <= T) & (fc2_data >  T))))
            df_.append(int(np.sum((obs_data >  T) & (fc1_data <= T))) -
                       int(np.sum((obs_data >  T) & (fc2_data <= T))))
        else:
            dh.append(int(np.sum((obs_data >= T) & (fc1_data >= T))) -
                      int(np.sum((obs_data >= T) & (fc2_data >= T))))
            dm.append(int(np.sum((obs_data >= T) & (fc1_data <  T))) -
                      int(np.sum((obs_data >= T) & (fc2_data <  T))))
            df_.append(int(np.sum((obs_data <  T) & (fc1_data >= T))) -
                       int(np.sum((obs_data <  T) & (fc2_data >= T))))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # positive_is_good=True  → positive means M1 better  (Hits)
    # positive_is_good=False → positive means M1 worse   (Misses, FAs)
    panel_data = [
        (axes[0], dh,  "ΔHits (M1−M2)",        "M1 better: more hits →",    "M2 better: more hits →",    True),
        (axes[1], dm,  "ΔMisses (M1−M2)",       "M1 worse: more misses →",   "M2 worse: more misses →",   False),
        (axes[2], df_, "ΔFalse Alarms (M1−M2)", "M1 worse: more FAs →",      "M2 worse: more FAs →",      False),
    ]
    for ax, vals, title, pos_lbl, neg_lbl, pos_good in panel_data:
        vals = np.array(vals, dtype=float)
        # Blue = M1 winning; Red = M2 winning — consistent with all other plots
        bar_colors = [_style.C_FC1 if (v >= 0) == pos_good else _style.C_FC2 for v in vals]
        ax.bar(x_vals, vals,
               color=bar_colors,
               alpha=0.75, width=(x_vals[1] - x_vals[0]) * 0.8 if len(x_vals) > 1 else 1)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count difference (M1 − M2)", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        # Label colours reflect who benefits: M1 colour if M1 wins, M2 colour if M2 wins
        m1_wins_color = _style.C_FC1
        m2_wins_color = _style.C_FC2
        ax.text(0.97, 0.93, pos_lbl, transform=ax.transAxes, ha='right',
                fontsize=8, color=m1_wins_color if pos_good else m2_wins_color)
        ax.text(0.97, 0.06, neg_lbl, transform=ax.transAxes, ha='right',
                fontsize=8, color=m2_wins_color if pos_good else m1_wins_color)

    fig.suptitle(
        f"Count Difference M1−M2 — {var_lbl} | Day {condition['forecast_day']} | "
        f"{get_threshold_description_from_condition(condition)}\n"
        f"M1={names[0]}   M2={names[1]}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    _savefig(fig, f"20_count_difference_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_detection_profile(fc1_data, fc2_data, obs_data, condition):
    """Plot 21: Normalised detection profile — stacked 100% bars per threshold.

    For each model, shows what fraction of ALL observations falls into:
      Hits, Misses, False Alarms, Correct Negatives
    at each threshold level.  Unlike plots 1/8 (ratios relative to events only),
    this expresses every category as a fraction of the total sample, making it
    easy to compare the 'detection landscape' of each model side by side.
    """
    print("[21/21] Normalised detection profile (stacked 100% bars)...")

    event_type = condition.get("event_type", "above")
    names      = [condition["expver1"], condition["expver2"]]
    var_lbl, unit = VARIABLE_LABELS.get(condition["var_short"], (condition["var_short"], ""))

    thresholds, _, xlabel, main_label, percentiles = \
        get_threshold_range_and_labels(condition, obs_data)
    x_vals = np.array(percentiles if (percentiles is not None) else thresholds, dtype=float)
    n_total = len(obs_data)

    def _counts(fc):
        hs, ms, fs, cns = [], [], [], []
        for T in thresholds:
            if event_type == "below":
                hs.append(np.sum((obs_data <= T) & (fc <= T)))
                ms.append(np.sum((obs_data <= T) & (fc >  T)))
                fs.append(np.sum((obs_data >  T) & (fc <= T)))
                cns.append(np.sum((obs_data >  T) & (fc >  T)))
            else:
                hs.append(np.sum((obs_data >= T) & (fc >= T)))
                ms.append(np.sum((obs_data >= T) & (fc <  T)))
                fs.append(np.sum((obs_data <  T) & (fc >= T)))
                cns.append(np.sum((obs_data <  T) & (fc <  T)))
        return (np.array(hs) / n_total * 100,
                np.array(ms) / n_total * 100,
                np.array(fs) / n_total * 100,
                np.array(cns) / n_total * 100)

    COLORS = {"Hits": "#2e7d32", "Misses": "#e65100",
              "False Alarms": "#6a1b9a", "Correct Neg.": "#bbdefb"}
    bar_w = (x_vals[1] - x_vals[0]) * 0.8 if len(x_vals) > 1 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, fc, name in zip(axes, [fc1_data, fc2_data], names):
        hs, ms, fs, cns = _counts(fc)
        bottom = np.zeros(len(x_vals))
        for label, vals in [("Correct Neg.", cns), ("Hits", hs),
                             ("Misses", ms), ("False Alarms", fs)]:
            ax.bar(x_vals, vals, bottom=bottom, width=bar_w,
                   color=COLORS[label], label=label, alpha=0.85)
            bottom += vals
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("% of all observations", fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
        f"Detection Profile (% of total sample) — {var_lbl} | Day {condition['forecast_day']} | "
        f"{get_threshold_description_from_condition(condition)}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _savefig(fig, f"21_detection_profile_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")


def plot_threshold_weighted_errors(fc1_data, fc2_data, obs_data, threshold, condition):
    print("[9/11] Conditional Error Analysis (MAE/MSE for extremes)...")
    var_short = condition["var_short"]
    mode = condition.get("threshold_mode", "percentile")
    pct = condition.get("threshold_percentile") if mode == "percentile" else None
    thresholds, _, xlabel, main_label, percentiles = get_threshold_range_and_labels(condition, obs_data)
    main_ref = condition.get("threshold_percentile", 95) if mode == "percentile" else condition.get("threshold_value", 0.0)

    mae1_r = np.mean(np.abs(fc1_data - obs_data))
    mae2_r = np.mean(np.abs(fc2_data - obs_data))
    mse1_r = np.mean((fc1_data - obs_data) ** 2)
    mse2_r = np.mean((fc2_data - obs_data) ** 2)
    mae1_w = threshold_conditional_mae(fc1_data, obs_data, threshold, pct, var_short)
    mae2_w = threshold_conditional_mae(fc2_data, obs_data, threshold, pct, var_short)
    mse1_w = threshold_conditional_mse(fc1_data, obs_data, threshold, pct, var_short)
    mse2_w = threshold_conditional_mse(fc2_data, obs_data, threshold, pct, var_short)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, cats, v1s, v2s, ylabel, title in [
        (axes[0, 0], ["Regular MAE", "Weighted MAE"], [mae1_r, mae1_w], [mae2_r, mae2_w],
         "MAE", "MAE Comparison"),
        (axes[0, 1], ["Regular MSE", "Weighted MSE"], [mse1_r, mse1_w], [mse2_r, mse2_w],
         "MSE", "MSE Comparison"),
    ]:
        x = np.arange(2); w = 0.35
        bars1 = ax.bar(x - w/2, v1s, w, label=condition["expver1"], alpha=0.8, color=_style.C_FC1)
        bars2 = ax.bar(x + w/2, v2s, w, label=condition["expver2"], alpha=0.8, color=_style.C_FC2)
        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + h*0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(cats)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    x_vals, mae1_ev, mae2_ev, mse1_ev, mse2_ev = [], [], [], [], []
    for i, thresh in enumerate(thresholds):
        pct_det = percentiles[i] if percentiles is not None else None
        x_val = percentiles[i] if mode == "percentile" else thresh
        x_vals.append(x_val)
        mae1_ev.append(threshold_conditional_mae(fc1_data, obs_data, thresh, pct_det, var_short))
        mae2_ev.append(threshold_conditional_mae(fc2_data, obs_data, thresh, pct_det, var_short))
        mse1_ev.append(threshold_conditional_mse(fc1_data, obs_data, thresh, pct_det, var_short))
        mse2_ev.append(threshold_conditional_mse(fc2_data, obs_data, thresh, pct_det, var_short))

    for ax, reg1, ev1, reg2, ev2, ylabel, title in [
        (axes[1, 0], mae1_r, mae1_ev, mae2_r, mae2_ev, "MAE", "MAE: All vs Extremes Only"),
        (axes[1, 1], mse1_r, mse1_ev, mse2_r, mse2_ev, "MSE", "MSE: All vs Extremes Only"),
    ]:
        ax.axhline(reg1, ls="--", color="lightblue",  alpha=0.9,
                   label=f"{condition['expver1']} (all)")
        ax.plot(x_vals, ev1, "o-", color=_style.C_FC1, lw=2, ms=5,
                label=f"{condition['expver1']} (extremes)")
        ax.axhline(reg2, ls="--", color="lightcoral", alpha=0.9,
                   label=f"{condition['expver2']} (all)")
        ax.plot(x_vals, ev2, "s-", color=_style.C_FC2,  lw=2, ms=5,
                label=f"{condition['expver2']} (extremes)")
        ax.axvline(main_ref, color=_style.C_THRESHOLD, ls="--", alpha=0.7, label=main_label)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, f"9_conditional_errors_{condition['expver1']}_vs_{condition['expver2']}"
                  f"_{condition['var_short']}_day{condition['forecast_day']}.png")

    print(f"\n  Conditional Error Summary (Extreme Events Only):")
    print(f"  {'Metric':<8} {'Model':<25} {'All Data':>10} {'Extremes':>12} {'Ratio':>8}")
    print(f"  {'-'*65}")
    for met, r1, w1, r2, w2 in [("MAE", mae1_r, mae1_w, mae2_r, mae2_w),
                                   ("MSE", mse1_r, mse1_w, mse2_r, mse2_w)]:
        for label, reg, wgt in [(condition["expver1"], r1, w1),
                                 (condition["expver2"], r2, w2)]:
            ratio = wgt / reg if reg else float("nan")
            print(f"  {met:<8} {label:<25} {reg:>10.4f} {wgt:>12.4f} {ratio:>8.3f}")


# ============================================================================
# DAILY AGGREGATION HELPER (copy from run.py to avoid metview import chain)
# ============================================================================

def _aggregate_to_daily_mean_local(data, config):
    """Aggregate sub-daily rows to daily means (mirrors run._aggregate_to_daily_mean)."""
    data = data.copy()
    data['forecast_day'] = ((data['step'] - 1) // 24).astype(int) + 1
    group_cols = ['lat', 'lon', 'date', 'forecast_day']
    agg_cols = [c for c in ['obs_value', 'fc1_value', 'fc2_value',
                             'fc1_value_uncorrected', 'fc2_value_uncorrected']
                if c in data.columns]
    member_cols = [c for c in data.columns
                   if c.startswith('fc1_member_') or c.startswith('fc2_member_')]
    agg_cols = agg_cols + member_cols
    meta_cols = [c for c in data.columns
                 if c not in agg_cols and c not in group_cols and c != 'step']
    agg_dict = {col: (col, 'mean') for col in agg_cols}
    agg_dict['step'] = ('step', 'mean')
    for col in meta_cols:
        agg_dict[col] = (col, 'first')
    agg = data.groupby(group_cols, sort=False).agg(**agg_dict).reset_index()
    freq = int(config.get('lead_time_frequency', 6)) if config else 6
    agg['step'] = (agg['forecast_day'] * 24 - (24 - freq) // 2).astype(int)
    return agg.reset_index(drop=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global SAVE_FIGURES, OUTPUT_PATH

    parser = argparse.ArgumentParser(
        prog="diagnose_extremes.py",
        description="Detailed extreme-event diagnostics from scorecards4extremes parquet data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Warm extremes (99th pct), day 3, all conditions:
  python diagnose_extremes.py --config config_2t_local_p99obsclim_aifs_ifs_single.yaml \\
      --day 3 --threshold-pct 99

  # Cold extremes (1st pct), day 5, DJF only, flat terrain:
  python diagnose_extremes.py --config config_2t_local_p99obsclim_aifs_ifs_single.yaml \\
      --day 5 --threshold-pct 1 --season DJF --orog flat

  # Fixed threshold (30mm precipitation):
  python diagnose_extremes.py --config config_tp24_local_p99obs.yaml \\
      --day 3 --threshold-value 30.0
""",
    )
    parser.add_argument("--config",          required=True,      help="YAML config file")
    parser.add_argument("--day",             type=int, default=3, help="Forecast day (default: 3)")
    parser.add_argument("--season",          default=None,       help="Season: DJF MAM JJA SON")
    parser.add_argument("--orog",            default=None,       help="Orography: flat/low/mid/hilly/high/complex")
    parser.add_argument("--threshold-pct",   type=float, default=None,
                        help="Threshold percentile (e.g. 99 for warm, 1 for cold)")
    parser.add_argument("--threshold-value", type=float, default=None,
                        help="Fixed threshold value (overrides --threshold-pct)")
    parser.add_argument("--output-dir",      default=None,
                        help="Output directory for plots (default: results dir from config)")
    parser.add_argument("--no-save",         action="store_true",
                        help="Display plots interactively instead of saving")
    args = parser.parse_args()

    if args.threshold_pct is None and args.threshold_value is None:
        parser.error("Specify --threshold-pct or --threshold-value")

    config       = load_config(args.config)
    fc1_name, fc2_name = get_model_names(config)
    variable     = config["variable"]

    _style.apply_style(save_dpi=300)

    SAVE_FIGURES = not args.no_save
    # Build a subdirectory that encodes all run parameters so different
    # combinations (season, orog, day, threshold) never overwrite each other.
    # Only auto-create the subdirectory when using the default path; if the user
    # explicitly passed --output-dir, use it verbatim.
    thr_tag  = (f"pct{int(args.threshold_pct)}" if args.threshold_pct is not None
                else f"fixed{args.threshold_value}")
    run_tag  = f"day{args.day}_{thr_tag}"
    if args.season:
        run_tag += f"_{args.season}"
    if args.orog:
        run_tag += f"_{args.orog}"
    if args.output_dir:
        OUTPUT_PATH = str(args.output_dir)
    else:
        base_out = config.get("save", {}).get("output_directory", "./results/diagnostics")
        OUTPUT_PATH = str(Path(base_out) / run_tag)
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Build condition dict (compatible with all plot functions)
    if args.threshold_value is not None:
        condition = {
            "expver1": fc1_name, "expver2": fc2_name,
            "var_short": variable, "forecast_day": args.day,
            "season": args.season, "terrain": args.orog or "all",
            "threshold_mode": "fixed", "threshold_value": args.threshold_value,
            "threshold_arr": None,
        }
    else:
        condition = {
            "expver1": fc1_name, "expver2": fc2_name,
            "var_short": variable, "forecast_day": args.day,
            "season": args.season, "terrain": args.orog or "all",
            "threshold_mode": "percentile", "threshold_percentile": args.threshold_pct,
            "threshold_arr": None,  # populated below if local_obs_climatology
        }
    # Store event_type from config so all plots use the correct tail direction
    _thr_cfg = config.get('threshold', {})
    if _thr_cfg.get('method') == 'fixed':
        condition['event_type'] = _thr_cfg.get('fixed', {}).get('event_type', 'above')
    else:
        condition['event_type'] = _thr_cfg.get('event_type', 'above')

    print("\n" + "=" * 70)
    print("EXTREME EVENTS DIAGNOSTIC — Scorecards4Extremes")
    print("=" * 70)
    print(f"  Config     : {args.config}")
    print(f"  Variable   : {variable}")
    print(f"  Models     : {fc1_name}  vs  {fc2_name}")
    print(f"  Day        : {args.day}")
    print(f"  Season     : {args.season or 'all'}")
    print(f"  Orography  : {args.orog or 'all'}")
    if args.threshold_value is not None:
        print(f"  Threshold  : fixed = {args.threshold_value}")
    else:
        print(f"  Threshold  : {args.threshold_pct}th percentile of obs")
    print(f"  Output dir : {OUTPUT_PATH}")

    # Load and filter data
    print(f"\nLoading day {args.day} data...")
    df, _, _ = load_day(config, args.day)
    df = filter_data(df, config, season=args.season, orog=args.orog)

    # Aggregate to daily means when the pipeline uses local_obs_climatology,
    # so that the twMAE values match those in the heatmap CSV (which is also
    # computed on daily-mean data after the same aggregation in run.py).
    thr_method_main = config.get('threshold', {}).get('method', '')
    lead_freq       = config.get('lead_time_frequency', 24)
    # Mirror run.py exactly so twMAE matches the heatmap CSV: the pipeline
    # aggregates sub-daily rows to daily means for local_obs_climatology
    # UNCONDITIONALLY (run.py STEP 5b, ~line 970), and for fixed thresholds only
    # when the data is sub-daily (lead_time_frequency < 24).  Grouping by
    # (lat, lon, date, forecast_day) is a no-op on already-daily data, so applying
    # it unconditionally is safe — and, crucially, still correct when a config
    # leaves lead_time_frequency unset (previously this branch was skipped because
    # the default 24 failed the `lead_freq < 24` test, silently desyncing twMAE
    # from the heatmap).
    if thr_method_main == 'local_obs_climatology' or (
            thr_method_main == 'fixed' and lead_freq < 24):
        n_before = len(df)
        df = _aggregate_to_daily_mean_local(df, config)
        print(f"  Daily aggregation ({thr_method_main}): "
              f"{n_before:,} sub-daily rows → {len(df):,} daily rows "
              "(matches main pipeline scoring)")

    # Downcast to float32 to halve peak memory (sufficient precision for all diagnostics)
    obs  = df["obs_value"].values.astype(np.float32)
    fc1  = df["fc1_value"].values.astype(np.float32)
    fc2  = df["fc2_value"].values.astype(np.float32)
    N    = len(obs)

    if N < 100:
        print(f"[ERROR] Only {N} rows after filtering — too few for diagnostics.")
        sys.exit(1)

    print(f"  N = {N:,}")

    # Compute threshold
    if args.threshold_value is not None:
        threshold = args.threshold_value
        threshold_arr = None
    else:
        threshold = np.percentile(obs, args.threshold_pct)
        print(f"  {args.threshold_pct}th percentile of obs → threshold = {threshold:.4f}")
        threshold_arr = None

    # If the config uses local_obs_climatology, compute per-station per-row thresholds
    # so that plot 10 (twMAE decomposition) exactly matches the main pipeline.
    thr_method = config.get('threshold', {}).get('method', '')
    if thr_method == 'local_obs_climatology' and args.threshold_value is None:
        print("\n  Computing per-station threshold from local_obs_climatology "
              "(matches main pipeline)...")
        try:
            import threshold as _thr_module
            _thr_series = _thr_module._compute_local_obs_climatology_threshold(config, df)
            threshold_arr = _thr_series.values.astype(np.float32)
            valid_thr = threshold_arr[~np.isnan(threshold_arr)]
            if len(valid_thr):
                print(f"  Per-station threshold — mean={np.mean(valid_thr):.3f}, "
                      f"p5={np.percentile(valid_thr, 5):.3f}, "
                      f"p95={np.percentile(valid_thr, 95):.3f}")
        except Exception as e:
            print(f"  WARNING: could not load local_obs_climatology thresholds: {e}")
            print("  Falling back to pooled percentile threshold.")
            threshold_arr = None

    # Store the per-station threshold array (if computed) in condition for plot 10.
    # Also keep df and config in condition so plot 11 can load per-station
    # thresholds for each percentile level in the sweep.
    condition['threshold_arr'] = threshold_arr
    condition['_df']     = df      # kept alive for plot 11; deleted after
    condition['_config'] = config

    verify_extreme_detection_logic(obs, condition)

    # Run all 9 plots
    print("\n" + "=" * 50)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 50)

    _plot_tally = {"ok": 0, "fail": 0}

    def _run(fn, *a, **kw):
        """Run one plot in isolation: a failure in a single plot must not abort
        the remaining plots (previously an exception here killed the whole run)."""
        try:
            result = fn(*a, **kw)
            _plot_tally["ok"] += 1
        except Exception as exc:
            _plot_tally["fail"] += 1
            print(f"  [WARN] {fn.__name__} failed: {exc!r} — skipping this plot")
            result = None
        finally:
            plt.close('all')
            gc.collect()
        return result

    # ── Curated ("moderate") figure set — duplicates retired ─────────────────
    # Retired to remove duplicate plot *types* (functions kept above for optional
    # re-enable, but no longer generated by default):
    #   • plot_ets_pss_comparison        → merged into plot_skill_scores_comparison
    #   • plot_frequency_bias_evolution  → redundant with the threshold-evolution sweep
    #   • plot_qq_plots                  → duplicated by the dedicated plot_qq.py tool
    #   • plot_twmae_percentile_decomposition → sweep view overlaps threshold-evolution + twMAE decomp
    #   • plot_twmae_component_fractions → this is panel 1 of plot_twmae_decomposition re-expressed as %
    #   • plot_miss_fa_severity          → error/severity view folded into error_distribution + conditional_bias_noise
    #   • plot_count_evolution           → the count difference view is the informative one
    #   • plot_detection_profile         → normalised restatement of the contingency counts
    #   • plot_error_depth_profile       → specialised; overlaps extreme_intensity_scatter
    _run(plot_skill_scores_comparison,       fc1, fc2, obs, threshold, condition)
    _run(plot_threshold_evolution,           fc1, fc2, obs, condition)
    _run(plot_error_distribution,            fc1, fc2, obs, condition)
    _run(plot_empirical_distributions,       fc1, fc2, obs, threshold, condition)
    _run(plot_contingency_table_comparison,  fc1, fc2, obs, threshold, condition)
    _run(plot_twmae_decomposition,           fc1, fc2, obs, threshold, condition)
    _run(plot_extreme_intensity_scatter,     fc1, fc2, obs, threshold, condition)
    _run(plot_count_difference,              fc1, fc2, obs, condition)
    _run(plot_conditional_bias_noise,        fc1, fc2, obs, condition)
    _run(plot_conditional_bias_decomposed,   fc1, fc2, obs, condition)
    _run(plot_twmae_skill_score,             fc1, fc2, obs, condition)
    _run(plot_summary_scorecard,             fc1, fc2, obs, threshold, condition)

    # df is no longer needed after all plots are done
    del condition['_df'], condition['_config']
    gc.collect()

    print("\n" + "=" * 70)
    if SAVE_FIGURES:
        msg = f"DONE — {_plot_tally['ok']} plots saved to: {OUTPUT_PATH}"
        if _plot_tally["fail"]:
            msg += f"  ({_plot_tally['fail']} failed — see [WARN] lines above)"
    else:
        msg = "DONE — display mode (no plots saved)"
    print(msg)
    print("=" * 70)


if __name__ == "__main__":
    main()
