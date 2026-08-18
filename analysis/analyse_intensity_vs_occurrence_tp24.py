"""
analyse_intensity_vs_occurrence_tp24.py — Does AIFS win on placement/occurrence
but under-forecast the intensity of the most extreme tp24 events?

Motivation
----------
Threshold-based scorecards (ETS, POD, twCRPS/twMAE at moderate thresholds)
reward correct occurrence + location and penalise displacement twice (miss +
false alarm). A smoother/regressive model can therefore "win" the scorecard
by being better placed while systematically capping the amplitude of the
most extreme events. This script separates OCCURRENCE skill (did it happen
here?) from INTENSITY skill (how big was it?) using the native-resolution
tp24 parquet (no 0.25deg regridding, so each model's true peak is preserved).

Produces two figures + one summary CSV:

  FIGURE 1 (vs threshold, pooled over all forecast days unless --day given):
    (a) Occurrence: ETS + POD vs threshold, IFS vs AIFS
    (b) Intensity:  bias + tail-weighted RMSE (chaining v(x)=max(x,T)) vs threshold
    (c) Amplitude ratio r(T) = median(fc/obs | obs>=T) and conditional bias
        vs threshold — the "does it cap the peak?" curve
    (d) Upper-tail Q-Q (p90..p99.9), one line per forecast day, IFS vs AIFS

  FIGURE 2 (vs forecast day, fixed reference threshold):
    Discrimination (deterministic-forecast ROC AUC — see roc_auc() docstring
    for the signal-detection methodology, no ensemble needed) vs conditional
    intensity bias among hits — the single-figure answer to
    "better predictability, not better intensity?"

  FIGURE 3 (vs threshold, one subplot per forecast day 1/3/5/7):
    Hits / misses / false alarms (% of samples) for each model — shows
    whether a higher hit rate comes paired with a higher false-alarm rate.

Usage
-----
    .venv/bin/python analysis/analyse_intensity_vs_occurrence_tp24.py \\
        --config configs/deterministic/config_tp24_local_fixed70mm_aifs_ifs_nhextrop_fullyear_orig.yaml \\
        --thresholds 30,50,70,80,100,120,150 \\
        --ref-threshold 70 \\
        --output-dir results/intensity_vs_occurrence_tp24_orig
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import rankdata

# ─── Palette (consistent with analyse_bias_orog.py) ───────────────────────────
C_IFS  = "#d7191c"   # red
C_AIFS = "#2c7bb6"   # blue
DAYS_STYLE = {1: ("-", 1.0), 3: ("--", 0.85), 5: ("-.", 0.7),
              7: (":", 0.55), 10: ((0, (3, 1, 1, 1)), 0.4)}

TAIL_QUANTILES = [90, 95, 97, 99, 99.5, 99.9]


def display_name(model_name):
    """Map a raw model identifier (e.g. 'ifs_oper', 'aifs1.0_oper') to a short
    display label ('IFS', 'AIFS') for plot titles/legends. Falls back to the
    raw name for anything that doesn't match, so this stays generic to other
    model pairs."""
    name_lower = model_name.lower()
    if "aifs" in name_lower:
        return "AIFS"
    if "ifs" in name_lower:
        return "IFS"
    return model_name


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Any tp24 deterministic YAML config "
                   "(used only for extract_points.output_path, model names, filter settings)")
    p.add_argument("--thresholds", default="30,50,70,80,100,120,150",
                   help="Comma list of thresholds (mm) for the sweep panels")
    p.add_argument("--ref-threshold", type=float, default=70.0,
                   help="Reference threshold (mm) for the AUC-vs-lead-day figure")
    p.add_argument("--orog", default=None, choices=[None, "flat", "hilly", "complex"],
                   help="Restrict to one orography class (default: all stations pooled)")
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help="Output directory (default: results/intensity_vs_occurrence_tp24)")
    return p.parse_args()


# ─── Data loading (mirrors filter.py's tp24-specific rules) ──────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_days(config, orog=None):
    """Load every forecast-day parquet, applying the same QC as filter.run_step4
    for tp24: coastal lsm cut + max_valid_precipitation (no outlier removal —
    filter.py explicitly skips it for tp24 to preserve genuine extremes).
    """
    rd = config["read_data"]
    fc1_name = rd["forecast_model1"]["name"]
    fc2_name = rd["forecast_model2"]["name"]
    variable = config["variable"]
    ep = Path(config["extract_points"]["output_path"])
    files = sorted(ep.glob(f"{variable}_{fc1_name}_vs_{fc2_name}_*day*.parquet"),
                   key=lambda p: int(p.name.split("day")[-1].split(".")[0]))
    if not files:
        raise SystemExit(f"No parquet files found in {ep}")

    cfg_f = config.get("filter", {})
    coastal_thresh = cfg_f.get("coastal_lsm_threshold", 0.9)
    max_precip = cfg_f.get("max_valid_precipitation", 800.0)
    orog_ranges = cfg_f.get("orography_ranges",
                             {"flat": [0, 40], "hilly": [40, 120], "complex": [120, 3000]})

    want_cols = ["date", "step", "lat", "lon", "obs_value", "fc1_value", "fc2_value",
                 "sdfor", "lsm"]

    frames = []
    for f in files:
        day = int(f.name.split("day")[-1].split(".")[0])
        avail = pq.ParquetFile(f).schema_arrow.names
        df = pd.read_parquet(f, columns=[c for c in want_cols if c in avail])
        df["forecast_day"] = day

        if "lsm" in df.columns:
            df = df[df["lsm"] > coastal_thresh]

        if "sdfor" in df.columns and orog and orog in orog_ranges:
            lo, hi = orog_ranges[orog]
            df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]

        if max_precip is not None:
            df = df[df["obs_value"] <= max_precip]

        df = df.dropna(subset=["obs_value", "fc1_value", "fc2_value"])
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data, fc1_name, fc2_name


# ─── Score primitives (self-contained, matching src/det_scores.py formulas) ──

def ets_pod(obs, fc, T):
    obs_ev, fc_ev = obs >= T, fc >= T
    hits = np.sum(obs_ev & fc_ev)
    misses = np.sum(obs_ev & ~fc_ev)
    fas = np.sum(~obs_ev & fc_ev)
    total = len(obs)
    hits_r = (hits + misses) * (hits + fas) / total if total else np.nan
    denom = hits + misses + fas - hits_r
    ets = (hits - hits_r) / denom if denom > 0 else np.nan
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    return ets, pod


def twrmse_bias(obs, fc, T):
    """Tail-weighted RMSE/bias, chaining v(x)=max(x,T) (same as det_scores.calculate_twrmse)."""
    obs_v = np.maximum(obs, T)
    fc_v = np.maximum(fc, T)
    twrmse = np.sqrt(np.mean((fc_v - obs_v) ** 2))
    bias = np.mean(fc - obs)
    return twrmse, bias


def amplitude_ratio_and_condbias(obs, fc, T):
    """r(T) = median(fc/obs) and mean(fc-obs) among cases where obs >= T."""
    mask = obs >= T
    if mask.sum() < 5:
        return np.nan, np.nan
    r = float(np.median(fc[mask] / obs[mask]))
    cond_bias = float(np.mean(fc[mask] - obs[mask]))
    return r, cond_bias


def hit_overprediction_stats(obs, fc, T):
    """Among HITS only (obs>=T and fc>=T), the fraction that OVERSHOOT the
    true magnitude (fc>obs) and the mean overshoot size. ETS/PSS cannot see
    this — both models count as an identical "hit" regardless of how far fc
    exceeds obs — but twMAE penalises |fc-obs| directly, so a model that
    overshoots more on its hits pays an extra twMAE cost on top of whatever
    the false-alarm count/severity difference already explains.
    """
    mask = (obs >= T) & (fc >= T)
    if mask.sum() < 5:
        return np.nan, np.nan
    o, f = obs[mask], fc[mask]
    over = f > o
    frac_over = float(over.mean())
    over_mae = float(np.mean(f[over] - o[over])) if over.sum() > 0 else np.nan
    return frac_over, over_mae


def roc_auc(scores, labels):
    """Deterministic-forecast discrimination AUC (signal-detection ROC, e.g.
    Mason & Graham 2002) — NOT the ensemble/probabilistic AUC.

    Rather than sweeping a probability of exceedance (which needs ensemble
    members), this sweeps the decision threshold over the raw continuous
    forecast VALUE itself ("warn if forecast >= x" for every possible x) and
    traces out the hit-rate vs false-alarm-rate curve. Its area equals the
    Mann-Whitney U statistic:
        AUC = P(forecast on an obs-extreme case > forecast on a non-extreme case)
            = (sum_ranks(positives) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    A single continuous value per case is sufficient — no ensemble required.
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def contingency_counts(obs, fc, T):
    """Hits / misses / false alarms (counts) for event obs>=T, fc>=T."""
    obs_ev, fc_ev = obs >= T, fc >= T
    hits = int(np.sum(obs_ev & fc_ev))
    misses = int(np.sum(obs_ev & ~fc_ev))
    fas = int(np.sum(~obs_ev & fc_ev))
    return hits, misses, fas


def hit_conditional_bias(obs, fc, T):
    """Mean (fc - obs) restricted to HITS ONLY (both obs and fc exceed T) —
    isolates intensity error from the cases already correctly detected,
    removing any influence of misses/false alarms on the bias estimate.
    """
    mask = (obs >= T) & (fc >= T)
    if mask.sum() < 5:
        return np.nan
    return float(np.mean(fc[mask] - obs[mask]))


# ─── Figure 1 ─────────────────────────────────────────────────────────────────

def fig_occurrence_vs_intensity(data, thresholds, fc1_name, fc2_name, out_dir):
    obs = data["obs_value"].values
    fc1 = data["fc1_value"].values
    fc2 = data["fc2_value"].values

    rows = []
    for T in thresholds:
        ets1, pod1 = ets_pod(obs, fc1, T)
        ets2, pod2 = ets_pod(obs, fc2, T)
        tw1, b1 = twrmse_bias(obs, fc1, T)
        tw2, b2 = twrmse_bias(obs, fc2, T)
        r1, cb1 = amplitude_ratio_and_condbias(obs, fc1, T)
        r2, cb2 = amplitude_ratio_and_condbias(obs, fc2, T)
        rows.append(dict(threshold=T, ets1=ets1, pod1=pod1, ets2=ets2, pod2=pod2,
                          twrmse1=tw1, bias1=b1, twrmse2=tw2, bias2=b2,
                          ratio1=r1, condbias1=cb1, ratio2=r2, condbias2=cb2))
    tbl = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Occurrence: ETS + POD vs threshold
    ax = axes[0, 0]
    ax.plot(tbl.threshold, tbl.ets1, color=C_IFS, marker="o", label=f"ETS {fc1_name}")
    ax.plot(tbl.threshold, tbl.ets2, color=C_AIFS, marker="o", label=f"ETS {fc2_name}")
    ax.plot(tbl.threshold, tbl.pod1, color=C_IFS, marker="s", ls="--", label=f"POD {fc1_name}")
    ax.plot(tbl.threshold, tbl.pod2, color=C_AIFS, marker="s", ls="--", label=f"POD {fc2_name}")
    ax.set_xlabel("Threshold (mm/24h)")
    ax.set_ylabel("Score")
    ax.set_title("(a) Occurrence skill vs threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Intensity: conditional bias (obs>=T) + twRMSE vs threshold
    # (unconditional bias is threshold-invariant by construction, so it is not
    # informative here — conditional-on-extreme bias shows the amplitude gap)
    ax = axes[0, 1]
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(tbl.threshold, tbl.condbias1, color=C_IFS, marker="o", label=f"cond. bias {fc1_name}")
    ax.plot(tbl.threshold, tbl.condbias2, color=C_AIFS, marker="o", label=f"cond. bias {fc2_name}")
    ax2 = ax.twinx()
    ax2.plot(tbl.threshold, tbl.twrmse1, color=C_IFS, marker="^", ls=":", label=f"twRMSE {fc1_name}")
    ax2.plot(tbl.threshold, tbl.twrmse2, color=C_AIFS, marker="^", ls=":", label=f"twRMSE {fc2_name}")
    ax.set_xlabel("Threshold (mm/24h)")
    ax.set_ylabel("mean(fc - obs | obs≥T)  (mm)")
    ax2.set_ylabel("tail-weighted RMSE (mm)")
    ax.set_title("(b) Intensity skill vs threshold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # (c) Amplitude ratio + conditional bias vs threshold ("caps the peak?")
    ax = axes[1, 0]
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="perfect (r=1)")
    ax.plot(tbl.threshold, tbl.ratio1, color=C_IFS, marker="o", label=f"r(T) {fc1_name}")
    ax.plot(tbl.threshold, tbl.ratio2, color=C_AIFS, marker="o", label=f"r(T) {fc2_name}")
    ax.set_xlabel("Threshold T (mm/24h)")
    ax.set_ylabel("median( fc / obs | obs≥T )")
    ax.set_title("(c) Amplitude ratio — does it under-forecast the peak?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Upper-tail Q-Q per forecast day
    ax = axes[1, 1]
    lo, hi = None, None
    for day, grp in data.groupby("forecast_day"):
        ls, alpha = DAYS_STYLE.get(int(day), ("-", 0.6))
        obs_q = np.percentile(grp["obs_value"].values, TAIL_QUANTILES)
        fc1_q = np.percentile(grp["fc1_value"].values, TAIL_QUANTILES)
        fc2_q = np.percentile(grp["fc2_value"].values, TAIL_QUANTILES)
        ax.plot(obs_q, fc1_q, color=C_IFS, ls=ls, alpha=alpha, marker="o", ms=4,
                label=f"{fc1_name} day{int(day)}")
        ax.plot(obs_q, fc2_q, color=C_AIFS, ls=ls, alpha=alpha, marker="o", ms=4,
                label=f"{fc2_name} day{int(day)}")
        vals = np.concatenate([obs_q, fc1_q, fc2_q])
        lo = float(vals.min()) if lo is None else min(lo, float(vals.min()))
        hi = float(vals.max()) if hi is None else max(hi, float(vals.max()))
    margin = (hi - lo) * 0.03
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=1.0)
    ax.set_xlabel("Observed quantile (mm)")
    ax.set_ylabel("Forecast quantile (mm)")
    ax.set_title(f"(d) Upper-tail Q-Q (p{TAIL_QUANTILES[0]}–p{TAIL_QUANTILES[-1]}) by lead day")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(f"tp24 (native resolution) — occurrence vs intensity skill\n"
                 f"{fc1_name} vs {fc2_name}", fontsize=13, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "1_occurrence_vs_intensity_by_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] Occurrence-vs-intensity sweep → {out.name}")
    return tbl


# ─── Figure 2 ─────────────────────────────────────────────────────────────────

def fig_discrimination_vs_intensity(data, ref_threshold, fc1_name, fc2_name, out_dir):
    days = sorted(data["forecast_day"].unique())
    rows = []
    for day in days:
        grp = data[data["forecast_day"] == day]
        obs = grp["obs_value"].values
        fc1 = grp["fc1_value"].values
        fc2 = grp["fc2_value"].values
        labels = obs >= ref_threshold

        auc1 = roc_auc(fc1, labels)
        auc2 = roc_auc(fc2, labels)
        cb1 = hit_conditional_bias(obs, fc1, ref_threshold)
        cb2 = hit_conditional_bias(obs, fc2, ref_threshold)
        rows.append(dict(day=day, auc1=auc1, auc2=auc2, condbias1=cb1, condbias2=cb2,
                          n_events=int(labels.sum())))
    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(tbl.day, tbl.auc1, color=C_IFS, marker="o", lw=2, label=f"AUC {fc1_name}")
    ax.plot(tbl.day, tbl.auc2, color=C_AIFS, marker="o", lw=2, label=f"AUC {fc2_name}")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("ROC AUC (discrimination of obs ≥ T)")
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.axhline(0, color="k", lw=0.8)
    ax2.plot(tbl.day, tbl.condbias1, color=C_IFS, marker="s", ls="--", lw=2,
             label=f"Hit-conditional bias {fc1_name}")
    ax2.plot(tbl.day, tbl.condbias2, color=C_AIFS, marker="s", ls="--", lw=2,
             label=f"Hit-conditional bias {fc2_name}")
    ax2.set_ylabel("Mean (fc - obs) among hits (mm)  [< 0 = under-forecasts peak]")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")
    ax.set_title(f"Discrimination (solid) vs intensity bias among hits (dashed)\n"
                 f"Reference threshold T={ref_threshold:.0f} mm/24h")
    fig.tight_layout()
    out = Path(out_dir) / "2_discrimination_vs_intensity_bias.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2] Discrimination-vs-intensity-bias → {out.name}")
    return tbl


# ─── Figure 3 ─────────────────────────────────────────────────────────────────

def fig_contingency_by_day(data, thresholds, fc1_name, fc2_name, out_dir,
                            days=(1, 3, 5, 7)):
    """Hits / false alarms (% of samples, LOG y-axis) vs threshold, one subplot
    per forecast day, one line per (model, event-type) combination.

    Misses are computed and kept in the returned table (misses = n_events -
    hits is a fixed-pool identity shared by both models, so it rarely adds
    visual information beyond what the hits line already shows) but are not
    plotted. Log scale keeps hits and false alarms comparable even though
    they differ by orders of magnitude across thresholds. The returned table
    includes explicit aifs/ifs ratio columns so the model comparison never
    depends on reading small percentages off the plot.
    """
    available_days = sorted(data["forecast_day"].unique())
    days = [d for d in days if d in available_days]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    rows = []

    for ax, day in zip(axes, days):
        grp = data[data["forecast_day"] == day]
        obs = grp["obs_value"].values
        fc1 = grp["fc1_value"].values
        fc2 = grp["fc2_value"].values
        n = len(obs)

        h1s, m1s, f1s, h2s, m2s, f2s = [], [], [], [], [], []
        for T in thresholds:
            h1, m1, f1 = contingency_counts(obs, fc1, T)
            h2, m2, f2 = contingency_counts(obs, fc2, T)
            h1s.append(100 * h1 / n); m1s.append(100 * m1 / n); f1s.append(100 * f1 / n)
            h2s.append(100 * h2 / n); m2s.append(100 * m2 / n); f2s.append(100 * f2 / n)
            rows.append(dict(
                day=day, threshold=T, n_samples=n,
                hits_ifs=h1, misses_ifs=m1, fa_ifs=f1,
                hits_aifs=h2, misses_aifs=m2, fa_aifs=f2,
                # Explicit model ratios (aifs/ifs) — the unambiguous way to
                # read "how many times more/fewer", independent of plot scale.
                ratio_hits=h2 / h1 if h1 else np.nan,
                ratio_misses=m2 / m1 if m1 else np.nan,
                ratio_fa=f2 / f1 if f1 else np.nan,
            ))

        ax.plot(thresholds, h1s, color=C_IFS, ls="-", marker="o", label=f"Hits {fc1_name}")
        ax.plot(thresholds, f1s, color=C_IFS, ls=":", marker="^", label=f"False alarms {fc1_name}")
        ax.plot(thresholds, h2s, color=C_AIFS, ls="-", marker="o", label=f"Hits {fc2_name}")
        ax.plot(thresholds, f2s, color=C_AIFS, ls=":", marker="^", label=f"False alarms {fc2_name}")

        ax.set_yscale("log")
        ax.set_xlabel("Threshold (mm/24h)")
        ax.set_ylabel("% of samples (log scale)")
        ax.set_title(f"Day {int(day)}")
        ax.grid(alpha=0.3, which="both")
        if ax is axes[0]:
            ax.legend(fontsize=7, ncol=1)

    fig.suptitle(f"Hits / false alarms vs threshold (log scale) — {fc1_name} vs {fc2_name}",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "3_contingency_hits_misses_fa_by_day.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] Hits/misses/FA by day → {out.name}")
    return pd.DataFrame(rows)


# ─── Figure 3b — presentation/slide version ──────────────────────────────────

def fig_contingency_presentation(tbl3, fc1_name, fc2_name, out_dir, days=(1, 3, 5, 7)):
    """Slide-friendly redraw of Figure 3 (hits vs false alarms) from the
    already-computed table: bigger fonts/lines/markers, one shared legend,
    16:9 figure sized for a PowerPoint slide.
    """
    available_days = sorted(tbl3["day"].unique())
    days = [d for d in days if d in available_days]

    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.ravel()
    handles_labels = None

    for ax, day in zip(axes, days):
        sub = tbl3[tbl3["day"] == day].sort_values("threshold")
        # tbl3 stores RAW counts (needed for the ratio columns) — convert to
        # % of samples here to match the axis label.
        n = sub.n_samples
        ax.plot(sub.threshold, 100 * sub.hits_ifs / n, color=C_IFS, ls="-", marker="o",
                lw=3, ms=9, label=f"Hits — {fc1_name}")
        ax.plot(sub.threshold, 100 * sub.fa_ifs / n, color=C_IFS, ls=":", marker="^",
                lw=3, ms=9, label=f"False alarms — {fc1_name}")
        ax.plot(sub.threshold, 100 * sub.hits_aifs / n, color=C_AIFS, ls="-", marker="o",
                lw=3, ms=9, label=f"Hits — {fc2_name}")
        ax.plot(sub.threshold, 100 * sub.fa_aifs / n, color=C_AIFS, ls=":", marker="^",
                lw=3, ms=9, label=f"False alarms — {fc2_name}")

        ax.set_yscale("log")
        ax.set_title(f"Day {int(day)}", fontsize=18, weight="bold")
        ax.set_xlabel("Threshold (mm/24h)", fontsize=15)
        ax.set_ylabel("% of samples (log)", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.25, which="major")
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    fig.suptitle(f"Hits vs. false alarms by lead time — {fc1_name} vs {fc2_name}",
                 fontsize=22, weight="bold", y=1.02)
    fig.legend(*handles_labels, loc="lower center", ncol=4, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    out = Path(out_dir) / "3b_contingency_hits_fa_presentation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})
    print(f"  [3b] Presentation-style hits/FA → {out.name}")


# ─── Figure 4 — POD bars + ratio line by threshold, one subplot per day ─────

def fig_hits_pod_and_ratio(tbl3, fc1_label, fc2_label, out_dir, days=(1, 3, 5, 7)):
    """Combined bars + line, one subplot per forecast day, threshold on the
    x-axis. Replaces the earlier design (one line per threshold, 2 side-by-
    side panels), which became cluttered with 7 threshold lines per panel.
    Faceting by day and using threshold as the categorical x-axis needs only
    2 bars + 1 line per subplot, regardless of how many thresholds are swept.

      bars (left y-axis, log scale) : POD per model = hits/(hits+misses), %
                                       — the absolute hit-rate context.
      line (right y-axis)           : ratio_hits (fc2/fc1) — the relative
                                       comparison, with a dashed line at 1
                                       (parity).
    """
    available_days = sorted(tbl3["day"].unique())
    days = [d for d in days if d in available_days]
    thresholds = sorted(tbl3["threshold"].unique())
    x = np.arange(len(thresholds))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    bar_handles_labels = None
    line_handle = None

    for ax_bar, day in zip(axes, days):
        sub = tbl3[tbl3["day"] == day].set_index("threshold").loc[thresholds]
        pod1 = 100 * sub.hits_ifs / (sub.hits_ifs + sub.misses_ifs)
        pod2 = 100 * sub.hits_aifs / (sub.hits_aifs + sub.misses_aifs)

        ax_bar.bar(x - width / 2, pod1, width, color=C_IFS, alpha=0.85, label=f"POD {fc1_label}")
        ax_bar.bar(x + width / 2, pod2, width, color=C_AIFS, alpha=0.85, label=f"POD {fc2_label}")
        ax_bar.set_yscale("log")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"{t:.0f}" for t in thresholds])
        ax_bar.set_xlabel("Threshold (mm/24h)")
        ax_bar.set_ylabel("POD (%, log scale)")
        ax_bar.set_title(f"Day {int(day)}", fontsize=13, weight="bold")
        ax_bar.grid(alpha=0.25, axis="y", which="both")

        ax_line = ax_bar.twinx()
        ax_line.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
        line, = ax_line.plot(x, sub.ratio_hits.values, color="black", marker="o",
                              lw=2, label=f"Ratio ({fc2_label}/{fc1_label})")
        ax_line.set_ylabel(f"Hits ratio ({fc2_label}/{fc1_label})")

        if bar_handles_labels is None:
            bar_handles_labels = ax_bar.get_legend_handles_labels()
            line_handle = line

    handles = bar_handles_labels[0] + [line_handle]
    labels = bar_handles_labels[1] + [line_handle.get_label()]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Absolute hit rate (bars) + relative ratio (line) by threshold\n"
                 f"{fc1_label} vs {fc2_label}", fontsize=15, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "4_hits_pod_and_ratio_by_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4] POD bars + ratio line by day → {out.name}")


# ─── Figure 5 — FAR bars + ratio line by threshold, one subplot per day ─────

def fig_fa_and_ratio(tbl3, fc1_label, fc2_label, out_dir, days=(1, 3, 5, 7)):
    """Same design as Figure 4 but for false alarms:
      bars (left y-axis, log scale) : FAR per model = FA/(hits+FA), % —
                                       the standard False Alarm Ratio, giving
                                       the absolute false-alarm context.
      line (right y-axis)           : ratio_fa (fc2/fc1) — the relative
                                       comparison, with a dashed line at 1
                                       (parity).
    """
    available_days = sorted(tbl3["day"].unique())
    days = [d for d in days if d in available_days]
    thresholds = sorted(tbl3["threshold"].unique())
    x = np.arange(len(thresholds))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    bar_handles_labels = None
    line_handle = None

    for ax_bar, day in zip(axes, days):
        sub = tbl3[tbl3["day"] == day].set_index("threshold").loc[thresholds]
        far1 = 100 * sub.fa_ifs / (sub.hits_ifs + sub.fa_ifs)
        far2 = 100 * sub.fa_aifs / (sub.hits_aifs + sub.fa_aifs)

        ax_bar.bar(x - width / 2, far1, width, color=C_IFS, alpha=0.85, label=f"FAR {fc1_label}")
        ax_bar.bar(x + width / 2, far2, width, color=C_AIFS, alpha=0.85, label=f"FAR {fc2_label}")
        ax_bar.set_yscale("log")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"{t:.0f}" for t in thresholds])
        ax_bar.set_xlabel("Threshold (mm/24h)")
        ax_bar.set_ylabel("FAR (%, log scale)")
        ax_bar.set_title(f"Day {int(day)}", fontsize=13, weight="bold")
        ax_bar.grid(alpha=0.25, axis="y", which="both")

        ax_line = ax_bar.twinx()
        ax_line.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
        line, = ax_line.plot(x, sub.ratio_fa.values, color="black", marker="o",
                              lw=2, label=f"Ratio ({fc2_label}/{fc1_label})")
        ax_line.set_ylabel(f"False-alarm ratio ({fc2_label}/{fc1_label})")

        if bar_handles_labels is None:
            bar_handles_labels = ax_bar.get_legend_handles_labels()
            line_handle = line

    handles = bar_handles_labels[0] + [line_handle]
    labels = bar_handles_labels[1] + [line_handle.get_label()]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"False Alarm Ratio (bars) + relative ratio (line) by threshold\n"
                 f"{fc1_label} vs {fc2_label}", fontsize=15, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "5_fa_and_ratio_by_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] FAR bars + ratio line by day → {out.name}")


# ─── Figure 6 — overprediction among hits ────────────────────────────────────

def fig_hit_overprediction(data, thresholds, fc1_label, fc2_label, out_dir,
                            days=(1, 3, 5, 7, 10)):
    """Among events BOTH models correctly detect (hits), shows how often and
    by how much each model overshoots the true magnitude. This is invisible
    to ETS/PSS (a hit is a hit regardless of overshoot size) but is exactly
    what twMAE penalises via |fc-obs| — the mechanism behind twMAE showing a
    consistent AIFS advantage even at thresholds where ETS/PSS do not.

      top row    : % of hits where fc > obs (bars, per model)
      bottom row : mean overshoot size among those cases (bars, per model)
      one column per forecast day
    """
    available_days = sorted(data["forecast_day"].unique())
    days = [d for d in days if d in available_days]
    x = np.arange(len(thresholds))
    width = 0.35
    rows = []

    fig, axes = plt.subplots(2, len(days), figsize=(4.2 * len(days), 8), sharex="col")

    for col, day in enumerate(days):
        grp = data[data["forecast_day"] == day]
        obs = grp["obs_value"].values
        fc1 = grp["fc1_value"].values
        fc2 = grp["fc2_value"].values

        frac1s, mae1s, frac2s, mae2s = [], [], [], []
        for T in thresholds:
            f1, m1 = hit_overprediction_stats(obs, fc1, T)
            f2, m2 = hit_overprediction_stats(obs, fc2, T)
            frac1s.append(100 * f1); mae1s.append(m1)
            frac2s.append(100 * f2); mae2s.append(m2)
            rows.append(dict(day=day, threshold=T,
                              frac_overshoot_ifs=100 * f1, overshoot_mm_ifs=m1,
                              frac_overshoot_aifs=100 * f2, overshoot_mm_aifs=m2))

        ax_top = axes[0, col]
        ax_top.bar(x - width / 2, frac1s, width, color=C_IFS, alpha=0.85, label=fc1_label)
        ax_top.bar(x + width / 2, frac2s, width, color=C_AIFS, alpha=0.85, label=fc2_label)
        ax_top.set_xticks(x)
        ax_top.set_title(f"Day {int(day)}", fontsize=12, weight="bold")
        ax_top.grid(alpha=0.25, axis="y")
        if col == 0:
            ax_top.set_ylabel("Hits that overshoot obs (%)")
            ax_top.legend(fontsize=8)

        ax_bot = axes[1, col]
        ax_bot.bar(x - width / 2, mae1s, width, color=C_IFS, alpha=0.85)
        ax_bot.bar(x + width / 2, mae2s, width, color=C_AIFS, alpha=0.85)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([f"{t:.0f}" for t in thresholds])
        ax_bot.set_xlabel("Threshold (mm/24h)")
        ax_bot.grid(alpha=0.25, axis="y")
        if col == 0:
            ax_bot.set_ylabel("Mean overshoot size (mm)")

    fig.suptitle(f"Overprediction among correctly-detected events (hits)\n"
                 f"{fc1_label} vs {fc2_label}", fontsize=15, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "6_hit_overprediction_by_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6] Hit overprediction by day → {out.name}")
    return pd.DataFrame(rows)


# ─── Figure 6b — same result as a day x threshold heatmap ───────────────────

def fig_hit_overprediction_heatmap(tbl6, fc1_label, fc2_label, out_dir):
    """Alternative view of Figure 6: two annotated heatmaps (day x threshold),
    showing IFS's excess over AIFS directly as a single number per cell —
    the same day/threshold grid layout as the production scorecards, so it
    reads the same way a scorecard does instead of needing 8 small bar charts.

      left  : IFS - AIFS, % of hits that overshoot obs (percentage points)
      right : IFS - AIFS, mean overshoot size among those hits (mm)

    Uses a DIVERGING colormap centred at zero (red = fc1 overshoots more,
    blue = fc2 overshoots more) so that cells where AIFS overshoots more are
    visible as blue rather than being indistinguishable near-white values on
    a sequential colormap. Missing/undersampled cells (NaN, <5 hits) are
    masked and drawn in a distinct grey so they cannot be misread as "AIFS
    wins here".
    """
    days = sorted(tbl6["day"].unique())
    thresholds = sorted(tbl6["threshold"].unique())

    diff_frac = np.full((len(days), len(thresholds)), np.nan)
    diff_mae = np.full((len(days), len(thresholds)), np.nan)
    for i, day in enumerate(days):
        for j, T in enumerate(thresholds):
            row = tbl6[(tbl6["day"] == day) & (tbl6["threshold"] == T)]
            if len(row) == 0:
                continue
            diff_frac[i, j] = row["frac_overshoot_ifs"].iloc[0] - row["frac_overshoot_aifs"].iloc[0]
            diff_mae[i, j] = row["overshoot_mm_ifs"].iloc[0] - row["overshoot_mm_aifs"].iloc[0]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mat, title, unit in [
        (ax1, diff_frac, "Overshoot frequency", "pp"),
        (ax2, diff_mae, "Overshoot magnitude", "mm"),
    ]:
        masked = np.ma.masked_invalid(mat)
        vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{t:.0f}" for t in thresholds])
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels([f"Day {int(d)}" for d in days])
        ax.set_xlabel("Threshold (mm/24h)")
        ax.set_title(f"{title} — {fc1_label} minus {fc2_label} ({unit})\n"
                      f"(red = {fc1_label} overshoots more, blue = {fc2_label} overshoots more, "
                      f"grey = too few hits)", fontsize=11, weight="bold")
        for i in range(len(days)):
            for j in range(len(thresholds)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > vmax * 0.55 else "black")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"Where {fc1_label} vs {fc2_label} overshoots correctly-detected events more",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    out = Path(out_dir) / "6b_hit_overprediction_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6b] Hit overprediction heatmap → {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config(args.config)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    out_dir = Path(args.output_dir or "results/intensity_vs_occurrence_tp24")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading native-resolution tp24 data (orog={args.orog or 'all'}) ...")
    data, fc1_name, fc2_name = load_all_days(config, orog=args.orog)
    print(f"  {len(data):,} rows across forecast days {sorted(data['forecast_day'].unique())}")

    # Raw names (fc1_name/fc2_name) are used only for data loading above;
    # everything plotted uses the short display labels (e.g. "IFS"/"AIFS").
    fc1_label, fc2_label = display_name(fc1_name), display_name(fc2_name)

    tbl1 = fig_occurrence_vs_intensity(data, thresholds, fc1_label, fc2_label, out_dir)
    tbl2 = fig_discrimination_vs_intensity(data, args.ref_threshold, fc1_label, fc2_label, out_dir)
    tbl3 = fig_contingency_by_day(data, thresholds, fc1_label, fc2_label, out_dir)
    fig_contingency_presentation(tbl3, fc1_label, fc2_label, out_dir)
    fig_hits_pod_and_ratio(tbl3, fc1_label, fc2_label, out_dir)
    fig_fa_and_ratio(tbl3, fc1_label, fc2_label, out_dir)
    tbl6 = fig_hit_overprediction(data, thresholds, fc1_label, fc2_label, out_dir)
    fig_hit_overprediction_heatmap(tbl6, fc1_label, fc2_label, out_dir)

    tbl1.to_csv(out_dir / "occurrence_vs_intensity_by_threshold.csv", index=False)
    tbl2.to_csv(out_dir / "discrimination_vs_intensity_bias_by_day.csv", index=False)
    tbl3.to_csv(out_dir / "contingency_hits_misses_fa_by_day.csv", index=False)
    tbl6.to_csv(out_dir / "hit_overprediction_by_threshold.csv", index=False)
    print(f"\n✓ Saved tables + figures to {out_dir}/")


if __name__ == "__main__":
    main()
