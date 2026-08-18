#!/usr/bin/env python3
"""
analyse_orography_position_ensemble.py — ensemble-mode counterpart of
analyse_orography_position.py.

Motivation
----------
The deterministic tool (analyse_orography_position.py) found that AIFS's
degradation relative to IFS in complex terrain is driven mainly by VALLEY
stations (AIFS warm bias), not mountain-top stations (where AIFS is actually
better than IFS's cold bias). This script checks whether the SAME mechanism
explains an analogous "AIFS-ENS worse than IFS-ENS in mid/high terrain from a
certain lead day" pattern seen in an ensemble twMAE scorecard.

Why a separate script (not just reusing analyse_orography_position.py)
------------------------------------------------------------------------
Ensemble parquets differ from deterministic ones in two ways that break the
deterministic tool's direct re-use:
  1. No single forecast value: use the ensemble MEAN across fc{1,2}_member_*
     columns as the model's central value (matches the pipeline's own
     `ens_mean_bias`/`ens_mean_mae` scores in src/ens_scores.py).
  2. No obs_height/fc1_height/fc2_height columns: the ensemble extractor
     applies the lapse-rate correction per-member at extraction time and
     does not retain height columns. This script recomputes the SAME signed
     elevation anomaly Δh = obs_height − model_orography_height directly via
     Metview (mv.elevations() on one observation geopoints file — station
     elevation is static — plus mv.nearest_gridpoint() on the config's
     auxiliary_fields orography GRIB, exactly mirroring
     extract_points_ensemble.py's own height computation), then joins it onto
     the ensemble parquet rows by rounded (lat, lon).

Usage
-----
  python analyse_orography_position_ensemble.py --config \\
      configs/ensemble/config_2t_ens_local_p99obsclim_aifsvsifs_commonperiod.yaml \\
      --days 1,3,6,10 --output-dir case_study_output/orog_position_ens_2t_aifsvsifs
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from analyse_orography_position import (      # noqa: E402
    classify, _stats, _fig_error_vs_dh, _fig_class_bias, _fig_class_map,
    _print_headline, CLASS_ORDER, CLASS_COLORS, SEASON_MONTHS,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
from case_studies.case_study_utils import (   # noqa: E402
    load_per_station_thresholds, extract_forecast_values,
)

# 4th category: classify() already labels anything that's neither valley,
# flat, nor mountain_top as "mid" (moderate |Δh| terrain, or low |Δh| but too
# rough in sdfor to count as flat) — include it explicitly instead of leaving
# it invisible in the per-class breakdown.
ALL_CLASSES = CLASS_ORDER + ["mid"]
ALL_CLASS_COLORS = dict(CLASS_COLORS, mid="#888888")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config file (mode: ensemble)")
    p.add_argument("--season", default=None, help="DJF | MAM | JJA | SON")
    p.add_argument("--output-dir", default=None, dest="output_dir")
    p.add_argument("--height-ref", default="mean", choices=["mean", "ifs", "aifs"],
                   help="Reference orography for classification (default: mean).")
    p.add_argument("--valley-dh", type=float, default=-75.0,
                   help="Δh (m) at/below which a station is a valley (default -75).")
    p.add_argument("--peak-dh", type=float, default=75.0,
                   help="Δh (m) at/above which a station is a mountain top (default +75).")
    p.add_argument("--flat-dh", type=float, default=40.0,
                   help="|Δh| (m) below which a station may be flat (default 40).")
    p.add_argument("--flat-sdfor", type=float, default=40.0,
                   help="sdfor (m) below which a low-|Δh| station is flat (default 40).")
    p.add_argument("--extremes", action="store_true",
                   help="Additionally stratify to obs beyond the per-station threshold.")
    p.add_argument("--keep-coastal", action="store_true",
                   help="Do NOT remove coastal stations (default: apply config lsm filter).")
    p.add_argument("--days", default="1,3,6,10",
                   help="Comma-separated forecast days (default: 1,3,6,10).")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_production_qc_filters(df, config):
    """Apply the SAME row-level QC filters src/run.py's ensemble streaming loop
    applies (2t physical-bounds QC, then the ±N-sigma outlier filter on
    obs_value), BEFORE thresholding/aggregation. Mirrors run.py lines ~565-585
    exactly so results are comparable to the production scorecard."""
    cfg_f = config.get("filter", {})
    if config.get("variable") == "2t":
        min_t = cfg_f.get("min_valid_temperature", -60.0)
        max_t = cfg_f.get("max_valid_temperature", 60.0)
        vm = (df["obs_value"] >= min_t) & (df["obs_value"] <= max_t)
        for c in df.columns:
            if c.startswith("fc1_member_") or c.startswith("fc2_member_"):
                vm &= (df[c] >= min_t) & (df[c] <= max_t)
        df = df[vm].reset_index(drop=True)
    if cfg_f.get("remove_outliers", False) and config.get("variable") != "tp24":
        std = cfg_f.get("outlier_threshold_std", 5.0)
        m, s = df["obs_value"].mean(), df["obs_value"].std()
        df = df[(df["obs_value"] - m).abs() <= std * s].reset_index(drop=True)
    return df


def _aggregate_to_daily_mean(data, threshold_value, config):
    """EXACT copy of src/run.py's `_aggregate_to_daily_mean` (duplicated here to
    avoid importing the full run.py/Metview import chain). Production applies
    this AFTER computing the per-station threshold and BEFORE scoring, for any
    `local_obs_climatology` config with sub-daily (< 24h) lead_time_frequency:
    it averages each station's members/obs across the multiple sub-daily steps
    that fall on the same calendar forecast_day. Without this step, twCRPS is
    computed on un-aggregated sub-daily samples — a DIFFERENT (and, for cold
    extremes, sign-flipping) quantity from what the real scorecard verifies.
    """
    data = data.copy()
    data["forecast_day"] = ((data["step"] - 1) // 24).astype(int) + 1
    group_cols = ["lat", "lon", "date", "forecast_day"]

    agg_cols = ["obs_value"] + [c for c in data.columns
                                if c.startswith("fc1_member_") or c.startswith("fc2_member_")]
    agg_cols = [c for c in agg_cols if c in data.columns]
    meta_cols = [c for c in data.columns
                 if c not in agg_cols and c not in group_cols and c != "step"]

    agg_dict = {col: (col, "mean") for col in agg_cols}
    agg_dict["step"] = ("step", "mean")
    for col in meta_cols:
        agg_dict[col] = (col, "first")

    agg = data.groupby(group_cols, sort=False).agg(**agg_dict).reset_index()

    freq = int(config.get("lead_time_frequency", 6)) if config else 6
    agg["step"] = (agg["forecast_day"] * 24 - (24 - freq) // 2).astype(int)

    if isinstance(threshold_value, pd.Series):
        thr_df = data[group_cols].copy()
        thr_df["_thr"] = threshold_value.values
        thr_map = thr_df.groupby(group_cols)["_thr"].first().reset_index()
        agg = agg.merge(thr_map, on=group_cols, how="left")
        new_threshold = agg["_thr"].values
        agg = agg.drop(columns=["_thr"])
    else:
        new_threshold = threshold_value

    agg = agg.reset_index(drop=True)
    print(f"    Daily aggregation: {len(data):,} sub-daily rows -> {len(agg):,} daily rows")
    return agg, new_threshold


def _keys(lats, lons):
    return (pd.Series(np.asarray(lats)).round(3).astype(str) + "_"
            + pd.Series(np.asarray(lons)).round(3).astype(str)).values


def _ens_spread(df, prefix):
    """Per-row ensemble spread (std across members, ddof=0) — matches
    src/ens_scores.py's `ens_spread` score definition."""
    cols = [c for c in df.columns if c.startswith(f"{prefix}_member_")]
    return df[cols].std(axis=1, ddof=0).values.astype(np.float64)


def _twcrps_fair(fc, obs, thr, event_type):
    """Per-row fair (tail-weighted) CRPS — IDENTICAL formula to
    src/ens_scores.py's _fair_crps_numpy + threshold-chaining, so results are
    directly comparable to the pipeline's own twCRPS score. `fc` is the raw
    (n_rows, n_members) member array for ONE model; `thr`/`obs` are (n_rows,).
    """
    m = fc.shape[1]
    if event_type == "below":
        fc_v = np.minimum(fc, thr[:, None])
        obs_v = np.minimum(obs, thr)
    else:
        fc_v = np.maximum(fc, thr[:, None])
        obs_v = np.maximum(obs, thr)
    term1 = np.mean(np.abs(fc_v - obs_v[:, None]), axis=1)
    s = np.sort(fc_v, axis=1)
    w = 2 * np.arange(m) - m + 1
    term2 = (s * w[None, :]).sum(axis=1) / (m * (m - 1))
    return term1 - term2


def build_station_dh_lookup(config, args):
    """dict {station_key: dh} via Metview, mirroring extract_points_ensemble.py's
    own height computation (mv.elevations for obs, mv.nearest_gridpoint on the
    auxiliary orography GRIB for model heights)."""
    import metview as mv

    variable = config["variable"]
    obs_path = Path(config["read_data"]["local_gpt"]["path"])
    start = pd.Timestamp(config["start_date"])
    candidate = None
    for offset in range(0, 14):
        d = start + pd.Timedelta(days=offset)
        f = obs_path / f"{variable}_obs_{d:%Y%m%d}00.geo"
        if f.exists():
            candidate = f
            break
    if candidate is None:
        raise FileNotFoundError(
            f"No {variable}_obs_*00.geo file found near {start:%Y%m%d} in {obs_path}")
    print(f"  Station elevations from: {candidate.name} (elevation is static per station)")

    obs_gpt = mv.read(str(candidate))
    obs_lats = np.array(mv.latitudes(obs_gpt))
    obs_lons = np.array(mv.longitudes(obs_gpt))
    obs_heights = np.array(mv.elevations(obs_gpt))

    cfg_aux = config.get("auxiliary_fields", {})
    orog1_path = cfg_aux.get("model1", {}).get("orog_path")
    orog2_path = cfg_aux.get("model2", {}).get("orog_path")
    fc1_h = np.array(mv.nearest_gridpoint(mv.read(orog1_path) / 9.80665,
                                          obs_lats, obs_lons)).flatten()
    fc2_h = np.array(mv.nearest_gridpoint(mv.read(orog2_path) / 9.80665,
                                          obs_lats, obs_lons)).flatten()

    if args.height_ref == "ifs":
        ref_h = fc1_h
    elif args.height_ref == "aifs":
        ref_h = fc2_h
    else:
        ref_h = 0.5 * (fc1_h + fc2_h)

    dh = obs_heights - ref_h
    lookup = dict(zip(_keys(obs_lats, obs_lons), dh))
    print(f"  Built Δh lookup for {len(lookup):,} stations (ref={args.height_ref})")
    return lookup


def _fig_class_bias_extreme(summ, m1, m2, out_dir, args, classes=ALL_CLASSES,
                             filename="4_class_bias_extreme.png", title_suffix=""):
    """Mean bias per position class vs lead time, EXTREME cases only
    (obs beyond the per-station threshold) — same style as the deterministic
    tool's _fig_class_bias, which only ever plots subset=='all'."""
    a = summ[summ.subset == "extreme"]
    if a.empty:
        return
    days = sorted(a.day.unique())
    x = np.arange(len(days))
    n_per_class = {c: int(np.nanmax(a[a.orog_class == c].n_stations.values))
                   if (a.orog_class == c).any() else 0 for c in classes}

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for c in classes:
        col = ALL_CLASS_COLORS[c]
        sub = a[a.orog_class == c].set_index("day").reindex(days)
        ax.plot(x, sub.bias_ifs, ls="--", color=col, lw=1.8, marker="o",
                ms=8, mfc="white", mec=col, mew=1.8, zorder=3)
        ax.plot(x, sub.bias_aifs, ls="-", color=col, lw=2.6, marker="s",
                ms=8, mfc=col, mec="k", mew=0.6, zorder=4)

    ax.axhline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f"day {d}" for d in days], fontsize=11)
    ax.set_ylabel("Mean bias  fc − obs  (°C) — EXTREME cases (obs ≥ per-station threshold)",
                  fontsize=11)
    ax.set_xlabel("Forecast lead time", fontsize=12)
    ax.grid(True, alpha=0.3)

    class_handles = [
        Patch(facecolor=ALL_CLASS_COLORS[c], edgecolor="k", lw=0.4,
              label=f"{c.replace('_', ' ')}  (n≈{n_per_class[c]:,} stations)")
        for c in classes]
    model_handles = [
        Line2D([0], [0], color="0.25", ls="--", lw=1.8, marker="o", ms=8,
               mfc="white", mec="0.25", mew=1.8, label=f"{m1}  (IFS)"),
        Line2D([0], [0], color="0.25", ls="-", lw=2.6, marker="s", ms=8,
               mfc="0.25", mec="k", mew=0.6, label=f"{m2}  (AIFS)")]
    leg1 = ax.legend(handles=class_handles, title="Topographic class",
                     fontsize=10, title_fontsize=10, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0., framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=model_handles, title="Model", fontsize=10,
              title_fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 0.45),
              borderaxespad=0., framealpha=0.95)

    ax.set_title(f"Mean bias by topographic position — EXTREME cases — "
                 f"{m1} vs {m2}{title_suffix} ({args.season or 'all seasons'})",
                 fontsize=12, weight="bold")
    fig.subplots_adjust(right=0.72)
    out = out_dir / filename
    fig.savefig(out, dpi=160, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"  [4] extreme-case class bias → {out.name}")


def _fig_class_spread(summ, m1, m2, out_dir, args, subset="all",
                       classes=ALL_CLASSES, filename=None, title_suffix=""):
    """Ensemble spread (std across members) per position class vs lead time —
    same visual style as the bias plots, but plotting spread_ifs/spread_aifs."""
    a = summ[summ.subset == subset]
    if a.empty:
        return
    days = sorted(a.day.unique())
    x = np.arange(len(days))
    n_per_class = {c: int(np.nanmax(a[a.orog_class == c].n_stations.values))
                   if (a.orog_class == c).any() else 0 for c in classes}

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for c in classes:
        col = ALL_CLASS_COLORS[c]
        sub = a[a.orog_class == c].set_index("day").reindex(days)
        ax.plot(x, sub.spread_ifs, ls="--", color=col, lw=1.8, marker="o",
                ms=8, mfc="white", mec=col, mew=1.8, zorder=3)
        ax.plot(x, sub.spread_aifs, ls="-", color=col, lw=2.6, marker="s",
                ms=8, mfc=col, mec="k", mew=0.6, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels([f"day {d}" for d in days], fontsize=11)
    subset_lbl = " — EXTREME cases (obs ≥ per-station threshold)" if subset == "extreme" else ""
    ax.set_ylabel(f"Ensemble spread  std(members)  (°C){subset_lbl}", fontsize=11)
    ax.set_xlabel("Forecast lead time", fontsize=12)
    ax.grid(True, alpha=0.3)

    class_handles = [
        Patch(facecolor=ALL_CLASS_COLORS[c], edgecolor="k", lw=0.4,
              label=f"{c.replace('_', ' ')}  (n≈{n_per_class[c]:,} stations)")
        for c in classes]
    model_handles = [
        Line2D([0], [0], color="0.25", ls="--", lw=1.8, marker="o", ms=8,
               mfc="white", mec="0.25", mew=1.8, label=f"{m1}  (IFS)"),
        Line2D([0], [0], color="0.25", ls="-", lw=2.6, marker="s", ms=8,
               mfc="0.25", mec="k", mew=0.6, label=f"{m2}  (AIFS)")]
    leg1 = ax.legend(handles=class_handles, title="Topographic class",
                     fontsize=10, title_fontsize=10, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0., framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=model_handles, title="Model", fontsize=10,
              title_fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 0.45),
              borderaxespad=0., framealpha=0.95)

    ax.set_title(f"Ensemble spread by topographic position{title_suffix} — "
                 f"{m1} vs {m2} ({args.season or 'all seasons'})",
                 fontsize=12, weight="bold")
    fig.subplots_adjust(right=0.72)
    fname = filename or f"6_class_spread_{subset}.png"
    out = out_dir / fname
    fig.savefig(out, dpi=160, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"  [6] {subset}-case class spread → {out.name}")


def _fig_class_twcrps(summ, m1, m2, out_dir, args, subset="all",
                      classes=ALL_CLASSES, filename=None, title_suffix=""):
    """twCRPS per position class vs lead time — the ACTUAL scorecard metric
    (fair, tail-weighted CRPS), decomposed by valley/flat/mid/mountain-top
    instead of the pipeline's sdfor-only low/mid/high bins."""
    a = summ[summ.subset == subset]
    if a.empty:
        return
    days = sorted(a.day.unique())
    x = np.arange(len(days))
    n_per_class = {c: int(np.nanmax(a[a.orog_class == c].n_stations.values))
                   if (a.orog_class == c).any() else 0 for c in classes}

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for c in classes:
        col = ALL_CLASS_COLORS[c]
        sub = a[a.orog_class == c].set_index("day").reindex(days)
        ax.plot(x, sub.twcrps_ifs, ls="--", color=col, lw=1.8, marker="o",
                ms=8, mfc="white", mec=col, mew=1.8, zorder=3)
        ax.plot(x, sub.twcrps_aifs, ls="-", color=col, lw=2.6, marker="s",
                ms=8, mfc=col, mec="k", mew=0.6, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels([f"day {d}" for d in days], fontsize=11)
    subset_lbl = " — EXTREME cases (obs ≥ per-station threshold)" if subset == "extreme" else ""
    ax.set_ylabel(f"twCRPS (lower = better){subset_lbl}", fontsize=11)
    ax.set_xlabel("Forecast lead time", fontsize=12)
    ax.grid(True, alpha=0.3)

    class_handles = [
        Patch(facecolor=ALL_CLASS_COLORS[c], edgecolor="k", lw=0.4,
              label=f"{c.replace('_', ' ')}  (n≈{n_per_class[c]:,} stations)")
        for c in classes]
    model_handles = [
        Line2D([0], [0], color="0.25", ls="--", lw=1.8, marker="o", ms=8,
               mfc="white", mec="0.25", mew=1.8, label=f"{m1}  (IFS)"),
        Line2D([0], [0], color="0.25", ls="-", lw=2.6, marker="s", ms=8,
               mfc="0.25", mec="k", mew=0.6, label=f"{m2}  (AIFS)")]
    leg1 = ax.legend(handles=class_handles, title="Topographic class",
                     fontsize=10, title_fontsize=10, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0., framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=model_handles, title="Model", fontsize=10,
              title_fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 0.45),
              borderaxespad=0., framealpha=0.95)

    ax.set_title(f"twCRPS by topographic position{title_suffix} — "
                 f"{m1} vs {m2} ({args.season or 'all seasons'})",
                 fontsize=12, weight="bold")
    fig.subplots_adjust(right=0.72)
    fname = filename or f"8_class_twcrps_{subset}.png"
    out = out_dir / fname
    fig.savefig(out, dpi=160, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"  [8] {subset}-case class twCRPS → {out.name}")


def main():
    args = parse_args()
    config = load_config(args.config)
    if config.get("mode") != "ensemble":
        print("ERROR: this tool is for mode: ensemble configs — use "
              "analyse_orography_position.py for deterministic configs.")
        sys.exit(1)

    m1 = config["read_data"]["forecast_model1"]["name"]
    m2 = config["read_data"]["forecast_model2"]["name"]
    parquet_dir = Path(config["extract_points"]["output_path"])
    fcfg = config.get("filter", {})
    lsm_thr = fcfg.get("coastal_lsm_threshold", 0.9)
    remove_coastal = fcfg.get("remove_coastal_stations", False) and not args.keep_coastal
    days = [int(d) for d in args.days.split(",")]

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("case_study_output") /
        f"orog_position_ens_{args.season or 'all'}_{m1}_vs_{m2}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}\n  ENSEMBLE OROGRAPHIC-POSITION SKILL ANALYSIS")
    print(f"  Config : {args.config}")
    print(f"  Models : {m1} (fc1) vs {m2} (fc2)  [ensemble-MEAN forecast]")
    print(f"  Season : {args.season or 'all'}   Days: {days}")
    print(f"  Δh ref : {args.height_ref}   valley<= {args.valley_dh}  "
          f"peak>= {args.peak_dh}  flat |Δh|<{args.flat_dh} & sdfor<{args.flat_sdfor}")
    print(f"  Coastal removal: {remove_coastal} (lsm>= {lsm_thr})")
    print(f"  Output : {out_dir}\n{'='*72}\n")

    dh_lookup = build_station_dh_lookup(config, args)

    DH_EDGES = np.array([-1e9, -600, -400, -250, -150, -75, -25,
                         25, 75, 150, 250, 400, 600, 1e9])
    DH_CENTERS = np.array([-700, -500, -325, -200, -112, -50, 0,
                           50, 112, 200, 325, 500, 700])
    nb = len(DH_CENTERS)
    acc = {m: {"n": np.zeros(nb), "sum_e": np.zeros(nb), "sum_ae": np.zeros(nb)}
           for m in ("fc1", "fc2")}

    summary_rows = []
    station_map = {}

    for day in days:
        cands = list(parquet_dir.glob(f"*_day{day}.parquet"))
        if not cands:
            print(f"  day {day}: no parquet found — skipping")
            continue

        all_cols = pq.ParquetFile(cands[0]).schema.names
        fc1_cols = [c for c in all_cols if c.startswith("fc1_member_")]
        fc2_cols = [c for c in all_cols if c.startswith("fc2_member_")]
        member_cols = fc1_cols + fc2_cols
        # station_id is required by threshold.py's local_obs_climatology matching;
        # without it, load_per_station_thresholds silently falls back to a
        # pooled percentile. Thresholds are needed unconditionally now (twCRPS
        # is inherently threshold-weighted, not just the --extremes subset).
        # 'step' is required for (a) valid-time-month threshold matching and
        # (b) the daily-mean aggregation below — both match production exactly.
        base_cols = [c for c in ("date", "step", "lat", "lon", "sdfor", "lsm",
                                 "obs_value", "station_id")
                    if c in all_cols]
        df = pd.read_parquet(cands[0], columns=base_cols + member_cols)

        if args.season in SEASON_MONTHS and "date" in df.columns:
            months = SEASON_MONTHS[args.season]
            df = df[df["date"].astype(str).str[4:6].astype(int).isin(months)]
        if remove_coastal and "lsm" in df.columns:
            df = df[df["lsm"] >= lsm_thr]
        df = df.dropna(subset=["obs_value", "lat", "lon"])
        if df.empty:
            print(f"  day {day}: empty after filters — skipping")
            continue

        df = _apply_production_qc_filters(df, config)
        if df.empty:
            print(f"  day {day}: empty after QC filters — skipping")
            continue

        # Per-station threshold + event type computed on SUB-DAILY rows (as
        # production does), THEN daily-mean aggregation (production's
        # `_aggregate_to_daily_mean`) — averaging each station's members/obs
        # across the sub-daily steps within the same calendar forecast_day.
        # Skipping this step (as the earlier version of this script did)
        # scores a fundamentally different, un-aggregated quantity and can
        # flip the sign of the model comparison for cold extremes.
        T_sub = load_per_station_thresholds(config, df)
        event = config.get("threshold", {}).get("event_type", "above")
        if "step" in df.columns and config.get("threshold", {}).get("method") == "local_obs_climatology":
            df, T = _aggregate_to_daily_mean(df, pd.Series(T_sub, index=df.index), config)
        else:
            df, T = df, T_sub

        obs = df["obs_value"].values.astype(np.float64)
        fc1, fc2 = extract_forecast_values(df, mode="ensemble")
        fc1 = fc1.astype(np.float64)
        fc2 = fc2.astype(np.float64)
        fc1_np = df[fc1_cols].values.astype(np.float64)
        fc2_np = df[fc2_cols].values.astype(np.float64)
        spread1 = _ens_spread(df, "fc1")
        spread2 = _ens_spread(df, "fc2")

        stn_key = _keys(df["lat"].values, df["lon"].values)
        dh = np.array([dh_lookup.get(k, np.nan) for k in stn_key])
        sdfor = df["sdfor"].values.astype(np.float64)
        valid = np.isfinite(dh)
        if (~valid).any():
            print(f"    ({int((~valid).sum()):,} rows with no Δh match dropped)")


        cls = np.full(len(df), "mid", dtype=object)
        cls[valid] = classify(dh[valid], sdfor[valid], args)

        # T/event were already computed above (on sub-daily rows, before the
        # daily-mean aggregation) — reused here unchanged.
        ext_mask = None
        if args.extremes:
            with np.errstate(invalid="ignore"):
                ext_mask = (obs <= T) if event == "below" else (obs >= T)
            ext_mask = ext_mask & np.isfinite(T)

        for c in ALL_CLASSES:
            cm = (cls == c) & valid
            n_st = np.unique(stn_key[cm]).size
            for tag, mask in ([("all", cm)] +
                              ([("extreme", cm & ext_mask)] if ext_mask is not None else [])):
                b1, a1, r1 = _stats(fc1[mask] - obs[mask])
                b2, a2, r2 = _stats(fc2[mask] - obs[mask])
                if mask.any() and np.isfinite(T[mask]).any():
                    tw1 = _twcrps_fair(fc1_np[mask], obs[mask], T[mask], event)
                    tw2 = _twcrps_fair(fc2_np[mask], obs[mask], T[mask], event)
                    twcrps_ifs = float(np.nanmean(tw1))
                    twcrps_aifs = float(np.nanmean(tw2))
                else:
                    twcrps_ifs = twcrps_aifs = np.nan
                summary_rows.append(dict(
                    day=day, orog_class=c, subset=tag, n_rows=int(mask.sum()),
                    n_stations=n_st,
                    bias_ifs=b1, mae_ifs=a1, rmse_ifs=r1,
                    bias_aifs=b2, mae_aifs=a2, rmse_aifs=r2,
                    mae_diff_aifs_minus_ifs=(a2 - a1) if (a1 == a1 and a2 == a2) else np.nan,
                    spread_ifs=float(np.mean(spread1[mask])) if mask.any() else np.nan,
                    spread_aifs=float(np.mean(spread2[mask])) if mask.any() else np.nan,
                    twcrps_ifs=twcrps_ifs, twcrps_aifs=twcrps_aifs,
                    twcrps_diff_aifs_minus_ifs=(twcrps_aifs - twcrps_ifs)
                        if (twcrps_ifs == twcrps_ifs and twcrps_aifs == twcrps_aifs) else np.nan))

        for mkey, fc in (("fc1", fc1), ("fc2", fc2)):
            idx = np.digitize(dh, DH_EDGES) - 1
            e = fc - obs
            ae = np.abs(e)
            for b in range(nb):
                bm = valid & (idx == b)
                if bm.any():
                    acc[mkey]["n"][b] += bm.sum()
                    acc[mkey]["sum_e"][b] += e[bm].sum()
                    acc[mkey]["sum_ae"][b] += ae[bm].sum()

        if not station_map:
            _, uniq_idx = np.unique(stn_key, return_index=True)
            lats_v, lons_v = df["lat"].values, df["lon"].values
            for i in uniq_idx:
                if valid[i]:
                    station_map[stn_key[i]] = (lats_v[i], lons_v[i], cls[i])

        counts = {c: int(((cls == c) & valid).sum()) for c in ALL_CLASSES}
        print(f"  day {day}: {len(df):,} rows  |  class rows {counts}")
        del df, obs, fc1, fc2, fc1_np, fc2_np, spread1, spread2, T, dh, sdfor, cls
        gc.collect()

    if not summary_rows:
        print("ERROR: no data produced — check filters / parquet paths / Δh match rate.")
        sys.exit(1)

    summ = pd.DataFrame(summary_rows)
    csv_path = out_dir / "orography_position_summary_ensemble.csv"
    summ.to_csv(csv_path, index=False)
    print(f"\n  ✓ summary table → {csv_path}")

    _print_headline(summ, m1, m2)
    print(f"\n  ── twCRPS by position class (subset=all, the ACTUAL scorecard metric) ──")
    print(f"  {'class':<13}{'day':>4}{'N_stn':>7}{m1[:9]:>12}{m2[:9]:>12}{'AIFS-IFS':>12}")
    a_all = summ[summ.subset == "all"]
    for c in ALL_CLASSES:
        for _, r in a_all[a_all.orog_class == c].sort_values("day").iterrows():
            print(f"  {c:<13}{int(r.day):>4}{int(r.n_stations):>7}"
                  f"{r.twcrps_ifs:>12.5f}{r.twcrps_aifs:>12.5f}"
                  f"{r.twcrps_diff_aifs_minus_ifs:>+12.5f}")
    _fig_error_vs_dh(acc, DH_CENTERS, m1, m2, out_dir, args)
    _fig_class_bias(summ, m1, m2, out_dir, args)
    _fig_class_map(station_map, out_dir, args, m1, m2)
    _fig_class_spread(summ, m1, m2, out_dir, args, subset="all",
                      filename="6_class_spread.png")
    _fig_class_twcrps(summ, m1, m2, out_dir, args, subset="all",
                      filename="8_class_twcrps.png")
    if args.extremes:
        _fig_class_bias_extreme(summ, m1, m2, out_dir, args)
        _fig_class_bias_extreme(
            summ, m1, m2, out_dir, args, classes=["valley", "flat"],
            filename="5_class_bias_extreme_valley_flat.png",
            title_suffix=" (valley vs flat only)")
        _fig_class_spread(summ, m1, m2, out_dir, args, subset="extreme",
                          filename="7_class_spread_extreme.png",
                          title_suffix=" — EXTREME cases")
        _fig_class_twcrps(summ, m1, m2, out_dir, args, subset="extreme",
                          filename="9_class_twcrps_extreme.png",
                          title_suffix=" — EXTREME cases")

    print(f"\n{'='*72}\n  DONE — outputs in {out_dir}\n{'='*72}\n")


if __name__ == "__main__":
    main()
