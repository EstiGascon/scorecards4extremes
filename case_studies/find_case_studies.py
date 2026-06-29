#!/usr/bin/env python3
"""
find_case_studies.py — Identify dates/steps where one model clearly outperforms
the other for extreme weather events.

For each date × forecast-day combination in the parquet files produced by a
scorecards4extremes config, this tool:

  1. Applies per-station thresholds (same method as the main pipeline).
  2. Classifies each station as hit / miss / false_alarm / correct_negative
     for both models.
  3. Computes a rich set of metrics:
       - FA count, FA severity (how far above threshold)
       - Miss count, miss severity
       - Hit intensity error
       - twMAE, POD, FAR, ETS
       - Dominant region of FA/miss events (NW/NE/CE/SW/SE Europe)
  4. Assigns a composite ranking score (high → Model 1 much worse; low → Model 2
     much worse; near zero → similar performance).
  5. Classifies the case type: M1_FALSE_ALARM, M2_MISS_COUNT, etc.

Output
------
  <output_dir>/case_study_ranking_<name>_day<N>.csv  — one file per forecast day
  <output_dir>/case_study_summary_<name>.txt          — top-N cases per day

Usage
-----
  python find_case_studies.py --config path/to/config.yaml [options]

Options
-------
  --config       YAML config file (required)
  --days         Forecast days to analyse, e.g. "1 3 5 7" (default: all found)
  --top-n        How many worst cases to print/show (default: 20)
  --output-dir   Where to save results (default: ./case_study_output)
  --min-stations Minimum stations per date to include (default: 50)
  --no-ensemble-prob  For ensemble: use mean instead of P>0.5 for classification
"""

import argparse
import copy
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from case_studies.case_study_utils import (
    load_per_station_thresholds,
    get_event_type,
    extract_forecast_values,
    extract_exceedance_probability,
    classify_events,
    classify_events_probabilistic,
    compute_date_metrics,
    add_composite_scores,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",        required=True, help="YAML config file")
    p.add_argument("--days",          nargs="+", type=int, default=None,
                   help="Forecast days to analyse (default: all found)")
    p.add_argument("--top-n",         type=int, default=20,
                   help="Top N worst cases to report (default: 20)")
    p.add_argument("--output-dir",    default=None,
                   help="Output directory (default: ./case_study_output)")
    p.add_argument("--min-stations",  type=int, default=50,
                   help="Minimum stations required per date (default: 50)")
    p.add_argument("--min-concentration", type=float, default=None,
                   dest="min_concentration",
                   help="Minimum spatial concentration of the dominant error "
                        "(fraction 0–1). Keeps only dates where the worst "
                        "model's FAs or misses are concentrated in one region. "
                        "E.g. 0.5 requires >50%% of FAs in one sub-region.")
    p.add_argument("--no-ensemble-prob", action="store_true",
                   help="For ensemble: use mean instead of P>0.5")
    p.add_argument("--season",        default=None,
                   help="Filter to season: DJF, MAM, JJA, SON")
    p.add_argument("--orog",          default=None,
                   help="Filter to orography: low, mid, high")
    return p.parse_args()


# ─── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_parquet_dir(config: dict) -> Path:
    return Path(config["extract_points"]["output_path"])


def get_model_names(config: dict) -> tuple[str, str]:
    m1 = config["read_data"]["forecast_model1"]["name"]
    m2 = config["read_data"]["forecast_model2"]["name"]
    return m1, m2


def get_mode(config: dict) -> str:
    return config.get("mode", "deterministic")


# ─── Parquet discovery ─────────────────────────────────────────────────────────

def discover_parquets(parquet_dir: Path, days: list[int] = None
                      ) -> list[tuple[int, Path]]:
    """Return list of (forecast_day, path) sorted by day."""
    patterns = ["*.parquet"]
    found = []
    for pat in patterns:
        found.extend(parquet_dir.glob(pat))
    result = []
    for f in sorted(set(found)):
        # Extract day from filename: *_day{N}.parquet
        name = f.stem
        try:
            day = int(name.split("_day")[-1])
        except ValueError:
            continue
        if days is None or day in days:
            result.append((day, f))
    if not result:
        print(f"  ⚠  No parquet files found in {parquet_dir}")
    return sorted(result)


# ─── Season / orog filter ─────────────────────────────────────────────────────

SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}

OROG_RANGES = {
    "low":  (0,   40),
    "mid":  (40,  120),
    "high": (120, 9999),
}


def apply_filters(df: pd.DataFrame, season: str = None,
                  orog: str = None, config: dict = None) -> pd.DataFrame:
    """Filter rows by season and/or orography type."""
    if season and season in SEASON_MONTHS:
        months = SEASON_MONTHS[season]
        dates = df["date"].astype(str)
        month_nums = dates.str[4:6].astype(int)
        df = df[month_nums.isin(months)]
    if orog and orog in OROG_RANGES:
        # Use config orography ranges if available, else defaults
        if config:
            cfg_ranges = (config.get("filter", {})
                          .get("orography_ranges", OROG_RANGES))
        else:
            cfg_ranges = OROG_RANGES
        lo, hi = cfg_ranges.get(orog, OROG_RANGES[orog])
        if "sdfor" in df.columns:
            df = df[(df["sdfor"] >= lo) & (df["sdfor"] <= hi)]
    return df


# ─── Core processing ───────────────────────────────────────────────────────────

def process_parquet(parquet_path: Path, config: dict, mode: str,
                    model1_name: str, model2_name: str,
                    min_stations: int, use_ensemble_prob: bool,
                    season: str = None, orog: str = None) -> pd.DataFrame:
    """Process one parquet file and return a DataFrame with per-date metrics."""
    print(f"  Loading: {parquet_path.name}")
    df_full = pd.read_parquet(parquet_path)

    # Apply season/orog filters
    df_full = apply_filters(df_full, season=season, orog=orog, config=config)
    if df_full.empty:
        print(f"    (empty after filtering)")
        return pd.DataFrame()

    dates = sorted(df_full["date"].unique())
    print(f"    {len(dates)} dates × {df_full['station_id'].nunique()} stations "
          f"({len(df_full):,} rows)")

    # Compute per-station thresholds once for the whole file
    # (local_obs_climatology is date-aware internally — it uses the month)
    # We pass the whole DataFrame; the threshold module handles month matching.
    event_type = get_event_type(config)
    print(f"    Computing per-station thresholds (event_type={event_type})...")
    T_full = load_per_station_thresholds(config, df_full)

    records = []
    for date_val in dates:
        mask_date = df_full["date"] == date_val
        df_d = df_full[mask_date].copy()
        T_d  = T_full[mask_date.values]

        if len(df_d) < min_stations:
            continue

        obs = df_d["obs_value"].values.astype(np.float32)

        # Exclude rows where threshold is NaN
        valid = ~np.isnan(T_d) & ~np.isnan(obs)
        if valid.sum() < min_stations:
            continue

        obs_v  = obs[valid]
        T_v    = T_d[valid]
        df_v   = df_d[valid]
        lats   = df_v["lat"].values.astype(np.float32)
        lons   = df_v["lon"].values.astype(np.float32)

        if mode == "ensemble" and not use_ensemble_prob:
            fc1_v, fc2_v = extract_forecast_values(df_v, "ensemble")
        elif mode == "ensemble":
            prob1, prob2 = extract_exceedance_probability(df_v, T_v, event_type)
            fc1_v, fc2_v = prob1, prob2   # will use probabilistic classify below
        else:
            fc1_v, fc2_v = extract_forecast_values(df_v, "deterministic")

        if mode == "ensemble" and use_ensemble_prob:
            # Use probability-based classification but still need fc mean for twMAE
            fc1_mean, fc2_mean = extract_forecast_values(df_v, "ensemble")
            metrics = compute_date_metrics(
                obs_v, fc1_mean, fc2_mean, T_v, event_type, lats, lons
            )
            # Override classification with probabilistic version
            masks1 = classify_events_probabilistic(obs_v, fc1_v, T_v, event_type)
            masks2 = classify_events_probabilistic(obs_v, fc2_v, T_v, event_type)
            # Patch counts
            metrics["n_hit1"]  = int(masks1["hit"].sum())
            metrics["n_miss1"] = int(masks1["miss"].sum())
            metrics["n_fa1"]   = int(masks1["false_alarm"].sum())
            metrics["n_hit2"]  = int(masks2["hit"].sum())
            metrics["n_miss2"] = int(masks2["miss"].sum())
            metrics["n_fa2"]   = int(masks2["false_alarm"].sum())
        else:
            metrics = compute_date_metrics(
                obs_v, fc1_v, fc2_v, T_v, event_type, lats, lons
            )

        step = int(df_d["step"].iloc[0]) if "step" in df_d.columns else 0
        forecast_day = int(df_d["forecast_day"].iloc[0]) \
                       if "forecast_day" in df_d.columns else 0

        records.append({
            "date": str(date_val),
            "step_h": step,
            "forecast_day": forecast_day,
            "model1": model1_name,
            "model2": model2_name,
            **metrics,
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ─── Reporting ─────────────────────────────────────────────────────────────────

def print_top_cases(df: pd.DataFrame, top_n: int,
                    model1: str, model2: str, day: int):
    """Print a compact human-readable summary of the top N extreme cases."""
    print(f"\n{'═'*80}")
    print(f"  TOP {top_n} EXTREME CASE STUDIES — Forecast Day {day}")
    print(f"  Positive score → {model1} WORSE  |  Negative → {model2} WORSE")
    print(f"{'═'*80}")
    top = df.sort_values("rank").head(top_n)

    print(f"  {'Rank':>4}  {'Date':>8}  {'Score':>7}  {'Type':<35}"
          f"  {'FA1/FA2':>8}  {'Miss1/2':>8}  {'twMAE1/2':>11}  {'Region FA1'}")
    print(f"  {'-'*105}")
    for _, row in top.iterrows():
        fa_str   = f"{row['n_fa1']:>3}/{row['n_fa2']:<3}"
        miss_str = f"{row['n_miss1']:>3}/{row['n_miss2']:<3}"
        twmae_str = f"{row['twmae1']:.4f}/{row['twmae2']:.4f}"
        print(f"  {int(row['rank']):>4}  {row['date']:>8}  "
              f"{row['composite_score']:>+7.3f}  {row['case_type']:<35}"
              f"  {fa_str:>8}  {miss_str:>8}  {twmae_str:>11}  {row['fa_region1']}")

    print(f"\n  Notes:")
    n_m1 = (df["composite_score"] > 0.5).sum()
    n_m2 = (df["composite_score"] < -0.5).sum()
    print(f"  • Dates where {model1} clearly worse (score>0.5): {n_m1}")
    print(f"  • Dates where {model2} clearly worse (score<-0.5): {n_m2}")
    print(f"  • Dates with negligible difference:  {len(df) - n_m1 - n_m2}")


def write_text_summary(all_results: dict[int, pd.DataFrame], output_dir: Path,
                       name: str, top_n: int, model1: str, model2: str):
    """Write a combined text summary for all days."""
    out = output_dir / f"case_study_summary_{name}.txt"
    lines = [
        f"Case Study Summary: {model1} vs {model2}",
        "=" * 70, ""
    ]
    for day, df in sorted(all_results.items()):
        if df.empty:
            continue
        n_m1 = (df["composite_score"] > 0.5).sum()
        n_m2 = (df["composite_score"] < -0.5).sum()
        lines += [
            f"Forecast Day {day}  ({len(df)} dates analysed)",
            f"  {model1} clearly worse: {n_m1} dates",
            f"  {model2} clearly worse: {n_m2} dates",
            f"  Top worst case:",
        ]
        worst = df.iloc[0]
        lines += [
            f"    Date {worst['date']} | score {worst['composite_score']:+.3f}"
            f" | {worst['case_type']}",
            f"    FA: {worst['n_fa1']}/{worst['n_fa2']} | "
            f"Miss: {worst['n_miss1']}/{worst['n_miss2']} | "
            f"twMAE: {worst['twmae1']:.5f}/{worst['twmae2']:.5f}",
            "",
        ]
    out.write_text("\n".join(lines))
    print(f"\n  ✓ Text summary → {out}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config(args.config)
    parquet_dir = get_parquet_dir(config)
    model1, model2 = get_model_names(config)
    mode = get_mode(config)
    event_type = get_event_type(config)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path("case_study_output") / Path(args.config).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_name = Path(args.config).stem
    use_prob = (mode == "ensemble") and (not args.no_ensemble_prob)

    print(f"\n{'='*70}")
    print(f"  CASE STUDY FINDER")
    print(f"  Config    : {args.config}")
    print(f"  Models    : {model1}  vs  {model2}")
    print(f"  Mode      : {mode} ({'P>0.5 classification' if use_prob else 'mean'})")
    print(f"  Event     : {event_type}")
    print(f"  Parquets  : {parquet_dir}")
    print(f"  Output    : {output_dir}")
    if args.season:
        print(f"  Season    : {args.season}")
    if args.orog:
        print(f"  Orog      : {args.orog}")
    print(f"{'='*70}\n")

    parquets = discover_parquets(parquet_dir, args.days)
    if not parquets:
        print("ERROR: No parquet files found. Check the config extract_points path.")
        sys.exit(1)

    all_results = {}
    for day, pq_path in parquets:
        print(f"\n── Forecast Day {day} {'─'*50}")
        df_day = process_parquet(
            pq_path, config, mode, model1, model2,
            args.min_stations, use_prob,
            season=args.season, orog=args.orog,
        )
        if df_day.empty:
            print(f"  No data after filtering.")
            continue

        df_ranked = add_composite_scores(df_day)

        # Optional spatial concentration filter
        if args.min_concentration is not None:
            c = args.min_concentration
            # For dates where M1 is worse (positive score): check M1 FA/miss conc
            # For dates where M2 is worse (negative score): check M2 FA/miss conc
            keep = (
                df_ranked["fa_conc1"].fillna(0).ge(c) |
                df_ranked["fa_conc2"].fillna(0).ge(c) |
                df_ranked["miss_conc1"].fillna(0).ge(c) |
                df_ranked["miss_conc2"].fillna(0).ge(c)
            )
            n_before = len(df_ranked)
            df_ranked = df_ranked[keep]
            print(f"  Concentration filter (>={c:.2f}): "
                  f"{len(df_ranked)}/{n_before} dates kept")
            if df_ranked.empty:
                print(f"  No dates pass concentration filter.")
                continue

        all_results[day] = df_ranked

        # Save CSV
        suffix = ""
        if args.season: suffix += f"_{args.season}"
        if args.orog:   suffix += f"_{args.orog}"
        csv_path = output_dir / f"case_study_ranking_{cfg_name}_day{day}{suffix}.csv"
        df_ranked.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"  ✓ Rankings saved → {csv_path.name}")

        print_top_cases(df_ranked, args.top_n, model1, model2, day)

    if all_results:
        write_text_summary(all_results, output_dir, cfg_name + suffix,
                           args.top_n, model1, model2)

    print(f"\n{'='*70}")
    print(f"  DONE — {sum(len(v) for v in all_results.values())} date-cases ranked "
          f"across {len(all_results)} forecast day(s)")
    print(f"  Results in: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
