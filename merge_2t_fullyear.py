#!/usr/bin/env python3
"""
merge_2t_fullyear.py
=====================
Merge the two 2t ensemble extractions into one unified full-year dataset:
  - extracted_points/2t_ens_local_p99obsclim_aifsvsifs/       (2025-07-02 -> 2026-03-19, fast local_grib mirror)
  - extracted_points/2t_ens_mars_p99obsclim_aifsvsifs_BACKWARD/ (2025-03-20 -> 2025-07-01, live MARS backfill)
into:
  - extracted_points/2t_ens_local_p99obsclim_aifsvsifs_fullyear/ (2025-03-20 -> 2026-03-19, 365 days)

Both sources are produced by the same extract_ensemble_points() function (see
src/extract_points_ensemble.py) with disjoint date ranges, so per-forecast-day
files (day1.parquet ... day10.parquet) are concatenated row-wise after
verifying the two sources' schemas and date ranges do not overlap.

Usage:
  python merge_2t_fullyear.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent
FAST_DIR = REPO / "extracted_points" / "2t_ens_local_p99obsclim_aifsvsifs"
BACKWARD_DIR = REPO / "extracted_points" / "2t_ens_mars_p99obsclim_aifsvsifs_BACKWARD"
OUT_DIR = REPO / "extracted_points" / "2t_ens_local_p99obsclim_aifsvsifs_fullyear"

FAST_PREFIX = "2t_ifsens4aifs_vs_aifs_ens_ens"
BACKWARD_PREFIX = "2t_ifsens4aifs_vs_aifs_ens_ens"  # same var/model names -> same filename pattern


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate schemas/date ranges only, don't write merged files.")
    args = ap.parse_args()

    if not FAST_DIR.exists():
        sys.exit(f"ERROR: fast dataset dir not found: {FAST_DIR}")
    if not BACKWARD_DIR.exists():
        sys.exit(f"ERROR: backward dataset dir not found: {BACKWARD_DIR} "
                 "(has the backward extraction job finished yet?)")

    fast_files = sorted(FAST_DIR.glob(f"{FAST_PREFIX}_day*.parquet"))
    back_files = sorted(BACKWARD_DIR.glob(f"{BACKWARD_PREFIX}_day*.parquet"))
    if not fast_files:
        sys.exit(f"ERROR: no day-parquet files found in {FAST_DIR}")
    if not back_files:
        sys.exit(f"ERROR: no day-parquet files found in {BACKWARD_DIR} "
                 "(extraction may still be in the retrieval phase, not yet extracted)")

    fast_days = {int(f.stem.rsplit("day", 1)[1]) for f in fast_files}
    back_days = {int(f.stem.rsplit("day", 1)[1]) for f in back_files}
    if fast_days != back_days:
        sys.exit(f"ERROR: forecast-day sets differ: fast={sorted(fast_days)} "
                 f"backward={sorted(back_days)}")

    print(f"Fast dataset:     {FAST_DIR}  ({len(fast_files)} day-files)")
    print(f"Backward dataset: {BACKWARD_DIR}  ({len(back_files)} day-files)")
    print(f"Output:           {OUT_DIR}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=not args.dry_run)

    for day in sorted(fast_days):
        fast_fp = FAST_DIR / f"{FAST_PREFIX}_day{day}.parquet"
        back_fp = BACKWARD_DIR / f"{BACKWARD_PREFIX}_day{day}.parquet"

        fast_schema = pq.ParquetFile(fast_fp).schema.names
        back_schema = pq.ParquetFile(back_fp).schema.names
        if set(fast_schema) != set(back_schema):
            only_fast = set(fast_schema) - set(back_schema)
            only_back = set(back_schema) - set(fast_schema)
            sys.exit(f"ERROR day{day}: schema mismatch. Only in fast: {only_fast}; "
                     f"only in backward: {only_back}")

        df_fast = pd.read_parquet(fast_fp)
        df_back = pd.read_parquet(back_fp)

        # Align column order (schema set matches, order may differ).
        df_back = df_back[df_fast.columns]

        fast_dates = set(df_fast["date"].astype(str))
        back_dates = set(df_back["date"].astype(str))
        overlap = fast_dates & back_dates
        if overlap:
            sys.exit(f"ERROR day{day}: {len(overlap)} overlapping dates between "
                     f"fast and backward sources (expected disjoint) — "
                     f"sample: {sorted(overlap)[:5]}")

        merged = pd.concat([df_back, df_fast], ignore_index=True)
        merged = merged.sort_values(["date", "station_id"]).reset_index(drop=True)

        n_dates = merged["date"].nunique()
        print(f"  day{day}: fast={len(df_fast):,} rows ({len(fast_dates)} dates) + "
              f"backward={len(df_back):,} rows ({len(back_dates)} dates) "
              f"-> merged={len(merged):,} rows ({n_dates} dates)")

        if not args.dry_run:
            out_fp = OUT_DIR / f"{FAST_PREFIX}_day{day}.parquet"
            merged.to_parquet(out_fp, index=False)
            print(f"    \u2713 wrote {out_fp}")

        del df_fast, df_back, merged

    if args.dry_run:
        print("\n[dry-run] Validation passed, no files written.")
    else:
        print(f"\n\u2713 Merge complete: {OUT_DIR}")


if __name__ == "__main__":
    main()
