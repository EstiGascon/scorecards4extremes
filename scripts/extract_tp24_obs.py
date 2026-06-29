#!/usr/bin/env python3
"""
Extract daily tp24 (24h precipitation) observations from STVL.

Retrieves all daily data from 1990 to 2025 at 00 UTC, month by month,
and saves to parquet files for later climatology computation.

Usage:
    python3 extract_tp24_obs.py --test          # Test with one month
    python3 extract_tp24_obs.py                 # Full extraction
    python3 extract_tp24_obs.py --start-year 2000 --end-year 2010  # Partial
"""

import sys
sys.path.insert(0, '/usr/local/apps/quaver/3.6.4/lib/python3.12/site-packages')

import os
import argparse
import traceback
from datetime import datetime

import calendar

import numpy as np
import pandas as pd
try:
    import vtb
    _VTB_AVAILABLE = True
except ImportError:
    vtb = None
    _VTB_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get('S4E_OBS_OUTPUT_DIR', './obs_climatology_raw')
START_YEAR = 1990
END_YEAR = 2025


def explore_result(result, label=""):
    """Print detailed structure info about an STVL result (test mode)."""
    print(f"\n--- Result structure for {label} ---")
    print(f"  Type: {type(result)}")
    print(f"  Length (number of fieldsets): {len(result)}")

    if len(result) == 0:
        print("  EMPTY RESULT")
        return

    fs = result[0]
    print(f"  Fieldset type: {type(fs)}")

    # Probe metadata attributes
    for attr in ('datetime', 'valid_datetime', 'date', 'time',
                 'startdate', 'enddate', 'step'):
        if hasattr(fs, attr):
            val = getattr(fs, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    val = '<callable, error>'
            print(f"  fs.{attr} = {val}")

    df = fs.to_dataframe()
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")
    print(f"  First 5 rows:\n{df.head()}")

    if 'stnid' in df.columns:
        print(f"  Unique stations: {df['stnid'].nunique()}")

    # Compare first two fieldsets (different days?)
    if len(result) > 1:
        df2 = result[1].to_dataframe()
        print(f"\n  Second fieldset shape: {df2.shape}")
        if 'stnid' in df.columns and 'stnid' in df2.columns:
            s1, s2 = set(df['stnid']), set(df2['stnid'])
            print(f"  Common stations: {len(s1 & s2)}/{len(s2)}")
        # Try metadata on second fieldset
        for attr in ('datetime', 'valid_datetime', 'date'):
            if hasattr(result[1], attr):
                val = getattr(result[1], attr)
                if callable(val):
                    try:
                        val = val()
                    except Exception:
                        continue
                print(f"  fs[1].{attr} = {val}")


def retrieve_month(year, month, output_dir, test_mode=False):
    """Retrieve all daily tp24 obs at 00 UTC for one year-month.

    STVL observation table returns data per exact date, so we must pass
    all days of the month as reference_datetimes.

    Each fieldset in the result corresponds to one day, containing all
    stations that reported on that day.
    """
    outfile = os.path.join(output_dir, f'tp24_{year:04d}_{month:02d}.parquet')

    if not test_mode and os.path.exists(outfile):
        print(f'  [{year}-{month:02d}] Skipping: already exists')
        return True

    t0 = datetime.now()
    print(f'  [{year}-{month:02d}] Retrieving...', end='', flush=True)

    # Build list of all days in this month
    n_days = calendar.monthrange(year, month)[1]
    ref_datetimes = [
        f'{year}-{month:02d}-{day:02d}T00:00:00'
        for day in range(1, n_days + 1)
    ]

    if not _VTB_AVAILABLE:
        raise ImportError(
            "extract_tp24_obs.py requires the 'vtb' package, which is an "
            "ECMWF-internal library not available on public PyPI. "
            "This script is only usable within the ECMWF environment."
        )
    result = vtb.media.stvl_retrieve(
        table='observation',
        parameter='tp',
        period=24,                    # 24h accumulation
        reference_datetimes=ref_datetimes,
        forecast_lengths=[0],
    )

    if test_mode:
        explore_result(result, f"{year}-{month:02d}")

    n_fields = len(result)
    if n_fields == 0:
        print(f' empty result ({datetime.now() - t0})')
        return False

    # ── Collect all fieldsets into one DataFrame ──
    dfs = []
    for i in range(n_fields):
        fs = result[i]
        try:
            df = fs.to_dataframe()
        except Exception as exc:
            print(f'\n    Warning: fieldset {i} error: {exc}')
            continue

        # Fieldsets are ordered by day (matching ref_datetimes order)
        day = i + 1  # day of month (1-indexed)
        df['day'] = day
        dfs.append(df)

    if not dfs:
        print(f' no valid data ({datetime.now() - t0})')
        return False

    all_data = pd.concat(dfs, ignore_index=True)
    all_data['year'] = year
    all_data['month'] = month

    # Drop rows with NaN observations (stations that reported but had no value)
    n_before = len(all_data)
    all_data = all_data.dropna(subset=['value_0'])
    n_nan = n_before - len(all_data)

    n_stations = all_data['stnid'].nunique()
    n_obs = len(all_data)
    elapsed = datetime.now() - t0

    if not test_mode:
        all_data.to_parquet(outfile, index=False)

    nan_str = f', {n_nan} NaN dropped' if n_nan > 0 else ''
    print(f' {n_obs} obs, {n_stations} stations, {n_fields} days{nan_str} ({elapsed})')

    if test_mode:
        vals = all_data['value_0'].astype(float)
        print(f"\n  Value statistics:")
        print(f"    min  = {vals.min():.4f}")
        print(f"    max  = {vals.max():.4f}")
        print(f"    mean = {vals.mean():.4f}")
        print(f"    med  = {vals.median():.4f}")
        neg = (vals < 0).sum()
        if neg > 0:
            print(f"    NEGATIVE VALUES: {neg}")
        zero = (vals == 0).sum()
        print(f"    zero values: {zero} ({100*zero/len(vals):.1f}%)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract tp24 observations from STVL')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: retrieve one month, print structure')
    parser.add_argument('--test-year', type=int, default=2020)
    parser.add_argument('--test-month', type=int, default=1)
    parser.add_argument('--start-year', type=int, default=START_YEAR)
    parser.add_argument('--end-year', type=int, default=END_YEAR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.test:
        print(f"=== TEST MODE: {args.test_year}-{args.test_month:02d} ===")
        retrieve_month(args.test_year, args.test_month,
                       args.output_dir, test_mode=True)
        return

    print("=" * 60)
    print("  tp24 Observation Extraction from STVL")
    print("=" * 60)
    print(f"  Period : {args.start_year} – {args.end_year}")
    print(f"  Output : {args.output_dir}")
    print(f"  Started: {datetime.now()}")
    print("=" * 60)

    total = 0
    errors = 0

    for year in range(args.start_year, args.end_year + 1):
        print(f"\n  Year {year}:")
        for month in range(1, 13):
            try:
                ok = retrieve_month(year, month, args.output_dir)
                if ok:
                    total += 1
            except Exception as exc:
                print(f'  [{year}-{month:02d}] ERROR: {exc}')
                traceback.print_exc()
                errors += 1

    print("\n" + "=" * 60)
    print(f"  Completed: {datetime.now()}")
    print(f"  Months retrieved: {total}")
    print(f"  Errors: {errors}")
    print("=" * 60)


if __name__ == '__main__':
    main()
