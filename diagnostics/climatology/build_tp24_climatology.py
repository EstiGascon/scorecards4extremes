#!/usr/bin/env python3
"""
Build tp24 station climatology from extracted observations.

Applies quality control following Rodwell et al. (2010):
  1. Latitudinal filtering of individual observations
  2. Percentile-based station rejection
  3. Minimum observation count

Computes percentiles p1-p99 for each station × calendar month
for three climatological periods: 1990-2020, 2000-2020, 2005-2025.

Station coordinate flexibility:
  WMO SYNOP stations are matched by station ID (stnid).
  Coordinates may vary slightly over time due to:
    - GPS precision improvements
    - Administrative metadata corrections
    - Minor relocations within the same compound
  References:
    - WMO (2018): Guide to the Global Observing System (WMO-No. 488).
      Stations relocated by <500 m horizontally keep the same ID.
    - Dunn et al. (2014): HadISD - a quality-controlled global synoptic report
      database. Uses 0.1° tolerance for station matching across metadata versions.
    - Menne et al. (2012): GHCN-Daily overview. Primary matching by station ID.
  We use stnid as primary key and flag stations where coordinates vary by >0.05°
  (~5.5 km at equator). Median coordinates are used as the official position.

Usage:
    python3 build_tp24_climatology.py               # Full build
    python3 build_tp24_climatology.py --period 2000-2020  # Single period
    python3 build_tp24_climatology.py --report-only  # Station report only
"""

import os
import sys
import argparse
import calendar
from datetime import datetime

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

RAW_DIR = '/ec/res4/scratch/$USER/obs_climatology_new/raw'
OUT_DIR = '/ec/res4/scratch/$USER/obs_climatology_new'

PERIODS = {
    '1990-2020': (1990, 2020),
    '2000-2020': (2000, 2020),
    '2005-2025': (2005, 2025),
    '2006-2025': (2006, 2025),
}

MIN_COVERAGE = 0.40    # Minimum fraction of expected days per station-month (intersection mode)
COORD_TOL = 0.05       # Coordinate tolerance in degrees (~5.5 km)
P99_ENVELOPE_0 = 200.0 # mm – p99 reference at equator (Rodwell eq. 1)
LAT_FILTER_MULT = 5.0  # Individual obs rejection: 5 × envelope
P99_STATION_MULT = 2.0 # Station rejection: p99 > 2 × envelope
RATIO_THRESHOLD = 10.0 # Consecutive percentile ratio limit near upper tail
BAD_VALUES = {420.0, 819.1}  # Known erroneous values in historical data


# ── QC functions following Rodwell et al. (2010) ──────────────────────────────

def p99_envelope(lat_deg):
    """Latitudinal envelope of p99 (Rodwell eq. 1): p99(0) × cos(lat).

    Returns the expected upper bound of p99 at a given latitude in mm.
    """
    lat_rad = np.radians(np.abs(lat_deg))
    return P99_ENVELOPE_0 * np.cos(lat_rad)


def filter_obs_latitudinal(values, lat_deg):
    """Reject individual observations by latitudinal filtering.

    - Reject negative values
    - Reject values > LAT_FILTER_MULT × p99_envelope(lat)
      i.e. 5 × 200 × cos(lat) = 1000 × cos(lat)

    Returns boolean mask (True = keep).
    """
    limit = LAT_FILTER_MULT * p99_envelope(lat_deg)
    return (values >= 0) & (values <= limit)


def qc_percentiles(pctls, lat_deg):
    """Apply percentile-based QC to a station-month.

    Returns (passed: bool, reason: str).

    Checks:
      a. Percentile values > 100 mm that occur more than once → reject
      b. Presence of known bad values (420.0, 819.1 mm) → reject
      c. p99 > 2 × envelope AND consecutive percentile ratio > 10
         near the upper tail → reject
    """
    # (a) Repeated high percentile values (> 100 mm appearing ≥2 times)
    high = pctls[pctls > 100]
    if len(high) > 0:
        unique, counts = np.unique(high, return_counts=True)
        if np.any(counts > 1):
            return False, 'repeated_high_percentile'

    # (b) Known erroneous values
    for bv in BAD_VALUES:
        if np.any(np.abs(pctls - bv) < 0.05):
            return False, f'bad_value_{bv}'

    # (c) p99 exceeding 2× envelope with extreme ratio near upper tail
    p99 = pctls[98]  # p99 (0-indexed: pctls[0]=p1, pctls[98]=p99)
    envelope = p99_envelope(lat_deg)
    if p99 > P99_STATION_MULT * envelope:
        # Check ratio between consecutive percentiles in upper tail (p90-p99)
        for i in range(89, 98):
            if pctls[i] > 0 and pctls[i + 1] / pctls[i] > RATIO_THRESHOLD:
                return False, 'extreme_tail_ratio'

    return True, 'ok'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_month_data(cal_month, start_year, end_year):
    """Load raw observations for a calendar month across years in a period.

    Returns a DataFrame with all daily obs from all years, or empty DataFrame.
    """
    dfs = []
    for year in range(start_year, end_year + 1):
        fpath = os.path.join(RAW_DIR, f'tp24_{year:04d}_{cal_month:02d}.parquet')
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ── Station analysis ─────────────────────────────────────────────────────────

def analyze_stations(df):
    """Analyze station coordinate stability and metadata.

    Returns a DataFrame indexed by stnid with columns:
      latitude, longitude, elevation, lat_range, lon_range, coord_moved, n_obs
    """
    records = []
    for stnid, grp in df.groupby('stnid'):
        lats = grp['latitude'].astype(float)
        lons = grp['longitude'].astype(float)
        lat_range = lats.max() - lats.min()
        lon_range = lons.max() - lons.min()

        rec = {
            'stnid': stnid,
            'latitude': lats.median(),
            'longitude': lons.median(),
            'elevation': grp['elevation'].median() if 'elevation' in grp else np.nan,
            'lat_range': lat_range,
            'lon_range': lon_range,
            'coord_moved': (lat_range > COORD_TOL) or (lon_range > COORD_TOL),
            'n_obs': len(grp),
        }
        records.append(rec)

    return pd.DataFrame(records).set_index('stnid')


def check_continuity(start_year, end_year, cal_months=range(1, 13)):
    """Check station data continuity across years for a period.

    Returns a DataFrame with stnid as index and columns:
      years_present, total_years, coverage_pct, continuous
    """
    # Count which years each station appears in
    station_years = {}
    total_years = end_year - start_year + 1

    for year in range(start_year, end_year + 1):
        for month in cal_months:
            fpath = os.path.join(RAW_DIR,
                                 f'tp24_{year:04d}_{month:02d}.parquet')
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_parquet(fpath, columns=['stnid'])
            except Exception:
                continue
            for sid in df['stnid'].unique():
                station_years.setdefault(sid, set()).add(year)

    records = []
    for sid, years in station_years.items():
        n_years = len(years)
        records.append({
            'stnid': sid,
            'years_present': n_years,
            'total_years': total_years,
            'coverage_pct': 100.0 * n_years / total_years,
            'continuous': n_years >= 0.8 * total_years,
        })

    return pd.DataFrame(records).set_index('stnid')


# ── Climatology computation ──────────────────────────────────────────────────

def compute_station_climatology(df, station_info, cal_month,
                                start_year, end_year,
                                valid_stations=None,
                                min_coverage_override=None):
    """Compute percentile climatology for one calendar month.

    Steps:
      1. (Optional) restrict to pre-selected valid_stations set
      2. Apply latitudinal filtering to individual observations
      3. Group by station, require ≥ coverage of expected days
      4. Apply percentile-based QC
      5. Return DataFrame with station metadata + p1..p99

    valid_stations: set of stnids that passed coverage in ALL months
                    (intersection). If None, applies per-month filter.
    min_coverage_override: override MIN_COVERAGE for this call.

    Returns (result_df, qc_report_dict).
    """
    cov = min_coverage_override if min_coverage_override is not None else MIN_COVERAGE
    # Expected number of days for this calendar month across the period
    n_years = end_year - start_year + 1
    days_in_month = calendar.monthrange(2020, cal_month)[1]  # use non-leap ref
    expected_days = n_years * days_in_month
    min_obs = int(np.ceil(cov * expected_days))
    # Ensure value column is float
    df = df.copy()
    df['value'] = df['value_0'].astype(float)

    # ── Step 0: Pre-filter to intersection station set ──
    if valid_stations is not None:
        df = df[df['stnid'].isin(valid_stations)].copy()

    # ── Step 1: Latitudinal filtering of individual observations ──
    # Use station median latitude for the filter
    stn_lat = station_info['latitude']
    df = df.merge(
        stn_lat.rename('stn_lat'),
        left_on='stnid', right_index=True, how='left'
    )
    # Reject negative values and latitudinally extreme values
    mask = filter_obs_latitudinal(df['value'].values, df['stn_lat'].values)
    n_rejected_obs = (~mask).sum()
    df = df[mask].copy()

    # ── Step 2: Group by station, compute percentiles ──
    results = []
    n_too_few = 0
    n_qc_fail = 0
    qc_reasons = {}

    for stnid, grp in df.groupby('stnid'):
        vals = grp['value'].values
        nobs = len(vals)

        if nobs < min_obs:
            n_too_few += 1
            continue

        # Compute p1..p99
        pctls = np.percentile(vals, range(1, 100))

        # Get station metadata
        if stnid in station_info.index:
            info = station_info.loc[stnid]
            lat = info['latitude']
            lon = info['longitude']
            elev = info.get('elevation', np.nan)
        else:
            lat = grp['latitude'].median()
            lon = grp['longitude'].median()
            elev = grp['elevation'].median() if 'elevation' in grp else np.nan

        # ── Step 3: Percentile-based QC ──
        passed, reason = qc_percentiles(pctls, lat)
        if not passed:
            n_qc_fail += 1
            qc_reasons[reason] = qc_reasons.get(reason, 0) + 1
            continue

        # Determine last observation date
        if 'obs_date' in grp.columns:
            last_obs = grp['obs_date'].max()
            if hasattr(last_obs, 'strftime'):
                last_obs_str = last_obs.strftime('%Y%m%d')
            else:
                last_obs_str = str(last_obs)[:10].replace('-', '')
        else:
            last_year = int(grp['year'].max())
            last_obs_str = f'{last_year}{cal_month:02d}28'

        rec = {
            'latitude': lat,
            'longitude': lon,
            'stnid': stnid,
            'last_obs_date': last_obs_str,
            'elevation': elev if not np.isnan(elev) else 0.0,
            'nobs': nobs,
        }
        for i, p in enumerate(pctls, start=1):
            rec[f'p{i}'] = p

        results.append(rec)

    report = {
        'month': cal_month,
        'total_obs_before_qc': len(df) + n_rejected_obs,
        'obs_rejected_latitudinal': n_rejected_obs,
        'expected_days': expected_days,
        'min_obs_required': min_obs,
        'stations_too_few_obs': n_too_few,
        'stations_qc_fail': n_qc_fail,
        'qc_fail_reasons': qc_reasons,
        'stations_retained': len(results),
    }

    if results:
        return pd.DataFrame(results), report
    return pd.DataFrame(), report


# ── Output ────────────────────────────────────────────────────────────────────

def write_percentile_file(df, output_file, min_coverage=None):
    """Write percentile file in Rodwell-compatible ASCII format.

    Format: lat lon sid yyyymmdd height nobs p1 p2 ... p99
    Stations ordered by decreasing latitude.
    """
    if df.empty:
        return

    cov = min_coverage if min_coverage is not None else MIN_COVERAGE
    df = df.sort_values('latitude', ascending=False)

    pcol = [f'p{i}' for i in range(1, 100)]

    with open(output_file, 'w') as f:
        # Two-line header
        f.write("lat lon sid yyyymmdd height nobs " +
                " ".join(pcol) + "\n")
        f.write(f"# {len(df)} stations, "
                f"min_coverage={cov:.0%}, "
                f"generated {datetime.now().strftime('%Y-%m-%d')}\n")

        for _, row in df.iterrows():
            parts = [
                f"{row['latitude']:.4f}",
                f"{row['longitude']:.4f}",
                f"{int(row['stnid'])}",
                f"{row['last_obs_date']}",
                f"{row['elevation']:.1f}",
                f"{int(row['nobs'])}",
            ]
            parts.extend(f"{row[c]:.4f}" for c in pcol)
            f.write(" ".join(parts) + "\n")


def write_station_report(reports, period_name, output_dir):
    """Write a summary report for the climatology build."""
    rpath = os.path.join(output_dir, f'station_report_{period_name}.txt')
    with open(rpath, 'w') as f:
        f.write(f"Station Climatology Report: {period_name}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 70 + "\n\n")

        for r in reports:
            f.write(f"Month {r['month']:02d}:\n")
            f.write(f"  Total observations (before QC) : {r['total_obs_before_qc']}\n")
            f.write(f"  Rejected (latitudinal)         : {r['obs_rejected_latitudinal']}\n")
            f.write(f"  Expected days in period         : {r['expected_days']}\n")
            f.write(f"  Min obs required ({MIN_COVERAGE:.0%} coverage): {r['min_obs_required']}\n")
            f.write(f"  Stations with insufficient data : {r['stations_too_few_obs']}\n")
            f.write(f"  Stations rejected (percentile QC)    : {r['stations_qc_fail']}\n")
            if r['qc_fail_reasons']:
                for reason, cnt in r['qc_fail_reasons'].items():
                    f.write(f"    - {reason}: {cnt}\n")
            f.write(f"  Stations retained              : {r['stations_retained']}\n\n")

    print(f"  Report written: {rpath}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_period(period_name, min_coverage=None, monthly_mode=False):
    """Build climatology for one period.

    Args:
        period_name: key in PERIODS dict.
        min_coverage: override MIN_COVERAGE (fraction, e.g. 0.50).
        monthly_mode: if True, apply coverage filter per month independently
                      (each month can have a different station set).
                      If False (default), use intersection across all 12 months.
    """
    start_year, end_year = PERIODS[period_name]
    cov = min_coverage if min_coverage is not None else MIN_COVERAGE
    mode_str = 'monthly' if monthly_mode else 'intersection'
    out_dir = os.path.join(OUT_DIR, 'climatology', period_name)
    if monthly_mode:
        out_dir = os.path.join(OUT_DIR, 'climatology',
                               f'{period_name}_p{int(cov*100):02d}_monthly')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Building climatology: {period_name} ({start_year}–{end_year})")
    print(f"  Coverage threshold : {cov:.0%}  |  Mode: {mode_str}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    # First pass: analyze all stations across all months for this period
    print("\n  Analyzing station metadata...")
    all_month_data = []
    for month in range(1, 13):
        mdata = load_month_data(month, start_year, end_year)
        if not mdata.empty:
            all_month_data.append(mdata)

    if not all_month_data:
        print("  ERROR: No raw data found!")
        return

    combined = pd.concat(all_month_data, ignore_index=True)
    station_info = analyze_stations(combined)
    del combined  # Free memory

    n_total = len(station_info)
    n_moved = station_info['coord_moved'].sum()
    print(f"  Total stations: {n_total}")
    print(f"  Stations with coordinate shift > {COORD_TOL}°: {n_moved}")
    if n_moved > 0:
        moved = station_info[station_info['coord_moved']]
        print(f"    Max lat range: {moved['lat_range'].max():.4f}°")
        print(f"    Max lon range: {moved['lon_range'].max():.4f}°")

    # Second pass: determine valid station set
    n_years = end_year - start_year + 1
    if monthly_mode:
        # Per-month mode: each month selects its own passing stations independently.
        # valid_stations=None tells compute_station_climatology to use per-month filter.
        print("\n  Monthly mode: coverage filter applied independently per month.")
        valid_all_months = None
    else:
        # Intersection mode: same station set for all months.
        print("\n  Computing station intersection across all 12 months...")
        valid_all_months = None
        for month in range(1, 13):
            days = calendar.monthrange(2020, month)[1]
            expected = n_years * days
            min_obs_m = int(np.ceil(cov * expected))
            mdata_tmp = load_month_data(month, start_year, end_year)
            if mdata_tmp.empty:
                continue
            counts = mdata_tmp.groupby('stnid').size()
            passing = set(counts[counts >= min_obs_m].index)
            if valid_all_months is None:
                valid_all_months = passing
            else:
                valid_all_months &= passing
        if not valid_all_months:
            print("  ERROR: intersection of valid stations is empty!")
            return
        print(f"  Intersection: {len(valid_all_months):,} stations pass "
              f"{cov:.0%} coverage in ALL 12 months")

    # Third pass: compute climatology month by month
    reports = []
    for month in range(1, 13):
        print(f"\n  Processing month {month:02d}...", flush=True)
        mdata = load_month_data(month, start_year, end_year)
        if mdata.empty:
            print(f"    No data for month {month:02d}")
            reports.append({
                'month': month,
                'total_obs_before_qc': 0,
                'obs_rejected_latitudinal': 0,
                'expected_days': 0,
                'min_obs_required': 0,
                'stations_too_few_obs': 0,
                'stations_qc_fail': 0,
                'qc_fail_reasons': {},
                'stations_retained': 0,
            })
            continue

        result_df, report = compute_station_climatology(
            mdata, station_info, month, start_year, end_year,
            valid_stations=valid_all_months,
            min_coverage_override=cov)
        reports.append(report)

        if not result_df.empty:
            outfile = os.path.join(out_dir, f'PPT24_percentiles_{month:02d}_00.txt')
            write_percentile_file(result_df, outfile, min_coverage=cov)
            print(f"    Written {outfile}: {len(result_df)} stations")
        else:
            print(f"    No stations passed QC for month {month:02d}")

    write_station_report(reports, period_name, out_dir)


def station_continuity_report():
    """Generate a station continuity report across all three periods."""
    report_dir = os.path.join(OUT_DIR, 'station_info')
    os.makedirs(report_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Station Continuity Report")
    print("=" * 60)

    for period_name, (start_year, end_year) in PERIODS.items():
        print(f"\n  Period: {period_name}")
        cont = check_continuity(start_year, end_year)
        n_total = len(cont)
        n_continuous = cont['continuous'].sum()
        print(f"    Total stations seen    : {n_total}")
        print(f"    Continuous (≥80% years): {n_continuous}")
        print(f"    Coverage distribution:")
        for pct in [100, 90, 80, 50, 20]:
            n = (cont['coverage_pct'] >= pct).sum()
            print(f"      ≥{pct:3d}% of years: {n:5d} stations")

        # Save details
        outfile = os.path.join(report_dir,
                               f'station_continuity_{period_name}.csv')
        cont.to_csv(outfile)
        print(f"    Saved: {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description='Build tp24 station climatology')
    parser.add_argument('--period', type=str, default=None,
                        choices=list(PERIODS.keys()),
                        help='Build only this period (default: all)')
    parser.add_argument('--report-only', action='store_true',
                        help='Only generate station continuity report')
    parser.add_argument('--monthly', action='store_true',
                        help='Per-month coverage filter (stations can differ by month)')
    parser.add_argument('--min-coverage', type=float, default=None,
                        metavar='FRAC',
                        help='Override min coverage fraction (e.g. 0.50)')
    args = parser.parse_args()

    cov = args.min_coverage if args.min_coverage is not None else MIN_COVERAGE

    print("=" * 60)
    print("  tp24 Station Climatology Builder")
    print("=" * 60)
    print(f"  Started: {datetime.now()}")
    print(f"  Raw data: {RAW_DIR}")
    print(f"  Output  : {OUT_DIR}")
    print(f"  QC: min_coverage={cov:.0%}, coord_tol={COORD_TOL}°")
    print(f"  Mode: {'monthly (per-month)' if args.monthly else 'intersection (all months)'}")

    # Always generate continuity report
    station_continuity_report()

    if args.report_only:
        return

    # Build climatology
    periods_to_build = ([args.period] if args.period
                        else list(PERIODS.keys()))
    for period_name in periods_to_build:
        build_period(period_name, min_coverage=cov, monthly_mode=args.monthly)

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
