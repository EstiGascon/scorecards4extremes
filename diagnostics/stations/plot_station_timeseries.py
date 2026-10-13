#!/usr/bin/env python3
"""
Station time series — 2m temperature, IFS vs AIFS vs obs.

Plots daily mean temperature (averaged over all steps of the requested
forecast day) for a single station across the DJF season.

Usage:
    python3 plot_station_timeseries.py \
        --config config_2t_local_fixed35_aifs_ifs_single.yaml \
        --station S260 \
        --day 3 --season DJF --orog flat \
        --output-dir case_study_output/timeseries
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path


SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
OROG_RANGES = {
    "flat":    (0,   40),
    "hilly":   (40,  120),
    "complex": (120, 3000),
    "low":     (0,   120),
    "high":    (120, 3000),
}
M1_COLOR = "#1565C0"   # IFS  — deep blue
M2_COLOR = "#D55E00"   # model 2 — Okabe-Ito vermillion (colorblind-safe)
OBS_COLOR = "black"    # obs  — black


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--station",    default=None,
                   help="Station ID (e.g. S260). If omitted, auto-selects the "
                        "station with the largest mean |fc1-fc2|.")
    p.add_argument("--day",        type=int, default=3)
    p.add_argument("--season",     default=None)
    p.add_argument("--orog",       default=None,
                   help="flat / hilly / complex / low / high")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Mark dates where obs exceeds this threshold.")
    p.add_argument("--output-dir", default="case_study_output/timeseries")
    p.add_argument("--lat-min",    type=float, default=None)
    p.add_argument("--lat-max",    type=float, default=None)
    p.add_argument("--lon-min",    type=float, default=None)
    p.add_argument("--lon-max",    type=float, default=None)
    p.add_argument("--lat",        type=float, default=None,
                   help="Select station by exact latitude (use with --lon).")
    p.add_argument("--lon",        type=float, default=None,
                   help="Select station by exact longitude (use with --lat).")
    p.add_argument("--hour",       type=int, default=None,
                   help="Plot only the synoptic hour (0, 6, 12, or 18 UTC) instead of the daily mean.")
    p.add_argument("--m1-label",   default=None,
                   help="Legend label for model 1 (default: name from config).")
    p.add_argument("--m2-label",   default=None,
                   help="Legend label for model 2 (default: name from config).")
    p.add_argument("--suffix",     default=None,
                   help="Extra string appended to the output filename to avoid overwriting existing plots.")
    return p.parse_args()


def load_and_filter(pq_path, day, season, orog, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    cols = ['date', 'step', 'valid_time', 'station_id', 'lat', 'lon',
            'obs_height', 'sdfor', 'fc1_value', 'fc2_value', 'obs_value']
    df = pd.read_parquet(pq_path, columns=cols)

    # Season filter
    if season:
        months = SEASON_MONTHS.get(season.upper(), None)
        if months:
            df['_month'] = df['date'].astype(str).str[4:6].astype(int)
            df = df[df['_month'].isin(months)]
            df = df.drop(columns=['_month'])

    # Orography filter — uses sdfor (std-dev of model orography), same as main codebase
    # flat: sdfor < 40, hilly: 40–120, complex: >= 120
    if orog:
        lo, hi = OROG_RANGES.get(orog.lower(), (None, None))
        if lo is not None:
            df = df[(df['sdfor'] >= lo) & (df['sdfor'] < hi)]

    # Geographic bounding box
    if lat_min is not None:
        df = df[df['lat'] >= lat_min]
    if lat_max is not None:
        df = df[df['lat'] <= lat_max]
    if lon_min is not None:
        df = df[df['lon'] >= lon_min]
    if lon_max is not None:
        df = df[df['lon'] <= lon_max]

    return df


def select_by_location(df):
    """Pick physical location (lat, lon) with largest mean |fc1-fc2| (>=30 unique dates)."""
    df2 = df.copy()
    df2['absdiff'] = (df2['fc1_value'] - df2['fc2_value']).abs()
    # Average over steps → one row per location per date
    by_date = df2.groupby(['lat', 'lon', 'obs_height', 'date'])[['absdiff']].mean().reset_index()
    stn = by_date.groupby(['lat', 'lon', 'obs_height']).agg(
        diff_mean=('absdiff', 'mean'),
        n=('date', 'count'),
    ).reset_index()
    stn = stn[stn['n'] >= 30]
    if stn.empty:
        stn = by_date.groupby(['lat', 'lon', 'obs_height']).agg(
            diff_mean=('absdiff', 'mean'),
            n=('date', 'count'),
        ).reset_index()
    best = stn.nlargest(1, 'diff_mean').iloc[0]
    print(f"  Auto-selected location: lat={best.lat:.3f}  lon={best.lon:.3f}  "
          f"elev={best.obs_height:.0f}m  mean|fc1-fc2|={best.diff_mean:.2f}°C  n={int(best.n)} dates")
    return float(best.lat), float(best.lon)


def main():
    args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    parquet_dir = cfg['extract_points']['output_path']
    var         = cfg.get('variable', '2t')
    m1 = cfg['read_data']['forecast_model1']['name']
    m2 = cfg['read_data']['forecast_model2']['name']
    lbl1 = args.m1_label if args.m1_label else m1
    lbl2 = args.m2_label if args.m2_label else m2
    T  = args.threshold
    if T is None and cfg.get('threshold', {}).get('method') == 'fixed':
        T = cfg['threshold']['fixed']['value']

    pq_file = Path(parquet_dir) / f"{var}_{m1}_vs_{m2}_day{args.day}.parquet"
    if not pq_file.exists():
        raise FileNotFoundError(f"Parquet not found: {pq_file}")

    print(f"Loading: {pq_file.name}")
    df = load_and_filter(pq_file, args.day, args.season, args.orog,
                         lat_min=args.lat_min, lat_max=args.lat_max,
                         lon_min=args.lon_min, lon_max=args.lon_max)
    print(f"  Rows after filter: {len(df):,}")

    # Determine location to plot
    if args.lat is not None and args.lon is not None:
        # Exact coordinate selection
        sel_lat, sel_lon = args.lat, args.lon
        tol = 0.02
        sdf = df[(df['lat'].between(sel_lat - tol, sel_lat + tol)) &
                 (df['lon'].between(sel_lon - tol, sel_lon + tol))]
        if sdf.empty:
            raise ValueError(f"No data near lat={sel_lat}, lon={sel_lon} (tol={tol}°)")
        print(f"  Selected location: lat={sdf['lat'].iloc[0]:.3f}  lon={sdf['lon'].iloc[0]:.3f}")
    elif args.station is not None:
        # Legacy station_id — pick best lat/lon for that ID
        sdf = df[df['station_id'] == args.station]
        if sdf.empty:
            raise ValueError(f"Station '{args.station}' not found after filtering.")
        if sdf[['lat','lon']].drop_duplicates().shape[0] > 1:
            df2 = sdf.copy()
            df2['absdiff'] = (df2['fc1_value'] - df2['fc2_value']).abs()
            best = df2.groupby(['lat','lon'])['absdiff'].mean().idxmax()
            sdf = sdf[(sdf['lat'] == best[0]) & (sdf['lon'] == best[1])]
            print(f"  Multiple locations for {args.station} — picked lat={best[0]:.3f} lon={best[1]:.3f}")
    else:
        # Auto-select best physical location
        sel_lat, sel_lon = select_by_location(df)
        tol = 0.001
        sdf = df[(df['lat'].between(sel_lat - tol, sel_lat + tol)) &
                 (df['lon'].between(sel_lon - tol, sel_lon + tol))]

    lat    = float(sdf['lat'].iloc[0])
    lon    = float(sdf['lon'].iloc[0])
    height = float(sdf['obs_height'].iloc[0])
    # Use the most common station_id at this location for labelling
    station_id = sdf['station_id'].mode().iloc[0]

    # Optionally restrict to a single synoptic hour before averaging
    if args.hour is not None:
        sdf = sdf[sdf['valid_time'].astype(str).str[-2:].astype(int) == args.hour % 100]
        if sdf.empty:
            raise ValueError(f"No data for hour={args.hour:02d}Z at this location.")

    # Average over steps → one row per date
    daily = sdf.groupby('date')[['fc1_value', 'fc2_value', 'obs_value']].mean()
    daily.index = pd.to_datetime(daily.index.astype(str), format='%Y%m%d')
    daily = daily.sort_index()

    fc1 = daily['fc1_value'].values
    fc2 = daily['fc2_value'].values
    obs = daily['obs_value'].values
    dates = daily.index

    mae1 = float(np.mean(np.abs(fc1 - obs)))
    mae2 = float(np.mean(np.abs(fc2 - obs)))
    bias1 = float(np.mean(fc1 - obs))
    bias2 = float(np.mean(fc2 - obs))
    rmse1 = float(np.sqrt(np.mean((fc1 - obs) ** 2)))
    rmse2 = float(np.sqrt(np.mean((fc2 - obs) ** 2)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_ts, ax_diff) = plt.subplots(
        2, 1, figsize=(18, 10),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    season_lbl = args.season.upper() if args.season else "All"
    orog_lbl   = args.orog.capitalize() if args.orog else "All"
    hour_lbl   = f"  |  {args.hour:02d} UTC" if args.hour is not None else ""
    fig.suptitle(
        f"lat {lat:.3f}°N, lon {lon:.3f}°E, elev {height:.0f} m  —  "
        f"Day {args.day} forecast  |  {season_lbl} season  |  {orog_lbl} orography{hour_lbl}",
        fontsize=24, y=1.01, fontweight='bold',
    )

    # ── Top panel: time series ────────────────────────────────────────────────
    ax_ts.plot(dates, obs, color=OBS_COLOR,  lw=3.5,  label='Observation', zorder=4)
    ax_ts.plot(dates, fc1, color=M1_COLOR,   lw=2.0,  label=lbl1, alpha=0.85, zorder=3)
    ax_ts.plot(dates, fc2, color=M2_COLOR,   lw=2.0,  label=lbl2, alpha=0.85, zorder=3)

    # Threshold line
    if T is not None:
        ax_ts.axhline(T, color='darkorange', lw=1.5, ls='--',
                      label=f'Threshold T = {T:.0f}°C', zorder=2)
        above = obs >= T
        if above.any():
            ax_ts.scatter(dates[above], obs[above], color='darkorange',
                          s=80, zorder=5, label='Obs ≥ T')

    ax_ts.set_ylabel('2m Temperature (°C)', fontsize=20)
    ax_ts.tick_params(axis='y', labelsize=18)
    ax_ts.grid(True, lw=0.4, alpha=0.4)
    ax_ts.legend(fontsize=17, loc='upper left', ncol=2)

    # Tight y-axis: 5% padding around actual data range
    all_vals = np.concatenate([fc1, fc2, obs])
    ypad = (all_vals.max() - all_vals.min()) * 0.07
    ax_ts.set_ylim(all_vals.min() - ypad, all_vals.max() + ypad)

    # Stats annotation
    stats_txt = (
        f"MAE:   {lbl1} = {mae1:.2f}°C    {lbl2} = {mae2:.2f}°C\n"
        f"Bias:  {lbl1} = {bias1:+.2f}°C   {lbl2} = {bias2:+.2f}°C\n"
        f"RMSE:  {lbl1} = {rmse1:.2f}°C    {lbl2} = {rmse2:.2f}°C"
    )
    ax_ts.text(0.99, 0.97, stats_txt, transform=ax_ts.transAxes,
               fontsize=16, va='top', ha='right', family='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#aaaaaa', alpha=0.9))

    # ── Bottom panel: model errors vs obs ────────────────────────────────────
    err1 = fc1 - obs   # IFS error
    err2 = fc2 - obs   # AIFS error
    ax_diff.bar(dates - pd.Timedelta(hours=9.6), err1,
                color=M1_COLOR, alpha=0.75, width=0.8, label=f'{lbl1} (bias={bias1:+.2f}°C)')
    ax_diff.bar(dates + pd.Timedelta(hours=9.6), err2,
                color=M2_COLOR, alpha=0.75, width=0.8, label=f'{lbl2} (bias={bias2:+.2f}°C)')
    ax_diff.axhline(0, color='black', lw=0.8)
    ax_diff.axhline(bias1, color=M1_COLOR, lw=1.5, ls='--', alpha=0.8)
    ax_diff.axhline(bias2, color=M2_COLOR, lw=1.5, ls='--', alpha=0.8)
    ax_diff.set_ylabel('Model − Obs\n(°C)', fontsize=20)
    ax_diff.tick_params(axis='y', labelsize=18)
    ax_diff.legend(fontsize=17, loc='upper right', ncol=2)
    ax_diff.grid(True, lw=0.4, alpha=0.4, axis='y')
    err_top = max(5.0, np.nanmax(np.concatenate([err1, err2])) * 1.15)
    err_bot = min(-5.0, np.nanmin(np.concatenate([err1, err2])) * 1.15)
    ax_diff.set_ylim(err_bot, err_top)

    ax_diff.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax_diff.xaxis.set_major_locator(mdates.MonthLocator())
    ax_diff.tick_params(axis='x', labelsize=18)
    fig.autofmt_xdate(rotation=0, ha='center')

    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    season_str = (args.season or 'all').lower()
    orog_str   = (args.orog   or 'all').lower()
    hour_str   = f"_{args.hour:02d}z" if args.hour is not None else ""
    suffix_str = f"_{args.suffix}" if args.suffix else ""
    loc_str    = f"{lat:.2f}N_{lon:.2f}E".replace('-', 'S').replace('.', 'p')
    out = out_dir / f"timeseries_{loc_str}_day{args.day}_{season_str}_{orog_str}{hour_str}{suffix_str}.png"
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {out}")


if __name__ == "__main__":
    main()
