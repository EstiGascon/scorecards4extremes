#!/usr/bin/env python3
"""
Maps of Europe: annual mean 98th percentile 10m wind speed per station,
filtered to stations where p98 >= 8 m/s and >= 12 m/s.
Also produces versions with coastal stations removed (lsm >= 0.9).
Output: case_study_output/timeseries/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from scipy.spatial import cKDTree

CLIM_DIR  = Path("./obs_clim_local")
PARQUET   = Path("./extracted_points/"
                 "10ff_local_p98obsclim_ifs_oper_aifs_new/"
                 "10ff_ifs_oper_vs_aifs1.0_oper_day1.parquet")
OUT_DIR   = Path("case_study_output/timeseries")

LAT_MIN, LAT_MAX =  30.0,  72.0
LON_MIN, LON_MAX = -15.0,  45.0
MISSING = 3e+38
LSM_THRESHOLD = 0.9


def load_all_months(pct_col='q98'):
    dfs = []
    for m in range(1, 13):
        fname = CLIM_DIR / f"clim_10ff_1_{m:02d}_20years_2005_2024_65"
        raw = pd.read_csv(fname, comment='#', sep=r'\s+', header=0)
        raw = raw[pd.to_numeric(raw['stnid'], errors='coerce').notna()].copy()
        for col in ['latitude', 'longitude', pct_col]:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
        raw.loc[raw[pct_col] >= MISSING * 0.9, pct_col] = np.nan
        dfs.append(raw[['stnid', 'latitude', 'longitude', pct_col]].rename(
            columns={pct_col: 'value'}))
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['stnid', 'latitude', 'longitude'])['value'].mean().reset_index()
    df = df[
        df['latitude'].between(LAT_MIN, LAT_MAX) &
        df['longitude'].between(LON_MIN, LON_MAX) &
        df['value'].notna() &
        (df['value'] >= 0)
    ]
    return df


def attach_lsm(df):
    """Nearest-neighbour join: assign lsm from parquet to each climatology station."""
    print("  Loading lsm values from parquet ...")
    pq = pd.read_parquet(PARQUET, columns=['station_id', 'lat', 'lon', 'lsm'])
    pq = pq.drop_duplicates('station_id')[['lat', 'lon', 'lsm']].dropna()
    tree = cKDTree(pq[['lat', 'lon']].values)
    _, idx = tree.query(df[['latitude', 'longitude']].values, k=1)
    df = df.copy()
    df['lsm'] = pq['lsm'].values[idx]
    return df


def make_map(df_plot, title, out_path):
    vmin = np.nanpercentile(df_plot['value'], 2)
    vmax = np.nanpercentile(df_plot['value'], 98)
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=15.0, central_latitude=50.0))
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    for feat, kw in [
        (cfeature.NaturalEarthFeature("physical", "ocean",    "50m"), dict(facecolor="#d6eaf8")),
        (cfeature.NaturalEarthFeature("physical", "land",     "50m"), dict(facecolor="#f0ede8")),
        (cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m"),
             dict(edgecolor="#888888", facecolor="none", lw=0.7)),
        (cfeature.NaturalEarthFeature("physical", "coastline","50m"),
             dict(edgecolor="#444444", facecolor="none", lw=0.8)),
    ]:
        ax.add_feature(feat, zorder=0, **kw)
    gl = ax.gridlines(draw_labels=True, lw=0.4, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 12}; gl.ylabel_style = {"size": 12}
    sc = ax.scatter(
        df_plot['longitude'], df_plot['latitude'],
        c=df_plot['value'], cmap='YlOrRd',
        vmin=vmin, vmax=vmax,
        s=18, alpha=0.9, linewidths=0,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.72, pad=0.02, extend='both')
    cbar.set_label("10m Wind Speed 98th percentile (m/s)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title(f"{title}\nn = {len(df_plot):,} stations", fontsize=14, fontweight='bold', pad=10)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}  (n={len(df_plot):,}, {df_plot['value'].min():.1f}–{df_plot['value'].max():.1f} m/s)")


# ── Load & annotate ────────────────────────────────────────────────────────────
print("Loading all 12 months of 10ff obs climatology — q98 ...")
df = load_all_months('q98')
print(f"Total stations (Europe): {len(df):,}")
df = attach_lsm(df)

title_base = "98th percentile 10m Wind Speed — Annual mean  (obs climatology 2005–2024)"

for thr, tag in [(8.0, "ge8ms"), (12.0, "ge12ms")]:
    print(f"\n--- p98 >= {thr:.0f} m/s ---")
    sub = df[df['value'] >= thr]
    print(f"  All stations:                     {len(sub):,}")
    make_map(sub,
             f"{title_base} | p98 ≥ {thr:.0f} m/s",
             OUT_DIR / f"map_10ff_p98_{tag}.png")

    sub_land = sub[sub['lsm'] >= LSM_THRESHOLD]
    print(f"  Coastal removed (lsm≥{LSM_THRESHOLD}):  {len(sub_land):,}  (removed {len(sub)-len(sub_land):,})")
    make_map(sub_land,
             f"{title_base} | p98 ≥ {thr:.0f} m/s | coastal stations removed (lsm≥{LSM_THRESHOLD})",
             OUT_DIR / f"map_10ff_p98_{tag}_no_coastal.png")
