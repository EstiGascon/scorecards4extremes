#!/usr/bin/env python3
"""
Two maps of Europe:
  1) DJF mean of q1  (1st  percentile) from obs climatology
  2) JJA mean of q99 (99th percentile) from obs climatology
Source: obs_clim_local/clim_2t_1_{MM}_20years_2005_2024_65
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

CLIM_DIR = Path("./obs_clim_local")
OUT_DIR  = Path("case_study_output/timeseries")

# Europe bounding box
LAT_MIN, LAT_MAX = 30.0, 72.0
LON_MIN, LON_MAX = -15.0, 45.0


def load_months(months, pct_col):
    """Load given months, average pct_col per station, filter to Europe."""
    dfs = []
    for m in months:
        fname = CLIM_DIR / f"clim_2t_1_{m:02d}_20years_2005_2024_65"
        raw = pd.read_csv(fname, comment='#', sep=r'\s+', header=0)
        # drop metadata rows (non-numeric stnid)
        raw = raw[pd.to_numeric(raw['stnid'], errors='coerce').notna()].copy()
        for col in ['latitude', 'longitude', pct_col]:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
        dfs.append(raw[['stnid', 'latitude', 'longitude', pct_col]].rename(
            columns={pct_col: 'value'}))
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['stnid', 'latitude', 'longitude'])['value'].mean().reset_index()
    df = df[
        df['latitude'].between(LAT_MIN, LAT_MAX) &
        df['longitude'].between(LON_MIN, LON_MAX) &
        df['value'].notna()
    ]
    return df


def make_map(df, title, cmap, out_path, cbar_label, vmin=None, vmax=None):
    vlo = vmin if vmin is not None else np.nanpercentile(df['value'], 2)
    vhi = vmax if vmax is not None else np.nanpercentile(df['value'], 98)

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

    gl = ax.gridlines(draw_labels=True, lw=0.4, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 12}
    gl.ylabel_style = {"size": 12}

    sc = ax.scatter(
        df['longitude'], df['latitude'],
        c=df['value'], cmap=cmap,
        vmin=vlo, vmax=vhi,
        s=18, alpha=0.9, linewidths=0,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical',
                        shrink=0.72, pad=0.02, extend='both')
    cbar.set_label(cbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    n = len(df)
    ax.set_title(f"{title}  (n = {n:,} stations)", fontsize=15, fontweight='bold', pad=10)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}  (n={n}  range {vlo:.1f}–{vhi:.1f} °C)")


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading DJF (months 12, 1, 2) — q1 ...")
djf = load_months([12, 1, 2], 'q1')

print("Loading JJA (months 6, 7, 8) — q99 ...")
jja = load_months([6, 7, 8], 'q99')

# ── Map 1: DJF q1 ─────────────────────────────────────────────────────────────
make_map(
    djf,
    title="1st percentile 2m Temperature — DJF  (obs climatology 2005–2024)",
    cmap='plasma_r',        # dark-purple = warmest, yellow = coldest
    out_path=OUT_DIR / "map_2t_p01_djf.png",
    cbar_label="2m Temperature 1st percentile (°C)",
)

# ── Map 2: JJA q99 ──────────────────────────────────────────────────────────────
make_map(
    jja,
    title="99th percentile 2m Temperature — JJA  (obs climatology 2005–2024)",
    cmap='plasma',          # dark-purple = coolest, bright-yellow = hottest
    out_path=OUT_DIR / "map_2t_p99_jja.png",
    cbar_label="2m Temperature 99th percentile (°C)",
)
