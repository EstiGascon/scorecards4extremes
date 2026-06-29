"""Plot q99, q99.5 and q99.9 from the Italy tp annual climatology (5yr, 50% avail)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

CLIM_FILE = os.path.join(os.path.dirname(__file__),
                         "clim_tp_1_01_5years_2020_2024_50")

PERCENTILES = [
    ("q99",   "99th",   "q99"),
    ("q99p5", "99.5th", "q99p5"),
    ("q99p9", "99.9th", "q99p9"),
]


# ── Parse the GEO NCOLS file ─────────────────────────────────────────────────
def parse_geo_ncols(fpath):
    rows, header = [], None
    in_data = False
    with open(fpath) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('#DATA'):
                in_data = True
                continue
            if not in_data:
                if not line.startswith('#') and header is None and 'stnid' in line:
                    header = line.split()
                continue
            if line.startswith('#') or not line.strip():
                continue
            rows.append(line.split())
    return pd.DataFrame(rows, columns=header[:len(rows[0])])


df = parse_geo_ncols(CLIM_FILE)
df['latitude']  = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

for col, label, _ in PERCENTILES:
    if col not in df.columns:
        raise KeyError(
            f"Column '{col}' not found in {CLIM_FILE}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Re-run the obsclim job after the code update to regenerate the files."
        )
    df[col] = df[col].astype(float).replace(3e38, np.nan)
    df.loc[df[col] > 1e37, col] = np.nan

df = df.dropna(subset=[c for c, _, _ in PERCENTILES])

# ── Print area-mean stats ────────────────────────────────────────────────────
print(f"Italy tp24 obs climatology — 5 years 2020–2024, ≥50% availability")
print(f"Number of stations: {len(df)}\n")
print(f"{'Percentile':<12} {'Mean (mm/24h)':>15} {'Min':>8} {'Max':>8}")
print("-" * 46)
for col, label, _ in PERCENTILES:
    vals = df[col].dropna()
    print(f"{label:<12} {vals.mean():>15.2f} {vals.min():>8.2f} {vals.max():>8.2f}")
print()

# ── 3-panel map ──────────────────────────────────────────────────────────────
PROJ   = ccrs.PlateCarree()
EXTENT = [5.5, 19.5, 35.5, 47.5]

# Colour palette: white→light-blue→yellow→orange→red→dark-purple
cmap = plt.get_cmap('plasma')
vmax_global = float(np.nanpercentile(df['q99p9'].values, 98))

fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                         subplot_kw={'projection': PROJ})
fig.suptitle(
    'Italy tp24 obs climatology — 5 years 2020–2024, ≥50% availability\n'
    f'n = {len(df)} stations',
    fontsize=13, y=1.02,
)

for ax, (col, label, _) in zip(axes, PERCENTILES):
    ax.set_extent(EXTENT, crs=PROJ)

    # Background
    ax.add_feature(cfeature.LAND,   facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN,  facecolor='#c8e6f5', zorder=0)
    ax.add_feature(cfeature.LAKES,  facecolor='#c8e6f5', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6, edgecolor='#444444',
                   linestyle='-', zorder=3)

    vals = df[col].values
    mean_val = np.nanmean(vals)

    sc = ax.scatter(
        df['longitude'].values, df['latitude'].values,
        c=vals, cmap=cmap, s=30,
        vmin=0, vmax=vmax_global,
        edgecolors='none', zorder=4,
        transform=PROJ,
    )
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal',
                        fraction=0.046, pad=0.04, aspect=30)
    cbar.set_label('mm / 24h', fontsize=10)

    ax.set_title(
        f'{label} percentile\nMean = {mean_val:.1f} mm/24h',
        fontsize=11,
    )

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "italy_tp_q99_q995_q999.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)
