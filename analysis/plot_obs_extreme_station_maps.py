"""Three station maps over Europe (observations only), JJA 2026 mx2t.

  1. observed p99 CLIMATOLOGY of daily-max 2m temperature (the same local obs
     climatology, q99, used for the p99 filtering: 20 years 2005-2024, JJA
     months 06/07/08 averaged) at each station.
  2. stations with observed mx2t > 35 C in JJA 2026, coloured by number of cases.
  3. stations with observed mx2t > 38 C in JJA 2026, coloured by number of cases.
"""
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, "src")
from threshold import _load_clim_file_for_month  # noqa: E402

SRC = "extracted_points/mx2t_ens_quaver_p99obsclim_aifsvsifs_JJA2026_europe/mx2t_ifsens4aifs_vs_aifs_ens_ens_day1.parquet"
CLIM_DIR = "./obs_clim_local"
OUTDIR = "results/obs_climatology_maps_JJA2026_europe"
EXTENT = [-25, 45, 32, 72]          # lon_min, lon_max, lat_min, lat_max
PROJ = ccrs.PlateCarree()

os.makedirs(OUTDIR, exist_ok=True)

# --- load observations only, one value per station and date ---
df = pq.read_table(SRC, columns=["station_id", "lat", "lon", "date", "obs_value"]).to_pandas()
df = df.dropna(subset=["obs_value"])
df = df.groupby(["station_id", "date"], as_index=False).agg(
    obs_value=("obs_value", "max"), lat=("lat", "first"), lon=("lon", "first"))
print(f"obs_value range: {df.obs_value.min():.1f} .. {df.obs_value.max():.1f}  "
      f"(median {df.obs_value.median():.1f})  -> assuming degrees C")

# --- per-station observed exceedance counts (JJA 2026) ---
g = df.groupby("station_id")
stat = pd.DataFrame({
    "lat": g["lat"].first(), "lon": g["lon"].first(),
    "n_gt35": g["obs_value"].apply(lambda s: int((s > 35).sum())),
    "n_gt38": g["obs_value"].apply(lambda s: int((s > 38).sum())),
}).reset_index()
stat = stat[(stat.lon.between(EXTENT[0], EXTENT[1])) & (stat.lat.between(EXTENT[2], EXTENT[3]))]
print(f"stations in domain: {len(stat)}  |  >35C: {(stat.n_gt35>0).sum()}  |  >38C: {(stat.n_gt38>0).sum()}")


def base_map(ax):
    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4", zorder=5)
    ax.set_extent(EXTENT, crs=PROJ)


def make_map(sub, value, cmap, title, cbar_label, fname, vmin=None, vmax=None, size=14):
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=PROJ)
    base_map(ax)
    sc = ax.scatter(sub.lon, sub.lat, c=value, cmap=cmap, s=size, vmin=vmin, vmax=vmax,
                    edgecolor="k", linewidth=0.2, transform=PROJ, zorder=10)
    cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.75, pad=0.02)
    cb.set_label(cbar_label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    out = f"{OUTDIR}/{fname}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# 1) p99 from the local obs climatology (q99, JJA months 06/07/08 averaged)
frames = []
for month in (6, 7, 8):
    dfc, _ = _load_clim_file_for_month(CLIM_DIR, "mx2t", 1, month, 20, "2005", "2024", 65)
    frames.append(dfc.reset_index()[["stnid", "latitude", "longitude", "q99"]])
allc = pd.concat(frames, ignore_index=True)
for c in ("latitude", "longitude", "q99"):
    allc[c] = pd.to_numeric(allc[c], errors="coerce")
clim = allc.groupby("stnid", as_index=False).agg(
    latitude=("latitude", "first"), longitude=("longitude", "first"), p99=("q99", "mean"))
clim = clim.dropna(subset=["p99", "latitude", "longitude"])
clim = clim[clim.longitude.between(EXTENT[0], EXTENT[1]) & clim.latitude.between(EXTENT[2], EXTENT[3])]
print(f"clim stations: {len(clim)}  q99 range {clim.p99.min():.1f}..{clim.p99.max():.1f} (median {clim.p99.median():.1f})")

fig = plt.figure(figsize=(11, 9))
ax = plt.axes(projection=PROJ)
base_map(ax)
levels = np.arange(15, 43, 2.0)
cmap = plt.get_cmap("turbo")
norm = BoundaryNorm(levels, cmap.N, extend="both")
sc = ax.scatter(clim.longitude, clim.latitude, c=clim.p99, cmap=cmap, norm=norm,
                s=16, edgecolor="k", linewidth=0.15, transform=PROJ, zorder=10)
cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.78, pad=0.02, ticks=levels, extend="both")
cb.set_label("p99 of daily-max 2m T (\u00b0C)  [obs climatology 2005-2024, JJA]")
ax.set_title("Observed p99 climatology of mx2t (JJA, 2005-2024) \u2014 Europe",
             fontsize=13, fontweight="bold")
out = f"{OUTDIR}/obs_p99_mx2t_JJA2026_europe.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", out)

# 2) > 35 C counts
s35 = stat[stat.n_gt35 > 0]
make_map(s35, s35.n_gt35, "hot_r",
         f"Stations with observed mx2t > 35 \u00b0C — JJA 2026, Europe  (n={len(s35)})",
         "number of days > 35 \u00b0C",
         "obs_gt35_counts_mx2t_JJA2026_europe.png",
         vmin=1, vmax=float(np.nanpercentile(s35.n_gt35, 98)), size=20)

# 3) > 38 C counts
s38 = stat[stat.n_gt38 > 0]
make_map(s38, s38.n_gt38, "hot_r",
         f"Stations with observed mx2t > 38 \u00b0C — JJA 2026, Europe  (n={len(s38)})",
         "number of days > 38 \u00b0C",
         "obs_gt38_counts_mx2t_JJA2026_europe.png",
         vmin=1, vmax=max(2, float(np.nanpercentile(s38.n_gt38, 98))), size=26)
