#!/usr/bin/env python3
"""
Map of the 3 case study locations used in the time series analysis.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

LOCATIONS = [
    {"lat": 48.055, "lon": 14.132, "label": "Flat inland\n(Linz, Austria)",  "marker": "o", "color": "#1565C0"},
    {"lat": 45.931, "lon":  7.663, "label": "Mountain top\n(Aosta Alps)",     "marker": "^", "color": "#b71c1c"},
    {"lat": 46.010, "lon": 11.660, "label": "Alpine valley\n(Adige, Trento)", "marker": "s", "color": "#e65100"},
]

# annotation offsets (points): positive x = right, positive y = up
OFFSETS = [
    ( 55,  15),   # Linz — right
    (-20,  45),   # Aosta — upper-left
    ( 55, -25),   # Trento — lower-right
]

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
    central_longitude=13.0, central_latitude=47.0))

ax.set_extent([3, 25, 42, 53], crs=ccrs.PlateCarree())

land_50m = cfeature.NaturalEarthFeature(
    "physical", "land", "50m", facecolor="#f5f0e8")
ocean_50m = cfeature.NaturalEarthFeature(
    "physical", "ocean", "50m", facecolor="#cde6f5")
borders_50m = cfeature.NaturalEarthFeature(
    "cultural", "admin_0_boundary_lines_land", "50m",
    edgecolor="#777777", facecolor="none", lw=0.8)
coast_50m = cfeature.NaturalEarthFeature(
    "physical", "coastline", "50m",
    edgecolor="#444444", facecolor="none", lw=0.9)
lakes_50m = cfeature.NaturalEarthFeature(
    "physical", "lakes", "50m", facecolor="#cde6f5", edgecolor="#90caf9", lw=0.4)
rivers_50m = cfeature.NaturalEarthFeature(
    "physical", "rivers_lake_centerlines", "50m",
    edgecolor="#90caf9", facecolor="none", lw=0.5)

for feat in [ocean_50m, land_50m, lakes_50m, rivers_50m, borders_50m, coast_50m]:
    ax.add_feature(feat, zorder=0)

gl = ax.gridlines(draw_labels=True, lw=0.4, color="gray",
                  alpha=0.5, linestyle="--", crs=ccrs.PlateCarree())
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 13}
gl.ylabel_style = {"size": 13}

for loc, (dx, dy) in zip(LOCATIONS, OFFSETS):
    ax.plot(loc["lon"], loc["lat"],
            transform=ccrs.PlateCarree(),
            marker=loc["marker"],
            color=loc["color"],
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=1.8,
            zorder=6,
            linestyle="none")

    txt = ax.annotate(
        loc["label"],
        xy=(loc["lon"], loc["lat"]),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=12, fontweight="bold", color=loc["color"],
        ha="left" if dx > 0 else "right",
        arrowprops=dict(arrowstyle="-", color=loc["color"],
                        lw=1.2, shrinkA=7, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=loc["color"], alpha=0.88, lw=1.3),
        zorder=7,
    )
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

ax.set_title("Case study locations — 2m temperature IFS vs AIFS",
             fontsize=16, fontweight="bold", pad=12)

out = Path("case_study_output/timeseries/casestudy_map.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
