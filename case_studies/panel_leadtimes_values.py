"""Lead-time panel of actual mx2t values for one heat case: obs / IFS / AIFS.

Fixed VALID date, initialisation shifted so the forecast verifies on the same day
at each lead (valid = init + (day-1)). Rows = lead time (5/7/10), columns =
Observed / IFS / AIFS. Dots are coloured by temperature on a shared scale; black
outline = station above the p99 extreme threshold. Shows IFS keeping/creating a
hot field at long lead while AIFS relaxes toward the background.
"""
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, ".")
from case_studies.case_study_utils import (
    load_per_station_thresholds, get_event_type, extract_forecast_values,
    aggregate_subdaily_to_daily,
)
from case_studies.find_case_studies import apply_filters

CFG = yaml.safe_load(open(
    "configs/deterministic/config_mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_JJA2026_europe.yaml"))
DDIR = "extracted_points/mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_JJA2026_europe"
OUTDIR = "case_study_output/mx2t_p99_JJA2026_flat_ifsvsaifs"
ET = get_event_type(CFG)
PROJ = ccrs.PlateCarree()
EXTENT = [-25, 40, 35, 72]
CMAP = "seismic"

CASES = {
    "A": ("2026-07-13", [("20260709", 5), ("20260707", 7), ("20260704", 10)]),
    "B": ("2026-08-02", [("20260729", 5), ("20260727", 7), ("20260724", 10)]),
    # AIFS-worse (more false alarms than IFS)
    "C": ("2026-06-20", [("20260616", 5), ("20260614", 7), ("20260611", 10)]),
    "D": ("2026-07-11", [("20260707", 5), ("20260705", 7), ("20260702", 10)]),
}


def load_case(init, day):
    df = pd.read_parquet(f"{DDIR}/mx2t_ifs_oper_vs_aifs1.0_oper_day{day}.parquet")
    df = apply_filters(df, season="JJA", orog="low", config=CFG)
    df = df[df["date"] == init].copy()
    df = aggregate_subdaily_to_daily(df, CFG)
    T = load_per_station_thresholds(CFG, df)
    obs = df["obs_value"].values.astype(float)
    v = ~np.isnan(T) & ~np.isnan(obs)
    df, T, obs = df[v], T[v], obs[v]
    f1, f2 = extract_forecast_values(df, "deterministic")
    return df["lat"].values, df["lon"].values, obs, f1, f2, T


def draw(ax, lats, lons, vals, ext, vmin, vmax):
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.4", zorder=5)
    sc = ax.scatter(lons, lats, c=vals, cmap=CMAP, vmin=vmin, vmax=vmax,
                    s=8, alpha=0.75, linewidths=0, transform=PROJ, zorder=3)
    if ext.sum() > 0:
        ax.scatter(lons[ext], lats[ext], c=vals[ext], cmap=CMAP, vmin=vmin, vmax=vmax,
                   s=34, alpha=0.95, linewidths=0.5, edgecolors="black",
                   transform=PROJ, zorder=4)
    return sc


def make_panel(tag, valid, specs):
    data = [(day,) + load_case(init, day) for init, day in specs]

    # shared diverging scale over all panels, centred on the median
    allv = np.concatenate([np.concatenate([d[3], d[4], d[5]]) for d in data])
    allv = allv[np.isfinite(allv)]
    vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)
    mid = np.median(allv)
    half = max(abs(vmax - mid), abs(mid - vmin))
    vmin, vmax = mid - half, mid + half

    fig, axes = plt.subplots(len(specs), 3, figsize=(16, 13),
                             subplot_kw={"projection": PROJ})
    col_names = ["Observed", "IFS", "AIFS"]
    sc = None
    for r, (day, lats, lons, obs, f1, f2, T) in enumerate(data):
        fields = [(obs, obs >= T), (f1, f1 >= T), (f2, f2 >= T)]
        for c, (vals, ext) in enumerate(fields):
            ax = axes[r, c]
            sc = draw(ax, lats, lons, vals, ext, vmin, vmax)
            ttl = col_names[c] if c == 0 else f"{col_names[c]}  ({int(ext.sum())} extreme)"
            ax.set_title(ttl, fontsize=11, fontweight="bold")
        axes[r, 0].text(-0.08, 0.5, f"Day {day}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold")

    cax = fig.add_axes([0.92, 0.20, 0.014, 0.6])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("mx2t  (°C)", fontsize=10)
    ext_marker = mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                               markersize=9, markeredgecolor="black", markeredgewidth=0.9,
                               label="station above p99 extreme threshold (black outline)")
    fig.legend(handles=[ext_marker], loc="lower center", fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"mx2t values valid {valid} (flat Europe) — forecast at day 5, 7, 10\n"
                 f"Observed vs IFS vs AIFS",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.04, 0.91, 0.95])
    out = f"{OUTDIR}/panel_leadtimes_values_case{tag}_valid{valid.replace('-', '')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


for tag, (valid, specs) in CASES.items():
    make_panel(tag, valid, specs)
