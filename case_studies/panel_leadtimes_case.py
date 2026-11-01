"""Lead-time panel for one heat case: IFS vs AIFS station maps at day 5/7/10.

For a fixed VALID date we shift the initialisation date so the forecast verifies
on the same day at each lead (valid = init + (day-1)). Rows = lead time, columns =
model. Each dot is a station coloured hit / miss / false alarm. The panel shows
that at short lead the two models are similar, and IFS's warm-extreme false alarms
grow with lead while AIFS stays conservative.
"""
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, ".")
from case_studies.case_study_utils import (
    load_per_station_thresholds, get_event_type,
    extract_forecast_values, classify_events,
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

COLORS = {"hit": "#2e7d32", "miss": "#e65100", "false_alarm": "#c62828", "correct_neg": "#cfcfcf"}

# (label, valid_date, [(init, lead), ...])  ordered short -> long lead
CASES = {
    "A": ("2026-07-13", [("20260709", 5), ("20260707", 7), ("20260704", 10)]),
    "B": ("2026-08-02", [("20260729", 5), ("20260727", 7), ("20260724", 10)]),
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


def draw(ax, lats, lons, obs, fc, T):
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.4", zorder=5)
    masks = classify_events(obs, fc, T, ET)
    for cat, size, a in [("correct_neg", 3, 0.35), ("hit", 16, 0.9),
                         ("miss", 16, 0.9), ("false_alarm", 16, 0.9)]:
        m = masks[cat]
        if m.sum() == 0:
            continue
        ax.scatter(lons[m], lats[m], c=COLORS[cat], s=size, alpha=a,
                   linewidths=0.15, edgecolors="white", transform=PROJ,
                   zorder=6 + ["correct_neg", "hit", "miss", "false_alarm"].index(cat))
    return int(masks["false_alarm"].sum()), int(masks["miss"].sum())


def make_panel(tag, valid, specs):
    fig, axes = plt.subplots(len(specs), 2, figsize=(11, 13),
                             subplot_kw={"projection": PROJ})
    for r, (init, day) in enumerate(specs):
        lats, lons, obs, f1, f2, T = load_case(init, day)
        for c, (fc, name) in enumerate([(f1, "IFS"), (f2, "AIFS")]):
            ax = axes[r, c]
            nfa, nmiss = draw(ax, lats, lons, obs, fc, T)
            ax.set_title(f"{name} — day {day}   (FA={nfa}, miss={nmiss})",
                         fontsize=11, fontweight="bold")
        axes[r, 0].text(-0.06, 0.5, f"Day {day}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=12, fontweight="bold")
    legend = [
        mpatches.Patch(color=COLORS["hit"], label="hit"),
        mpatches.Patch(color=COLORS["miss"], label="miss"),
        mpatches.Patch(color=COLORS["false_alarm"], label="false alarm"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Warm extremes valid {valid} (mx2t > p99, flat Europe) — "
                 f"forecast at day 5, 7, 10\nIFS (left) vs AIFS (right)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.95])
    out = f"{OUTDIR}/panel_leadtimes_case{tag}_valid{valid.replace('-', '')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


for tag, (valid, specs) in CASES.items():
    make_panel(tag, valid, specs)
