#!/usr/bin/env python3
"""
Two-message slide diagnostics for daily-maximum 10 m wind (IFS-oper, iekm, AIFS-oper).
=====================================================================================
Identical cases for the three models (merged pairwise extractions, see
threshold_sweep_3models.py), 6-hourly wind -> daily maximum per location and forecast
day (as mx10ff is scored), forecast days 1-5, Europe, 2025-03-01 to 2026-02-28.

1. Amplitude: distribution of daily-max wind, forecast vs observed, at increasing
   percentiles (pooled and per-location), by orography class.
2. Detection: hits / misses / false alarms at the per-station p99 of the observed
   daily-max climatology (clim_mx10ff_*, same lat/lon matching as the pipeline).
   Misses and false alarms are split into "near" (an event in the other source within
   +/-1 day at the same location, or the same day within --radius degrees) and "far",
   to quantify the timing/position-error (double-penalty) share.

Writes CSVs and two slide figures to --out.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import _style  # noqa: E402
import threshold_sweep_3models as T  # noqa: E402
from threshold import _load_clim_file_for_month  # noqa: E402

MODELS = T.MODELS
COLORS = T.COLORS
OROG = T.OROG
PCTS = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
BIAS_PCTS = [90, 95, 98, 99]
BIAS_FIXED = [8, 10, 12, 14, 16, 18, 20]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--clim-dir", default="obs_clim_local")
    p.add_argument("--clim-param", default="mx10ff")
    p.add_argument("--percentile", type=int, default=99)
    p.add_argument("--radius", type=float, default=0.25, help="Spatial tolerance (degrees) for 'near' events")
    p.add_argument("--out", default="results/mx10ff_3model_slides_d1to5")
    p.add_argument("--plot-only", action="store_true", help="Only redraw figures from the CSVs in --out")
    return p.parse_args()


def slide_amplitude(amp, out_png, max_pct=99.5):
    """Message 1: forecast strong winds are weaker than observed, increasingly so; AIFS most."""
    s = amp[(amp.method == "pooled") & (amp.terrain == "all") & (amp.pct <= max_pct)]
    obs = s[s.model == MODELS[0]].set_index("pct")["obs"]
    x = np.arange(len(obs))
    with plt.rc_context({"font.size": 15, "axes.labelsize": 15}):
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axhline(0, color="black", lw=1.2)
        ax.text(x[-1], 0.4, "observed", fontsize=13, va="bottom", ha="center")
        for m in MODELS:
            v = s[s.model == m].set_index("pct").loc[obs.index, "rel_diff_pct"].values
            ax.plot(x, v, "o-", color=COLORS[m], lw=3, ms=8, label=m)
            ax.annotate(f"{v[-1]:.0f}%", (x[-1], v[-1]), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=14, color=COLORS[m], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{o:.0f} m/s\n(p{p:g})" for p, o in obs.items()])
        ax.set_xlim(-0.3, len(x) - 0.4)
        ax.set_xlabel("Observed daily maximum 10 m wind (percentile of all observations)")
        ax.set_ylabel("Forecast minus observed (%)")
        ax.set_title("The stronger the wind, the more all models underestimate it",
                     fontsize=18, fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", frameon=False)
        fig.text(0.5, -0.02, "Same percentile of the forecast and of the observed distribution. Europe, all stations, "
                 "2025-03 to 2026-02, forecast days 1-5, identical cases", ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def slide_detection(tot, pct, out_png):
    """Message 2: detection (hits) and false alarms per 100 observed extremes, by terrain."""
    terrains = list(OROG)
    x = np.arange(len(terrains))
    w = 0.26
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, (col, title) in zip(axes, (("hits_per100obs", "Extremes detected\n(per 100 observed; the rest are missed)"),
                                         ("false_alarms_per100obs", "False alarms\n(per 100 observed extremes)"))):
            for i, m in enumerate(MODELS):
                t = tot[tot.model == m].set_index("terrain").reindex(terrains)
                v = t[col].values
                bars = ax.bar(x + (i - 1) * w, v, w, color=COLORS[m], label=m)
                if col == "false_alarms_per100obs":
                    ax.bar(x + (i - 1) * w, t["false_alarms_near_per100obs"].values, w, color="white",
                           alpha=0.45, hatch="//", edgecolor=COLORS[m], lw=0)
                ax.bar_label(bars, labels=[f"{u:.0f}" for u in v], padding=3, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([t.capitalize() for t in terrains])
            ax.set_title(title)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(loc="upper right", frameon=False)
        from matplotlib.patches import Patch
        axes[1].legend(handles=[Patch(facecolor="white", hatch="//", edgecolor="#555555",
                                      label="near miss (±1 day or ~25 km)")], loc="upper right", frameon=False)
        fig.suptitle(f"Daily maximum 10 m wind above the station {pct}th percentile, forecast days 1-5",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, "Threshold: per-station p99 of the observed daily-max wind climatology (2005-2024). "
                 "Europe, 2025-03 to 2026-02, identical cases", ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def slide_tradeoff(tot, pct, out_png):
    """Alternative to slide 2: mirrored bars, false alarms (left) vs detected extremes (right), per terrain."""
    terrains = list(OROG)
    ys, labels, groups = [], [], []
    y = 0.0
    for terrain in terrains:
        groups.append((terrain, y + 1))
        for m in MODELS:
            ys.append(y)
            labels.append(m)
            y += 1
        y += 0.9
    ys = np.array(ys)
    fa = np.concatenate([tot[tot.terrain == t].set_index("model").reindex(MODELS)["false_alarms_per100obs"].values
                         for t in terrains])
    hit = np.concatenate([tot[tot.terrain == t].set_index("model").reindex(MODELS)["hits_per100obs"].values
                          for t in terrains])
    colors = [COLORS[m] for m in labels]
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True,
                                       gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.32})
        axl.barh(ys, fa, color=colors, height=0.8)
        axr.barh(ys, hit, color=colors, height=0.8)
        for yy, v in zip(ys, fa):
            axl.text(v + 2, yy, f"{v:.0f}", va="center", ha="right", fontsize=14)
        for yy, v in zip(ys, hit):
            axr.text(v + 0.4, yy, f"{v:.0f}", va="center", ha="left", fontsize=14)
        axl.set_xlim(fa.max() * 1.18, 0)
        axr.set_xlim(0, hit.max() * 1.25)
        axl.set_title("\u2190 False alarms", color="#b03a2e", fontweight="bold")
        axr.set_title("Extremes detected \u2192", color="#2a7a2a", fontweight="bold")
        axl.set_xlabel("per 100 observed extremes")
        axr.set_xlabel("per 100 observed extremes (the rest are missed)")
        for ax in (axl, axr):
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.grid(axis="y", visible=False)
            ax.spines[["top", "right", "left"]].set_visible(False)
        for yy, m in zip(ys, labels):
            axr.text(-0.03, yy, m, transform=axr.get_yaxis_transform(), ha="right", va="center",
                     fontsize=14, fontweight="bold", color=COLORS[m])
        for terrain, yc in groups:
            axl.text(-0.03, yc, terrain.capitalize(), transform=axl.get_yaxis_transform(), rotation=90,
                     va="center", ha="right", fontsize=17, fontweight="bold", color="#444444")
        axl.invert_yaxis()
        fig.suptitle("Per 100 observed wind extremes: false alarms raised and extremes detected",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, f"Daily maximum 10 m wind above the station p{pct} of the 2005-2024 daily-max climatology. "
                 "Europe, 2025-03 to 2026-02, forecast days 1-5, identical cases",
                 ha="center", fontsize=10, color=_style.C_REF)
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.97)
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def slide_frequency_bias(counts, kind, out_png):
    """Forecast extreme events per 100 observed events, against threshold, by terrain."""
    s = counts[(counts.kind == kind) & (counts.terrain != "all")].copy()
    s["per100"] = 100 * s.n_fc / s.n_obs
    levels = sorted(s.level.unique())
    x = np.arange(len(levels))
    if kind == "fixed":
        fmt, xlabel = (lambda v: f"{v:g}"), "Threshold on daily maximum 10 m wind (m/s)"
    else:
        fmt, xlabel = (lambda v: f"p{v:g}"), "Threshold: station percentile of the daily-max wind climatology"
    ymax = max(110, np.nanmax(s["per100"].values) * 1.1)
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = plt.subplots(1, len(OROG), figsize=(16, 6.2), sharey=True)
        for ax, terrain in zip(axes, OROG):
            t = s[s.terrain == terrain]
            ax.axhline(100, color="black", lw=1.3, ls="--")
            ax.text(x[-1], 101.5, "as many as observed", fontsize=11, ha="right", va="bottom")
            for m in MODELS:
                v = t[t.model == m].set_index("level").reindex(levels)["per100"].values
                ax.plot(x, v, "o-", color=COLORS[m], lw=3, ms=8, label=m)
            n_obs = t[t.model == MODELS[0]].set_index("level").reindex(levels)["n_obs"].values
            ax.set_xticks(x)
            ax.set_xticklabels([f"{fmt(v)}\n{n / 1000:.1f}k" if n < 10000 else f"{fmt(v)}\n{n / 1000:.0f}k"
                                for v, n in zip(levels, n_obs)], fontsize=13)
            ax.set_ylim(0, ymax)
            ax.set_title(terrain.capitalize())
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Forecast events per 100 observed events")
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925))
        fig.supxlabel(f"{xlabel}   (second row: number of observed events)", fontsize=14, y=0.06)
        fig.suptitle("How many extreme-wind events do the models forecast, compared with what was observed?",
                     fontsize=18, fontweight="bold")
        fig.text(0.5, 0.005, "Above 100: forecast too often; below 100: too rarely (frequency bias x 100). "
                 "Daily max wind, Europe, 2025-03 to 2026-02, forecast days 1-5, identical cases",
                 ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.07, 1, 0.86))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def make_slides(out, pct):
    _style.apply_style()
    slide_amplitude(pd.read_csv(out / "amplitude_quantiles_3models.csv"), out / "slide1_underestimation_vs_wind_strength.png")
    tot = pd.read_csv(out / "detection_totals_3models.csv")
    slide_detection(tot, pct, out / "slide2_detection_false_alarms_by_terrain.png")
    slide_tradeoff(tot, pct, out / "slide2b_detection_vs_false_alarms.png")
    counts_csv = out / "threshold_event_counts_3models.csv"
    if counts_csv.exists():
        counts = pd.read_csv(counts_csv)
        slide_frequency_bias(counts, "fixed", out / "slide3_frequency_bias_fixed_thresholds.png")
        slide_frequency_bias(counts, "station_pct", out / "slide3b_frequency_bias_station_percentiles.png")


def daily_max(df, day):
    """6-hourly rows -> daily max per (location, init date) for one forecast day."""
    g = df.groupby(["lat", "lon", "date"], sort=False)
    out = g[["obs_value"] + MODELS].max()
    out["sdfor"] = g["sdfor"].first()
    out = out.reset_index()
    # Month of the forecast day (valid day = init + day - 1, 06-24 UTC window).
    valid = pd.to_datetime(out["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(day - 1, unit="D")
    out["valid_month"] = valid.dt.month.values
    out["date_idx"] = (valid - pd.Timestamp("2025-01-01")).dt.days.values
    return out


def attach_threshold(df, args, cache):
    """Per-row station-percentile thresholds thr_q{p}; 'thr' is the main --percentile one."""
    pcts = sorted(set(BIAS_PCTS) | {args.percentile})
    thr = {p: np.full(len(df), np.nan) for p in pcts}
    for month in np.unique(df["valid_month"]):
        if month not in cache:
            clim, _ = _load_clim_file_for_month(args.clim_dir, args.clim_param, 1, int(month), 20, "2005", "2024", 65)
            cache[month] = ({p: clim[f"q{p}"].astype(float).values for p in pcts},
                            cKDTree(np.column_stack([clim["latitude"].astype(float), clim["longitude"].astype(float)])))
        vals, tree = cache[month]
        rows = np.where(df["valid_month"].values == month)[0]
        d, i = tree.query(np.column_stack([df["lat"].values[rows], df["lon"].values[rows]]))
        for p in pcts:
            thr[p][rows] = np.where(d <= 0.1, vals[p][i], np.nan)
    for p in pcts:
        df[f"thr_q{p}"] = thr[p]
    df["thr"] = thr[args.percentile]
    return df[np.isfinite(df["thr"])].reset_index(drop=True)


def near_flags(loc, date_idx, lat, lon, cand_mask, target_mask, radius):
    """Full-length mask: candidate rows with a target event within +/-1 day (same loc) or same day within radius."""
    n_dates = date_idx.max() + 2
    target_keys = loc[target_mask].astype(np.int64) * n_dates + date_idx[target_mask]
    cand = np.where(cand_mask)[0]
    ck = loc[cand].astype(np.int64) * n_dates + date_idx[cand]
    near = np.isin(ck - 1, target_keys) | np.isin(ck + 1, target_keys)
    for d in np.unique(date_idx[cand]):
        tsel = target_mask & (date_idx == d)
        if not tsel.any():
            continue
        tree = cKDTree(np.column_stack([lat[tsel], lon[tsel]]))
        csel = cand[date_idx[cand] == d]
        hits = tree.query_ball_point(np.column_stack([lat[csel], lon[csel]]), r=radius, return_length=True)
        near[np.searchsorted(cand, csel)] |= hits > 0
    full = np.zeros(len(loc), bool)
    full[cand] = near
    return full


def main():
    args = parse_args()
    _style.apply_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        make_slides(out, args.percentile)
        return

    loader = argparse.Namespace(
        iekm_dir="extracted_points/10ff_local_p98obsclim_ifs_oper_iekm_new",
        iekm_file="10ff_ifs_oper_vs_iekm_day{d}.parquet",
        aifs_dir="extracted_points/10ff_local_p99obsclim_ifs_oper_aifs_new",
        aifs_file="10ff_ifs_oper_vs_aifs1.0_oper_day{d}.parquet")
    cache, frames = {}, []
    for d in args.days:
        dm = daily_max(T.load_day(loader, d), d)
        dm = attach_threshold(dm, args, cache)
        dm["day"] = d
        frames.append(dm)
        print(f"  day {d}: {len(dm):,} location-days with a p{args.percentile} threshold")
        gc.collect()
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["terrain"] = pd.cut(df["sdfor"], [OROG["flat"][0], OROG["hilly"][0], OROG["complex"][0], OROG["complex"][1]],
                           right=False, labels=list(OROG)).astype(str)

    # ---------------- 1. amplitude ----------------
    rows = []
    for terrain in ["all"] + list(OROG):
        s = df if terrain == "all" else df[df.terrain == terrain]
        obs_q = np.percentile(s["obs_value"], PCTS)
        for m in MODELS:
            fq = np.percentile(s[m], PCTS)
            for p, o, f in zip(PCTS, obs_q, fq):
                rows.append(dict(method="pooled", terrain=terrain, pct=p, model=m, obs=o, fc=f,
                                 rel_diff_pct=100 * (f - o) / o))
        # Per-location percentile, then median over locations (removes between-site mixing).
        g = s.groupby(["lat", "lon"])
        n = g.size()
        ok = n[n >= 300].index
        sl = s.set_index(["lat", "lon"]).loc[ok]
        gq = sl.groupby(level=[0, 1])
        for p in [90, 99]:
            oq = gq["obs_value"].quantile(p / 100)
            for m in MODELS:
                fq = gq[m].quantile(p / 100)
                rel = 100 * (fq - oq) / oq.where(oq > 0)
                rows.append(dict(method="per_location_median", terrain=terrain, pct=p, model=m,
                                 obs=float(oq.median()), fc=float(fq.median()),
                                 rel_diff_pct=float(rel.median())))
    amp = pd.DataFrame(rows)
    amp.to_csv(out / "amplitude_quantiles_3models.csv", index=False)

    # ---------------- 2. detection ----------------
    df["loc"] = df.groupby(["lat", "lon"], sort=False).ngroup()
    o_ev = (df["obs_value"] >= df["thr"]).values
    flags = {}
    for m in MODELS:
        f_ev = (df[m] >= df["thr"]).values
        miss_near = np.zeros(len(df), bool)
        fa_near = np.zeros(len(df), bool)
        for day in args.days:
            idx = np.where(df["day"].values == day)[0]
            loc, di = df["loc"].values[idx], df["date_idx"].values[idx]
            lat, lon = df["lat"].values[idx], df["lon"].values[idx]
            o, f = o_ev[idx], f_ev[idx]
            miss_near[idx] = near_flags(loc, di, lat, lon, o & ~f, f, args.radius)
            fa_near[idx] = near_flags(loc, di, lat, lon, ~o & f, o, args.radius)
        flags[m] = (f_ev, miss_near, fa_near)
        print(f"  near-event flags done for {m}")
    rows = []
    for (terrain, day), s_idx in df.groupby(["terrain", "day"]).groups.items():
        idx = np.asarray(s_idx)
        o = o_ev[idx]
        for m in MODELS:
            f, mn, fn = (a[idx] for a in flags[m])
            miss, fa = o & ~f, ~o & f
            rows.append(dict(terrain=terrain, day=day, model=m, n_obs_events=int(o.sum()),
                             hits=int((o & f).sum()), misses=int(miss.sum()), false_alarms=int(fa.sum()),
                             misses_near=int((miss & mn).sum()), false_alarms_near=int((fa & fn).sum())))
    det = pd.DataFrame(rows)
    det.to_csv(out / "detection_counts_by_day_3models.csv", index=False)
    tot = det.groupby(["terrain", "model"], as_index=False).sum(numeric_only=True).drop(columns="day")
    for c in ["hits", "misses", "false_alarms", "misses_near", "false_alarms_near"]:
        tot[f"{c}_per100obs"] = 100 * tot[c] / tot["n_obs_events"]
    tot["POD"] = tot.hits / tot.n_obs_events
    tot["FAR"] = tot.false_alarms / (tot.hits + tot.false_alarms)
    tot["freq_bias"] = (tot.hits + tot.false_alarms) / tot.n_obs_events
    tot["near_share_of_FA"] = tot.false_alarms_near / tot.false_alarms
    tot["near_share_of_misses"] = tot.misses_near / tot.misses
    tot.to_csv(out / "detection_totals_3models.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Amplitude, pooled: forecast vs observed daily-max wind at percentiles (rel. diff %) ===")
    print(amp[amp.method == "pooled"].pivot_table(index=["terrain", "pct"], columns="model",
                                                  values="rel_diff_pct").round(1).to_string())
    print("\nObserved percentile values (m/s):")
    print(amp[(amp.method == "pooled") & (amp.model == MODELS[0])].pivot_table(index="pct", columns="terrain",
                                                                              values="obs").round(1).to_string())
    print("\n=== Amplitude, per-location percentile (median over locations, rel. diff %) ===")
    print(amp[amp.method != "pooled"].pivot_table(index=["terrain", "pct"], columns="model",
                                                  values="rel_diff_pct").round(1).to_string())
    print(f"\n=== Detection at station p{args.percentile} of daily-max climatology, days {args.days} pooled ===")
    cols = ["terrain", "model", "n_obs_events", "hits_per100obs", "misses_per100obs", "false_alarms_per100obs",
            "POD", "FAR", "freq_bias", "near_share_of_FA", "near_share_of_misses"]
    print(tot[cols].round(3).to_string(index=False))

    # ---------------- 3. event counts per threshold (frequency bias) ----------------
    rows = []
    for terrain in ["all"] + list(OROG):
        s = df if terrain == "all" else df[df.terrain == terrain]
        levels = [("fixed", v, np.full(len(s), float(v))) for v in BIAS_FIXED] + \
                 [("station_pct", p, s[f"thr_q{p}"].values) for p in BIAS_PCTS]
        for kind, lvl, thr in levels:
            ok = np.isfinite(thr)
            o = s["obs_value"].values[ok] >= thr[ok]
            for m in MODELS:
                f = s[m].values[ok] >= thr[ok]
                rows.append(dict(kind=kind, level=lvl, terrain=terrain, model=m, n_cases=int(ok.sum()),
                                 n_obs=int(o.sum()), n_fc=int(f.sum()), hits=int((o & f).sum())))
    counts = pd.DataFrame(rows)
    counts.to_csv(out / "threshold_event_counts_3models.csv", index=False)
    c = counts[counts.model == MODELS[0]]
    print("\n=== Observed exceedance rate of the climatological station percentiles (expected 100-p %) ===")
    print(c[c.kind == "station_pct"].assign(obs_rate_pct=lambda t: 100 * t.n_obs / t.n_cases)
          .pivot_table(index="level", columns="terrain", values="obs_rate_pct").round(2).to_string())
    print("\n=== Forecast events per 100 observed ===")
    print(counts.assign(per100=lambda t: 100 * t.n_fc / t.n_obs)
          .pivot_table(index=["kind", "terrain", "level"], columns="model", values="per100").round(0).to_string())
    make_slides(out, args.percentile)


if __name__ == "__main__":
    main()
