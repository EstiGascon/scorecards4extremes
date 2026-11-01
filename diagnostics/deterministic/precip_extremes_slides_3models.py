#!/usr/bin/env python3
"""
Why does AIFS score better than iekm (and IFS) for extreme 24 h precipitation?
==============================================================================
Three models on identical cases (the IFS-vs-iekm and IFS-vs-AIFS native-grid quaver
extractions share the IFS column), 24 h precipitation, forecast days 1-5, Europe,
2025-03-09 to 2026-03-08, coastal stations removed (lsm > 0.9), seasons by init month
(as in the scorecards). Four slide diagnostics:

  P1  twMAE split into misses / hits / false alarms (read from the scorecard CSVs)
  P2  forecast events per 100 observed events, fixed thresholds 1-80 mm/24 h
  P3  forecast vs observed precipitation amount at the same percentile
  P4  hits and false alarms when an event within r km (same day) also counts, at the
      per-station p99 of the 2005-2024 observed climatology (position-error tolerance)

Usage:
    python diagnostics/deterministic/precip_extremes_slides_3models.py [--plot-only]
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
from matplotlib.patches import Patch

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import _style  # noqa: E402
from threshold import _load_clim_file_for_month  # noqa: E402

MODELS = ["IFS-oper", "iekm", "AIFS-oper"]
COLORS = {"IFS-oper": _style.C_FC1, "AIFS-oper": _style.C_FC2, "iekm": "#CC79A7"}
# Names shown on slides; internal keys (and CSV contents) stay unchanged.
LABELS = {"IFS-oper": "IFS-oper", "iekm": "IFS 4.4 km", "AIFS-oper": "AIFS-oper"}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
OROG = {"flat": (0, 40), "hilly": (40, 120), "complex": (120, 3000)}
FIXED_MM = [1, 5, 10, 20, 30, 50, 80]
QPCTS = [90, 95, 98, 99, 99.5, 99.9]
RADII_KM = [0, 25, 50, 100]
FOOT = "Europe, 2025-03 to 2026-03, forecast days 1-5, identical cases, coastal stations removed"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iekm-dir", default="extracted_points/tp24_method3_quaverextract_p99obsclim_ifsvsiekm_fullyear")
    p.add_argument("--aifs-dir", default="extracted_points/tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear")
    p.add_argument("--iekm-results", default="results/tp24_method3_quaverextract_p99obsclim_ifsvsiekm_fullyear")
    p.add_argument("--aifs-results", default="results/tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear")
    p.add_argument("--days", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--percentile", type=int, default=99)
    p.add_argument("--out", default="results/tp24_3model_slides_d1to5")
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def load_day(args, d, cache):
    cols = ["date", "step", "lat", "lon", "sdfor", "lsm", "obs_value", "fc1_value", "fc2_value"]
    a = pd.read_parquet(Path(args.iekm_dir) / f"tp24_ifs_oper_vs_iekm_day{d}.parquet", columns=cols)
    b = pd.read_parquet(Path(args.aifs_dir) / f"tp24_ifs_oper_vs_aifs1.0_oper_day{d}.parquet", columns=cols)
    keys = ["date", "step", "lat", "lon"]
    aligned = (len(a) == len(b) and all(np.array_equal(a[k].values, b[k].values) for k in keys)
               and np.allclose(a["obs_value"].values, b["obs_value"].values, equal_nan=True))
    if aligned:
        df = a.rename(columns={"fc1_value": "IFS-oper", "fc2_value": "iekm"})
        df["AIFS-oper"] = b["fc2_value"].values
        ifs_b = b["fc1_value"].values
    else:
        print(f"  day {d}: rows not aligned, merging on {keys}")
        b = b.drop_duplicates(keys)[keys + ["fc1_value", "fc2_value"]]
        df = a.drop_duplicates(keys).merge(b, on=keys, suffixes=("", "_b"))
        df = df.rename(columns={"fc1_value": "IFS-oper", "fc2_value": "iekm", "fc2_value_b": "AIFS-oper"})
        ifs_b = df.pop("fc1_value_b").values
    # Both extractions carry the same IFS field; a few rows can differ where a station is
    # equidistant between grid points (different nearest point). Drop those; fail if many.
    diff = np.abs(df["IFS-oper"].values - ifs_b) > 1e-3
    if diff.mean() > 1e-4:
        raise RuntimeError(f"IFS differs between extractions on day {d} in {diff.sum()} rows")
    df = df.loc[~diff]
    mism = int(diff.sum())
    del a, b
    ok = (df["lsm"] > 0.9) & df["obs_value"].between(0, 800)
    for m in MODELS:
        ok &= df[m].between(0, 800)
    df = df.loc[ok].reset_index(drop=True)
    init = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["season"] = init.dt.month.map({m: s for s, ms in SEASONS.items() for m in ms}).values
    df["valid_month"] = (init + pd.to_timedelta(df["step"], unit="h")).dt.month.values
    df["date_idx"] = (init - pd.Timestamp("2025-01-01")).dt.days.values
    df["terrain"] = pd.cut(df["sdfor"], [0, 40, 120, 3000], right=False, labels=list(OROG)).astype(str)
    df["thr"] = station_threshold(df, args, cache)
    df["day"] = d
    print(f"  day {d}: {len(df):,} pairs ({mism} rows dropped: IFS differs between files)")
    return df


def station_threshold(df, args, cache):
    thr = np.full(len(df), np.nan)
    col = f"q{args.percentile}"
    for month in np.unique(df["valid_month"]):
        if month not in cache:
            clim, _ = _load_clim_file_for_month("obs_clim_local", "tp", 1, int(month), 20, "2005", "2024", 65)
            cache[month] = (clim[col].astype(float).values,
                            cKDTree(np.column_stack([clim["latitude"].astype(float), clim["longitude"].astype(float)])))
        vals, tree = cache[month]
        rows = np.where(df["valid_month"].values == month)[0]
        dist, idx = tree.query(np.column_stack([df["lat"].values[rows], df["lon"].values[rows]]))
        thr[rows] = np.where(dist <= 0.1, vals[idx], np.nan)
    return thr


def unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def tolerance_counts(df):
    """Per season and radius: observed events detected within r, forecast events with no observed event within r."""
    df = df[np.isfinite(df["thr"].values)]
    o_all = (df["obs_value"] >= df["thr"]).values
    f_all = {m: (df[m] >= df["thr"]).values for m in MODELS}
    xyz = unit_xyz(df["lat"].values, df["lon"].values)
    chord = {r: 2 * np.sin(r / 6371.0 / 2) for r in RADII_KM if r > 0}
    keys = df["date_idx"].values.astype(np.int64) * 10 + df["day"].values
    order = np.argsort(keys, kind="stable")
    bounds = np.flatnonzero(np.diff(keys[order])) + 1
    seasons = df["season"].values
    acc = {}
    for grp in np.split(order, bounds):
        s = seasons[grp[0]]
        o = o_all[grp]
        oi = np.flatnonzero(o)
        otree = cKDTree(xyz[grp][oi]) if len(oi) else None
        for m in MODELS:
            f = f_all[m][grp]
            fi = np.flatnonzero(f)
            ftree = cKDTree(xyz[grp][fi]) if len(fi) else None
            for r in RADII_KM:
                k = (s, m, r)
                c = acc.setdefault(k, np.zeros(4, np.int64))  # n_obs, detected, n_fc, false_alarms
                c[0] += len(oi)
                c[2] += len(fi)
                if r == 0:
                    c[1] += int((o & f).sum())
                    c[3] += int((f & ~o).sum())
                    continue
                if len(oi) and ftree is not None:
                    c[1] += int((ftree.query_ball_point(xyz[grp][oi], chord[r], return_length=True) > 0).sum())
                if len(fi):
                    near = (otree.query_ball_point(xyz[grp][fi], chord[r], return_length=True) > 0) if otree else 0
                    c[3] += int(len(fi) - np.sum(near))
    rows = [dict(season=s, model=m, radius_km=r, n_obs=v[0], detected=v[1], n_fc=v[2], false_alarms=v[3])
            for (s, m, r), v in acc.items()]
    out = pd.DataFrame(rows)
    allsea = out.groupby(["model", "radius_km"], as_index=False)[["n_obs", "detected", "n_fc", "false_alarms"]].sum()
    allsea["season"] = "all"
    return pd.concat([out, allsea], ignore_index=True)


# ----------------------------------------------------------------------------- slides
def _season_axes(n=4, h=5.6):
    return plt.subplots(1, n, figsize=(17, h), sharey=True)


def slide_p1(args, out_png):
    """twMAE components, pooled over terrain (weighted by sample size), days 1-5, per season."""
    rows = []
    for s in SEASONS:
        acc = {m: np.zeros(3) for m in MODELS}
        w_tot = 0.0
        for o in OROG:
            a = pd.read_csv(Path(args.aifs_results) / f"scores_by_leadtime_tp24_{s}_{o}.csv")
            a = a[a.forecast_day.isin(args.days)].set_index("forecast_day")
            i = pd.read_csv(Path(args.iekm_results) / f"scores_by_leadtime_tp24_{s}_{o}.csv")
            i = i[i.forecast_day.isin(args.days)].set_index("forecast_day").reindex(a.index)
            w = a["n_samples"].values
            w_tot += w.sum()
            for m, t, suf in (("IFS-oper", a, "fc1"), ("AIFS-oper", a, "fc2"), ("iekm", i, "fc2")):
                acc[m] += [np.sum(t[f"twMAE_{c}_{suf}"].values * w) for c in ("misses", "hits", "FA")]
        for m in MODELS:
            mis, hit, fa = acc[m] / w_tot
            rows.append(dict(season=s, model=m, misses=mis, hits=hit, FA=fa))
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out) / "twmae_components_3models.csv", index=False)
    comp_col = {"misses": _style.C_MISS, "hits": _style.C_HIT, "FA": _style.C_FA}
    comp_lbl = {"misses": "misses", "hits": "hits (amount error)", "FA": "false alarms"}
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = _season_axes()
        x = np.arange(len(MODELS))
        for ax, s in zip(axes, SEASONS):
            t = df[df.season == s].set_index("model").reindex(MODELS)
            bottom = np.zeros(len(MODELS))
            for c in ("misses", "hits", "FA"):
                ax.bar(x, t[c].values, 0.65, bottom=bottom, color=comp_col[c], edgecolor="white")
                bottom += t[c].values
            for xi, tot in zip(x, bottom):
                ax.text(xi, tot * 1.01, f"{tot:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=12)
            for lbl, m in zip(ax.get_xticklabels(), MODELS):
                lbl.set_color(COLORS[m])
                lbl.set_fontweight("bold")
            ax.set_title(s, pad=14)
            ax.set_ylim(0, df[["misses", "hits", "FA"]].sum(axis=1).max() * 1.15)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("twMAE (mm/24 h), lower is better")
        fig.legend(handles=[Patch(color=comp_col[c], label=comp_lbl[c]) for c in ("misses", "hits", "FA")],
                   loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925))
        fig.suptitle("Where the error for extreme rain comes from: misses, hits and false alarms",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, f"Threshold-weighted MAE above the station p{args.percentile} (2005-2024 climatology), "
                 f"all terrain classes pooled. {FOOT}", ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.03, 1, 0.86))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def slide_p2(counts, out_png):
    s = counts[counts.terrain == "all"].copy()
    s["per100"] = 100 * s.n_fc / s.n_obs
    x = np.arange(len(FIXED_MM))
    ymax = max(110, np.nanmax(s["per100"]) * 1.1)
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = _season_axes(h=6.0)
        for ax, season in zip(axes, SEASONS):
            t = s[s.season == season]
            ax.axhline(100, color="black", lw=1.3, ls="--")
            ax.text(x[-1], 101.5, "as many as observed", fontsize=10, ha="right", va="bottom")
            for m in MODELS:
                v = t[t.model == m].set_index("threshold").reindex(FIXED_MM)["per100"].values
                ax.plot(x, v, "o-", color=COLORS[m], lw=3, ms=7, label=LABELS[m])
            n_obs = t[t.model == MODELS[0]].set_index("threshold").reindex(FIXED_MM)["n_obs"].values
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v}\n{n / 1000:.0f}k" if n >= 1000 else f"{v}\n{n}" for v, n in zip(FIXED_MM, n_obs)],
                               fontsize=11)
            ax.set_ylim(0, ymax)
            ax.set_title(season)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Forecast events per 100 observed")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925))
        fig.supxlabel("Threshold on 24 h precipitation (mm)   (second row: number of observed events)",
                      fontsize=13, y=0.06)
        fig.suptitle("How often does each model forecast heavy rain, compared with what was observed?",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, f"Above 100: forecast too often; below 100: too rarely. {FOOT}",
                 ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.07, 1, 0.86))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def slide_p3(q, out_png):
    s = q[q.terrain == "all"]
    x = np.arange(len(QPCTS))
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = _season_axes(h=6.0)
        for ax, season in zip(axes, SEASONS):
            t = s[s.season == season]
            obs = t[t.model == MODELS[0]].set_index("pct").reindex(QPCTS)["obs"].values
            ax.axhline(0, color="black", lw=1.3)
            for m in MODELS:
                v = t[t.model == m].set_index("pct").reindex(QPCTS)["rel_diff_pct"].values
                ax.plot(x, v, "o-", color=COLORS[m], lw=3, ms=7, label=LABELS[m])
            ax.set_xticks(x)
            ax.set_xticklabels([f"p{p:g}\n{o:.0f} mm" for p, o in zip(QPCTS, obs)], fontsize=11)
            ax.set_title(season)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Forecast minus observed amount (%)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925))
        fig.supxlabel("Percentile of 24 h precipitation (observed amount below)", fontsize=13, y=0.06)
        fig.suptitle("Heavy-rain amounts: forecast versus observed at the same percentile",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, f"Same percentile of the forecast and of the observed distribution (all days). {FOOT}",
                 ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.07, 1, 0.86))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def slide_p4(tol, pct, out_png, season="all"):
    t = tol[tol.season == season].copy()
    t["detected_pct"] = 100 * t.detected / t.n_obs
    t["fa_pct"] = 100 * t.false_alarms / t.n_fc
    x = np.arange(len(RADII_KM))
    labels = ["same station" if r == 0 else f"within {r} km" for r in RADII_KM]
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, (col, title) in zip(axes, (("detected_pct", "Observed extremes detected (%)"),
                                         ("fa_pct", "Forecast extremes that are false alarms (%)"))):
            vals = {m: t[t.model == m].set_index("radius_km").reindex(RADII_KM)[col].values for m in MODELS}
            for m in MODELS:
                ax.plot(x, vals[m], "o-", color=COLORS[m], lw=3, ms=8, label=LABELS[m])
            # Stagger end labels vertically (by value rank) so close values stay readable.
            for end, xi, dx, ha in ((0, x[0], -12, "right"), (-1, x[-1], 12, "left")):
                order = sorted(MODELS, key=lambda mm: vals[mm][end])
                for k, m in enumerate(order):
                    ax.annotate(f"{vals[m][end]:.0f}", (xi, vals[m][end]), xytext=(dx, (k - 1) * 13),
                                textcoords="offset points", ha=ha, va="center", fontsize=12,
                                color=COLORS[m], fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlim(-0.5, len(x) - 0.5)
            ax.set_ylim(0, 100)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_xlabel("An extreme counts as forecast if the model has one ...")
        axes[1].set_xlabel("A forecast extreme counts as wrong if no extreme was observed ...")
        axes[0].legend(loc="upper left", frameon=False)
        what = "" if season == "all" else f" ({season})"
        fig.suptitle(f"Allowing for position errors: hits and false alarms of extreme rain{what}",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, f"Extreme: 24 h precipitation above the station p{pct} (2005-2024 climatology); "
                 f"same day, neighbouring stations. {FOOT}", ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


FAR_TITLE = "False alarm ratio\n(% of forecast extremes that did not happen)"


def slide_p1_counts(tol, pct, out_png, ratio=False):
    """Simple version of P1: hits and false alarms per 100 observed extremes (or false alarm ratio), by season."""
    t = tol[tol.radius_km == 0].copy()
    t["hits"] = 100 * t.detected / t.n_obs
    t["fa"] = 100 * t.false_alarms / (t.n_fc if ratio else t.n_obs)
    fa_title = FAR_TITLE if ratio else "False alarms\n(per 100 observed extremes)"
    groups = list(SEASONS) + ["all"]
    names = {**{s: s for s in SEASONS}, "all": "Whole year"}
    x = np.arange(len(groups))
    w = 0.26
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
        for ax, (col, title) in zip(axes, (("hits", "Extremes detected\n(per 100 observed; the rest are missed)"),
                                         ("fa", fa_title))):
            for i, m in enumerate(MODELS):
                v = t[t.model == m].set_index("season").reindex(groups)[col].values
                bars = ax.bar(x + (i - 1) * w, v, w, color=COLORS[m], label=LABELS[m])
                ax.bar_label(bars, labels=[f"{u:.0f}" for u in v], padding=3, fontsize=13)
            ax.axvspan(len(groups) - 1.5, len(groups) - 0.5, color="#eeeeee", zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([names[g] for g in groups])
            ax.set_ylim(0, 100)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(loc="upper left", frameon=False, ncol=3, fontsize=13)
        fig.suptitle(f"Extreme 24 h rain (above the station p{pct}): extremes detected and false alarms",
                     fontsize=19, fontweight="bold")
        fig.text(0.5, 0.005, f"Per-station p{pct} of the 2005-2024 observed climatology. {FOOT}",
                 ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def slide_detection_by_amount(counts, out_png, season="all", ratio=False):
    """Detected and false alarms (per 100 observed, or false alarm ratio) against fixed rain thresholds."""
    t = counts[(counts.terrain == "all") & (counts.season == season) & (counts.threshold >= 10)].copy()
    t["detected"] = 100 * t.hits / t.n_obs
    t["fa"] = 100 * (t.n_fc - t.hits) / (t.n_fc if ratio else t.n_obs)
    fa_title = FAR_TITLE if ratio else "False alarms\n(per 100 observed events)"
    levels = sorted(t.threshold.unique())
    x = np.arange(len(levels))
    n_obs = t[t.model == MODELS[0]].set_index("threshold").reindex(levels)["n_obs"].values
    xt = [f"\u2265 {v:g} mm\n{n / 1000:.0f}k obs" if n >= 1000 else f"\u2265 {v:g} mm\n{n} obs" for v, n in zip(levels, n_obs)]
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
        for ax, (col, title) in zip(axes, (("detected", "Extremes detected\n(per 100 observed; the rest are missed)"),
                                         ("fa", fa_title))):
            vals = {m: t[t.model == m].set_index("threshold").reindex(levels)[col].values for m in MODELS}
            for m in MODELS:
                ax.plot(x, vals[m], "o-", color=COLORS[m], lw=3, ms=8, label=LABELS[m])
            for k, m in enumerate(sorted(MODELS, key=lambda mm: vals[mm][-1])):
                ax.annotate(f"{vals[m][-1]:.0f}", (x[-1], vals[m][-1]), xytext=(10, (k - 1) * 12),
                            textcoords="offset points", ha="left", va="center", fontsize=13,
                            color=COLORS[m], fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(xt, fontsize=12)
            ax.set_xlim(-0.3, len(x) - 0.5)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylim(0, 100 if ratio else 70)
        axes[0].legend(loc="upper right", frameon=False)
        when = "whole year" if season == "all" else season
        fig.suptitle(f"AIFS detects moderate rain extremes as well as IFS, but fewer of the heaviest ({when})",
                     fontsize=18, fontweight="bold")
        fig.text(0.5, 0.005, f"24 h precipitation at or above each amount, same station and day. {FOOT}",
                 ha="center", fontsize=10, color=_style.C_REF)
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        fig.savefig(out_png)
        plt.close(fig)
    print(f"✓ {out_png}")


def make_slides(args):
    _style.apply_style()
    out = Path(args.out)
    slide_p1(args, out / "slideP1_twmae_misses_hits_falsealarms.png")
    counts = pd.read_csv(out / "event_counts_fixed_mm_3models.csv")
    slide_p2(counts, out / "slideP2_frequency_bias_fixed_mm.png")
    slide_detection_by_amount(counts, out / "slideP1c_detection_by_rain_amount.png")
    slide_detection_by_amount(counts, out / "slideP1c_detection_by_rain_amount_JJA.png", season="JJA")
    slide_detection_by_amount(counts, out / "slideP1e_detection_FAR_by_rain_amount.png", ratio=True)
    slide_detection_by_amount(counts, out / "slideP1e_detection_FAR_by_rain_amount_JJA.png", season="JJA", ratio=True)
    slide_p3(pd.read_csv(out / "amount_quantiles_3models.csv"), out / "slideP3_amounts_vs_observed.png")
    tol = pd.read_csv(out / "position_tolerance_3models.csv")
    slide_p1_counts(tol, args.percentile, out / "slideP1b_detected_false_alarms_by_season.png")
    slide_p1_counts(tol, args.percentile, out / "slideP1d_detected_FAR_by_season.png", ratio=True)
    slide_p4(tol, args.percentile, out / "slideP4_position_tolerance.png")
    for s in SEASONS:
        slide_p4(tol, args.percentile, out / f"slideP4_position_tolerance_{s}.png", season=s)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        make_slides(args)
        return
    cache, frames = {}, []
    for d in args.days:
        frames.append(load_day(args, d, cache))
        gc.collect()
    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"  total {len(df):,} pairs; observed p{args.percentile} exceedance rate "
          f"{100 * np.mean(df['obs_value'][np.isfinite(df['thr'])] >= df['thr'][np.isfinite(df['thr'])]):.2f}%")

    rows, qrows = [], []
    for season in list(SEASONS) + ["all"]:
        ss = df if season == "all" else df[df.season == season]
        for terrain in ["all"] + list(OROG):
            s = ss if terrain == "all" else ss[ss.terrain == terrain]
            obs = s["obs_value"].values
            for thr in FIXED_MM:
                o = obs >= thr
                for m in MODELS:
                    f = s[m].values >= thr
                    rows.append(dict(season=season, terrain=terrain, threshold=thr, model=m,
                                     n_obs=int(o.sum()), n_fc=int(f.sum()), hits=int((o & f).sum())))
            oq = np.percentile(obs, QPCTS)
            for m in MODELS:
                fq = np.percentile(s[m].values, QPCTS)
                for p, ov, fv in zip(QPCTS, oq, fq):
                    qrows.append(dict(season=season, terrain=terrain, pct=p, model=m, obs=ov, fc=fv,
                                      rel_diff_pct=100 * (fv - ov) / ov if ov > 0 else np.nan))
    pd.DataFrame(rows).to_csv(out / "event_counts_fixed_mm_3models.csv", index=False)
    pd.DataFrame(qrows).to_csv(out / "amount_quantiles_3models.csv", index=False)

    tol = tolerance_counts(df)
    tol.to_csv(out / "position_tolerance_3models.csv", index=False)

    pd.set_option("display.width", 220)
    c = pd.DataFrame(rows)
    c = c[c.terrain == "all"].assign(per100=lambda t: 100 * t.n_fc / t.n_obs)
    print("\n=== Forecast events per 100 observed (fixed mm, all terrain) ===")
    print(c.pivot_table(index=["season", "threshold"], columns="model", values="per100").round(0).to_string())
    q = pd.DataFrame(qrows)
    print("\n=== Amount at percentile, forecast vs observed (%) ===")
    print(q[q.terrain == "all"].pivot_table(index=["season", "pct"], columns="model", values="rel_diff_pct")
          .round(1).to_string())
    t = tol.assign(det=lambda d: 100 * d.detected / d.n_obs, fa=lambda d: 100 * d.false_alarms / d.n_fc)
    print("\n=== Position tolerance: % detected / % false alarms ===")
    print(t.pivot_table(index=["season", "radius_km"], columns="model", values=["det", "fa"]).round(1).to_string())
    make_slides(args)


if __name__ == "__main__":
    main()
