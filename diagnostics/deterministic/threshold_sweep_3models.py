#!/usr/bin/env python3
"""
Three-model extreme 10 m wind verification on identical cases: calibration vs discrimination.
=============================================================================================
Relates IFS-oper, iekm (DestinE) and AIFS-oper by merging the two existing pairwise
extractions (both have ifs_oper as model 1), on 6-hourly 10 m wind.

SEDI is only base-rate independent for unbiased forecasts (Ferro & Stephenson 2011), so
the tail behaviour is scored as two separate aspects:

  1. Calibration (raw forecasts): frequency bias B = (hits + FA) / (hits + misses) at
     per-station observed percentiles (each station's own p90..p99.5 of 6-hourly obs)
     and at fixed thresholds (8..20 m/s). B < 1: extremes forecast too rarely.
  2. Discrimination (bias removed): per-station frequency matching. At each station the
     forecast event is the model's own top-n values, n = number of observed events there,
     so B = 1 by construction and SEDI measures only the placement of the events in time
     and space. With B = 1 and a common base rate, SEDI, ETS, PSS and hit rate give the
     same model ranking.

Station percentiles and matching use the verification year itself (in-sample), per
forecast day. 95% intervals: paired date-block bootstrap (initialisation dates resampled,
same draw for all models); paired differences iekm-IFS, AIFS-IFS and iekm-AIFS are
tested as well. Orography classes (sdfor) and wind QC (0-100 m/s on obs and all
forecasts) match the main pipeline (src/filter.py).

Usage:
    python diagnostics/deterministic/threshold_sweep_3models.py [--days 1 2 3 4 5]
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style  # noqa: E402

MODELS = ["IFS-oper", "iekm", "AIFS-oper"]
COLORS = {"IFS-oper": _style.C_FC1, "AIFS-oper": _style.C_FC2, "iekm": "#CC79A7"}
OROG = {"flat": (0, 40), "hilly": (40, 120), "complex": (120, 3000)}
PAIRS = [("iekm", "IFS-oper"), ("AIFS-oper", "IFS-oper"), ("iekm", "AIFS-oper")]
COLS = ["date", "step", "lat", "lon", "sdfor", "obs_value", "fc1_value", "fc2_value"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iekm-dir", default="extracted_points/10ff_local_p98obsclim_ifs_oper_iekm_new")
    p.add_argument("--iekm-file", default="10ff_ifs_oper_vs_iekm_day{d}.parquet")
    p.add_argument("--aifs-dir", default="extracted_points/10ff_local_p99obsclim_ifs_oper_aifs_new")
    p.add_argument("--aifs-file", default="10ff_ifs_oper_vs_aifs1.0_oper_day{d}.parquet")
    p.add_argument("--days", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--thresholds", type=float, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    p.add_argument("--percentiles", type=float, nargs="+", default=[90, 95, 98, 99, 99.5])
    p.add_argument("--lead-percentile", type=float, default=99.0,
                   help="Percentile used for the matched-SEDI-vs-forecast-day figure")
    p.add_argument("--min-station-samples", type=int, default=200,
                   help="Stations with fewer 6-hourly pairs per forecast day are excluded from the percentile scores")
    p.add_argument("--min-events", type=int, default=30,
                   help="Mask levels with fewer observed events than this")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/10ff_3model_extremes_d1to5")
    return p.parse_args()


def load_day(args, d):
    """Merge the two pairwise extractions for forecast day d into one 3-model frame."""
    a = pd.read_parquet(Path(args.iekm_dir) / args.iekm_file.format(d=d), columns=COLS)
    b = pd.read_parquet(Path(args.aifs_dir) / args.aifs_file.format(d=d), columns=COLS)
    keys = ["date", "step", "lat", "lon"]
    aligned = (len(a) == len(b)
               and all(np.array_equal(a[k].values, b[k].values) for k in keys)
               and np.allclose(a["obs_value"].values, b["obs_value"].values, equal_nan=True))
    if aligned:
        df = a.rename(columns={"fc1_value": "IFS-oper", "fc2_value": "iekm"})
        df["AIFS-oper"] = b["fc2_value"].values
        ifs_b = b["fc1_value"].values
    else:
        print(f"  day {d}: rows not aligned, merging on {keys}")
        a = a.drop_duplicates(keys)
        b = b.drop_duplicates(keys)[keys + ["fc1_value", "fc2_value"]]
        df = a.merge(b, on=keys, suffixes=("", "_b"))
        df = df.rename(columns={"fc1_value": "IFS-oper", "fc2_value": "iekm", "fc2_value_b": "AIFS-oper"})
        ifs_b = df.pop("fc1_value_b").values
    # Both extractions use the same IFS fields; a mismatch would invalidate the pairing.
    mism = np.nanmax(np.abs(df["IFS-oper"].values - ifs_b))
    print(f"  day {d}: {len(df):,} rows, max |IFS(iekm file) - IFS(aifs file)| = {mism:.2e}")
    if mism > 1e-3:
        raise RuntimeError(f"IFS values differ between extractions on day {d} (max {mism})")
    del a, b
    valid = df["obs_value"].between(0, 100)
    for m in MODELS:
        valid &= df[m].between(0, 100)
    return df.loc[valid].reset_index(drop=True)


def date_counts(df, date_idx, n_dates, thresholds):
    """Per-date contingency counts: array (n_dates, n_models, n_thr, 4) = H, M, F, CN."""
    out = np.zeros((n_dates, len(MODELS), len(thresholds), 4), dtype=np.int64)
    obs = df["obs_value"].values
    for t, thr in enumerate(thresholds):
        o = obs >= thr
        for m, name in enumerate(MODELS):
            f = df[name].values >= thr
            for k, cell in enumerate((o & f, o & ~f, ~o & f, ~o & ~f)):
                out[:, m, t, k] = np.bincount(date_idx[cell], minlength=n_dates)
    return out


def _fill(out, date_idx, n_dates, m, t, o, f):
    for k, cell in enumerate((o & f, o & ~f, ~o & f, ~o & ~f)):
        out[:, m, t, k] = np.bincount(date_idx[cell], minlength=n_dates)


def percentile_counts(df, date_idx, n_dates, pcts, min_station_samples):
    """Per-date counts for per-station percentile events: (raw, frequency-matched).

    Observed event: obs >= station's own percentile of obs. Raw forecast event: forecast
    >= that same value. Matched forecast event: among the model's top-n values at the
    station, n = number of observed events there (B = 1 per station).
    """
    # station_id in the local 10ff extractions is a per-date row label, so locations are keyed by lat/lon.
    codes = df.groupby(["lat", "lon"], sort=False).ngroup().values
    n_per_station = np.bincount(codes)
    keep = n_per_station[codes] >= min_station_samples
    codes, date_idx = pd.factorize(codes[keep])[0], date_idx[keep]
    obs = df["obs_value"].values[keep]
    fc = {m: df[m].values[keep] for m in MODELS}

    qs = [p / 100.0 for p in pcts]
    q = pd.Series(obs).groupby(codes).quantile(qs).unstack()   # rows: station code, cols: qs
    # 'first' breaks ties by position, so exactly n forecasts are flagged per station.
    ranks = {m: pd.Series(fc[m]).groupby(codes).rank(ascending=False, method="first").values
             for m in MODELS}

    shape = (n_dates, len(MODELS), len(pcts), 4)
    raw, matched = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    for t in range(len(pcts)):
        thr = q.iloc[:, t].values[codes]
        o = obs >= thr
        n_ev = np.bincount(codes, weights=o)[codes]
        for m, name in enumerate(MODELS):
            _fill(raw, date_idx, n_dates, m, t, o, fc[name] >= thr)
            _fill(matched, date_idx, n_dates, m, t, o, ranks[name] <= n_ev)
    return raw, matched


def sedi_bias(c):
    """SEDI and frequency bias from counts (..., 4); NaN where undefined."""
    h, m, f, cn = (c[..., i].astype(float) for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        H = h / (h + m)
        F = f / (f + cn)
        ok = (H > 0) & (H < 1) & (F > 0) & (F < 1)
        lH, lF, l1H, l1F = (np.log(np.where(ok, x, 0.5)) for x in (H, F, 1 - H, 1 - F))
        sedi = np.where(ok, (lF - lH - l1F + l1H) / (lF + lH + l1F + l1H), np.nan)
        bias = np.where(h + m > 0, (h + f) / (h + m), np.nan)
    return sedi, bias


def summarise(counts, n_boot, rng):
    """Point estimates and paired date-block bootstrap draws for SEDI and bias."""
    active = counts.reshape(len(counts), -1).sum(1) > 0
    c = counts[active]
    n = len(c)
    total = c.sum(0)
    sedi, bias = sedi_bias(total)
    draws = rng.integers(0, n, size=(n_boot, n))
    w = np.stack([np.bincount(r, minlength=n) for r in draws])          # (B, n)
    boot = np.tensordot(w, c, axes=(1, 0))                               # (B, models, thr, 4)
    b_sedi, b_bias = sedi_bias(boot)
    return total, sedi, bias, b_sedi, b_bias, n


def ci(x):
    return np.nanpercentile(x, 2.5, axis=0), np.nanpercentile(x, 97.5, axis=0)


def build_tables(label, terrain, kind, levels, res, min_events):
    """Long-format score rows and paired-difference rows for one (lead, terrain, kind)."""
    total, sedi, bias, b_sedi, b_bias, n_dates = res
    s_lo, s_hi = ci(b_sedi)
    f_lo, f_hi = ci(b_bias)
    n_obs = total[0, :, 0] + total[0, :, 1]
    rows, drows = [], []
    for t, lvl in enumerate(levels):
        enough = n_obs[t] >= min_events
        nan = np.nan
        for m, name in enumerate(MODELS):
            h, mi, fa = (int(total[m, t, k]) for k in range(3))
            rows.append(dict(kind=kind, lead=label, terrain=terrain, level=lvl, model=name,
                             n_dates=n_dates, n_obs_events=int(n_obs[t]),
                             hits=h, misses=mi, false_alarms=fa,
                             POD=h / (h + mi) if (enough and h + mi) else nan,
                             SEDI=sedi[m, t] if enough else nan,
                             SEDI_lo=s_lo[m, t] if enough else nan,
                             SEDI_hi=s_hi[m, t] if enough else nan,
                             freq_bias=bias[m, t] if enough else nan,
                             freq_bias_lo=f_lo[m, t] if enough else nan,
                             freq_bias_hi=f_hi[m, t] if enough else nan))
        for a, b in PAIRS:
            ia, ib = MODELS.index(a), MODELS.index(b)
            for score, point, draw in (("SEDI", sedi, b_sedi), ("freq_bias", bias, b_bias)):
                d = draw[:, ia, t] - draw[:, ib, t]
                lo, hi = np.nanpercentile(d, [2.5, 97.5]) if enough else (nan, nan)
                drows.append(dict(kind=kind, lead=label, terrain=terrain, level=lvl, score=score,
                                  pair=f"{a} - {b}", n_obs_events=int(n_obs[t]),
                                  diff=(point[ia, t] - point[ib, t]) if enough else nan,
                                  ci_lo=lo, ci_hi=hi,
                                  significant=bool(enough and not (lo <= 0 <= hi))))
    return rows, drows


def _band(ax, x, s, score, name):
    ax.plot(x, s[score].values, "o-", color=COLORS[name], label=name, lw=1.8, ms=4)
    ax.fill_between(x, s[f"{score}_lo"].values, s[f"{score}_hi"].values,
                    color=COLORS[name], alpha=0.2, lw=0)


def _event_counts(ax, x, sub):
    ev = sub[sub.model == MODELS[0]]
    for xi, n in zip(x, ev.n_obs_events.values):
        ax.annotate(f"{n:,}", (xi, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7, color=_style.C_REF)


def plot_main(scores, pcts, out_png, days):
    """Rows: terrain. Left: raw frequency bias. Right: SEDI after per-station frequency matching."""
    pooled = scores[scores.lead == "D1-5"]
    x = np.arange(len(pcts))
    panels = (("raw_pct", "freq_bias", "Frequency bias (raw forecasts)"),
              ("matched", "SEDI", "SEDI (frequency-matched, B = 1)"))
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    for r, terrain in enumerate(OROG):
        lo, hi = OROG[terrain]
        for c, (kind, score, ylabel) in enumerate(panels):
            ax = axes[r, c]
            sub = pooled[(pooled.terrain == terrain) & (pooled.kind == kind)]
            for name in MODELS:
                _band(ax, x, sub[sub.model == name].set_index("level").reindex(pcts), score, name)
            if score == "freq_bias":
                ax.axhline(1.0, color=_style.C_REF, lw=1, ls="--")
            else:
                _event_counts(ax, x, sub.set_index("level").loc[pcts].reset_index())
            ax.set_ylabel(ylabel)
            ax.set_title(f"{terrain} (sdfor {lo}-{hi} m)")
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"p{p:g}" for p in pcts])
        ax.set_xlabel("Event: station percentile of 6-hourly observed 10 m wind")
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"10 m wind, Europe, 2025-03-01 to 2026-02-28, forecast days {days[0]}-{days[-1]} pooled\n"
                 "Shading: 95% date-block bootstrap interval. Numbers: observed events",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_fixed_bias(scores, thresholds, out_png, days):
    pooled = scores[(scores.lead == "D1-5") & (scores.kind == "fixed")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, terrain in zip(axes, OROG):
        lo, hi = OROG[terrain]
        sub = pooled[pooled.terrain == terrain]
        for name in MODELS:
            _band(ax, thresholds, sub[sub.model == name].set_index("level").reindex(thresholds), "freq_bias", name)
        ax.axhline(1.0, color=_style.C_REF, lw=1, ls="--")
        _event_counts(ax, thresholds, sub.set_index("level").loc[thresholds].reset_index())
        ax.set_title(f"{terrain} (sdfor {lo}-{hi} m)")
        ax.set_xlabel("Threshold on 6-hourly 10 m wind (m/s)")
    axes[0].set_ylabel("Frequency bias (raw forecasts)")
    axes[0].legend(loc="best")
    fig.suptitle(f"Frequency bias at fixed thresholds, forecast days {days[0]}-{days[-1]} pooled "
                 "(95% date-block bootstrap; numbers: observed events)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_lead(scores, pct, out_png):
    per_day = scores[(scores.lead != "D1-5") & (scores.kind == "matched") & (scores.level == pct)].copy()
    per_day["day"] = per_day.lead.str[1:].astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, terrain in zip(axes, OROG):
        sub = per_day[per_day.terrain == terrain]
        for name in MODELS:
            s = sub[sub.model == name].sort_values("day")
            ax.errorbar(s.day, s.SEDI, yerr=[s.SEDI - s.SEDI_lo, s.SEDI_hi - s.SEDI],
                        fmt="o-", color=COLORS[name], label=name, capsize=3, lw=1.6, ms=4)
        ax.set_title(terrain)
        ax.set_xlabel("Forecast day")
        ax.set_xticks(sorted(sub.day.unique()))
    axes[0].set_ylabel(f"SEDI, frequency-matched (station p{pct:g})")
    axes[0].legend()
    fig.suptitle(f"Frequency-matched SEDI by forecast day, events above station p{pct:g} of 6-hourly 10 m wind "
                 "(95% date-block bootstrap)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    args = parse_args()
    _style.apply_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    thresholds = sorted(args.thresholds)
    pcts = sorted(set(args.percentiles) | {args.lead_percentile})
    levels = {"fixed": thresholds, "raw_pct": pcts, "matched": pcts}

    all_dates = None
    pooled = {}
    rows, drows = [], []
    for d in args.days:
        df = load_day(args, d)
        if all_dates is None:
            all_dates = np.sort(df["date"].unique())
        date_idx = np.searchsorted(all_dates, df["date"].values)
        if (date_idx >= len(all_dates)).any() or not np.array_equal(all_dates[date_idx], df["date"].values):
            raise RuntimeError(f"day {d} has initialisation dates not present in day {args.days[0]}")
        for terrain, (lo, hi) in OROG.items():
            sel = ((df["sdfor"] >= lo) & (df["sdfor"] < hi)).values
            sub, di = df.loc[sel], date_idx[sel]
            counts = {"fixed": date_counts(sub, di, len(all_dates), thresholds)}
            counts["raw_pct"], counts["matched"] = percentile_counts(sub, di, len(all_dates), pcts,
                                                                     args.min_station_samples)
            for kind, c in counts.items():
                key = (terrain, kind)
                pooled[key] = c if key not in pooled else pooled[key] + c
                r, dr = build_tables(f"D{d}", terrain, kind, levels[kind],
                                     summarise(c, args.n_boot, rng), args.min_events)
                rows += r
                drows += dr
            print(f"  day {d} {terrain}: {sel.sum():,} pairs")
        del df
        gc.collect()

    for (terrain, kind), c in pooled.items():
        r, dr = build_tables("D1-5", terrain, kind, levels[kind], summarise(c, args.n_boot, rng),
                             args.min_events)
        rows += r
        drows += dr

    scores = pd.DataFrame(rows)
    diffs = pd.DataFrame(drows)
    scores.to_csv(out / "scores_3models.csv", index=False)
    diffs.to_csv(out / "paired_differences_3models.csv", index=False)
    plot_main(scores, pcts, out / "calibration_discrimination_3models.png", args.days)
    plot_fixed_bias(scores, thresholds, out / "freqbias_fixed_thresholds_3models.png", args.days)
    plot_lead(scores, args.lead_percentile, out / f"sedi_matched_p{args.lead_percentile:g}_by_leadtime_3models.png")

    pd.set_option("display.width", 220)
    p = scores[scores.lead == "D1-5"]
    print("\nRaw frequency bias at station percentiles, pooled D1-5:\n",
          p[p.kind == "raw_pct"].pivot_table(index=["terrain", "level"], columns="model",
                                             values="freq_bias").round(3).to_string())
    print("\nFrequency-matched SEDI, pooled D1-5:\n",
          p[p.kind == "matched"].pivot_table(index=["terrain", "level", "n_obs_events"], columns="model",
                                             values="SEDI").round(3).to_string())
    sig = diffs[(diffs.lead == "D1-5") & (diffs.kind == "matched") & (diffs.score == "SEDI")]
    print("\nPaired differences, frequency-matched SEDI, pooled D1-5:\n",
          sig[["terrain", "level", "pair", "diff", "ci_lo", "ci_hi", "significant"]].round(3).to_string(index=False))
    print(f"\n✓ Outputs written to {out}")


if __name__ == "__main__":
    main()
