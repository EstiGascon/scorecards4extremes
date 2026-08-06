#!/usr/bin/env python3
"""
analyse_orography_position.py — Generalise the valley / mountain-top / flat
case-study comparison to ALL stations.

Motivation
----------
A single case study (Monte Rosa summit, Trentino valley, Paris-basin flat)
suggested AIFS performs worse in valleys.  This tool turns that anecdote into a
population-level statistic by classifying *every* station into a topographic
position class and comparing IFS vs AIFS skill within each class and as a
continuous function of terrain misrepresentation.

Key idea — signed elevation anomaly
------------------------------------
A model's smoothed orography height at a station approximates the gridbox-mean
(≈ regional-mean) terrain height.  The signed anomaly

        Δh = obs_height − model_orography_height

is therefore a Topographic-Position-Index-like measure that is directly
available in the extracted parquet (no external DEM needed):

    Δh ≪ 0  → the true station sits BELOW the model surface  → VALLEY
              (model can't resolve the incised valley floor)
    Δh ≫ 0  → the true station sits ABOVE the model surface  → MOUNTAIN TOP
              (model smooths the peak away)
    Δh ≈ 0  → station is representative of its gridbox        → FLAT / repr.

Δh is exactly the height difference the lapse-rate correction fights, and it is
model-specific (IFS and AIFS smooth the orography differently).

Classification (all configurable)
---------------------------------
Using a model-agnostic reference height (default: mean of the two models):
    mountain_top : Δh >=  peak_dh
    valley       : Δh <= valley_dh
    flat         : |Δh| <  flat_dh  AND  sdfor < flat_sdfor
    (mid/slope)  : everything else — reported but not a focus class

Outputs (to --output-dir)
-------------------------
  1_error_vs_dh.png        MAE & mean-bias as a continuous function of each
                           model's OWN Δh (the "general conclusion" plot).
  2_class_bias.png         Mean bias per position class, IFS vs AIFS, per day.
  3_class_map.png          Map of classified stations (valley/flat/peak).
  orography_position_summary.csv   full (class × day × model) metric table.

Usage
-----
  python analyse_orography_position.py --config \\
      configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single_nhextrop.yaml \\
      --season DJF --output-dir case_study_output/orog_position_DJF

  # extremes-only stratification (obs beyond the per-station threshold):
  python analyse_orography_position.py --config <cfg> --extremes
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from case_studies.case_study_utils import load_per_station_thresholds  # noqa: E402

# ─── Palette / constants ──────────────────────────────────────────────────────
C_IFS = "#d7191c"    # model1 — red
C_AIFS = "#2c7bb6"   # model2 — blue
CLASS_COLORS = {"valley": "#e66101", "flat": "#4d9221", "mountain_top": "#5e3c99"}
CLASS_ORDER = ["valley", "flat", "mountain_top"]
SEASON_MONTHS = {"DJF": {12, 1, 2}, "MAM": {3, 4, 5},
                 "JJA": {6, 7, 8}, "SON": {9, 10, 11}}

# Columns we actually need (keeps the 27M-row parquets manageable).
# NB: station_id is deliberately excluded — it is not a stable per-location key
# in these parquets and is a heavy string column; we key stations by (lat, lon).
NEEDED_COLS = ["date", "step", "lat", "lon",
               "obs_height", "fc1_height", "fc2_height", "sdfor", "lsm",
               "obs_value", "fc1_value", "fc2_value"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--season", default=None, help="DJF | MAM | JJA | SON")
    p.add_argument("--output-dir", default=None, dest="output_dir")
    p.add_argument("--height-ref", default="mean", choices=["mean", "ifs", "aifs"],
                   help="Reference orography for classification (default: mean).")
    p.add_argument("--valley-dh", type=float, default=-75.0,
                   help="Δh (m) at/below which a station is a valley (default -75).")
    p.add_argument("--peak-dh", type=float, default=75.0,
                   help="Δh (m) at/above which a station is a mountain top (default +75).")
    p.add_argument("--flat-dh", type=float, default=40.0,
                   help="|Δh| (m) below which a station may be flat (default 40).")
    p.add_argument("--flat-sdfor", type=float, default=40.0,
                   help="sdfor (m) below which a low-|Δh| station is flat (default 40).")
    p.add_argument("--extremes", action="store_true",
                   help="Additionally stratify to obs beyond the per-station threshold.")
    p.add_argument("--keep-coastal", action="store_true",
                   help="Do NOT remove coastal stations (default: apply config lsm filter).")
    p.add_argument("--days", default=None,
                   help="Comma-separated forecast days (default: config forecast_days).")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def classify(dh, sdfor, args):
    """Vectorised topographic-position classification from signed anomaly Δh."""
    cls = np.full(len(dh), "mid", dtype=object)
    cls[dh <= args.valley_dh] = "valley"
    cls[dh >= args.peak_dh] = "mountain_top"
    flat_mask = (np.abs(dh) < args.flat_dh) & (sdfor < args.flat_sdfor)
    cls[flat_mask] = "flat"
    return cls


def ref_height(df, which):
    if which == "ifs":
        return df["fc1_height"].values
    if which == "aifs":
        return df["fc2_height"].values
    return 0.5 * (df["fc1_height"].values + df["fc2_height"].values)


def _stats(err):
    """Return (bias, mae, rmse) for an error array fc-obs."""
    if len(err) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(err)), float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def main():
    args = parse_args()
    config = load_config(args.config)

    m1 = config["read_data"]["forecast_model1"]["name"]
    m2 = config["read_data"]["forecast_model2"]["name"]
    parquet_dir = Path(config["extract_points"]["output_path"])
    fcfg = config.get("filter", {})
    lsm_thr = fcfg.get("coastal_lsm_threshold", 0.9)
    remove_coastal = fcfg.get("remove_coastal_stations", False) and not args.keep_coastal

    if args.days:
        days = [int(d) for d in args.days.split(",")]
    else:
        days = config.get("forecast_days", [1, 3, 5, 7, 10])

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("case_study_output") /
        f"orog_position_{args.season or 'all'}_{m1}_vs_{m2}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}\n  OROGRAPHIC-POSITION SKILL ANALYSIS")
    print(f"  Config : {args.config}")
    print(f"  Models : {m1} (fc1) vs {m2} (fc2)")
    print(f"  Season : {args.season or 'all'}   Days: {days}")
    print(f"  Δh ref : {args.height_ref}   valley<= {args.valley_dh}  "
          f"peak>= {args.peak_dh}  flat |Δh|<{args.flat_dh} & sdfor<{args.flat_sdfor}")
    print(f"  Coastal removal: {remove_coastal} (lsm>= {lsm_thr})")
    print(f"  Output : {out_dir}\n{'='*72}\n")

    # Continuous Δh bins (each model vs its OWN anomaly), streamed across days.
    DH_EDGES = np.array([-1e9, -600, -400, -250, -150, -75, -25,
                         25, 75, 150, 250, 400, 600, 1e9])
    DH_CENTERS = np.array([-700, -500, -325, -200, -112, -50, 0,
                           50, 112, 200, 325, 500, 700])
    nb = len(DH_CENTERS)
    # accumulators [model][bin] -> running sums
    acc = {m: {"n": np.zeros(nb), "sum_e": np.zeros(nb), "sum_ae": np.zeros(nb)}
           for m in ("fc1", "fc2")}

    summary_rows = []
    station_map = {}   # station_id -> (lat, lon, class) for the map (from first day seen)

    for day in days:
        cands = list(parquet_dir.glob(f"*_day{day}.parquet"))
        if not cands:
            print(f"  day {day}: no parquet found — skipping")
            continue
        cols = [c for c in NEEDED_COLS]
        df = pd.read_parquet(cands[0], columns=cols)

        # Season filter
        if args.season in SEASON_MONTHS:
            months = SEASON_MONTHS[args.season]
            df = df[df["date"].astype(str).str[4:6].astype(int).isin(months)]
        # Coastal filter
        if remove_coastal and "lsm" in df.columns:
            df = df[df["lsm"] >= lsm_thr]
        # Need finite heights + values
        df = df.dropna(subset=["obs_height", "fc1_height", "fc2_height",
                               "obs_value", "fc1_value", "fc2_value"])
        if df.empty:
            print(f"  day {day}: empty after filters — skipping")
            continue

        obs = df["obs_value"].values.astype(np.float64)
        fc1 = df["fc1_value"].values.astype(np.float64)
        fc2 = df["fc2_value"].values.astype(np.float64)
        dh_ref = df["obs_height"].values - ref_height(df, args.height_ref)
        dh1 = df["obs_height"].values - df["fc1_height"].values
        dh2 = df["obs_height"].values - df["fc2_height"].values
        cls = classify(dh_ref, df["sdfor"].values, args)

        # NOTE: station_id is NOT a stable per-location key in these parquets
        # (IDs like S0/S1 are reused across different physical sites). The
        # reliable physical key is rounded (lat, lon).
        stn_key = (df["lat"].round(3).astype(str) + "_"
                   + df["lon"].round(3).astype(str)).values

        # Optional extremes mask (obs beyond per-station threshold)
        ext_mask = None
        if args.extremes:
            T = load_per_station_thresholds(config, df)
            event = config.get("threshold", {}).get("event_type", "above")
            with np.errstate(invalid="ignore"):
                ext_mask = (obs <= T) if event == "below" else (obs >= T)
            ext_mask = ext_mask & np.isfinite(T)

        # ── Per-class metrics (this day) ──
        for c in CLASS_ORDER:
            cm = cls == c
            n_st = np.unique(stn_key[cm]).size
            for tag, mask in ([("all", cm)] +
                              ([("extreme", cm & ext_mask)] if ext_mask is not None else [])):
                b1, a1, r1 = _stats(fc1[mask] - obs[mask])
                b2, a2, r2 = _stats(fc2[mask] - obs[mask])
                summary_rows.append(dict(
                    day=day, orog_class=c, subset=tag, n_rows=int(mask.sum()),
                    n_stations=n_st,
                    bias_ifs=b1, mae_ifs=a1, rmse_ifs=r1,
                    bias_aifs=b2, mae_aifs=a2, rmse_aifs=r2,
                    mae_diff_aifs_minus_ifs=(a2 - a1) if (a1 == a1 and a2 == a2) else np.nan))

        # ── Continuous Δh accumulation (each model vs own anomaly) ──
        for mkey, dh, fc in (("fc1", dh1, fc1), ("fc2", dh2, fc2)):
            idx = np.digitize(dh, DH_EDGES) - 1
            e = fc - obs
            ae = np.abs(e)
            for b in range(nb):
                bm = idx == b
                if bm.any():
                    acc[mkey]["n"][b] += bm.sum()
                    acc[mkey]["sum_e"][b] += e[bm].sum()
                    acc[mkey]["sum_ae"][b] += ae[bm].sum()

        # ── Station map (record class once, from the first day) ──
        if not station_map:
            uniq = df.drop_duplicates(["lat", "lon"])
            ucls = classify(
                uniq["obs_height"].values - ref_height(uniq, args.height_ref),
                uniq["sdfor"].values, args)
            for la, lo, cc in zip(uniq["lat"].values, uniq["lon"].values, ucls):
                station_map[(round(float(la), 3), round(float(lo), 3))] = (la, lo, cc)

        counts = {c: int((cls == c).sum()) for c in CLASS_ORDER}
        print(f"  day {day}: {len(df):,} rows  |  class rows {counts}")
        del df, obs, fc1, fc2, dh_ref, dh1, dh2, cls

    if not summary_rows:
        print("ERROR: no data produced — check filters / parquet paths.")
        sys.exit(1)

    summ = pd.DataFrame(summary_rows)
    csv_path = out_dir / "orography_position_summary.csv"
    summ.to_csv(csv_path, index=False)
    print(f"\n  ✓ summary table → {csv_path}")

    _print_headline(summ, m1, m2)
    _fig_error_vs_dh(acc, DH_CENTERS, m1, m2, out_dir, args)
    _fig_class_bias(summ, m1, m2, out_dir, args)
    _fig_class_map(station_map, out_dir, args, m1, m2)

    print(f"\n{'='*72}\n  DONE — outputs in {out_dir}\n{'='*72}\n")


def _print_headline(summ, m1, m2):
    """Concise stdout table of the 'all' subset for immediate conclusions."""
    print(f"\n  ── Headline: MAE by position class (subset=all) ──")
    print(f"  {'class':<13}{'day':>4}{'N_stn':>7}"
          f"{m1[:9]:>10}{m2[:9]:>10}{'AIFS-IFS':>10}")
    a = summ[summ.subset == "all"]
    for c in CLASS_ORDER:
        for _, r in a[a.orog_class == c].sort_values("day").iterrows():
            print(f"  {c:<13}{int(r.day):>4}{int(r.n_stations):>7}"
                  f"{r.mae_ifs:>10.3f}{r.mae_aifs:>10.3f}"
                  f"{r.mae_diff_aifs_minus_ifs:>+10.3f}")


def _fig_error_vs_dh(acc, centers, m1, m2, out_dir, args):
    """MAE & mean bias as a continuous function of each model's own Δh."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True)
    lines = []
    for mkey, name, col in (("fc1", m1, C_IFS), ("fc2", m2, C_AIFS)):
        n = acc[mkey]["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            bias = np.where(n > 0, acc[mkey]["sum_e"] / n, np.nan)
            mae = np.where(n > 0, acc[mkey]["sum_ae"] / n, np.nan)
        m = n >= 30
        ln, = axes[0].plot(centers[m], bias[m], "o-", color=col, lw=2.4, ms=7,
                            mec="k", mew=0.5, label=name)
        lines.append(ln)
        axes[1].plot(centers[m], mae[m], "o-", color=col, lw=2.4, ms=7,
                     mec="k", mew=0.5, label=name)

    for ax, title, ylab in ((axes[0], "Mean bias (fc−obs) vs Δh", "Mean bias (°C)"),
                            (axes[1], "MAE vs Δh", "MAE (°C)")):
        ax.axvline(args.valley_dh, color="#e66101", ls=":", lw=1.2)
        ax.axvline(args.peak_dh, color="#5e3c99", ls=":", lw=1.2)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Signed elevation anomaly  Δh = obs − model_orog  (m)"
                      "\n←  valley                    flat                    peak  →")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3)
    # Single shared legend for the whole figure (both panels use the same
    # model/colour mapping) placed OUTSIDE the axes, between the suptitle and
    # the plots, so it never overlaps the data lines.
    fig.legend(handles=lines, fontsize=11, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), frameon=True, framealpha=0.95)
    fig.suptitle(f"Forecast error vs terrain misrepresentation — {m1} vs {m2} "
                 f"({args.season or 'all seasons'})", fontsize=12, weight="bold",
                 y=1.0)
    fig.tight_layout()
    out = out_dir / "1_error_vs_dh.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] error vs Δh → {out.name}")


def _fig_class_bias(summ, m1, m2, out_dir, args):
    """Mean bias per position class vs lead time, IFS vs AIFS.

    Single panel.  Colour encodes the topographic class, line style encodes the
    model (IFS = dashed / open circles, AIFS = solid / filled squares), so the
    two models are easy to tell apart within each class.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    a = summ[summ.subset == "all"]
    days = sorted(a.day.unique())
    x = np.arange(len(days))

    # Representative case count per class (station count is ~constant across days).
    n_per_class = {c: int(np.nanmax(a[a.orog_class == c].n_stations.values))
                   if (a.orog_class == c).any() else 0
                   for c in CLASS_ORDER}

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for c in CLASS_ORDER:
        col = CLASS_COLORS[c]
        sub = a[a.orog_class == c].set_index("day").reindex(days)
        # IFS — dashed, open circle
        ax.plot(x, sub.bias_ifs, ls="--", color=col, lw=1.8, marker="o",
                ms=8, mfc="white", mec=col, mew=1.8, zorder=3)
        # AIFS — solid, filled square
        ax.plot(x, sub.bias_aifs, ls="-", color=col, lw=2.6, marker="s",
                ms=8, mfc=col, mec="k", mew=0.6, zorder=4)

    ax.axhline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f"day {d}" for d in days], fontsize=11)
    ax.set_ylabel("Mean bias  fc − obs  (°C)", fontsize=12)
    ax.set_xlabel("Forecast lead time", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Two-part legend: (1) class colour + case count, (2) model line style.
    # Both are placed OUTSIDE the plot axes (to the right) so they never
    # overlap the data lines.
    class_handles = [
        Patch(facecolor=CLASS_COLORS[c], edgecolor="k", lw=0.4,
              label=f"{c.replace('_', ' ')}  (n≈{n_per_class[c]:,} stations)")
        for c in CLASS_ORDER]
    model_handles = [
        Line2D([0], [0], color="0.25", ls="--", lw=1.8, marker="o", ms=8,
               mfc="white", mec="0.25", mew=1.8, label=f"{m1}  (IFS)"),
        Line2D([0], [0], color="0.25", ls="-", lw=2.6, marker="s", ms=8,
               mfc="0.25", mec="k", mew=0.6, label=f"{m2}  (AIFS)")]
    leg1 = ax.legend(handles=class_handles, title="Topographic class",
                     fontsize=10, title_fontsize=10, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0.,
                     framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=model_handles, title="Model", fontsize=10,
              title_fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 0.45),
              borderaxespad=0., framealpha=0.95)

    ax.set_title(f"Mean bias by topographic position — {m1} vs {m2} "
                 f"({args.season or 'all seasons'})",
                 fontsize=12, weight="bold")
    # Reserve room on the right for the two external legends (tight_layout()
    # doesn't know about bbox_to_anchor artists placed outside the axes, so we
    # fix the right margin explicitly instead — avoids the legend text being
    # clipped by its own frame).
    fig.subplots_adjust(right=0.72)
    out = out_dir / "2_class_bias.png"
    # NOTE: bbox_inches='tight' alone does not know about legend `leg1` (added
    # via ax.add_artist rather than the axes' tracked legend) — pass both
    # legends explicitly via bbox_extra_artists or their text gets clipped.
    fig.savefig(out, dpi=160, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"  [2] class bias → {out.name}")


def _fig_class_map(station_map, out_dir, args, m1, m2):
    """Map of classified stations (valley / flat / mountain_top)."""
    if not station_map:
        return
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except Exception:
        use_cartopy = False

    lats = np.array([v[0] for v in station_map.values()])
    lons = np.array([v[1] for v in station_map.values()])
    cls = np.array([v[2] for v in station_map.values()])

    fig = plt.figure(figsize=(16, 8))
    if use_cartopy:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f0e8")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#cde6f5")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="#888")
        tk = dict(transform=ccrs.PlateCarree())
        if lons.size:
            pad = 3.0
            lon_min = max(-180.0, lons.min() - pad)
            lon_max = min(180.0, lons.max() + pad)
            lat_min = max(-90.0, lats.min() - pad)
            lat_max = min(90.0, lats.max() + pad)
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)
        tk = {}
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

    for c in CLASS_ORDER:
        m = cls == c
        ax.scatter(lons[m], lats[m], s=14, color=CLASS_COLORS[c],
                   edgecolor="white", lw=0.3, label=f"{c} (n={int(m.sum())})",
                   zorder=5, **tk)
    # Legend BELOW the map (outside the data area) so it never covers stations.
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, framealpha=0.95, markerscale=1.8)
    ax.set_title(f"Station topographic-position classes  "
                 f"(Δh ref = {args.height_ref})", fontsize=12, weight="bold")
    fig.tight_layout()
    out = out_dir / "3_class_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] class map → {out.name}")


if __name__ == "__main__":
    main()
