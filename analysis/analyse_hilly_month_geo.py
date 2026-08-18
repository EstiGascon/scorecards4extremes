#!/usr/bin/env python3
"""
analyse_hilly_month_geo.py — decompose the ensemble "hilly is red / mountain
is blue" twCRPS scorecard pattern two more ways:

  1. By MONTH: is the hilly (sdfor 40-120) twCRPS reversal concentrated in a
     particular month of the commonperiod, or present throughout?
  2. Geographically: map per-station AIFS-vs-IFS error within the hilly bin
     only, to check for regional clustering (e.g. one mountain range/country)
     rather than a terrain-height effect.

Uses the pipeline's OWN sdfor bins (LOW <40, MID 40-120, HIGH >=120) — same
bins as the real scorecard — and the exact fair/tail-weighted CRPS formula
from src/ens_scores.py (via analyse_orography_position_ensemble._twcrps_fair),
so results are directly comparable to the actual twCRPS_diff heatmap.

Never overwrites a previous run's plots: existing filenames get a _v2, _v3...
suffix automatically (see _unique_path).

Usage
-----
  python analyse_hilly_month_geo.py --config \\
      configs/ensemble/config_2t_ens_local_p1obsclim_aifsvsifs_commonperiod.yaml \\
      --days 1,3,6,10 --output-dir case_study_output/hilly_month_geo_2t_p1cold
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyse_orography_position_ensemble import (       # noqa: E402
    _twcrps_fair, _keys, _apply_production_qc_filters, _aggregate_to_daily_mean,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
from case_studies.case_study_utils import load_per_station_thresholds  # noqa: E402

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
SDFOR_BIN_NAMES = ["LOW", "MID", "HIGH"]
SDFOR_BIN_COLORS = {"LOW": "#2c7bb6", "MID": "#d7191c", "HIGH": "#5e3c99"}


def _unique_path(path):
    """Return `path` if free, else the same name with a _v2, _v3... suffix —
    so reruns never clobber a previous run's plots."""
    path = Path(path)
    if not path.exists():
        return path
    i = 2
    while True:
        cand = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def _sdfor_bin(sdfor):
    b = np.full(len(sdfor), "LOW", dtype=object)
    b[(sdfor >= 40) & (sdfor < 120)] = "MID"
    b[sdfor >= 120] = "HIGH"
    return b


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config file (mode: ensemble)")
    p.add_argument("--days", default="1,3,6,10",
                   help="Comma-separated forecast days (default: 1,3,6,10).")
    p.add_argument("--output-dir", default=None, dest="output_dir")
    p.add_argument("--keep-coastal", action="store_true")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    if config.get("mode") != "ensemble":
        print("ERROR: this tool is for mode: ensemble configs.")
        sys.exit(1)

    m1 = config["read_data"]["forecast_model1"]["name"]
    m2 = config["read_data"]["forecast_model2"]["name"]
    parquet_dir = Path(config["extract_points"]["output_path"])
    fcfg = config.get("filter", {})
    lsm_thr = fcfg.get("coastal_lsm_threshold", 0.9)
    remove_coastal = fcfg.get("remove_coastal_stations", False) and not args.keep_coastal
    event = config.get("threshold", {}).get("event_type", "above")
    days = [int(d) for d in args.days.split(",")]

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("case_study_output") / f"hilly_month_geo_{m1}_vs_{m2}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}\n  HILLY-BIN MONTH x GEOGRAPHY DECOMPOSITION")
    print(f"  Config : {args.config}")
    print(f"  Models : {m1} (fc1) vs {m2} (fc2)")
    print(f"  Days   : {days}")
    print(f"  Output : {out_dir}\n{'='*72}\n")

    monthly_rows = []
    geo_parts = []  # per-day station-level MID-bin dataframes, concatenated at the end

    for day in days:
        cands = list(parquet_dir.glob(f"*_day{day}.parquet"))
        if not cands:
            print(f"  day {day}: no parquet found — skipping")
            continue

        all_cols = pq.ParquetFile(cands[0]).schema.names
        fc1_cols = [c for c in all_cols if c.startswith("fc1_member_")]
        fc2_cols = [c for c in all_cols if c.startswith("fc2_member_")]
        # 'step' is required for valid-time-month threshold matching AND the
        # daily-mean aggregation below — both must match production exactly.
        base_cols = [c for c in ("date", "step", "lat", "lon", "sdfor", "lsm",
                                 "obs_value", "station_id")
                    if c in all_cols]
        df = pd.read_parquet(cands[0], columns=base_cols + fc1_cols + fc2_cols)

        if remove_coastal and "lsm" in df.columns:
            df = df[df["lsm"] >= lsm_thr]
        df = df.dropna(subset=["obs_value", "lat", "lon"])
        if df.empty:
            print(f"  day {day}: empty after filters — skipping")
            continue

        df = _apply_production_qc_filters(df, config)
        if df.empty:
            print(f"  day {day}: empty after QC filters — skipping")
            continue

        # Per-station threshold computed on SUB-DAILY rows (as production
        # does), then daily-mean aggregation (production's own
        # `_aggregate_to_daily_mean`) BEFORE scoring. Skipping this averages
        # sub-daily steps into a smoothed daily forecast — omitting it scores
        # a different quantity and can flip the AIFS-vs-IFS sign for cold
        # extremes (this was the bug in the previous version of this script).
        T_sub = load_per_station_thresholds(config, df)
        if "step" in df.columns and config.get("threshold", {}).get("method") == "local_obs_climatology":
            df, T = _aggregate_to_daily_mean(df, pd.Series(T_sub, index=df.index), config)
        else:
            df, T = df, T_sub

        month = df["date"].astype(str).str[4:6].astype(int).values
        obs = df["obs_value"].values.astype(np.float64)
        fc1_np = df[fc1_cols].values.astype(np.float64)
        fc2_np = df[fc2_cols].values.astype(np.float64)
        fc1_mean = fc1_np.mean(axis=1)
        fc2_mean = fc2_np.mean(axis=1)
        sdfor = df["sdfor"].values.astype(np.float64)
        sbin = _sdfor_bin(sdfor)
        T = np.asarray(T)

        # ---- month x sdfor-bin twCRPS ----
        for mo in sorted(np.unique(month)):
            mo_mask = month == mo
            for bname in SDFOR_BIN_NAMES:
                mask = mo_mask & (sbin == bname)
                if mask.sum() < 20:
                    continue
                tw1 = _twcrps_fair(fc1_np[mask], obs[mask], T[mask], event)
                tw2 = _twcrps_fair(fc2_np[mask], obs[mask], T[mask], event)
                monthly_rows.append(dict(
                    day=day, month=MONTH_NAMES.get(int(mo), mo), month_num=int(mo),
                    sdfor_bin=bname, n_rows=int(mask.sum()),
                    twcrps_ifs=float(np.nanmean(tw1)),
                    twcrps_aifs=float(np.nanmean(tw2))))

        # ---- geographic accumulation, MID (hilly) bin only ----
        mid_mask = sbin == "MID"
        if mid_mask.any():
            stn_key = _keys(df["lat"].values[mid_mask], df["lon"].values[mid_mask])
            geo_parts.append(pd.DataFrame({
                "key": stn_key,
                "lat": df["lat"].values[mid_mask],
                "lon": df["lon"].values[mid_mask],
                "abs_err_diff": np.abs(fc2_mean[mid_mask] - obs[mid_mask])
                              - np.abs(fc1_mean[mid_mask] - obs[mid_mask]),
            }))

        counts = {b: int((sbin == b).sum()) for b in SDFOR_BIN_NAMES}
        print(f"  day {day}: {len(df):,} rows  |  months {sorted(set(month))}  |  bin rows {counts}")
        del df, fc1_np, fc2_np, obs, T
        gc.collect()

    if not monthly_rows:
        print("ERROR: no data produced — check filters / parquet paths.")
        sys.exit(1)

    monthly = pd.DataFrame(monthly_rows)
    monthly["twcrps_diff"] = monthly["twcrps_aifs"] - monthly["twcrps_ifs"]
    csv_path = _unique_path(out_dir / "hilly_month_sdfor_summary.csv")
    monthly.to_csv(csv_path, index=False)
    print(f"\n  ✓ month x sdfor summary → {csv_path}")

    print(f"\n  ── twCRPS_diff (AIFS-IFS) by month x sdfor bin, pooled over requested days ──")
    pooled = monthly.groupby(["month_num", "month", "sdfor_bin"], as_index=False).agg(
        n_rows=("n_rows", "sum"),
        twcrps_ifs=("twcrps_ifs", "mean"),
        twcrps_aifs=("twcrps_aifs", "mean"))
    pooled["twcrps_diff"] = pooled["twcrps_aifs"] - pooled["twcrps_ifs"]
    pooled = pooled.sort_values(["month_num", "sdfor_bin"])
    print(pooled[["month", "sdfor_bin", "n_rows", "twcrps_ifs", "twcrps_aifs", "twcrps_diff"]]
          .to_string(index=False))

    _fig_month_sdfor(pooled, m1, m2, out_dir)

    if geo_parts:
        geo = pd.concat(geo_parts, ignore_index=True)
        geo_agg = geo.groupby("key", as_index=False).agg(
            lat=("lat", "first"), lon=("lon", "first"),
            mean_abs_err_diff=("abs_err_diff", "mean"), n=("abs_err_diff", "size"))
        geo_csv = _unique_path(out_dir / "hilly_bin_station_geo.csv")
        geo_agg.to_csv(geo_csv, index=False)
        print(f"  ✓ per-station hilly-bin geo table → {geo_csv} ({len(geo_agg):,} stations)")
        _fig_hilly_geo_map(geo_agg, m1, m2, out_dir)

    print(f"\n{'='*72}\n  DONE — outputs in {out_dir}\n{'='*72}\n")


def _fig_month_sdfor(pooled, m1, m2, out_dir):
    """twCRPS_diff by month, one line per sdfor bin — directly shows whether
    the hilly (MID) reversal is concentrated in specific months."""
    months = sorted(pooled.month_num.unique())
    month_labels = [MONTH_NAMES[mo] for mo in months]
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for bname in SDFOR_BIN_NAMES:
        sub = pooled[pooled.sdfor_bin == bname].set_index("month_num").reindex(months)
        ax.plot(x, sub.twcrps_diff, ls="-", color=SDFOR_BIN_COLORS[bname], lw=2.4,
                marker="o", ms=9, mec="k", mew=0.6, label=f"{bname} (sdfor bin)")

    ax.axhline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(month_labels, fontsize=11)
    ax.set_ylabel("twCRPS_diff = AIFS − IFS  (negative = AIFS better)", fontsize=11)
    ax.set_xlabel("Month", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best", framealpha=0.95)
    ax.set_title(f"twCRPS_diff by month and sdfor bin — {m1} vs {m2}\n"
                 f"(pooled over requested lead days)", fontsize=12, weight="bold")
    fig.tight_layout()
    out = _unique_path(out_dir / "monthly_sdfor_twcrps_diff.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [month] twCRPS_diff by month x sdfor → {out.name}")


def _fig_hilly_geo_map(geo_agg, m1, m2, out_dir):
    """Map of per-station AIFS-vs-IFS |error| difference, HILLY (sdfor 40-120)
    stations only. Red = AIFS worse there, blue = AIFS better."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except Exception:
        use_cartopy = False

    lats = geo_agg["lat"].values
    lons = geo_agg["lon"].values
    val = geo_agg["mean_abs_err_diff"].values
    vmax = np.nanpercentile(np.abs(val), 95) or 1.0

    fig = plt.figure(figsize=(16, 8))
    if use_cartopy:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f0e8")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#cde6f5")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="#888")
        tk = dict(transform=ccrs.PlateCarree())
        pad = 3.0
        ax.set_extent([max(-180, lons.min() - pad), min(180, lons.max() + pad),
                       max(-90, lats.min() - pad), min(90, lats.max() + pad)],
                      crs=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)
        tk = {}
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

    sc = ax.scatter(lons, lats, c=val, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=16, edgecolor="k", lw=0.2, zorder=5, **tk)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cb.set_label("mean(|AIFS err| − |IFS err|)  °C  —  red = AIFS worse here", fontsize=10)
    ax.set_title(f"HILLY bin (sdfor 40-120) stations — AIFS vs IFS mean |error| difference\n"
                 f"{m1} vs {m2} — {len(geo_agg):,} stations", fontsize=12, weight="bold")
    fig.tight_layout()
    out = _unique_path(out_dir / "hilly_bin_geo_map.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [geo] hilly-bin station map → {out.name}")


if __name__ == "__main__":
    main()
