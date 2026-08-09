#!/usr/bin/env python3
"""
plot_station_coverage_map.py — World map of observation station coverage.

Plots every unique observation location (deduplicated by rounded lat/lon,
since station_id is not a stable key across days — see repo notes) found in
a CAMS variable's extracted parquet directory, ignoring any area filter in
the config so the FULL observing network is shown.

Usage
-----
  python analysis/plot_station_coverage_map.py --config configs/cams/config_go3_icki_vs_oper_fixed70_nh_extratropics.yaml
  python analysis/plot_station_coverage_map.py --config <cfg> --output-dir ./plots/station_coverage
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diagnostics"))
import _style

VARIABLE_LABELS = {
    "aod500": "AOD 500 nm",
    "go3":    "Ozone (O3)",
    "pm2p5":  "PM2.5",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_unique_stations(config: dict) -> pd.DataFrame:
    """Union of unique (lat, lon) station locations across all day-parquets."""
    var = config["variable"]
    fc1 = config["read_data"]["forecast_model1"]["name"]
    fc2 = config["read_data"]["forecast_model2"]["name"]
    base = Path(config["extract_points"]["output_path"])

    day_files = sorted(base.glob(f"{var}_{fc1}_vs_{fc2}_day*.parquet"))
    if not day_files:
        raise FileNotFoundError(f"No day-parquet files found in {base}")

    frames = []
    for fp in day_files:
        df = pd.read_parquet(fp, columns=["lat", "lon"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["lat_r"] = all_df["lat"].round(2)
    all_df["lon_r"] = all_df["lon"].round(2)
    stations = all_df.drop_duplicates(subset=["lat_r", "lon_r"])[["lat", "lon"]]
    return stations.reset_index(drop=True)


def plot_coverage_map(stations: pd.DataFrame, variable: str, output_path: Path, dpi: int = 300):
    var_label = VARIABLE_LABELS.get(variable, variable)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4", zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="grey", alpha=0.4, linestyle="--")

    ax.scatter(stations["lon"], stations["lat"], transform=ccrs.PlateCarree(),
               s=8, color=_style.C_OBS, alpha=0.7, edgecolors="none", zorder=3)

    ax.set_title(f"{var_label} — Observation Station Coverage  (N = {len(stations):,} stations)",
                 fontsize=13, fontweight="bold", pad=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}  ({len(stations):,} unique stations)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--output-dir", default="./plots/station_coverage",
                   help="Directory to save the map (default: ./plots/station_coverage)")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    config = load_config(args.config)
    variable = config["variable"]
    print(f"Loading station locations for '{variable}' from {config['extract_points']['output_path']} ...")
    stations = load_unique_stations(config)

    out_path = Path(args.output_dir) / f"coverage_{variable}.png"
    plot_coverage_map(stations, variable, out_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
