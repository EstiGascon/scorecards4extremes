#!/usr/bin/env python3
"""
Orography Elevation Representation Check
=========================================
Tests the hypothesis: "AIFS's own orography under-represents station elevation
in complex terrain more than IFS's orography does, causing insufficient
lapse-rate cooling and a warm bias there."

Compares fc1_height (model1 orog, e.g. hres_orog for ifs_oper) and fc2_height
(model2 orog, e.g. aifs_orog for aifs1.0_oper) against obs_height (true station
elevation), stratified by the SAME sdfor-based orography classes
(flat/hilly/complex) used by the main scorecards pipeline (filter.py).

Usage
-----
  python analysis/orog_elevation_representation.py --config <config.yaml> [--day 3]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

# Same color convention as diagnostics/_style.py (model1=blue, model2=vermillion)
C_FC1 = "#0072B2"
C_FC2 = "#D55E00"
CLASS_COLORS = {"flat": "#009E73", "hilly": "#E69F00", "complex": "#CC79A7"}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot_orog_representation(st, res, orog_ranges, fc1_name, fc2_name, save_dir):
    """Two-panel figure:
      Left  — obs_height vs model_height scatter per station, coloured by
              orography class, with the 1:1 line (perfect representation).
      Right — bar chart of signed mean error (bias) and |mean error| per
              orography class for both models, so the elevation-representation
              hypothesis is directly visible alongside the CSV numbers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- Left: scatter, obs_height vs model height ----
    ax = axes[0]
    cls_of = pd.Series(index=st.index, dtype=object)
    for cls, (lo, hi) in orog_ranges.items():
        cls_of[(st["sdfor"] >= lo) & (st["sdfor"] < hi)] = cls
    lims = [0, max(st["obs_height"].max(), st["fc1_height"].max(), st["fc2_height"].max()) * 1.05]
    ax.plot(lims, lims, color="grey", ls="--", lw=1, label="1:1 (perfect)", zorder=1)
    for cls in orog_ranges:
        m = cls_of == cls
        if m.sum() == 0:
            continue
        col = CLASS_COLORS.get(cls, "black")
        ax.scatter(st.loc[m, "obs_height"], st.loc[m, "fc1_height"], s=10, marker="o",
                   facecolors="none", edgecolors=col, alpha=0.5, linewidths=0.7,
                   label=f"{cls} — {fc1_name}")
        ax.scatter(st.loc[m, "obs_height"], st.loc[m, "fc2_height"], s=10, marker="x",
                   color=col, alpha=0.5, linewidths=0.7,
                   label=f"{cls} — {fc2_name}")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True station elevation, obs_height (m)")
    ax.set_ylabel("Model orography height (m)")
    ax.set_title(f"Model orography vs true elevation\n(o = {fc1_name}, x = {fc2_name})", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left", ncol=1)
    ax.grid(True, alpha=0.3)

    # ---- Right: bias / |bias| bar chart per orography class ----
    ax = axes[1]
    classes = list(res["cls"])
    x = np.arange(len(classes)); w = 0.2
    ax.bar(x - 1.5*w, res["err1_mean"],     w, color=C_FC1, alpha=0.55, label=f"{fc1_name} — mean err (signed)")
    ax.bar(x - 0.5*w, res["err1_abs_mean"], w, color=C_FC1, alpha=1.0,  label=f"{fc1_name} — |mean err|")
    ax.bar(x + 0.5*w, res["err2_mean"],     w, color=C_FC2, alpha=0.55, label=f"{fc2_name} — mean err (signed)")
    ax.bar(x + 1.5*w, res["err2_abs_mean"], w, color=C_FC2, alpha=1.0,  label=f"{fc2_name} — |mean err|")
    ax.axhline(0, color="black", lw=0.8)
    for xi, v1, v2 in zip(x, res["err1_mean"], res["err2_mean"]):
        ax.text(xi - 1.5*w, v1 + np.sign(v1)*3, f"{v1:+.0f}", ha="center",
                va="bottom" if v1 >= 0 else "top", fontsize=8)
        ax.text(xi + 0.5*w, v2 + np.sign(v2)*3, f"{v2:+.0f}", ha="center",
                va="bottom" if v2 >= 0 else "top", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Elevation representation error, model − obs (m)")
    ax.set_title("Orography representation error by terrain class\n"
                 "(negative = model terrain too LOW → under-applied lapse-rate cooling → warm bias)",
                 fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Orography Elevation Representation — {fc1_name} vs {fc2_name}", fontsize=13, weight="bold")
    plt.tight_layout()
    out = save_dir / f"orog_elevation_representation_{fc1_name}_vs_{fc2_name}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--day", type=int, default=3, help="Forecast day to read (heights are static per station)")
    args = parser.parse_args()

    config = load_config(args.config)
    fc1_name = config["read_data"]["forecast_model1"]["name"]
    fc2_name = config["read_data"]["forecast_model2"]["name"]
    variable = config["variable"]
    out_path = Path(config["extract_points"]["output_path"])
    pq_path = out_path / f"{variable}_{fc1_name}_vs_{fc2_name}_day{args.day}.parquet"

    print(f"Loading {pq_path} ...")
    df = pd.read_parquet(pq_path, columns=[
        "station_id", "lat", "lon", "obs_height", "fc1_height", "fc2_height", "sdfor", "lsm",
    ])

    filt = config.get("filter", {})
    coastal_thr = filt.get("coastal_lsm_threshold", 0.9)
    if filt.get("remove_coastal_stations", False):
        df = df[df["lsm"] > coastal_thr]

    # One row per station (heights/sdfor are static -> dedupe)
    st = df.drop_duplicates(subset="station_id").copy()
    print(f"Unique stations (post coastal filter): {len(st):,}")

    orog_ranges = filt.get("orography_ranges", {
        "flat": [0, 40], "hilly": [40, 120], "complex": [120, 3000],
    })

    st["err1"] = st["fc1_height"] - st["obs_height"]  # model1 (e.g. IFS/hres_orog) representation error
    st["err2"] = st["fc2_height"] - st["obs_height"]  # model2 (e.g. AIFS/aifs_orog) representation error

    print("\n" + "=" * 96)
    print(f"OROGRAPHY ELEVATION REPRESENTATION — {fc1_name} vs {fc2_name}")
    print("=" * 96)
    print(f"{'Class':<10}{'sdfor range':<14}{'N':>8}{'obs_h mean':>12}"
          f"{'err1 mean':>12}{'err1 |mean|':>12}{'err1 std':>10}"
          f"{'err2 mean':>12}{'err2 |mean|':>12}{'err2 std':>10}")
    rows = []
    for cls, (lo, hi) in orog_ranges.items():
        sub = st[(st["sdfor"] >= lo) & (st["sdfor"] < hi)]
        if len(sub) == 0:
            continue
        row = dict(
            cls=cls, lo=lo, hi=hi, n=len(sub),
            obs_h_mean=sub["obs_height"].mean(),
            err1_mean=sub["err1"].mean(), err1_abs_mean=sub["err1"].abs().mean(), err1_std=sub["err1"].std(),
            err2_mean=sub["err2"].mean(), err2_abs_mean=sub["err2"].abs().mean(), err2_std=sub["err2"].std(),
        )
        rows.append(row)
        print(f"{cls:<10}{f'[{lo},{hi})':<14}{row['n']:>8,}{row['obs_h_mean']:>12.1f}"
              f"{row['err1_mean']:>12.1f}{row['err1_abs_mean']:>12.1f}{row['err1_std']:>10.1f}"
              f"{row['err2_mean']:>12.1f}{row['err2_abs_mean']:>12.1f}{row['err2_std']:>10.1f}")

    res = pd.DataFrame(rows)
    print("\nInterpretation:")
    print("  err = model_orog_height - true_station_elevation. Negative mean => model")
    print("  terrain is SMOOTHED/LOWER than reality (under-represents elevation), which")
    print("  under-applies lapse-rate cooling (T_corrected = T_model + lapse_rate*(obs_h - model_h),")
    print("  lapse_rate<0, so model_h too LOW  => correction too WARM).")

    save_dir = Path(config.get("save", {}).get("output_directory", "./results/diagnostics"))
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_out = save_dir / f"orog_elevation_representation_{fc1_name}_vs_{fc2_name}.csv"
    res.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    plot_orog_representation(st, res, orog_ranges, fc1_name, fc2_name, save_dir)


if __name__ == "__main__":
    main()
