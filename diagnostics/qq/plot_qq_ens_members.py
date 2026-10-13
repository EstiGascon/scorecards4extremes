#!/usr/bin/env python3
"""
plot_qq_ens_members.py
=======================
Within-model Q-Q comparison: control member vs a single perturbed member vs
the full pooled ensemble, for ONE model, plotted together on the same axes so
their behaviour can be compared directly (as opposed to plot_qq_ens.py, which
always compares the two MODELS against each other for one fixed member
selection).

Reuses the data loading / filtering / styling machinery of plot_qq_ens.py.

Usage
-----
  # Both models (IFS-ENS and AIFS-ENS), both orographies -> 4 plots:
  python plot_qq_ens_members.py --config config_2t_ens.yaml --orog low high

  # One model only, one orography:
  python plot_qq_ens_members.py --config config_2t_ens.yaml --model fc1 --orog low

  # Restrict to a season / lead time, like plot_qq_ens.py:
  python plot_qq_ens_members.py --config config_10ff_ens.yaml --season DJF --orog high
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent            # this script's subfolder
_DIAG = _HERE.parent                               # diagnostics/  (shared _style)
_REPO = _DIAG.parent                               # repository root
for _p in (_REPO, _REPO / "src", _DIAG, _HERE):
    sys.path.insert(0, str(_p))
import _style

from plot_qq import (  # noqa: E402
    VARIABLE_LABELS,
    load_config,
    get_model_names,
    get_threshold_from_config,
)
from plot_qq_ens import (  # noqa: E402
    SEASON_MONTHS,
    OROGRAPHY_RANGES,
    AREAS,
    BASE_COLUMNS,
    IMPOSSIBLE_VALUE,
    filter_chunk,
    _forecast_arrays,
    _paired_stats,
    _condition_label,
)

# Colourblind-safe triad (Okabe-Ito): control / member-1 / full ensemble.
SERIES_COLORS = {
    "control": _style.C_FC1,       # blue
    "member1": _style.C_FC2,       # vermillion
    "full":    "#009E73",          # bluish green
}
SERIES_LABELS = {
    "control": "Control (member 0)",
    "member1": "Member 1",
    "full":    "Full ensemble (pooled)",
}


# ============================================================================
# DATA SOURCE DISCOVERY — ALL member columns for ONE model
# ============================================================================

def discover_model_source(config: dict, fc1_name: str, fc2_name: str, model: str):
    """Locate the extracted ensemble data and resolve every member column for
    ONE model (model='fc1' or 'fc2'). Returns the same 'kind' dict shape as
    plot_qq_ens.discover_ens_source, minus the member-selection resolution
    (we always load ALL of that model's members up front, then slice the
    control/member-1/full series out of the same loaded frame)."""
    var = config["variable"]
    point_data_path = Path(config["extract_points"]["output_path"])

    day_pattern = f"{var}_{fc1_name}_vs_{fc2_name}_ens_day*.parquet"
    day_files = sorted(point_data_path.glob(day_pattern))
    if not day_files:
        day_pattern = f"{var}_{fc1_name}_vs_{fc2_name}_day*.parquet"
        day_files = sorted(point_data_path.glob(day_pattern))
    if not day_files:
        raise FileNotFoundError(
            f"No extracted ENSEMBLE data found in '{point_data_path}'.\n"
            f"Expected files matching: {point_data_path / day_pattern}")

    available = pq.ParquetFile(str(day_files[0])).schema.names
    prefix = f"{model}_member_"
    member_cols = sorted([c for c in available if c.startswith(prefix)],
                        key=lambda c: int(c.rsplit("_", 1)[1]))
    if not member_cols:
        raise SystemExit(f"ERROR: no '{prefix}*' columns found in {day_files[0].name}")

    needed = [c for c in BASE_COLUMNS if c in available] + member_cols
    return {"day_files": day_files, "needed": needed, "member_cols": member_cols}


def load_model_data(source, config, *, season=None, orog_type=None, lead_time=None,
                    start_date=None, end_date=None) -> pd.DataFrame:
    """Stream-load + filter one model's member columns across all forecast-day files."""
    parts, total_raw = [], 0
    for fp in source["day_files"]:
        chunk = pd.read_parquet(fp, columns=source["needed"])
        total_raw += len(chunk)
        chunk = filter_chunk(
            chunk, config, source["member_cols"],
            season=season, orog_type=orog_type, lead_time=lead_time,
            start_date=start_date, end_date=end_date,
        )
        if len(chunk):
            parts.append(chunk)
        del chunk
        gc.collect()
    if not parts:
        raise RuntimeError(
            "No data remains after filtering. Adjust filter criteria or check "
            "that the data covers the selected conditions.")
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"  Streamed {len(source['day_files'])} file(s): {total_raw:,} rows read "
          f"→ {len(df):,} rows after filtering")
    return df


# ============================================================================
# PLOTTING
# ============================================================================

def plot_qq_ens_members(
    df: pd.DataFrame,
    config: dict,
    model_name: str,
    member_cols,
    *,
    season=None,
    orog_type=None,
    lead_time=None,
    output_dir=None,
    threshold=None,
    dpi: int = 150,
) -> Path:
    variable = config["variable"]
    var_label, unit = VARIABLE_LABELS.get(variable, (variable, ""))
    condition = _condition_label(season, orog_type, lead_time)

    obs = df["obs_value"].to_numpy(dtype=np.float32)

    # Slice the three series out of the SAME loaded frame (no re-read).
    series = {}
    control_col = member_cols[0] if member_cols[0].endswith("_0") else None
    mem1_col = next((c for c in member_cols if c.endswith("_1")), None)
    if control_col is not None:
        series["control"] = _forecast_arrays(df, [control_col])
    else:
        print("  Note: no control member (member_0) in this dataset — skipping control curve.")
    if mem1_col is not None:
        series["member1"] = _forecast_arrays(df, [mem1_col])
    else:
        print("  Note: no member 1 in this dataset — skipping member-1 curve.")
    series["full"] = _forecast_arrays(df, member_cols)

    has_cold_tail = variable == "2t"
    dot_levels = np.arange(1, 99.5, 1.0) if has_cold_tail else np.arange(0, 99.5, 1.0)
    x_levels = np.linspace(99.0, 99.9, 10)
    cold_x_levels = np.linspace(0.1, 1.0, 10) if has_cold_tail else None

    obs_dots = np.nanpercentile(obs, dot_levels)
    obs_x = np.nanpercentile(obs, x_levels)
    obs_cold_x = np.nanpercentile(obs, cold_x_levels) if has_cold_tail else None

    computed = {}
    all_capped = [obs_dots, obs_x]
    if has_cold_tail:
        all_capped.append(obs_cold_x)
    for key, (dist, mat) in series.items():
        dots = np.nanpercentile(dist, dot_levels)
        xs = np.nanpercentile(dist, x_levels)
        cold_xs = np.nanpercentile(dist, cold_x_levels) if has_cold_tail else None
        corr, bias, rmse = _paired_stats(mat, obs)
        computed[key] = dict(dots=dots, xs=xs, cold_xs=cold_xs,
                             corr=corr, bias=bias, rmse=rmse)
        all_capped += [dots, xs] + ([cold_xs] if has_cold_tail else [])

    all_capped = np.concatenate(all_capped)
    vmin, vmax = np.nanmin(all_capped), np.nanmax(all_capped)
    margin = (vmax - vmin) * 0.05
    lim = (vmin - margin, vmax + margin)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Q-Q Plot (ENS)  —  {var_label}  |  {model_name}  |  {condition}  "
        f"(N = {len(df):,})",
        fontsize=12, fontweight="bold",
    )

    ax.plot(lim, lim, color=_style.C_REF, linestyle="--", linewidth=1.2, zorder=1)

    for key, c in computed.items():
        color = SERIES_COLORS[key]
        ax.scatter(obs_dots, c["dots"], color=color, s=20, zorder=3,
                   marker="o", label=SERIES_LABELS[key])
        ax.scatter(obs_x, c["xs"], color=color, s=60, zorder=4,
                   marker="x", linewidths=2)
        if has_cold_tail:
            ax.scatter(obs_cold_x, c["cold_xs"], color=color, s=60, zorder=4,
                       marker="x", linewidths=2)

    if threshold is not None:
        ax.axvline(threshold, color="green", linestyle=":", linewidth=1.3, alpha=0.9, zorder=1)
        ax.axhline(threshold, color="green", linestyle=":", linewidth=1.3, alpha=0.9, zorder=1)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed [{unit}]", fontsize=11)
    ax.set_ylabel(f"Forecast [{unit}]", fontsize=11)
    ax.grid(True, alpha=0.3)

    legend = ax.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    fig.canvas.draw()
    lb = legend.get_window_extent().transformed(ax.transAxes.inverted())
    marker_note = (
        "o = 0–99th percentile\nx = 0.1–1st & 99–99.9th (Δ 0.1%)"
        if has_cold_tail else
        "o = 0–99th percentile\nx = 99–99.9th (Δ 0.1%)"
    )
    ax.text(lb.x1, lb.y1 + 0.005, marker_note, transform=ax.transAxes, fontsize=9,
           verticalalignment="bottom", horizontalalignment="right",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="lightgray", alpha=0.85))

    # Stats box: corr / bias / RMSE per series (each in its own colour).
    lines = []
    for key in ("control", "member1", "full"):
        if key not in computed:
            continue
        c = computed[key]
        lines.append((SERIES_COLORS[key],
                      f"{SERIES_LABELS[key]}:  \u03c1={c['corr']:.3f}  "
                      f"bias={c['bias']:+.2f}  RMSE={c['rmse']:.2f}"))
    y0 = 0.985
    for color, text in lines:
        ax.text(0.02, y0, text, transform=ax.transAxes, fontsize=9,
                color=color, fontweight="bold", va="top", ha="left")
        y0 -= 0.045

    if output_dir is None:
        output_dir = Path(config.get("save", {}).get("output_directory", "./results"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fn_parts = [f"qq_members_{variable}_{model_name}"]
    if season:
        fn_parts.append(season.upper())
    if orog_type:
        fn_parts.append(orog_type.lower())
    if lead_time is not None:
        steps = [lead_time] if isinstance(lead_time, int) else list(lead_time)
        fn_parts.append("lt" + "_".join(str(s) for s in steps) + "h")
    filename = "_".join(fn_parts) + ".png"

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")
    return out_path


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Within-model ensemble Q-Q comparison: control vs member-1 "
                    "vs full ensemble, for one model at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model", nargs="+", choices=["fc1", "fc2"], default=["fc1", "fc2"],
                  help="Which model(s) to plot (default: both, one plot each).")
    p.add_argument("--season", nargs="+", choices=["DJF", "MAM", "JJA", "SON"], default=None)
    p.add_argument("--orog", dest="orog_type", nargs="+", choices=["low", "mid", "high"],
                  default=None)
    p.add_argument("--lead-time", type=int, nargs="+", dest="lead_time", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    _style.apply_style()
    config = load_config(args.config)
    fc1_name, fc2_name = get_model_names(config)
    model_names = {"fc1": fc1_name, "fc2": fc2_name}

    threshold = args.threshold if args.threshold is not None else get_threshold_from_config(config)

    print("\n" + "=" * 70)
    print("Q-Q PLOT TOOL (ENSEMBLE, WITHIN-MODEL MEMBER COMPARISON)")
    print("=" * 70)
    print(f"  Config   : {args.config}")
    print(f"  Variable : {config['variable']}")
    print(f"  Models   : {fc1_name} (fc1)  /  {fc2_name} (fc2)")

    seasons = args.season or [None]
    orogs = args.orog_type or [None]

    for model in args.model:
        model_name = model_names[model]
        print(f"\n--- Model: {model_name} ({model}) ---")
        source = discover_model_source(config, fc1_name, fc2_name, model)
        print(f"  {len(source['day_files'])} forecast-day file(s); "
              f"{len(source['member_cols'])} member column(s)")

        for season in seasons:
            for orog in orogs:
                label = f"{season or 'ALL'} × {orog or 'ALL'}"
                print(f"\n  [{model_name}] Plotting: {label}")
                try:
                    df = load_model_data(
                        source, config, season=season, orog_type=orog,
                        lead_time=args.lead_time,
                        start_date=args.start_date, end_date=args.end_date,
                    )
                    plot_qq_ens_members(
                        df, config, model_name, source["member_cols"],
                        season=season, orog_type=orog, lead_time=args.lead_time,
                        output_dir=args.output_dir, threshold=threshold, dpi=args.dpi,
                    )
                    del df
                    gc.collect()
                except RuntimeError as exc:
                    print(f"    Skipping {label}: {exc}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
