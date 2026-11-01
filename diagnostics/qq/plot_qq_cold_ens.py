#!/usr/bin/env python3
"""
QQ Plot for Cold Extremes — Ensemble adaptation.

Loads multiple forecast days and combines them for a richer sample.
Compares two representations of the ensemble forecast vs observations:

  1. Pooled ensemble  : all 51 member values from all cases concatenated.
                        Represents the full marginal forecast distribution.
                        On the 1:1 line = the ensemble is marginally calibrated.

  2. Ensemble mean    : per-case mean of 51 members, then pooled across cases.
                        Narrower distribution (mean shrinks extremes).

Cold tail quantile levels:
  p0.1, 0.2, …, 0.9   (×-markers, sub-percentile extremes)
  p1.0, 2.0, …, 10.0  (●-markers, integer percentiles)

Usage:
  python plot_qq_cold_ens.py --config <yaml> \\
      [--days 1 2 3 4] [--season DJF] [--orog low] [--output-dir <dir>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

# ── constants ─────────────────────────────────────────────────────────────────
VARIABLE_LABELS = {
    "2t":   ("2m Temperature",    "°C"),
    "10ff": ("10m Wind Speed",    "m/s"),
    "tp24": ("24h Precipitation", "mm"),
}

SEASON_MONTHS = {
    "DJF": [12, 1, 2], "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],  "SON": [9, 10, 11],
    "ASO": [8, 9, 10],
}

OROGRAPHY_RANGES = {
    "flat": (0, 40), "low": (0, 40),
    "hilly": (40, 120), "mid": (40, 120),
    "complex": (120, 3000), "high": (120, 3000),
}

C1 = "#1f77b4"   # model 1 (blue)
C2 = "#d62728"   # model 2 (red)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _month(date_int):
    """Extract month from YYYYMMDD integer/string."""
    return int(str(date_int)[4:6])


def load_day_ensemble(config, day, season=None, orog=None, max_samples=300_000, seed=42):
    """Load a single forecast-day parquet in chunks to avoid OOM.

    Applies season + orography filters per chunk, then stacks the kept rows.
    With max_samples > 0, randomly subsamples the filtered rows.

    Returns (obs_arr, fc1_np, fc2_np, fc1_name, fc2_name) or None on failure.
    """
    var  = config["variable"]
    fc1  = config["read_data"]["forecast_model1"]["name"]
    fc2  = config["read_data"]["forecast_model2"]["name"]
    base = Path(config["extract_points"]["output_path"])

    fp = None
    for pat in [
        f"{var}_{fc1}_vs_{fc2}_ens_day{day}.parquet",
        f"{var}_{fc1}_vs_{fc2}_day{day}.parquet",
        f"{var}_{fc1}_vs_{fc2}_99th_day{day}.parquet",
    ]:
        candidate = base / pat
        if candidate.exists():
            fp = candidate
            break

    if fp is None:
        print(f"  [day {day}] No parquet file found in {base}")
        return None

    print(f"  [day {day}] Streaming {fp.name}")
    pf = pq.ParquetFile(str(fp))

    season_months = set(SEASON_MONTHS[season.upper()]) if season else None
    orog_lo, orog_hi = (OROGRAPHY_RANGES[orog.lower()]
                        if orog and orog in OROGRAPHY_RANGES else (None, None))

    fcfg = config.get("filter", {})
    t_lo = fcfg.get("min_valid_temperature", -60.0) if var == "2t" else None
    t_hi = fcfg.get("max_valid_temperature",  60.0) if var == "2t" else None

    chunks = []
    n_raw  = 0
    for batch in pf.iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        n_raw += len(chunk)

        if season_months:
            chunk = chunk[chunk["date"].apply(_month).isin(season_months)]
        if orog_lo is not None and "sdfor" in chunk.columns:
            chunk = chunk[(chunk["sdfor"] >= orog_lo) & (chunk["sdfor"] < orog_hi)]
        if chunk.empty:
            continue

        # QC obs
        chunk = chunk.dropna(subset=["obs_value"])
        if t_lo is not None:
            chunk = chunk[(chunk["obs_value"] >= t_lo) & (chunk["obs_value"] <= t_hi)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        print(f"  [day {day}] Empty after filters")
        return None

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    # Identify member columns
    fc1_cols = sorted([c for c in df.columns if c.startswith("fc1_member_")],
                      key=lambda c: int(c.split("_")[-1]))
    fc2_cols = sorted([c for c in df.columns if c.startswith("fc2_member_")],
                      key=lambda c: int(c.split("_")[-1]))

    if not fc1_cols or not fc2_cols:
        print(f"  [day {day}] No member columns found")
        return None

    # Drop rows with any NaN or corrupt members (sentinel < -200)
    all_member_cols = fc1_cols + fc2_cols
    df = df.dropna(subset=["obs_value"] + all_member_cols)
    member_np  = df[all_member_cols].values
    corrupt    = (member_np < -200).any(axis=1)
    if corrupt.any():
        print(f"  [day {day}] Removing {corrupt.sum()} corrupt rows")
        df = df[~corrupt].reset_index(drop=True)

    # Optional subsampling
    if max_samples and len(df) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_samples, replace=False)
        idx.sort()
        df = df.iloc[idx].reset_index(drop=True)

    print(f"  [day {day}] {len(df):,} cases kept  (from {n_raw:,} raw)")

    obs_arr = df["obs_value"].values.astype(np.float64)
    fc1_np  = df[fc1_cols].values.astype(np.float64)
    fc2_np  = df[fc2_cols].values.astype(np.float64)
    return obs_arr, fc1_np, fc2_np, fc1, fc2


# ── Q-Q computation ───────────────────────────────────────────────────────────

def member_mean(fc_np):
    """Per-case mean of ensemble members (N, M) → (N,)."""
    return np.nanmean(fc_np, axis=1)


def qq_percentiles(obs_all, fc_members_all, fc_mean_all, levels):
    """Compute Q-Q percentile values for obs, pooled ensemble, and ensemble mean.

    Parameters
    ----------
    obs_all        : (N,) observed values
    fc_members_all : (N, M) ensemble member values  (may contain NaN)
    fc_mean_all    : (N,) per-case ensemble mean
    levels         : percentile levels (0–100)

    Returns
    -------
    obs_q, pooled_q, mean_q – each shape (len(levels),)
    """
    obs_q    = np.nanpercentile(obs_all, levels)
    pooled   = fc_members_all.ravel()
    pooled   = pooled[np.isfinite(pooled)]
    pooled_q = np.percentile(pooled, levels)
    mean_q   = np.nanpercentile(fc_mean_all, levels)
    return obs_q, pooled_q, mean_q


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_qq_panel(ax, obs_q_dot, obs_q_x,
                   fc1_pool_dot, fc1_pool_x, fc1_mean_dot, fc1_mean_x,
                   fc2_pool_dot, fc2_pool_x, fc2_mean_dot, fc2_mean_x,
                   dot_levels, x_levels,
                   m1_name, m2_name, unit,
                   title, label_levels=None):
    """Draw a single Q-Q panel with pooled and mean ensemble lines."""
    all_vals = np.concatenate([
        obs_q_dot, obs_q_x,
        fc1_pool_dot, fc1_pool_x, fc1_mean_dot, fc1_mean_x,
        fc2_pool_dot, fc2_pool_x, fc2_mean_dot, fc2_mean_x,
    ])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = max((vmax - vmin) * 0.08, 0.5)
    lim = (vmin - margin, vmax + margin)

    # 1:1 reference
    ax.plot(lim, lim, color="black", ls="--", lw=1.2, zorder=1, label="1:1 (perfect)")

    # p1 threshold line (obs value at p1)
    p1_idx = np.searchsorted(np.sort(np.concatenate([dot_levels, x_levels])), 1.0)
    p1_obs = float(np.nanpercentile(np.concatenate([obs_q_dot, obs_q_x]),
                                    50.0))  # rough; label_levels handles exact
    # Find exact p1 obs value from dot_levels (p1 is first integer level)
    if len(dot_levels) > 0:
        p1_obs = obs_q_dot[0]
        ax.axvline(p1_obs, color="gray", lw=0.9, ls=":", zorder=2, alpha=0.7)
        ax.text(p1_obs, lim[1] - (lim[1] - lim[0]) * 0.04, " p1",
                color="gray", fontsize=8, va="top", ha="left")

    for (pool_dot, pool_x, mean_dot, mean_x, color, name) in [
        (fc1_pool_dot, fc1_pool_x, fc1_mean_dot, fc1_mean_x, C1, m1_name),
        (fc2_pool_dot, fc2_pool_x, fc2_mean_dot, fc2_mean_x, C2, m2_name),
    ]:
        # Pooled ensemble: solid line + markers
        all_obs = np.concatenate([obs_q_x, obs_q_dot])
        all_pool = np.concatenate([pool_x, pool_dot])
        sort_idx = np.argsort(all_obs)
        ax.plot(all_obs[sort_idx], all_pool[sort_idx],
                color=color, lw=1.8, ls="-", zorder=3, alpha=0.85)
        ax.scatter(obs_q_dot, pool_dot, color=color, s=35, zorder=4, marker="o")
        ax.scatter(obs_q_x,   pool_x,   color=color, s=70, zorder=5, marker="x",
                   linewidths=2)

        # Ensemble mean: dashed line + hollow markers
        all_mean = np.concatenate([mean_x, mean_dot])
        ax.plot(all_obs[sort_idx], all_mean[sort_idx],
                color=color, lw=1.5, ls="--", zorder=3, alpha=0.65)
        ax.scatter(obs_q_dot, mean_dot, color=color, s=25, zorder=4,
                   marker="o", facecolors="none", linewidths=1.5)
        ax.scatter(obs_q_x,   mean_x,   color=color, s=55, zorder=5,
                   marker="x", linewidths=1.5, alpha=0.65)

    # Percentile level annotations
    if label_levels is None:
        label_levels = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
    all_lvl = np.concatenate([x_levels, dot_levels])
    all_obs_q = np.concatenate([obs_q_x, obs_q_dot])
    for lvl, ov in zip(all_lvl, all_obs_q):
        if lvl in label_levels:
            ax.annotate(f"p{lvl:g}",
                        xy=(ov, ov), xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color="dimgray", zorder=7)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed [{unit}]", fontsize=11)
    ax.set_ylabel(f"Forecast [{unit}]", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Legend
    legend_handles = [
        mlines.Line2D([], [], color=C1, lw=1.8, ls="-",  marker="o", ms=5,
                      label=f"{m1_name} (pooled ens)"),
        mlines.Line2D([], [], color=C1, lw=1.5, ls="--", marker="o", ms=5,
                      markerfacecolor="none",
                      label=f"{m1_name} (ens mean)"),
        mlines.Line2D([], [], color=C2, lw=1.8, ls="-",  marker="o", ms=5,
                      label=f"{m2_name} (pooled ens)"),
        mlines.Line2D([], [], color=C2, lw=1.5, ls="--", marker="o", ms=5,
                      markerfacecolor="none",
                      label=f"{m2_name} (ens mean)"),
        mlines.Line2D([], [], color="black", lw=1.2, ls="--",
                      label="1:1 (perfect)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right")
    ax.text(0.02, 0.98,
            "●  = p1–p10 (Δ1%)\n×  = p0.1–p0.9 (Δ0.1%)\n"
            "Solid  = pooled members\nDashed = ensemble mean",
            transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="lightgray", alpha=0.85))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cold-extreme Q-Q plot for ensemble forecasts (multi-day)")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--days",       nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Forecast days to combine (default: 1 2 3 4)")
    parser.add_argument("--season",     default=None)
    parser.add_argument("--orog",       default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    config   = load_config(args.config)
    variable = config["variable"]
    var_label, unit = VARIABLE_LABELS.get(variable, (variable, ""))

    # ── Load and combine days ─────────────────────────────────────────────────
    obs_parts, fc1_parts, fc2_parts = [], [], []
    m1_name = m2_name = None
    total_rows = 0

    for day in args.days:
        print(f"\nLoading day {day}...")
        result = load_day_ensemble(config, day,
                                   season=args.season, orog=args.orog,
                                   max_samples=300_000, seed=42)
        if result is None:
            continue
        obs_arr, fc1_np, fc2_np, m1, m2 = result
        m1_name, m2_name = m1, m2
        obs_parts.append(obs_arr)
        fc1_parts.append(fc1_np)
        fc2_parts.append(fc2_np)
        total_rows += len(obs_arr)

    if total_rows == 0:
        sys.exit("ERROR: No data loaded.")

    obs_all      = np.concatenate(obs_parts)
    fc1_np_all   = np.vstack(fc1_parts)
    fc2_np_all   = np.vstack(fc2_parts)
    fc1_mean_all = member_mean(fc1_np_all)
    fc2_mean_all = member_mean(fc2_np_all)

    print(f"\nCombined: {len(obs_all):,} cases from days {args.days}")
    print(f"  {m1_name} vs {m2_name}")
    print(f"  Ensemble members: {fc1_np_all.shape[1]}")

    # ── Percentile levels ─────────────────────────────────────────────────────
    # Fine tail: 0.1 to 0.9 (x-markers)
    x_levels   = np.round(np.arange(0.1, 1.0, 0.1), 2)   # 0.1…0.9
    # Integer: 1 to 10 (dot-markers)
    dot_levels = np.arange(1.0, 11.0, 1.0)                # 1…10

    # Wide view: 0.1 to 10
    def _compute(obs_a, fc1_m, fc1_mn, fc2_m, fc2_mn, levels):
        obs_q, p1_q, m1_q = qq_percentiles(obs_a, fc1_m, fc1_mn, levels)
        _,     p2_q, m2_q = qq_percentiles(obs_a, fc2_m, fc2_mn, levels)
        return obs_q, p1_q, m1_q, p2_q, m2_q

    obs_x,   p1x,  m1x,  p2x,  m2x  = _compute(
        obs_all, fc1_np_all, fc1_mean_all, fc2_np_all, fc2_mean_all, x_levels)
    obs_dot, p1d,  m1d,  p2d,  m2d  = _compute(
        obs_all, fc1_np_all, fc1_mean_all, fc2_np_all, fc2_mean_all, dot_levels)

    # Zoomed: 0.1 to 1.0 only (finer, sub-percentile focus)
    zoom_levels = np.round(np.arange(0.1, 1.05, 0.1), 2)   # 0.1, 0.2, …, 1.0
    obs_zoom, p1z, m1z, p2z, m2z = _compute(
        obs_all, fc1_np_all, fc1_mean_all, fc2_np_all, fc2_mean_all, zoom_levels)
    # Split for marker style: ×=0.1–0.9, ●=1.0
    zoom_x_mask   = zoom_levels < 1.0
    zoom_dot_mask = zoom_levels >= 1.0

    # ── Build figure ──────────────────────────────────────────────────────────
    condition_parts = [f"Days {','.join(str(d) for d in args.days)}"]
    if args.season:
        condition_parts.append(args.season.upper())
    if args.orog:
        condition_parts.append(f"{args.orog} terrain")
    condition = " | ".join(condition_parts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Cold-extreme Q-Q  —  {var_label}  |  {condition}\n"
        f"N = {len(obs_all):,} cases × {fc1_np_all.shape[1]} members  |  "
        f"{m1_name} (blue)  vs  {m2_name} (red)",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # Panel 1: full cold range p0.1–p10
    _plot_qq_panel(
        ax1,
        obs_q_dot=obs_dot, obs_q_x=obs_x,
        fc1_pool_dot=p1d, fc1_pool_x=p1x,
        fc1_mean_dot=m1d, fc1_mean_x=m1x,
        fc2_pool_dot=p2d, fc2_pool_x=p2x,
        fc2_mean_dot=m2d, fc2_mean_x=m2x,
        dot_levels=dot_levels, x_levels=x_levels,
        m1_name=m1_name, m2_name=m2_name, unit=unit,
        title="Cold tail Q-Q  (p0.1 – p10)",
        label_levels={0.1, 0.5, 1.0, 2.0, 5.0, 10.0},
    )

    # Panel 2: zoomed p0.1–p1.0 only
    _plot_qq_panel(
        ax2,
        obs_q_dot=obs_zoom[zoom_dot_mask],
        obs_q_x=obs_zoom[zoom_x_mask],
        fc1_pool_dot=p1z[zoom_dot_mask], fc1_pool_x=p1z[zoom_x_mask],
        fc1_mean_dot=m1z[zoom_dot_mask], fc1_mean_x=m1z[zoom_x_mask],
        fc2_pool_dot=p2z[zoom_dot_mask], fc2_pool_x=p2z[zoom_x_mask],
        fc2_mean_dot=m2z[zoom_dot_mask], fc2_mean_x=m2z[zoom_x_mask],
        dot_levels=dot_levels[dot_levels <= 1.0],
        x_levels=zoom_levels[zoom_x_mask],
        m1_name=m1_name, m2_name=m2_name, unit=unit,
        title="Extreme cold tail zoom  (p0.1 – p1.0)",
        label_levels={0.1, 0.2, 0.3, 0.5, 0.7, 1.0},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"qq_cold_ens"
    if args.season:
        stem += f"_{args.season}"
    if args.orog:
        stem += f"_{args.orog}"
    stem += f"_days{''.join(str(d) for d in args.days)}"

    cfg_stem = Path(args.config).stem
    out_path = out_dir / f"{cfg_stem}_{stem}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ Saved: {out_path}")


if __name__ == "__main__":
    main()
