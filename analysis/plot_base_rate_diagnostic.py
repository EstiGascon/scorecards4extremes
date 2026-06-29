#!/usr/bin/env python3
"""
Base-Rate Diagnostic Plot
=========================
Reads score CSV files produced by the scorecards4extremes pipeline and
produces a two-panel diagnostic figure:

  Left panel  — Actual obs exceedance rate (%) vs the expected climatological
                rate (1 % for p99 / p1 thresholds).  A large deviation
                explains why ETS / PSS values are outside their normal range.

  Right panel — ETS lead-time profile with a shaded band showing the
                literature-typical range for well-calibrated extremes
                verification (Hamill & Juras 2006; Mittermaier & Roberts 2010;
                ECMWF Tech Memo; Bouallegue et al. 2023).

USAGE
-----
  python plot_base_rate_diagnostic.py --results-dir <path> [options]

  # example:
  python plot_base_rate_diagnostic.py \\
      --results-dir ./results/tp24_local_p99obsclim_destine50r1 \\
      --variable tp24 --expected-rate 1.0 --label "tp24 p99 obs-clim"

  python plot_base_rate_diagnostic.py \\
      --results-dir ./results/2t_local_p1obsclim_destine50r1 \\
      --variable 2t --expected-rate 1.0 --label "2t p1 obs-clim (cold)"
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Literature ETS ranges per variable + percentile (day1, day3, day5)
# Sources: ECMWF Tech Memo; Mittermaier & Roberts 2010 QJRMS;
#          Hamill 1999 WAF; Bouallegue et al. 2023 BAMS
# ============================================================================
LITERATURE_ETS = {
    # (variable, expected_rate): {day: (low, high)}
    ("tp24", 1.0): {1: (0.05, 0.20), 3: (0.03, 0.12), 5: (0.00, 0.08)},
    ("2t",   1.0): {1: (0.20, 0.45), 3: (0.12, 0.28), 5: (0.05, 0.20)},
    ("10ff", 1.0): {1: (0.10, 0.30), 3: (0.05, 0.20), 5: (0.02, 0.12)},
}

COLOR_FC1 = "#1f77b4"
COLOR_FC2 = "#d62728"
COLOR_EXP = "#2ca02c"    # green for expected rate line
COLOR_LIT  = "#ff7f0e"   # orange shading for literature range


def load_scores(results_dir: Path, variable: str) -> pd.DataFrame:
    """
    Concatenate all scores_by_leadtime_{variable}_*.csv files found in
    results_dir.  Adds a 'subset' column with the season_orog label.
    """
    pattern = f"scores_by_leadtime_{variable}_*.csv"
    files = sorted(results_dir.glob(pattern))
    if not files:
        # Try without season/orog suffix (e.g. 2t has no season split)
        pattern = f"scores_by_leadtime_{variable}.csv"
        files = list(results_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No score files matching 'scores_by_leadtime_{variable}*.csv' "
            f"in '{results_dir}'"
        )

    parts = []
    for fp in files:
        df = pd.read_csv(fp)
        # extract subset label from filename
        stem = fp.stem  # e.g. scores_by_leadtime_tp24_ASO_flat
        suffix = stem.replace(f"scores_by_leadtime_{variable}", "").lstrip("_")
        df["subset"] = suffix if suffix else "all"
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


def compute_base_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add base_rate_pct column if n_exceedances and n_samples are present."""
    if "n_exceedances" in df.columns and "n_samples" in df.columns:
        df = df.copy()
        df["base_rate_pct"] = df["n_exceedances"] / df["n_samples"] * 100.0
    return df


def interpolate_lit_range(lit_dict: dict, days: list) -> tuple:
    """
    Linearly interpolate literature ETS range to arbitrary lead-time days.
    Returns (lo_array, hi_array).
    """
    anchor_days = sorted(lit_dict.keys())
    lo_anchors  = [lit_dict[d][0] for d in anchor_days]
    hi_anchors  = [lit_dict[d][1] for d in anchor_days]
    lo = np.interp(days, anchor_days, lo_anchors)
    hi = np.interp(days, anchor_days, hi_anchors)
    return lo, hi


def make_diagnostic_figure(
    df: pd.DataFrame,
    variable: str,
    expected_rate: float,
    label: str,
    fc1_name: str,
    fc2_name: str,
    output_path: Path,
    dpi: int = 150,
):
    """Build and save the two-panel diagnostic figure."""

    subsets = df["subset"].unique().tolist()
    lit_key = (variable, expected_rate)
    lit_range = LITERATURE_ETS.get(lit_key)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Base-Rate Diagnostic — {label}\n"
        f"Models: {fc1_name} (blue)  vs  {fc2_name} (red)",
        fontsize=12, fontweight="bold",
    )

    ax_rate, ax_ets = axes

    # -----------------------------------------------------------------------
    # Left panel: observed exceedance rate by lead time
    # -----------------------------------------------------------------------
    ax_rate.axhline(expected_rate, color=COLOR_EXP, linewidth=2,
                    linestyle="--", label=f"Expected ({expected_rate:.1f}%)")

    has_rate_data = "base_rate_pct" in df.columns
    cmap = plt.cm.tab10
    for i, subset in enumerate(subsets):
        sub = df[df["subset"] == subset].sort_values("forecast_day")
        if has_rate_data:
            ax_rate.plot(
                sub["forecast_day"], sub["base_rate_pct"],
                marker="o", linewidth=1.5, color=cmap(i),
                label=subset if len(subsets) > 1 else "observed rate",
            )

    if not has_rate_data:
        ax_rate.text(0.5, 0.5, "n_exceedances / n_samples\nnot available",
                     transform=ax_rate.transAxes, ha="center", va="center",
                     fontsize=11, color="gray")

    ax_rate.set_xlabel("Forecast day", fontsize=11)
    ax_rate.set_ylabel("Obs exceedance rate [%]", fontsize=11)
    ax_rate.set_title("Actual vs Expected threshold exceedance rate", fontsize=10)
    ax_rate.legend(fontsize=9)
    ax_rate.grid(True, alpha=0.3)

    # Annotation: average ratio
    if has_rate_data and not df["base_rate_pct"].isna().all():
        avg_rate = df["base_rate_pct"].mean()
        ratio = avg_rate / expected_rate
        color_txt = "red" if ratio > 2.0 else "darkorange" if ratio > 1.3 else "green"
        ax_rate.text(
            0.97, 0.97,
            f"Mean observed: {avg_rate:.1f}%\n"
            f"Expected:       {expected_rate:.1f}%\n"
            f"Ratio:          {ratio:.1f}×",
            transform=ax_rate.transAxes,
            ha="right", va="top", fontsize=10,
            color=color_txt, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color_txt, alpha=0.85),
        )

    # -----------------------------------------------------------------------
    # Right panel: ETS vs lead time
    # -----------------------------------------------------------------------
    all_days = sorted(df["forecast_day"].unique())

    if lit_range:
        lo, hi = interpolate_lit_range(lit_range, all_days)
        ax_ets.fill_between(
            all_days, lo, hi,
            color=COLOR_LIT, alpha=0.25,
            label="Literature range\n(well-calibrated extremes)",
        )

    for i, subset in enumerate(subsets):
        sub = df[df["subset"] == subset].sort_values("forecast_day")
        ax_ets.plot(
            sub["forecast_day"], sub["ETS_fc1"],
            marker="o", linewidth=1.8, color=COLOR_FC1,
            label=f"{fc1_name}" + (f" ({subset})" if len(subsets) > 1 else ""),
        )
        ax_ets.plot(
            sub["forecast_day"], sub["ETS_fc2"],
            marker="s", linewidth=1.8, linestyle="--", color=COLOR_FC2,
            label=f"{fc2_name}" + (f" ({subset})" if len(subsets) > 1 else ""),
        )

    ax_ets.set_xlabel("Forecast day", fontsize=11)
    ax_ets.set_ylabel("ETS", fontsize=11)
    ax_ets.set_title("ETS vs lead time  (orange band = literature range\nfor a well-calibrated p99/p1 threshold)", fontsize=10)
    ax_ets.set_ylim(bottom=0)
    ax_ets.legend(fontsize=9, loc="upper right")
    ax_ets.grid(True, alpha=0.3)

    # Explanation box
    if has_rate_data and not df["base_rate_pct"].isna().all():
        avg_rate = df["base_rate_pct"].mean()
        ratio = avg_rate / expected_rate
        if ratio > 1.5:
            msg = (
                f"⚠ Threshold exceeded {ratio:.1f}× more often than\n"
                f"expected ({avg_rate:.1f}% vs {expected_rate:.1f}%).\n"
                f"High ETS reflects an anomalous period,\n"
                f"not typical model skill at this percentile."
            )
            ax_ets.text(
                0.97, 0.97, msg,
                transform=ax_ets.transAxes,
                ha="right", va="top", fontsize=9,
                color="red",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="red", alpha=0.9),
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


def print_summary_table(df: pd.DataFrame, expected_rate: float, label: str):
    """Print a compact ASCII summary table."""
    print(f"\n{'='*70}")
    print(f"BASE-RATE DIAGNOSTIC SUMMARY — {label}")
    print(f"{'='*70}")
    print(f"{'Day':>4}  {'Subset':<18}  {'Base rate':>10}  {'Ratio':>6}  {'ETS fc1':>8}  {'ETS fc2':>8}")
    print("-" * 70)

    has_rate = "base_rate_pct" in df.columns
    for subset in sorted(df["subset"].unique()):
        sub = df[df["subset"] == subset].sort_values("forecast_day")
        for _, row in sub.iterrows():
            rate_str = f"{row['base_rate_pct']:.2f}%" if has_rate else "  n/a  "
            ratio_str = f"{row['base_rate_pct']/expected_rate:.1f}×" if has_rate else "  n/a"
            print(
                f"{int(row['forecast_day']):>4}  {subset:<18}  {rate_str:>10}  "
                f"{ratio_str:>6}  {row['ETS_fc1']:>8.4f}  {row['ETS_fc2']:>8.4f}"
            )
    print("=" * 70)
    if has_rate:
        avg = df["base_rate_pct"].mean()
        ratio = avg / expected_rate
        status = "ANOMALOUS" if ratio > 2.0 else "ELEVATED" if ratio > 1.3 else "OK"
        print(f"\nAverage observed rate: {avg:.2f}%  (expected: {expected_rate:.1f}%)")
        print(f"Average ratio:         {ratio:.1f}×  → {status}")
        if ratio > 1.5:
            print(
                f"\n⚠  WARNING: The verification period had {ratio:.1f}× more threshold\n"
                f"   exceedances than the historical climatology expects. ETS values\n"
                f"   are inflated and should not be compared to results from\n"
                f"   climatologically normal periods without this context."
            )


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="plot_base_rate_diagnostic.py",
        description=(
            "Diagnostic plot: observed exceedance rate vs expected, and ETS "
            "vs literature range. Helps explain anomalously high ETS values "
            "when the verification period is climatologically anomalous."
        ),
    )
    parser.add_argument("--results-dir", required=True, metavar="DIR",
                        help="Results directory containing scores_by_leadtime_*.csv files.")
    parser.add_argument("--variable", required=True, choices=["tp24", "2t", "10ff"],
                        help="Variable name (tp24, 2t, 10ff).")
    parser.add_argument("--expected-rate", type=float, default=1.0, metavar="PCT",
                        help="Expected threshold exceedance rate in %% (default: 1.0 for p99/p1).")
    parser.add_argument("--label", default=None,
                        help="Descriptive label for plot title.")
    parser.add_argument("--fc1-name", default="fc1",
                        help="Name of forecast model 1 (for legend).")
    parser.add_argument("--fc2-name", default="fc2",
                        help="Name of forecast model 2 (for legend).")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Output PNG path. Default: <results-dir>/base_rate_diagnostic.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-plot", action="store_true",
                        help="Print summary table only, do not save a figure.")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    label = args.label or results_dir.name

    print(f"\nLoading scores from: {results_dir}")
    df = load_scores(results_dir, args.variable)
    df = compute_base_rate(df)
    print(f"  Loaded {len(df)} rows across {df['subset'].nunique()} subset(s).")

    # Try to infer model names from CSV columns or filename
    fc1_name = args.fc1_name
    fc2_name = args.fc2_name
    # e.g. directory name: tp24_local_p99obsclim_destine50r1 → not enough info;
    # the user should supply --fc1-name / --fc2-name

    print_summary_table(df, args.expected_rate, label)

    if not args.no_plot:
        out = Path(args.output) if args.output else results_dir / "base_rate_diagnostic.png"
        make_diagnostic_figure(
            df, args.variable, args.expected_rate,
            label, fc1_name, fc2_name,
            out, dpi=args.dpi,
        )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
