#!/usr/bin/env python3
"""
6-panel Q-Q figure for the paper: IFS vs AIFS, day-3 (72h) forecasts,
flat vs high/complex terrain, for 2t / 10ff / tp24.

Rows    : 2m Temperature | 10m Wind Speed | 24h Precipitation
Columns : Flat terrain (sdfor < 40 m) | High/complex terrain (sdfor >= 120 m)

Reuses the exact data-loading/filtering logic from diagnostics/qq/plot_qq.py
(load_config, get_model_names, load_data, apply_filters) so the panel figure
is numerically identical to the individual qq_*.png plots already produced.
Legend always reads "IFS" / "AIFS" regardless of the config's internal model
names (ifs_oper, aifs1.0_oper, ifs_oper_0.25degree, ...).

Output: plots/qq_day3_ifsvsaifs/panel_qq_ifs_vs_aifs_day3.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

_HERE = Path(__file__).resolve().parent            # diagnostics/qq
_DIAG = _HERE.parent                               # diagnostics/  (shared _style)
ROOT = _DIAG.parent                                # repository root
for _p in (ROOT, ROOT / "src", _DIAG, _HERE):
    sys.path.insert(0, str(_p))
import _style  # noqa: E402
from plot_qq import (  # noqa: E402
    load_config, get_model_names, load_data, apply_filters,
    get_threshold_from_config, VARIABLE_LABELS,
)

OUT = ROOT / "plots/qq_day3_ifsvsaifs/panel_qq_ifs_vs_aifs_day3.png"
# Day-3 lead times per variable: 2t/10ff are 6-hourly (lead_time_frequency: 6),
# so day 3 spans steps 54/60/66/72h; tp24 is a 24h accumulation
# (lead_time_frequency: 24), so day 3 is step 72h only.
DAY3_LEAD_TIMES = {"2t": [54, 60, 66, 72], "10ff": [54, 60, 66, 72], "tp24": [72]}
OROGS = [("low", "Flat terrain (sdfor < 40 m)"), ("high", "High / complex terrain (sdfor \u2265 120 m)")]
ROWS = [
    ("2t",   ROOT / "configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml"),
    ("10ff", ROOT / "configs/deterministic/config_10ff_local_p99obsclim_aifs_ifs_single.yaml"),
    ("tp24", ROOT / "configs/deterministic/config_tp24_local_p99obsclim_aifs_ifs_single.yaml"),
]

LABEL_FC1, LABEL_FC2 = "IFS", "AIFS"
PANEL_LETTERS = "abcdef"


def _percentile_series(obs, fc1, fc2, variable):
    dot_levels = np.arange(1, 99.5, 1.0) if variable == "2t" else np.arange(0, 99.5, 1.0)
    x_levels = np.linspace(99.0, 99.9, 10)
    out = {
        "obs_dots": np.nanpercentile(obs, dot_levels), "fc1_dots": np.nanpercentile(fc1, dot_levels),
        "fc2_dots": np.nanpercentile(fc2, dot_levels),
        "obs_x": np.nanpercentile(obs, x_levels), "fc1_x": np.nanpercentile(fc1, x_levels),
        "fc2_x": np.nanpercentile(fc2, x_levels),
    }
    if variable == "2t":
        cold_levels = np.linspace(0.1, 1.0, 10)
        out["obs_cold_x"] = np.nanpercentile(obs, cold_levels)
        out["fc1_cold_x"] = np.nanpercentile(fc1, cold_levels)
        out["fc2_cold_x"] = np.nanpercentile(fc2, cold_levels)
    return out


def _draw_panel(ax, df, variable, var_label, unit, threshold, letter):
    obs, fc1, fc2 = df["obs_value"].values, df["fc1_value"].values, df["fc2_value"].values
    p = _percentile_series(obs, fc1, fc2, variable)
    has_cold = variable == "2t"

    corr1 = float(np.corrcoef(fc1, obs)[0, 1]); bias1 = float(np.mean(fc1 - obs))
    rmse1 = float(np.sqrt(np.mean((fc1 - obs) ** 2)))
    corr2 = float(np.corrcoef(fc2, obs)[0, 1]); bias2 = float(np.mean(fc2 - obs))
    rmse2 = float(np.sqrt(np.mean((fc2 - obs) ** 2)))
    frac_rmse = (rmse2 - rmse1) / rmse1 * 100.0 if rmse1 else 0.0
    frac_corr = (corr2 - corr1) / abs(corr1) * 100.0 if corr1 else 0.0
    frac_bias = (abs(bias2) - abs(bias1)) / abs(bias1) * 100.0 if bias1 else 0.0

    all_capped = [p["obs_dots"], p["obs_x"], p["fc1_dots"], p["fc1_x"], p["fc2_dots"], p["fc2_x"]]
    if has_cold:
        all_capped += [p["obs_cold_x"], p["fc1_cold_x"], p["fc2_cold_x"]]
    all_capped = np.concatenate(all_capped)
    vmin, vmax = np.nanmin(all_capped), np.nanmax(all_capped)
    margin = (vmax - vmin) * 0.05
    lim = (vmin - margin, vmax + margin)

    ax.plot(lim, lim, color=_style.C_REF, linestyle="--", linewidth=1.1, zorder=1)

    for dots, x, cold_x, color, label in [
        (p["fc1_dots"], p["fc1_x"], p.get("fc1_cold_x"), _style.C_FC1, LABEL_FC1),
        (p["fc2_dots"], p["fc2_x"], p.get("fc2_cold_x"), _style.C_FC2, LABEL_FC2),
    ]:
        ax.scatter(p["obs_dots"], dots, color=color, s=16, zorder=3, marker="o", label=label)
        ax.scatter(p["obs_x"], x, color=color, s=42, zorder=4, marker="x", linewidths=1.8)
        if has_cold:
            ax.scatter(p["obs_cold_x"], cold_x, color=color, s=42, zorder=4, marker="x", linewidths=1.8)

    if threshold is not None:
        ax.axvline(threshold, color="green", linestyle=":", linewidth=1.1, alpha=0.85, zorder=1)
        ax.axhline(threshold, color="green", linestyle=":", linewidth=1.1, alpha=0.85, zorder=1)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed [{unit}]", fontsize=12)
    ax.set_ylabel(f"Forecast [{unit}]", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left")
    ax.text(0.97, 0.03, f"N = {len(df):,}", transform=ax.transAxes, fontsize=9.5,
            va="bottom", ha="right", color="0.25")

    # ------------------------------------------------ inset stats bar chart
    ax_ins = ax.inset_axes([0.11, 0.66, 0.30, 0.20])
    bar_x, bar_vals = [0.0, 0.27, 0.52], [frac_rmse, frac_corr, frac_bias]
    ax_ins.bar(bar_x, bar_vals, width=0.20, color=_style.C_FC2)
    ax_ins.axhline(0, color="black", linewidth=0.7)
    ax_ins.set_xticks(bar_x); ax_ins.set_xticklabels(["RMSE", "\u03c1", "|b|"], fontsize=9)
    ax_ins.set_ylabel("\u0394 [%]", fontsize=9, labelpad=1)
    ax_ins.tick_params(axis="y", labelsize=8, pad=1)
    trans = mtransforms.blended_transform_factory(ax_ins.transData, ax_ins.transAxes)
    for i, (v2, v1) in enumerate(zip([rmse2, corr2, bias2], [rmse1, corr1, bias1])):
        ax_ins.text(bar_x[i], 1.24, f"{v2:.2f}", transform=trans, ha="center", va="bottom",
                    fontsize=8.5, color=_style.C_FC2, weight="bold", clip_on=False)
        ax_ins.text(bar_x[i], 1.05, f"{v1:.2f}", transform=trans, ha="center", va="bottom",
                    fontsize=8.5, color=_style.C_FC1, weight="bold", clip_on=False)

    return ax.get_legend_handles_labels()


def main():
    _style.apply_style()
    plt.rcParams.update({"font.size": 12})

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 11))

    for col_idx, (variable, config_path) in enumerate(ROWS):
        var_label, _ = VARIABLE_LABELS.get(variable, (variable, ""))
        axes[0, col_idx].set_title(var_label, fontsize=14, fontweight="bold", pad=6)

    handles_labels = None
    for col_idx, (variable, config_path) in enumerate(ROWS):
        config = load_config(str(config_path))
        var_label, unit = VARIABLE_LABELS.get(variable, (variable, ""))
        fc1_name, fc2_name = get_model_names(config)
        threshold = get_threshold_from_config(config)

        print(f"\nLoading {variable} ...")
        df_full = load_data(config, fc1_name, fc2_name)

        for row_idx, (orog, _) in enumerate(OROGS):
            filtered = apply_filters(df_full, config, season=None, orog_type=orog,
                                      lead_time=DAY3_LEAD_TIMES[variable], start_date=None, end_date=None)
            ax = axes[row_idx, col_idx]
            letter = PANEL_LETTERS[row_idx * len(ROWS) + col_idx]
            hl = _draw_panel(ax, filtered, variable, var_label, unit, threshold, letter)
            if handles_labels is None:
                handles_labels = hl
        del df_full

    # Row label (terrain type) to the left of each row
    for row_idx, (_, orog_title) in enumerate(OROGS):
        axes[row_idx, 0].annotate(
            orog_title, xy=(-0.20, 0.5), xycoords="axes fraction",
            fontsize=13, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

    handles, labels = handles_labels
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, fontsize=16, frameon=True, markerscale=1.8)
    fig.suptitle(
        "Q-Q plots: IFS vs AIFS  \u2014  day-3 (72h) forecasts vs SYNOP observations, Europe\n"
        "2025-03-01 to 2026-02-28",
        fontsize=15, fontweight="bold", y=1.02,
    )

    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\n\u2713 Saved: {OUT}")


if __name__ == "__main__":
    main()
