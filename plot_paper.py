"""
Paper-ready heatmap variant of plot.create_smooth_multicolumn_heatmap.

Kept as a separate module so plot.py — used by the full pipeline for every
other config — is never touched. Differences from the standard smooth
multicolumn heatmap:
  - Only the central percentage-difference value is annotated per cell
    (no per-cell sample count / raw score shown).
  - Larger, paper-legible fonts; title sits close to the heatmap instead
    of floating in blank space above it.
  - Score name kept as its short code (e.g. "twMAE") rather than spelled out.
  - Model names can be overridden with a human-readable label via
    model_names['fc1_display'] / model_names['fc2_display'].
  - Output filename gets a "_paper" suffix so it never overwrites the
    standard heatmap PNG produced by plot.py.
"""
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import _get_color_limit, _ERROR_SCORES, _BOUNDED_ERROR_SCORES, _HIGHER_IS_BETTER_SCORES


def create_smooth_multicolumn_heatmap_paper(all_results, variable, threshold_value, output_dir,
                                             model_names, score_type='twMAE', season=None, config=None):
    if not all_results or not all(r['results'].get('by_leadtime') is not None for r in all_results):
        print(f"  ⚠ Cannot create paper heatmap for {score_type}")
        return

    for result_set in all_results:
        if result_set['results']['by_leadtime'].empty:
            print(f"  ⚠ Skipping {score_type} paper heatmap – no data")
            return
        if f'{score_type}_diff' not in result_set['results']['by_leadtime'].columns:
            print(f"  ⚠ Skipping {score_type} paper heatmap – missing diff column")
            return

    units = {'2t': '°C', '10ff': 'm/s', 'tp24': 'mm'}
    unit = units.get(variable, '')

    fc1_col = f'{score_type}_fc1'
    fc2_col = f'{score_type}_fc2'
    sig_col = f'{score_type}_is_significant'

    num_cols = len(all_results)
    orog_labels = []
    season_labels = []
    for r in all_results:
        season_labels.append(r.get('season', '') or '')
        orog_labels.append(r['orog_type'].upper() if r['orog_type'] else 'ALL')

    lead_times = sorted(all_results[0]['results']['by_leadtime']['lead_time'].unique())
    has_forecast_day = 'forecast_day' in all_results[0]['results']['by_leadtime'].columns
    pivot_index = 'forecast_day' if has_forecast_day else 'lead_time'
    all_pivot_vals = set()
    for r in all_results:
        all_pivot_vals.update(r['results']['by_leadtime'][pivot_index].unique())
    row_keys = sorted(all_pivot_vals)

    _plot_days = (config or {}).get('plot', {}).get('forecast_days') or (config or {}).get('forecast_days')
    if _plot_days:
        _filter_vals = set(_plot_days) if pivot_index == 'forecast_day' else None
        if _filter_vals:
            row_keys = sorted([v for v in row_keys if v in _filter_vals])

    # ---- Fixed box size ----
    box_size = 0.6  # inches per cell
    num_rows = len(row_keys)
    # Fixed margins (inches) reserved above/below the heatmap itself, independent
    # of num_rows, so the title sits close to the plot instead of floating in a
    # blank region that scales up with the number of rows.
    top_margin_in = 0.75     # two-line title, kept tight above the column headers
    bottom_margin_in = 0.55  # season labels below the heatmap
    fig_width = box_size * num_cols + 1.4
    fig_height = box_size * num_rows + top_margin_in + bottom_margin_in

    # Font floors sized for legibility in a printed scientific figure.
    main_fontsize = max(8, int(box_size * 10))
    label_fontsize = max(10, int(box_size * 10))
    # title_fontsize sizes the column headers (FLAT/HILLY/COMPLEX), y-axis label,
    # and colorbar label — constrained by the (narrow) column width, so only a
    # modest bump. The big figure title uses its own, larger size below.
    title_fontsize = max(8, int(box_size * 11))
    suptitle_fontsize = max(10, int(box_size * 11))
    sig_linewidth = box_size * 3.5

    # ---- Colour limits ----
    color_limit_pct = float(_get_color_limit(score_type))
    color_limit_raw = color_limit_pct / 100.0

    # ---- Smooth continuous colour palette ----
    if score_type in _ERROR_SCORES or score_type in _BOUNDED_ERROR_SCORES:
        palette = [
            "#0044cf", "#274ed3", "#3a5ad6", "#4865da", "#546fdd", "#5f7be1",
            "#6886e4", "#7291e7", "#7a9ceb", "#83a8ee", "#8ab3f1", "#92c0f4",
            "#99ccf7", "#a0d8fa", "#a7e4fd", "#d3d3d3",
            "#fbd5c0", "#f8c7b1", "#f4bba2", "#f0ad94", "#eba086", "#e69478",
            "#e1876a", "#db7a5d", "#d56d4f", "#cf6043", "#c85235", "#c14329",
            "#b9341c", "#b2210e", "#aa0000",
        ]
    else:
        palette = [
            "#aa0000", "#b2210e", "#b9341c", "#c14329", "#c85235", "#cf6043",
            "#d56d4f", "#db7a5d", "#e1876a", "#e69478", "#eba086", "#f0ad94",
            "#f4bba2", "#f8c7b1", "#fbd5c0", "#d3d3d3",
            "#a7e4fd", "#a0d8fa", "#99ccf7", "#92c0f4", "#8ab3f1", "#83a8ee",
            "#7a9ceb", "#7291e7", "#6886e4", "#5f7be1", "#546fdd", "#4865da",
            "#3a5ad6", "#274ed3", "#0044cf",
        ]
    smooth_cmap = LinearSegmentedColormap.from_list('smooth', palette)

    # ---- Collect data for each column ----
    combined_data = []
    threshold_value_for_plot = None

    for result_set in all_results:
        rlt = result_set['results']['by_leadtime'].copy()

        if (score_type in _BOUNDED_ERROR_SCORES or score_type in _ERROR_SCORES
                or score_type in _HIGHER_IS_BETTER_SCORES):
            rlt['pct_diff'] = ((rlt[fc2_col] - rlt[fc1_col]) / rlt[fc1_col].replace(0, np.nan)) * 100
        else:
            rlt['pct_diff'] = ((rlt[fc2_col] - rlt[fc1_col]) / (1 - rlt[fc1_col]).replace(0, np.nan)) * 100

        pivot_diff = rlt.pivot_table(values='pct_diff', index=pivot_index, aggfunc='first').reindex(row_keys)

        if sig_col in rlt.columns:
            pivot_sig = rlt.pivot_table(
                values=sig_col, index=pivot_index, aggfunc='first').reindex(row_keys).fillna(False)
        else:
            pivot_sig = pd.Series([False] * len(row_keys), index=row_keys)

        if 'threshold' in rlt.columns and threshold_value_for_plot is None:
            threshold_value_for_plot = rlt['threshold'].iloc[0]

        combined_data.append({
            'pct_diff': pivot_diff.iloc[:, 0] if isinstance(pivot_diff, pd.DataFrame) and len(pivot_diff.columns) > 0 else pivot_diff if not isinstance(pivot_diff, pd.DataFrame) else pd.Series(np.nan, index=row_keys),
            'significance': pivot_sig.iloc[:, 0] if isinstance(pivot_sig, pd.DataFrame) and len(pivot_sig.columns) > 0 else pivot_sig if not isinstance(pivot_sig, pd.DataFrame) else pd.Series(np.nan, index=row_keys),
        })

    # ---- Build matrix ----
    combined_matrix = np.column_stack([cd['pct_diff'].values for cd in combined_data])
    color_matrix = np.clip(combined_matrix, -color_limit_pct, color_limit_pct) / 100.0

    # ---- Figure layout ----
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_bottom = bottom_margin_in / fig_height
    ax_height = (box_size * num_rows) / fig_height
    ax = fig.add_axes([0.10, ax_bottom, 0.78, ax_height])

    im = ax.imshow(color_matrix, cmap=smooth_cmap,
                    vmin=-color_limit_raw, vmax=color_limit_raw,
                    aspect='equal', interpolation='nearest')

    cbar_ax = fig.add_axes([0.90, ax_bottom, 0.02, ax_height])
    cbar = plt.colorbar(im, cax=cbar_ax, extend='both')
    cbar.set_label('Percentage Difference (%)', fontsize=title_fontsize, weight='bold')
    cbar.ax.tick_params(labelsize=label_fontsize)
    tick_pcts = np.linspace(-color_limit_pct, color_limit_pct, 9)
    cbar.set_ticks([t / 100.0 for t in tick_pcts])
    cbar.set_ticklabels([f'{int(t)}' if t == int(t) else f'{t:.1f}' for t in tick_pcts])

    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Annotations: percentage difference only ----
    for col_idx in range(num_cols):
        for row_idx in range(num_rows):
            pct_diff = combined_data[col_idx]['pct_diff'].iloc[row_idx]
            is_sig = combined_data[col_idx]['significance'].iloc[row_idx]

            if not np.isnan(pct_diff):
                ax.text(col_idx, row_idx, f'{pct_diff:.1f}',
                        ha='center', va='center',
                        fontsize=main_fontsize, fontweight='bold', color='black')

            if is_sig:
                rect = plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                      fill=False, edgecolor='black',
                                      lw=sig_linewidth, zorder=10)
                ax.add_patch(rect)

    # ---- Axis ticks and labels ----
    if has_forecast_day:
        day_labels = [f"Day {int(rk)}" for rk in row_keys]
    else:
        day_labels = [f"{int(rk)}h" for rk in row_keys]

    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(day_labels, rotation=0, fontsize=label_fontsize)
    ax.set_ylabel("Forecast Period", fontsize=title_fontsize, weight='bold')

    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(orog_labels, fontsize=title_fontsize, weight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', length=0, pad=3)
    ax.tick_params(axis='y', length=0, pad=5)

    for col_idx in range(num_cols):
        ax.text(col_idx, num_rows - 0.45, season_labels[col_idx],
                ha='center', va='top', fontsize=label_fontsize, weight='bold',
                transform=ax.transData, clip_on=False)

    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)

    from matplotlib.patches import Rectangle as MplRect
    frame = MplRect((-0.5, -0.5), num_cols, num_rows,
                     fill=False, edgecolor='black', linewidth=2, zorder=1001)
    ax.add_patch(frame)

    for i in range(1, num_cols):
        if orog_labels[i] != orog_labels[i - 1]:
            ax.axvline(x=i - 0.5, ymin=0, ymax=1, color='white', linewidth=2, zorder=1000)

    # ---- Title ----
    var_display_map = {'2t': '2m Temperature', '10ff': '10m Wind Speed', 'tp24': '24h Precipitation'}
    var_display = var_display_map.get(variable, variable)
    season_title = f" - {season}" if season and season.lower() != 'null' else ""

    threshold_info = ""
    threshold_method = config.get('threshold', {}).get('method', 'fixed') if config else 'fixed'
    if threshold_method == 'local_obs_climatology':
        percentile = config.get('threshold', {}).get('local_obs_climatology', {}).get('percentile') if config else None
        if percentile is not None:
            threshold_info = f" - p{percentile} obs climatology (per station)"
        else:
            threshold_info = " - per-station obs climatology"
    elif threshold_method in ('dataset_climatology', 'station_climatology'):
        percentile = config.get('threshold', {}).get(threshold_method, {}).get('percentile') if config else None
        if percentile is not None:
            ordinal = {1: 'st', 2: 'nd', 3: 'rd'}.get(percentile, 'th')
            qualifier = '(per station)' if threshold_method == 'station_climatology' else '(pooled obs)'
            threshold_info = f" - {percentile}{ordinal} percentile {qualifier}"
        else:
            threshold_info = " - percentile threshold"
    else:
        thr_val = threshold_value_for_plot if threshold_value_for_plot is not None else threshold_value
        if thr_val is not None and not (isinstance(thr_val, float) and np.isnan(thr_val)):
            threshold_info = f" - thr: {thr_val:.1f}{unit}"

    fc1_disp = model_names.get('fc1_display', model_names['fc1_name'])
    fc2_disp = model_names.get('fc2_display', model_names['fc2_name'])
    # Score kept as its short code (e.g. "twMAE"), not spelled out.
    title = (f"{score_type} - {var_display}{season_title}{threshold_info} (as %)\n"
             f"{fc1_disp} vs {fc2_disp}")
    # Anchor the title just inside the figure's top margin so it sits close to
    # the column headers instead of floating in blank space.
    title_y = 1.0 - (0.08 / fig_height)
    plt.suptitle(title, fontsize=suptitle_fontsize, fontweight='bold', y=title_y)

    # ---- Save (never overwrites the standard plot.py output) ----
    season_suffix = f"_{season}" if season else "_all_conditions"
    output_file = output_dir / (
        f"heatmap_smooth_{score_type}_{variable}_"
        f"{model_names['fc1_name']}_vs_{model_names['fc2_name']}{season_suffix}_paper.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_file.name}")
