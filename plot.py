"""
STEP 9: PLOT RESULTS
====================
Create heatmaps comparing two forecast models
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd


def _save_figure(output_dir, filename_stem, config=None):
    """Save the current matplotlib figure, honoring plot.dpi and plot.format.

    Reads `plot.dpi` (default 300) and `plot.format` (default 'png') from the
    config so users can control output resolution and file type (e.g. 'pdf',
    'svg'). Returns the written file's name (with extension).
    """
    plot_cfg = (config or {}).get('plot', {})
    dpi = plot_cfg.get('dpi', 300)
    fmt = str(plot_cfg.get('format', 'png')).lower().lstrip('.')
    output_file = output_dir / f"{filename_stem}.{fmt}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    return output_file.name


# All known score names in both deterministic and ensemble modes
_DETERMINISTIC_SCORES = ['ETS', 'PSS', 'twMAE', 'twMAE_hits', 'twMAE_misses', 'twMAE_FA',
                         'twMAE_hit_mae', 'twMAE_miss_severity', 'twMAE_fa_severity',
                         'twRMSE', 'bias', 'mae', 'rmse', 'correlation']
# Per-level tw_quantile_score keys: tw_quantile_score_q001/q005/q010 (below)
# and tw_quantile_score_q090/q095/q098/q099 (above). The mean summary key is
# tw_quantile_score. All are registered so _detect_plottable_scores finds them.
_TW_QS_LEVEL_SCORES = [
    'tw_quantile_score_q001', 'tw_quantile_score_q005', 'tw_quantile_score_q010',
    'tw_quantile_score_q090', 'tw_quantile_score_q095', 'tw_quantile_score_q098',
    'tw_quantile_score_q099',
]
_ENSEMBLE_SCORES = ['CRPS', 'twCRPS', 'Brier', 'BSS', 'tw_quantile_score', 'diagonal_score',
                    'ens_mean_bias', 'ens_mean_mae', 'ens_mean_rmse', 'ens_spread',
                    'extreme_spread_skill_ratio'] + _TW_QS_LEVEL_SCORES
_ALL_KNOWN_SCORES = _DETERMINISTIC_SCORES + _ENSEMBLE_SCORES

# Scores where lower is better (error metrics) - negative % diff = blue = better
_ERROR_SCORES = {'twMAE', 'twMAE_hits', 'twMAE_misses', 'twMAE_FA',
                 'twMAE_hit_mae', 'twMAE_miss_severity', 'twMAE_fa_severity',
                 'twRMSE', 'mae', 'rmse', 'bias', 'CRPS', 'twCRPS',
                 'tw_quantile_score', 'diagonal_score',
                 'ens_mean_bias', 'ens_mean_mae', 'ens_mean_rmse'} | set(_TW_QS_LEVEL_SCORES)

# Scores where HIGHER is better but using (fc2-fc1)/fc1 formula (not skill-score denominator).
# positive % diff = fc2 better = blue. ens_spread > 1.0 K/m·s so (1-fc1) would flip sign.
# extreme_spread_skill_ratio: R=1 is perfect; typical NWP is under-dispersive (R<1) so higher→better.
_HIGHER_IS_BETTER_SCORES = {'ens_spread', 'extreme_spread_skill_ratio'}

# Bounded error scores [0,1] where lower is better - use (fc1-fc2)/fc1 normalization
# (like skill scores but with 0 as perfect instead of 1)
_BOUNDED_ERROR_SCORES = {'Brier'}

# Human-readable display names for heatmap titles
_SCORE_DISPLAY_NAMES = {
    'twMAE_hits':              'twMAE — Hit contribution',
    'twMAE_misses':            'twMAE — Miss contribution',
    'twMAE_FA':                'twMAE — False Alarm contribution',
    'tw_quantile_score':       'twQS (mean, extreme tail)',
    'tw_quantile_score_q001':  'twQS \u03b1=0.01 (cold extreme)',
    'tw_quantile_score_q005':  'twQS \u03b1=0.05 (cold extreme)',
    'tw_quantile_score_q010':  'twQS \u03b1=0.10 (cold extreme)',
    'tw_quantile_score_q090':  'twQS \u03b1=0.90 (warm extreme)',
    'tw_quantile_score_q095':  'twQS \u03b1=0.95 (warm extreme)',
    'tw_quantile_score_q098':  'twQS \u03b1=0.98 (warm extreme)',
    'tw_quantile_score_q099':  'twQS \u03b1=0.99 (warm extreme)',
}

# Per-score colorbar limits (±%) for heatmaps.
# Ensemble scores tend to show larger relative differences than deterministic ones.
_SCORE_COLOR_LIMITS = {
    'twMAE':                  50,
    'twMAE_hits':             50,
    'twMAE_misses':           50,
    'twMAE_FA':               50,
    'Brier':                  20,
    'ens_spread':             20,
    'extreme_spread_skill_ratio':  50,
    'tw_quantile_score':      20,
    'tw_quantile_score_q001': 20,
    'tw_quantile_score_q005': 20,
    'tw_quantile_score_q010': 20,
    'tw_quantile_score_q090': 20,
    'tw_quantile_score_q095': 20,
    'tw_quantile_score_q098': 20,
    'tw_quantile_score_q099': 20,
    'twCRPS':                 20,
    'CRPS':                   30,
    'diagonal_score':         30,
}
_DEFAULT_COLOR_LIMIT = 20   # % for error/bounded-error scores not listed above
_DEFAULT_SKILL_LIMIT = 20   # % for skill scores (ETS, BSS, PSS, …)


def _get_color_limit(score_type):
    """Return the symmetric ±limit (%) for a given score's heatmap colorbar."""
    if (score_type in _ERROR_SCORES or score_type in _BOUNDED_ERROR_SCORES
            or score_type in _HIGHER_IS_BETTER_SCORES):
        return _SCORE_COLOR_LIMITS.get(score_type, _DEFAULT_COLOR_LIMIT)
    return _DEFAULT_SKILL_LIMIT


def _make_bounds(limit):
    """Generate 12 proportionally-spaced boundary values within ±limit."""
    fracs = [1.0, 0.75, 0.50, 0.25, 0.10, 0.025]
    return sorted([-f * limit for f in fracs]) + sorted([f * limit for f in fracs])


def _detect_plottable_scores(by_leadtime_df):
    """Detect which scores are available in a by_leadtime DataFrame."""
    available = []
    for score_name in _ALL_KNOWN_SCORES:
        if f'{score_name}_fc1' in by_leadtime_df.columns:
            available.append(score_name)
    # Also pick up any tw_quantile_score_q* keys not pre-listed (forward-compatible)
    for col in by_leadtime_df.columns:
        if col.endswith('_fc1') and col.startswith('tw_quantile_score_q'):
            name = col[:-4]
            if name not in available:
                available.append(name)
    return available


def create_heatmap(results_by_leadtime, variable, threshold_value, output_dir, model_names, score_type='twMAE', orog_type=None, config=None):
    """
    Create heatmap showing score differences between two models across lead times
    Similar to the old analysis script style
    """
    if results_by_leadtime is None or results_by_leadtime.empty:
        print("  ⚠ No lead time results to plot")
        return
    
    # Units for display
    units = {'2t': '°C', '10ff': 'm/s', 'tp24': 'mm'}
    unit = units.get(variable, '')
    
    # Orography suffix for filename and title
    orog_suffix = f"_{orog_type}" if orog_type else ""
    orog_title = f" ({orog_type.upper()} terrain)" if orog_type else ""
    
    # Get the score columns for the selected score_type
    fc1_col = f'{score_type}_fc1'
    fc2_col = f'{score_type}_fc2'
    diff_col = f'{score_type}_diff'
    sig_col = f'{score_type}_is_significant'
    
    if fc1_col not in results_by_leadtime.columns:
        print(f"  ⚠ Score {score_type} not found in results")
        return
    
    # Create pivot table with lead_time as rows
    # Note: significance is only computed for overall scores, not per lead time
    cols_to_use = ['lead_time', fc1_col, fc2_col, diff_col]
    if sig_col in results_by_leadtime.columns:
        cols_to_use.append(sig_col)
        has_significance = True
    else:
        has_significance = False
    
    df = results_by_leadtime[cols_to_use].copy()
    
    # Convert lead_time to forecast day
    df['day'] = df['lead_time'] // 24
    
    # Filter to only forecast_days if available in config (global variable)
    try:
        from inspect import currentframe, getouterframes
        outer_frames = getouterframes(currentframe())
        config = None
        for f in outer_frames:
            if 'config' in f.frame.f_locals:
                config = f.frame.f_locals['config']
                break
        if config and 'forecast_days' in config:
            forecast_days = set(config['forecast_days'])
            df = df[df['day'].isin(forecast_days)]
    except Exception:
        pass
    
    # Calculate percentage difference based on score type
    if (score_type in _BOUNDED_ERROR_SCORES or score_type in _ERROR_SCORES
            or score_type in _HIGHER_IS_BETTER_SCORES):
        # Error metrics and spread: use relative difference (fc2-fc1)/fc1
        df['pct_diff'] = ((df[fc2_col] - df[fc1_col]) / df[fc1_col].replace(0, np.nan)) * 100
    else:
        # Skill scores (ETS, PSS etc.): (score2 - score1) / (1 - score1) * 100
        df['pct_diff'] = ((df[fc2_col] - df[fc1_col]) / (1 - df[fc1_col]).replace(0, np.nan)) * 100
    
    # Create matrix for plotting (days x 1 column for now)
    pivot_diff = df.pivot_table(index='lead_time', values='pct_diff', aggfunc='mean')
    pivot_fc2 = df.pivot_table(index='lead_time', values=fc2_col, aggfunc='mean')
    
    if has_significance:
        pivot_sig = df.pivot_table(index='lead_time', values=sig_col, aggfunc='first')
    else:
        # No significance data available for per-leadtime analysis
        pivot_sig = pd.DataFrame(False, index=pivot_diff.index, columns=[0])
    
    # Calculate square box size for professional look
    box_size = 2.5  # inches per box
    num_rows = len(pivot_diff)
    fig_width = box_size * 1
    fig_height = box_size * num_rows
    
    # Create figure with square boxes
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    bounds = _make_bounds(_get_color_limit(score_type))

    if score_type in _ERROR_SCORES or score_type in _BOUNDED_ERROR_SCORES:
        # Error metrics: negative=blue (better), positive=red (worse)
        colors = ['#000099', '#0000cc', '#0000ff', '#4d4dff', '#8080ff', '#ffffff', '#ff8080', '#ff4d4d', '#ff0000', '#cc0000', '#990000']
    else:
        # Skill scores: negative=red (worse), positive=blue (better)
        colors = ['#990000', '#cc0000', '#ff0000', '#ff4d4d', '#ff8080', '#ffffff', '#8080ff', '#4d4dff', '#0000ff', '#0000cc', '#000099']
    
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Calculate font size proportional to box size
    main_fontsize = int(box_size * 10)
    small_fontsize = int(box_size * 6)
    tiny_fontsize = int(box_size * 5)
    
    # Create annotations with percentage symbol and dynamic text color
    annot_matrix = []
    text_colors = []
    for row in pivot_diff.values:
        annot_row = []
        color_row = []
        for val in row:
            if np.isnan(val):
                annot_row.append("")
                color_row.append('black')
            else:
                annot_row.append(f"{val:.1f}")
                # Use white text for very dark colors (dark blues or dark reds)
                if val <= -10 or val >= 10:
                    color_row.append('white')
                else:
                    color_row.append('black')
        annot_matrix.append(annot_row)
        text_colors.append(color_row)
    
    # Create heatmap
    sns.heatmap(
        pivot_diff,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        norm=norm,
        cbar=False,
        annot_kws={'fontsize': main_fontsize, 'weight': 'bold'},
        linewidths=0.5,
        linecolor='lightgray',
        square=True,
        ax=ax
    )
    
    # Update text colors for dark backgrounds
    for i, row_colors in enumerate(text_colors):
        for j, color in enumerate(row_colors):
            if color == 'white':
                ax.texts[i * len(row_colors) + j].set_color('white')
    
    # Add fc2 scores in bottom-right corner
    for i, lead_time in enumerate(pivot_diff.index):
        fc2_val = pivot_fc2.loc[lead_time].values[0]
        if not np.isnan(fc2_val):
            ax.text(0.95, i + 0.95, f"{fc2_val:.2f}", 
                   ha='right', va='bottom', fontsize=small_fontsize, color='slategray', weight='bold')
    
    # Add threshold value in top-right corner
    for i, lead_time in enumerate(pivot_diff.index):
        ax.text(0.95, i + 0.05, f"{threshold_value:.1f}", 
               ha='right', va='top', fontsize=tiny_fontsize, color='darkblue', weight='bold')
    
    # Highlight significant cells
    sig_linewidth = box_size * 2.5
    for i, lead_time in enumerate(pivot_diff.index):
        if pivot_sig.loc[lead_time].values[0]:
            rect = plt.Rectangle((0, i), 1, 1, fill=False, edgecolor='black', lw=sig_linewidth)
            ax.add_patch(rect)
    
    # Set labels
    # Check if we have forecast_day information for better labels
    if 'forecast_day' in results_by_leadtime.columns:
        day_labels = []
        for lt in pivot_diff.index:
            day = int(results_by_leadtime[results_by_leadtime['lead_time'] == lt]['forecast_day'].iloc[0])
            day_labels.append(f"Day {day}")
    else:
        day_labels = [f"{int(lt)}h" for lt in pivot_diff.index]
    
    ax.set_yticklabels(day_labels, rotation=0, fontsize=9)
    ax.set_xticklabels([f"{model_names['fc1_name']} vs {model_names['fc2_name']}"], fontsize=10)
    ax.set_ylabel("Forecast Period", fontsize=11)
    
    # Title
    var_display_map = {'2t': '2m Temperature', '10ff': '10m Wind Speed', 'tp24': '24h Precipitation'}
    var_display = var_display_map.get(variable, variable)
    orog_title = f" ({orog_type.upper()} terrain)" if orog_type else ""
    score_display = _SCORE_DISPLAY_NAMES.get(score_type, score_type.upper())
    title = f"{score_display} - {var_display} - Threshold: {threshold_value:.1f}{unit} (as %){orog_title}"
    plt.title(title, fontsize=12, pad=15)
    
    plt.tight_layout()
    
    # Save
    orog_suffix = f"_{orog_type}" if orog_type else ""
    stem = f"heatmap_{score_type}_{variable}_{model_names['fc1_name']}_vs_{model_names['fc2_name']}{orog_suffix}"
    saved_name = _save_figure(output_dir, stem, config)
    plt.close()
    
    print(f"  ✓ Saved heatmap: {saved_name}")


def plot_summary(data, results_by_leadtime, variable, threshold, output_dir, model_names, orog_type=None, config=None):
    """Create comprehensive summary plot comparing both models"""
    units = {'2t': '°C', '10ff': 'm/s', 'tp24': 'mm'}
    unit = units.get(variable, '')
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Scatter - Model 1
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(data['obs_value'], data['fc1_value'], alpha=0.2, s=5, c='blue', label=model_names['fc1_name'])
    lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
            max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lims, lims, 'r--', alpha=0.75, label='1:1')
    ax1.axvline(threshold, color='green', linestyle='--', alpha=0.7)
    ax1.axhline(threshold, color='green', linestyle='--', alpha=0.7)
    ax1.set_xlabel(f'Observed ({unit})')
    ax1.set_ylabel(f'Forecast ({unit})')
    ax1.set_title(f'Model 1: {model_names["fc1_name"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter - Model 2
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(data['obs_value'], data['fc2_value'], alpha=0.2, s=5, c='red', label=model_names['fc2_name'])
    ax2.plot(lims, lims, 'r--', alpha=0.75, label='1:1')
    ax2.axvline(threshold, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(threshold, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel(f'Observed ({unit})')
    ax2.set_ylabel(f'Forecast ({unit})')
    ax2.set_title(f'Model 2: {model_names["fc2_name"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bias histograms
    ax3 = plt.subplot(3, 3, 3)
    bias1 = data['fc1_value'] - data['obs_value']
    bias2 = data['fc2_value'] - data['obs_value']
    ax3.hist([bias1, bias2], bins=50, alpha=0.5, label=[model_names['fc1_name'], model_names['fc2_name']], edgecolor='black')
    ax3.axvline(np.mean(bias1), color='blue', linestyle='--', linewidth=2, label=f'{model_names["fc1_name"]} mean')
    ax3.axvline(np.mean(bias2), color='red', linestyle='--', linewidth=2, label=f'{model_names["fc2_name"]} mean')
    ax3.set_xlabel(f'Bias ({unit})')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Bias Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-9. Score evolution plots
    score_plots = [
        ('ETS', 'Equitable Threat Score', 4),
        ('PSS', 'Peirce Skill Score', 5),
        ('twMAE', f'Threshold-weighted MAE ({unit})', 6),
        ('mae', f'MAE ({unit})', 7),
        ('rmse', f'RMSE ({unit})', 8),
        ('correlation', 'Correlation', 9)
    ]
    
    for score_name, score_title, subplot_idx in score_plots:
        fc1_col = f'{score_name}_fc1'
        fc2_col = f'{score_name}_fc2'
        # Filter to only forecast_days if available in config (global variable)
        filtered = results_by_leadtime
        try:
            from inspect import currentframe, getouterframes
            outer_frames = getouterframes(currentframe())
            config = None
            for f in outer_frames:
                if 'config' in f.frame.f_locals:
                    config = f.frame.f_locals['config']
                    break
            if config and 'forecast_days' in config:
                forecast_days = set(config['forecast_days'])
                if 'forecast_day' in filtered.columns:
                    filtered = filtered[filtered['forecast_day'].isin(forecast_days)]
        except Exception:
            pass
        if fc1_col in filtered.columns and fc2_col in filtered.columns:
            ax = plt.subplot(3, 3, subplot_idx)
            ax.plot(filtered['lead_time'], filtered[fc1_col], 
                   'o-', linewidth=2, label=model_names['fc1_name'], color='blue')
            ax.plot(filtered['lead_time'], filtered[fc2_col], 
                   's-', linewidth=2, label=model_names['fc2_name'], color='red')
            ax.set_xlabel('Representative Lead Time (h)')
            ax.set_ylabel(score_name.upper())
            ax.set_title(score_title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            if score_name in ['ETS', 'PSS', 'correlation']:
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    orog_title = f" ({orog_type.upper()} terrain)" if orog_type else ""
    fig.suptitle(f'Verification Summary - {variable} - Comparing Models{orog_title}', fontsize=16)
    plt.tight_layout()
    
    orog_suffix = f"_{orog_type}" if orog_type else ""
    stem = f"summary_{variable}_{model_names['fc1_name']}_vs_{model_names['fc2_name']}{orog_suffix}"
    saved_name = _save_figure(output_dir, stem, config)
    plt.close()
    
    print(f"  ✓ Saved: {saved_name}")


def create_multicolumn_heatmap(all_results, variable, threshold_value, output_dir, model_names, score_type='twMAE', season=None, config=None):
    """
    Create multi-column heatmap for different orography types side by side
    """
    if not all_results or not all(r['results'].get('by_leadtime') is not None for r in all_results):
        print(f"  ⚠ Cannot create multi-column heatmap for {score_type}")
        return
    
    # Check if all results have data and the required columns
    for result_set in all_results:
        results_by_leadtime = result_set['results']['by_leadtime']
        if results_by_leadtime.empty:
            print(f"  ⚠ Skipping {score_type} heatmap - no data available")
            return
        
        # Check if required columns exist
        diff_col = f'{score_type}_diff'
        if diff_col not in results_by_leadtime.columns:
            print(f"  ⚠ Skipping {score_type} heatmap - missing {diff_col} column")
            return
    
    # Units for display
    units = {'2t': '°C', '10ff': 'm/s', 'tp24': 'mm'}
    unit = units.get(variable, '')
    
    # Get columns
    fc1_col = f'{score_type}_fc1'
    fc2_col = f'{score_type}_fc2'
    diff_col = f'{score_type}_diff'
    sig_col = f'{score_type}_is_significant'
    
    # Prepare data for each orography type
    num_cols = len(all_results)
    # Create separate labels for orography (top) and season (bottom)
    orog_labels = []
    season_labels = []
    for r in all_results:
        season_labels.append(r.get('season', '') or '')
        orog_labels.append(r['orog_type'].upper() if r['orog_type'] else 'ALL')
    
    # Get lead times from first result (should be same for all)
    lead_times = sorted(all_results[0]['results']['by_leadtime']['lead_time'].unique())
    
    # Check if we have forecast_day information
    has_forecast_day = 'forecast_day' in all_results[0]['results']['by_leadtime'].columns

    # Filter rows by forecast_days if specified in config (plot.forecast_days or top-level)
    _plot_days = (config or {}).get('plot', {}).get('forecast_days') or (config or {}).get('forecast_days')
    if _plot_days and has_forecast_day:
        _ref = all_results[0]['results']['by_leadtime']
        _day_to_lt = _ref.drop_duplicates('forecast_day').set_index('forecast_day')['lead_time']
        lead_times = sorted([_day_to_lt[d] for d in _plot_days if d in _day_to_lt.index])

    # Calculate box size
    box_size = 0.6  # inches per box (very compact for better proportion)
    num_rows = len(lead_times)
    fig_width = box_size * num_cols + 1.2  # Add space for colorbar
    fig_height = box_size * num_rows
    
    # Create single figure with one axis for all data combined
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.08, 0.18, 0.80, 0.68])  # [left, bottom, width, height] - more width
    
    # Calculate font sizes proportional to box size
    main_fontsize = int(box_size * 12)  # Main percentage value
    small_fontsize = int(box_size * 8)   # Model score
    tiny_fontsize = int(box_size * 7)    # Threshold
    label_fontsize = int(box_size * 10)  # Axis labels
    title_fontsize = int(box_size * 11)  # Titles
    sig_linewidth = box_size * 3.5
    
    # Collect all data into single arrays
    all_pct_diff = []
    all_fc2_scores = []
    all_thresholds = []
    all_significance = []
    
    # Collect data from all results
    combined_data = []
    threshold_value_for_plot = None
    
    for idx, result_set in enumerate(all_results):
        results_by_leadtime = result_set['results']['by_leadtime']
        # Filter to only forecast_days if available in config (plot.forecast_days or top-level)
        _plot_days = (config or {}).get('plot', {}).get('forecast_days') or (config or {}).get('forecast_days')
        if _plot_days and 'forecast_day' in results_by_leadtime.columns:
            results_by_leadtime = results_by_leadtime[
                results_by_leadtime['forecast_day'].isin(set(_plot_days))]
        
        # Calculate percentage difference based on score type
        df = results_by_leadtime.copy()
        if (score_type in _BOUNDED_ERROR_SCORES or score_type in _ERROR_SCORES
                or score_type in _HIGHER_IS_BETTER_SCORES):
            df['pct_diff'] = ((df[fc2_col] - df[fc1_col]) / df[fc1_col].replace(0, np.nan)) * 100
        else:
            df['pct_diff'] = ((df[fc2_col] - df[fc1_col]) / (1 - df[fc1_col]).replace(0, np.nan)) * 100
        
        # Prepare pivot tables
        pivot_diff = df.pivot_table(values='pct_diff', index='lead_time', aggfunc='first').reindex(lead_times)
        pivot_fc2 = df.pivot_table(values=fc2_col, index='lead_time', aggfunc='first').reindex(lead_times)
        
        if sig_col in results_by_leadtime.columns:
            pivot_sig = results_by_leadtime.pivot_table(values=sig_col, index='lead_time', aggfunc='first').reindex(lead_times).fillna(False)
        else:
            pivot_sig = pd.Series([False] * len(lead_times), index=lead_times)
        
        if 'threshold' in df.columns:
            pivot_threshold = df.pivot_table(values='threshold', index='lead_time', aggfunc='first').reindex(lead_times)
            if threshold_value_for_plot is None:
                threshold_value_for_plot = pivot_threshold.iloc[0, 0] if isinstance(pivot_threshold, pd.DataFrame) else pivot_threshold.iloc[0]
        else:
            pivot_threshold = pd.Series([threshold_value] * len(lead_times), index=lead_times)
            threshold_value_for_plot = threshold_value
        
        # Collect column data
        combined_data.append({
            'pct_diff': pivot_diff.iloc[:, 0] if isinstance(pivot_diff, pd.DataFrame) else pivot_diff,
            'fc2_scores': pivot_fc2.iloc[:, 0] if isinstance(pivot_fc2, pd.DataFrame) else pivot_fc2,
            'thresholds': pivot_threshold.iloc[:, 0] if isinstance(pivot_threshold, pd.DataFrame) else pivot_threshold,
            'significance': pivot_sig.iloc[:, 0] if isinstance(pivot_sig, pd.DataFrame) else pivot_sig
        })
    
    # Create combined matrix for plotting (rows x cols)
    combined_matrix = np.column_stack([cd['pct_diff'].values for cd in combined_data])
    
    # Define color map
    climit = _get_color_limit(score_type)
    bounds = _make_bounds(climit)
    if score_type in _ERROR_SCORES or score_type in _BOUNDED_ERROR_SCORES:
        # Error metrics (lower is better): negative=blue (fc2 better), positive=red (fc2 worse)
        colors = ['#000099', '#0000cc', '#0000ff', '#4d4dff', '#8080ff', '#ffffff', '#ff8080', '#ff4d4d', '#ff0000', '#cc0000', '#990000']
    else:
        # For skill scores (ETS, PSS): negative=red (fc2 worse), positive=blue (fc2 better)
        colors = ['#990000', '#cc0000', '#ff0000', '#ff4d4d', '#ff8080', '#ffffff', '#8080ff', '#4d4dff', '#0000ff', '#0000cc', '#000099']
    
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Plot single heatmap with all data
    im = ax.imshow(combined_matrix, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
    
    # Add colorbar right next to heatmap
    cbar_ax = fig.add_axes([0.86, 0.18, 0.02, 0.68])  # [left, bottom, width, height] - close and narrow
    cbar = plt.colorbar(im, cax=cbar_ax, extend='both')  # extend shows arrows for out-of-range values
    cbar.set_label('Percentage Difference (%)', fontsize=title_fontsize, weight='bold')
    cbar.ax.tick_params(labelsize=label_fontsize)
    # Set explicit ticks to show min and max values
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([f'{b:.1f}' if abs(b) < 1 else f'{int(b)}' for b in bounds])
    
    # Remove frame and adjust appearance
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add text annotations for all cells
    for col_idx in range(num_cols):
        for row_idx in range(num_rows):
            pct_diff = combined_data[col_idx]['pct_diff'].iloc[row_idx]
            fc2_score = combined_data[col_idx]['fc2_scores'].iloc[row_idx]
            threshold_val = combined_data[col_idx]['thresholds'].iloc[row_idx]
            is_significant = combined_data[col_idx]['significance'].iloc[row_idx]
            
            # Main percentage text
            text_color = 'white' if (pct_diff <= -10 or pct_diff >= 10) else 'black'
            ax.text(col_idx, row_idx, f'{pct_diff:.1f}',
                   ha='center', va='center', fontsize=main_fontsize, fontweight='bold', color=text_color)
            
            # Top right: threshold
            ax.text(col_idx + 0.45, row_idx - 0.35, f'{threshold_val:.0f}',
                   ha='right', va='top', fontsize=tiny_fontsize, style='italic', color='gray')
            
            # Bottom right: model 2 score
            ax.text(col_idx + 0.45, row_idx + 0.35, f'{fc2_score:.2f}',
                   ha='right', va='bottom', fontsize=small_fontsize, color='darkblue')
            
            # Mark significant cells with black border
            if is_significant:
                rect = plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, 
                                    edgecolor='black', lw=sig_linewidth, zorder=10)
                ax.add_patch(rect)
    
    # Set axis labels and ticks
    if has_forecast_day:
        day_labels = []
        first_result_lt = all_results[0]['results']['by_leadtime']
        for lt in lead_times:
            day = int(first_result_lt[first_result_lt['lead_time'] == lt]['forecast_day'].iloc[0])
            day_labels.append(f"Day {day}")
    else:
        day_labels = [f"{int(lt)}h" for lt in lead_times]
    
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(day_labels, rotation=0, fontsize=label_fontsize)
    ax.set_ylabel("Forecast Period", fontsize=title_fontsize, weight='bold')
    
    # Set x-axis labels: orography on top
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(orog_labels, fontsize=title_fontsize, weight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', length=0, pad=3)  # Reduced padding to bring labels closer
    ax.tick_params(axis='y', length=0, pad=5)
    
    # Add season labels below the heatmap
    for col_idx in range(num_cols):
        ax.text(col_idx, num_rows - 0.45, season_labels[col_idx], 
               ha='center', va='top', fontsize=label_fontsize, weight='bold',
               transform=ax.transData, clip_on=False)
    
    # Set axis limits
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)
    
    # Add black frame around the entire heatmap
    from matplotlib.patches import Rectangle
    frame = Rectangle((-0.5, -0.5), num_cols, num_rows, 
                     fill=False, edgecolor='black', linewidth=2, zorder=1001)
    ax.add_patch(frame)
    
    # Draw white vertical separator lines between orography changes - very thin
    for i in range(1, num_cols):
        if orog_labels[i] != orog_labels[i-1]:
            ax.axvline(x=i - 0.5, ymin=0, ymax=1, color='white', linewidth=2, zorder=1000)
    
    # Set colorbar properties if it exists
    # (Already done in loop above)
    
    # Overall title
    var_display_map = {'2t': '2m Temperature', '10ff': '10m Wind Speed', 'tp24': '24h Precipitation'}
    var_display = var_display_map.get(variable, variable)
    # If season is None, it means we're showing all seasons in columns
    season_title = f" - {season}" if season and season.lower() != 'null' else ""
    
    # Add threshold info to title
    units = {'2t': '°C', '10ff': 'm/s', 'tp24': 'mm'}
    unit = units.get(variable, '')
    threshold_info = ""
    threshold_method = config.get('threshold', {}).get('method', 'fixed') if config else 'fixed'
    if threshold_method in ('dataset_climatology', 'station_climatology'):
        percentile = config.get('threshold', {}).get(threshold_method, {}).get('percentile') if config else None
        if percentile is not None:
            ordinal = {1: 'st', 2: 'nd', 3: 'rd'}.get(percentile, 'th')
            threshold_info = f" - {percentile}{ordinal} percentile (per station)"
        else:
            threshold_info = " - percentile threshold"
    else:
        # Fixed threshold: show the actual value used
        thr_val = threshold_value_for_plot if threshold_value_for_plot is not None else threshold_value
        if thr_val is not None and not (isinstance(thr_val, float) and np.isnan(thr_val)):
            threshold_info = f" - thr: {thr_val:.1f}{unit}"

    title = f"{_SCORE_DISPLAY_NAMES.get(score_type, score_type.upper())} - {var_display}{season_title}{threshold_info} (as %)\n{model_names['fc1_name']} vs {model_names['fc2_name']}"
    plt.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=1.05)  # Even higher to clear labels
    
    # Save (constrained_layout handles spacing automatically, no need for tight_layout)
    season_suffix = f"_{season}" if season else "_all_conditions"
    stem = f"heatmap_{score_type}_{variable}_{model_names['fc1_name']}_vs_{model_names['fc2_name']}{season_suffix}"
    saved_name = _save_figure(output_dir, stem, config)
    plt.close()
    
    print(f"    ✓ Saved: {saved_name}")


def create_smooth_multicolumn_heatmap(all_results, variable, threshold_value, output_dir, model_names, score_type='twMAE', season=None, config=None):
    """
    Smooth-style multi-column heatmap with continuous colour gradient.
    Box size is fixed regardless of the number of rows/columns, and font
    sizes scale proportionally with the box size.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if not all_results or not all(r['results'].get('by_leadtime') is not None for r in all_results):
        print(f"  ⚠ Cannot create smooth heatmap for {score_type}")
        return

    for result_set in all_results:
        if result_set['results']['by_leadtime'].empty:
            print(f"  ⚠ Skipping {score_type} smooth heatmap – no data")
            return
        if f'{score_type}_diff' not in result_set['results']['by_leadtime'].columns:
            print(f"  ⚠ Skipping {score_type} smooth heatmap – missing diff column")
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
    # When forecast_day is available, use it as the pivot index instead of lead_time.
    # lead_time can differ by ±1 h between orography types (different mean step per group)
    # whereas forecast_day is always consistently [1, 2, 3, …] for all orog types.
    pivot_index = 'forecast_day' if has_forecast_day else 'lead_time'
    # Take row_keys from the UNION of all results so that a season with fewer days
    # (e.g. DJF with contaminated parquets) doesn't silently drop rows present in
    # other seasons.
    all_pivot_vals = set()
    for r in all_results:
        all_pivot_vals.update(r['results']['by_leadtime'][pivot_index].unique())
    row_keys = sorted(all_pivot_vals)

    # Filter rows by forecast_days if specified in config (plot.forecast_days or top-level)
    _plot_days = (config or {}).get('plot', {}).get('forecast_days') or (config or {}).get('forecast_days')
    if _plot_days:
        _filter_vals = set(_plot_days) if pivot_index == 'forecast_day' else None
        if _filter_vals:
            row_keys = sorted([v for v in row_keys if v in _filter_vals])

    # ---- Fixed box size (same as normal style) ----
    box_size = 0.6  # inches per cell
    num_rows = len(row_keys)
    fig_width = box_size * num_cols + 1.4   # extra for colorbar
    fig_height = box_size * num_rows + 0.6  # extra for labels

    main_fontsize = max(4, int(box_size * 12))
    small_fontsize = max(3, int(box_size * 8))
    tiny_fontsize  = max(3, int(box_size * 7))
    label_fontsize = max(4, int(box_size * 10))
    title_fontsize = max(4, int(box_size * 11))
    sig_linewidth  = box_size * 3.5

    # ---- Colour limits ----
    color_limit_pct = float(_get_color_limit(score_type))
    color_limit_raw = color_limit_pct / 100.0

    # ---- Smooth continuous colour palette ----
    if score_type in _ERROR_SCORES or score_type in _BOUNDED_ERROR_SCORES:
        # error metrics: negative (fc2 better) → blue, positive (fc2 worse) → red
        palette = [
            "#0044cf", "#274ed3", "#3a5ad6", "#4865da", "#546fdd", "#5f7be1",
            "#6886e4", "#7291e7", "#7a9ceb", "#83a8ee", "#8ab3f1", "#92c0f4",
            "#99ccf7", "#a0d8fa", "#a7e4fd", "#d3d3d3",
            "#fbd5c0", "#f8c7b1", "#f4bba2", "#f0ad94", "#eba086", "#e69478",
            "#e1876a", "#db7a5d", "#d56d4f", "#cf6043", "#c85235", "#c14329",
            "#b9341c", "#b2210e", "#aa0000",
        ]
    else:
        # skill scores: negative (fc2 worse) → red, positive (fc2 better) → blue
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
        # Optionally filter to configured forecast_days
        try:
            from inspect import currentframe, getouterframes
            outer_frames = getouterframes(currentframe())
            cfg_local = None
            for f in outer_frames:
                if 'config' in f.frame.f_locals:
                    cfg_local = f.frame.f_locals['config']
                    break
            if cfg_local and 'forecast_days' in cfg_local:
                fd = set(cfg_local['forecast_days'])
                if 'forecast_day' in rlt.columns:
                    rlt = rlt[rlt['forecast_day'].isin(fd)]
        except Exception:
            pass

        # Percentage difference
        if (score_type in _BOUNDED_ERROR_SCORES or score_type in _ERROR_SCORES
                or score_type in _HIGHER_IS_BETTER_SCORES):
            rlt['pct_diff'] = ((rlt[fc2_col] - rlt[fc1_col]) / rlt[fc1_col].replace(0, np.nan)) * 100
        else:
            rlt['pct_diff'] = ((rlt[fc2_col] - rlt[fc1_col]) / (1 - rlt[fc1_col]).replace(0, np.nan)) * 100

        pivot_diff = rlt.pivot_table(values='pct_diff', index=pivot_index, aggfunc='first').reindex(row_keys)
        pivot_fc2  = rlt.pivot_table(values=fc2_col,    index=pivot_index, aggfunc='first').reindex(row_keys)

        if sig_col in rlt.columns:
            pivot_sig = result_set['results']['by_leadtime'].pivot_table(
                values=sig_col, index=pivot_index, aggfunc='first').reindex(row_keys).fillna(False)
        else:
            pivot_sig = pd.Series([False] * len(row_keys), index=row_keys)

        if 'threshold' in rlt.columns:
            pivot_thr = rlt.pivot_table(values='threshold', index=pivot_index, aggfunc='first').reindex(row_keys)
            if threshold_value_for_plot is None:
                v = pivot_thr.iloc[0, 0] if isinstance(pivot_thr, pd.DataFrame) else pivot_thr.iloc[0]
                threshold_value_for_plot = v
        else:
            pivot_thr = pd.Series([threshold_value] * len(row_keys), index=row_keys)
            threshold_value_for_plot = threshold_value

        if 'n_exceedances' in rlt.columns:
            pivot_nexc = rlt.pivot_table(values='n_exceedances', index=pivot_index, aggfunc='sum').reindex(row_keys)
        else:
            pivot_nexc = pd.Series([np.nan] * len(row_keys), index=row_keys)

        combined_data.append({
            'pct_diff':      pivot_diff.iloc[:, 0] if isinstance(pivot_diff, pd.DataFrame) and len(pivot_diff.columns) > 0 else pivot_diff if not isinstance(pivot_diff, pd.DataFrame) else pd.Series(np.nan, index=row_keys),
            'fc2_scores':    pivot_fc2.iloc[:, 0]  if isinstance(pivot_fc2,  pd.DataFrame) and len(pivot_fc2.columns) > 0 else pivot_fc2 if not isinstance(pivot_fc2, pd.DataFrame) else pd.Series(np.nan, index=row_keys),
            'thresholds':    pivot_thr.iloc[:, 0]  if isinstance(pivot_thr,  pd.DataFrame) and len(pivot_thr.columns) > 0 else pivot_thr if not isinstance(pivot_thr, pd.DataFrame) else pd.Series(np.nan, index=row_keys),
            'significance':  pivot_sig.iloc[:, 0]  if isinstance(pivot_sig,  pd.DataFrame) and len(pivot_sig.columns) > 0 else pivot_sig if not isinstance(pivot_sig, pd.DataFrame) else pd.Series(np.nan, index=row_keys),
            'n_exceedances': pivot_nexc.iloc[:, 0] if isinstance(pivot_nexc, pd.DataFrame) and len(pivot_nexc.columns) > 0 else pivot_nexc if not isinstance(pivot_nexc, pd.DataFrame) else pd.Series(np.nan, index=row_keys),
        })

    # ---- Build matrix ----
    combined_matrix = np.column_stack([cd['pct_diff'].values for cd in combined_data])
    # Colour matrix: clipped to limits then normalised to [-raw, +raw]
    color_matrix = np.clip(combined_matrix, -color_limit_pct, color_limit_pct) / 100.0

    # ---- Figure layout ----
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax  = fig.add_axes([0.10, 0.20, 0.78, 0.66])

    im = ax.imshow(color_matrix, cmap=smooth_cmap,
                   vmin=-color_limit_raw, vmax=color_limit_raw,
                   aspect='equal', interpolation='nearest')

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.66])
    cbar = plt.colorbar(im, cax=cbar_ax, extend='both')
    cbar.set_label('Percentage Difference (%)', fontsize=title_fontsize, weight='bold')
    cbar.ax.tick_params(labelsize=label_fontsize)
    # Tick values in percentage space
    tick_pcts = np.linspace(-color_limit_pct, color_limit_pct, 9)
    cbar.set_ticks([t / 100.0 for t in tick_pcts])
    cbar.set_ticklabels([f'{int(t)}' if t == int(t) else f'{t:.1f}' for t in tick_pcts])

    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Annotations ----
    for col_idx in range(num_cols):
        for row_idx in range(num_rows):
            pct_diff   = combined_data[col_idx]['pct_diff'].iloc[row_idx]
            fc2_score  = combined_data[col_idx]['fc2_scores'].iloc[row_idx]
            thr_val    = combined_data[col_idx]['thresholds'].iloc[row_idx]
            n_exc      = combined_data[col_idx]['n_exceedances'].iloc[row_idx]
            is_sig     = combined_data[col_idx]['significance'].iloc[row_idx]

            if not np.isnan(pct_diff):
                # Centre: main percentage value
                ax.text(col_idx, row_idx, f'{pct_diff:.1f}',
                        ha='center', va='center',
                        fontsize=max(3, int(main_fontsize * 0.80)), fontweight='bold', color='black')

            # Top-right: number of exceedances (or threshold if exceedances unavailable)
            if not np.isnan(n_exc):
                ax.text(col_idx + 0.45, row_idx - 0.35,
                        f'{int(n_exc)}',
                        ha='right', va='top', fontsize=tiny_fontsize,
                        style='italic', color='gray')
            elif not np.isnan(thr_val):
                ax.text(col_idx + 0.45, row_idx - 0.35,
                        f'{thr_val:.0f}' if thr_val >= 1 else f'{thr_val:.2f}',
                        ha='right', va='top', fontsize=tiny_fontsize,
                        style='italic', color='gray')

            if not np.isnan(fc2_score):
                # Bottom-right: fc2 score (slightly lower, no bold)
                ax.text(col_idx + 0.45, row_idx + 0.44, f'{fc2_score:.2f}',
                        ha='right', va='bottom', fontsize=small_fontsize,
                        color='darkblue', weight='normal')

            # Significance border
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

    # Season labels below heatmap
    for col_idx in range(num_cols):
        ax.text(col_idx, num_rows - 0.45, season_labels[col_idx],
                ha='center', va='top', fontsize=label_fontsize, weight='bold',
                transform=ax.transData, clip_on=False)

    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)

    # Outer frame
    from matplotlib.patches import Rectangle as MplRect
    frame = MplRect((-0.5, -0.5), num_cols, num_rows,
                    fill=False, edgecolor='black', linewidth=2, zorder=1001)
    ax.add_patch(frame)

    # White separator lines between orography groups
    for i in range(1, num_cols):
        if orog_labels[i] != orog_labels[i - 1]:
            ax.axvline(x=i - 0.5, ymin=0, ymax=1, color='white', linewidth=2, zorder=1000)

    # ---- Title ----
    var_display_map = {'2t': '2m Temperature', '10ff': '10m Wind Speed', 'tp24': '24h Precipitation'}
    var_display = var_display_map.get(variable, variable)
    season_title = f" - {season}" if season and season.lower() != 'null' else ""

    units = {'2t': '°C', '10ff': 'm/s', 'tp24': 'mm'}
    unit = units.get(variable, '')
    threshold_info = ""
    threshold_method = config.get('threshold', {}).get('method', 'fixed') if config else 'fixed'
    if threshold_method == 'local_obs_climatology':
        percentile = config.get('threshold', {}).get('local_obs_climatology', {}).get('percentile') if config else None
        if percentile is not None:
            ordinal = {1: 'st', 2: 'nd', 3: 'rd'}.get(percentile % 10 if percentile % 100 not in (11, 12, 13) else 0, 'th')
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

    title = (f"{_SCORE_DISPLAY_NAMES.get(score_type, score_type.upper())} - {var_display}{season_title}{threshold_info} (as %)\n"
             f"{model_names['fc1_name']} vs {model_names['fc2_name']}")
    plt.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=1.05)

    # ---- Save ----
    season_suffix = f"_{season}" if season else "_all_conditions"
    stem = (
        f"heatmap_smooth_{score_type}_{variable}_"
        f"{model_names['fc1_name']}_vs_{model_names['fc2_name']}{season_suffix}"
    )
    saved_name = _save_figure(output_dir, stem, config)
    plt.close()
    print(f"    ✓ Saved: {saved_name}")


def create_smooth_panel_heatmap(all_results, variable, threshold_value, output_dir, model_names, season=None, config=None):
    """
    2×2 panel of smooth heatmaps:
      (A) twCRPS  (B) TW Spread/Skill Ratio
      (C) twQS q95/q05  (D) twQS q99/q01
    One figure per test case for easy cross-score comparison.
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Resolve event_type from config to pick correct tail labels.
    # Falls back to 'above' (upper tail) when not specified.
    _event_type = 'above'
    if config:
        _thr_cfg = config.get('threshold', {})
        _event_type = _thr_cfg.get('event_type', 'above')

    # Panel C/D keys depend on tail direction:
    #   above → q095 (C) and q099 (D)
    #   below → q005 (C) and q001 (D)
    _mid_key  = 'tw_quantile_score_q095' if _event_type == 'above' else 'tw_quantile_score_q005'
    _extr_key = 'tw_quantile_score_q099' if _event_type == 'above' else 'tw_quantile_score_q001'

    # Fall back to whatever level is present if the preferred one is missing
    available_cols_check = all_results[0]['results']['by_leadtime'].columns
    _all_level_keys = sorted(
        c[:-4] for c in available_cols_check
        if c.startswith('tw_quantile_score_q') and c.endswith('_fc1')
    )
    if f'{_mid_key}_fc1' not in available_cols_check and _all_level_keys:
        # pick the second-most-extreme level available
        _mid_key = _all_level_keys[-2] if len(_all_level_keys) >= 2 else _all_level_keys[0]
    if f'{_extr_key}_fc1' not in available_cols_check and _all_level_keys:
        # pick the most-extreme level available
        _extr_key = _all_level_keys[-1]

    # Panel order: A=twCRPS, B=spread-skill, C=twQS mid-tail, D=twQS extreme-tail
    PANEL_SCORES = ['twCRPS', 'extreme_spread_skill_ratio', _mid_key, _extr_key]
    PANEL_LABELS = ['(A)', '(B)', '(C)', '(D)']
    SCORE_TITLES = {
        'twCRPS':                     'twCRPS',
        'extreme_spread_skill_ratio': 'TW Spread/Skill Ratio',
        _mid_key:                     _SCORE_DISPLAY_NAMES.get(_mid_key, 'twQS mid-tail'),
        _extr_key:                    _SCORE_DISPLAY_NAMES.get(_extr_key, 'twQS extreme-tail'),
    }

    # Only keep scores present in data
    available_cols = all_results[0]['results']['by_leadtime'].columns
    panel_scores = [s for s in PANEL_SCORES
                    if f'{s}_fc1' in available_cols and f'{s}_diff' in available_cols]
    if len(panel_scores) < 2:
        print(f"  ⚠ Not enough scores for panel plot (found: {panel_scores}), skipping")
        return

    # Colour palettes: error = negative→blue (lower is better)
    #                  skill = negative→red (higher is better, incl. spread)
    _err_pal = [
        "#0044cf", "#274ed3", "#3a5ad6", "#4865da", "#546fdd", "#5f7be1",
        "#6886e4", "#7291e7", "#7a9ceb", "#83a8ee", "#8ab3f1", "#92c0f4",
        "#99ccf7", "#a0d8fa", "#a7e4fd", "#d3d3d3",
        "#fbd5c0", "#f8c7b1", "#f4bba2", "#f0ad94", "#eba086", "#e69478",
        "#e1876a", "#db7a5d", "#d56d4f", "#cf6043", "#c85235", "#c14329",
        "#b9341c", "#b2210e", "#aa0000",
    ]
    _skill_pal = list(reversed(_err_pal))  # positive→blue for higher-is-better scores

    # Grid metadata
    lead_times     = sorted(all_results[0]['results']['by_leadtime']['lead_time'].unique())
    num_rows       = len(lead_times)
    num_cols       = len(all_results)
    has_fd         = 'forecast_day' in all_results[0]['results']['by_leadtime'].columns
    orog_labels    = [r['orog_type'].upper() if r['orog_type'] else 'ALL' for r in all_results]
    season_labels  = [r.get('season', '') or '' for r in all_results]

    if has_fd:
        first_rlt  = all_results[0]['results']['by_leadtime']
        day_labels = [f"Day {int(first_rlt[first_rlt['lead_time']==lt]['forecast_day'].iloc[0])}"
                      for lt in lead_times]
    else:
        day_labels = [f"{int(lt)}h" for lt in lead_times]

    # Layout
    n_panels    = len(panel_scores)
    ncols_grid  = min(2, n_panels)
    nrows_grid  = (n_panels + 1) // 2
    box_size    = 0.52          # inches per cell
    panel_w     = box_size * num_cols
    panel_h     = box_size * num_rows
    fig_w       = ncols_grid * (panel_w + 1.3) + 0.5
    fig_h       = nrows_grid * (panel_h + 1.4) + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        nrows_grid, ncols_grid,
        hspace = 1.4 / (panel_h if panel_h > 0 else 1),
        wspace = 1.3 / (panel_w if panel_w > 0 else 1),
        figure = fig,
    )

    threshold_value_for_plot = None

    main_fs  = max(5, int(box_size * 11))
    small_fs = max(4, int(box_size * 7))
    label_fs = max(5, int(box_size * 10))
    sig_lw   = box_size * 3.0

    for panel_idx, score_type in enumerate(panel_scores):
        row_g = panel_idx // ncols_grid
        col_g = panel_idx % ncols_grid
        ax = fig.add_subplot(gs[row_g, col_g])

        fc1_col = f'{score_type}_fc1'
        fc2_col = f'{score_type}_fc2'
        sig_col = f'{score_type}_is_significant'

        # ---- Collect per-column data ----
        combined_data = []
        for result_set in all_results:
            rlt = result_set['results']['by_leadtime'].copy()

            if (score_type in _BOUNDED_ERROR_SCORES or score_type in _ERROR_SCORES
                    or score_type in _HIGHER_IS_BETTER_SCORES):
                rlt['pct_diff'] = ((rlt[fc2_col] - rlt[fc1_col]) / rlt[fc1_col].replace(0, np.nan)) * 100
            else:
                rlt['pct_diff'] = ((rlt[fc2_col] - rlt[fc1_col]) / (1 - rlt[fc1_col]).replace(0, np.nan)) * 100

            pivot_diff = rlt.pivot_table(values='pct_diff', index='lead_time', aggfunc='first').reindex(lead_times)
            pivot_fc2  = rlt.pivot_table(values=fc2_col,    index='lead_time', aggfunc='first').reindex(lead_times)

            if sig_col in rlt.columns:
                pivot_sig = (result_set['results']['by_leadtime']
                             .pivot_table(values=sig_col, index='lead_time', aggfunc='first')
                             .reindex(lead_times).fillna(False))
            else:
                pivot_sig = pd.Series([False] * num_rows, index=lead_times)

            if 'threshold' in rlt.columns:
                pivot_thr = rlt.pivot_table(values='threshold', index='lead_time', aggfunc='first').reindex(lead_times)
                if threshold_value_for_plot is None:
                    v = pivot_thr.iloc[0, 0] if isinstance(pivot_thr, pd.DataFrame) else pivot_thr.iloc[0]
                    threshold_value_for_plot = v
            else:
                pivot_thr = pd.Series([threshold_value] * num_rows, index=lead_times)
                if threshold_value_for_plot is None:
                    threshold_value_for_plot = threshold_value

            combined_data.append({
                'pct_diff':     pivot_diff.iloc[:, 0] if isinstance(pivot_diff, pd.DataFrame) else pivot_diff,
                'fc2_scores':   pivot_fc2.iloc[:, 0]  if isinstance(pivot_fc2,  pd.DataFrame) else pivot_fc2,
                'significance': pivot_sig.iloc[:, 0]  if isinstance(pivot_sig,  pd.DataFrame) else pivot_sig,
            })

        # ---- Plot ----
        combined_matrix = np.column_stack([cd['pct_diff'].values for cd in combined_data])
        climit      = float(_get_color_limit(score_type))
        color_matrix = np.clip(combined_matrix, -climit, climit) / 100.0
        # Higher-is-better scores (spread): positive pct_diff = fc2 better = blue
        _cpal = _skill_pal if score_type in _HIGHER_IS_BETTER_SCORES else _err_pal
        _cmap = LinearSegmentedColormap.from_list('panel_cmap', _cpal)

        im = ax.imshow(color_matrix, cmap=_cmap,
                       vmin=-climit / 100.0, vmax=climit / 100.0,
                       aspect='equal', interpolation='nearest')

        # Colorbar
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="9%", pad=0.06)
        cbar    = plt.colorbar(im, cax=cbar_ax, extend='both')
        tick_pcts = np.linspace(-climit, climit, 5)
        cbar.set_ticks([t / 100.0 for t in tick_pcts])
        cbar.set_ticklabels([f'{int(round(t))}' for t in tick_pcts])
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('%', fontsize=6)

        # Annotations
        for ci in range(num_cols):
            for ri in range(num_rows):
                pct  = combined_data[ci]['pct_diff'].iloc[ri]
                fc2v = combined_data[ci]['fc2_scores'].iloc[ri]
                sig  = combined_data[ci]['significance'].iloc[ri]
                if not np.isnan(pct):
                    ax.text(ci, ri, f'{pct:.1f}',
                            ha='center', va='center',
                            fontsize=main_fs, fontweight='bold', color='black')
                if not np.isnan(fc2v):
                    ax.text(ci + 0.45, ri + 0.44, f'{fc2v:.2f}',
                            ha='right', va='bottom',
                            fontsize=small_fs, color='darkblue', weight='normal')
                if sig:
                    rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                         fill=False, edgecolor='black',
                                         lw=sig_lw, zorder=10)
                    ax.add_patch(rect)

        # Axes styling
        ax.set_frame_on(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_yticks(range(num_rows))
        if col_g == 0:
            ax.set_yticklabels(day_labels, rotation=0, fontsize=label_fs)
            ax.set_ylabel("Forecast Period", fontsize=label_fs, weight='bold')
        else:
            ax.set_yticklabels([])

        ax.set_xticks(range(num_cols))
        ax.set_xticklabels(orog_labels, fontsize=label_fs, weight='bold')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', length=0, pad=2)
        ax.tick_params(axis='y', length=0, pad=3)

        # Season labels below cells
        for ci in range(num_cols):
            ax.text(ci, num_rows - 0.45, season_labels[ci],
                    ha='center', va='top', fontsize=label_fs, weight='bold',
                    transform=ax.transData, clip_on=False)

        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(num_rows - 0.5, -0.5)

        from matplotlib.patches import Rectangle as MplRect
        frame = MplRect((-0.5, -0.5), num_cols, num_rows,
                        fill=False, edgecolor='black', linewidth=1.5, zorder=1001)
        ax.add_patch(frame)

        for i in range(1, num_cols):
            if orog_labels[i] != orog_labels[i - 1]:
                ax.axvline(x=i - 0.5, color='white', linewidth=1.5, zorder=1000)

        score_title = SCORE_TITLES.get(score_type, score_type.upper())
        ax.set_title(f'{PANEL_LABELS[panel_idx]} {score_title}',
                     fontsize=label_fs + 1, fontweight='bold', pad=16)

    # Overall title
    var_display_map = {'2t': '2m Temperature', '10ff': '10m Wind Speed', 'tp24': '24h Precipitation'}
    var_display = var_display_map.get(variable, variable)
    units_map   = {'2t': '°C', '10ff': 'm/s', 'tp24': 'mm'}

    threshold_info = ''
    if config:
        thr_method = config.get('threshold', {}).get('method', 'fixed')
        if thr_method in ('dataset_climatology', 'station_climatology'):
            pct = config.get('threshold', {}).get(thr_method, {}).get('percentile')
            if pct is not None:
                ordinal = {1: 'st', 2: 'nd', 3: 'rd'}.get(pct, 'th')
                qualifier = '(per station)' if thr_method == 'station_climatology' else '(pooled obs)'
                threshold_info = f' — {pct}{ordinal} percentile {qualifier}'
        else:
            u = units_map.get(variable, '')
            tv = threshold_value_for_plot
            if tv is not None and not (isinstance(tv, float) and np.isnan(tv)):
                threshold_info = f' — thr: {tv:.1f}{u}'

    fig.suptitle(
        f'{var_display}{threshold_info}\n{model_names["fc1_name"]} vs {model_names["fc2_name"]}',
        fontsize=10, fontweight='bold', y=1.01,
    )

    season_suffix = f'_{season}' if season else '_all_conditions'
    stem = (
        f'heatmap_smooth_panel_{variable}_'
        f'{model_names["fc1_name"]}_vs_{model_names["fc2_name"]}{season_suffix}'
    )
    saved_name = _save_figure(output_dir, stem, config)
    plt.close()
    print(f'    ✓ Saved: {saved_name}')


def run_step9(config, data_or_results, threshold_value, output_dir, model_names, season=None):
    """
    Execute Step 9: Plot Results
    Can handle either single result (dict) or multiple results (list) for multi-column heatmaps
    """
    print("\n" + "="*80)
    print("STEP 9: PLOT RESULTS")
    print("="*80)
    
    cfg = config.get('plot', {})
    
    if not cfg.get('enabled', True):
        print("\n  Skipped (disabled in config)")
        return
    
    print("\nCreating plots...")
    
    # Determine if we have single or multiple results
    heatmap_style = cfg.get('heatmap_style', 'normal')

    if isinstance(data_or_results, list):
        # Multiple orography types - create multi-column heatmaps
        all_results = data_or_results
        style_label = 'smooth' if heatmap_style == 'smooth' else 'normal'
        print(f"\n  Creating multi-column heatmaps ({style_label}) for {len(all_results)} conditions:")
        
        # Filter out results with empty by_leadtime DataFrames (e.g. seasons with no data)
        non_empty_results = [r for r in all_results
                             if r['results'].get('by_leadtime') is not None
                             and not r['results']['by_leadtime'].empty]
        if not non_empty_results:
            print("  ⚠ All results are empty, skipping heatmaps")
            return

        # Detect available scores from the first non-empty result's by_leadtime columns
        scores_to_plot = _detect_plottable_scores(non_empty_results[0]['results']['by_leadtime'])

        heatmap_fn = create_smooth_multicolumn_heatmap if heatmap_style == 'smooth' else create_multicolumn_heatmap

        for score_type in scores_to_plot:
            # Check if all non-empty results have this score
            if all(f'{score_type}_fc1' in result['results']['by_leadtime'].columns
                   for result in non_empty_results if 'by_leadtime' in result['results']):
                heatmap_fn(
                    non_empty_results,
                    config['variable'],
                    threshold_value,
                    output_dir,
                    model_names,
                    score_type=score_type,
                    season=season,
                    config=config
                )

        # 4-score panel plot (smooth style only)
        if heatmap_style == 'smooth':
            create_smooth_panel_heatmap(
                non_empty_results,
                config['variable'],
                threshold_value,
                output_dir,
                model_names,
                season=season,
                config=config,
            )
    else:
        # Single result - original single-column plotting
        result_set = data_or_results
        data = result_set['data']
        results = result_set['results']
        orog_type = result_set.get('orog_type', None)
        
        # Create heatmaps for key scores
        if 'by_leadtime' in results and results['by_leadtime'] is not None:
            print("\n  Creating heatmaps for key scores:")
            
            # Detect available scores from the data
            scores_to_plot = _detect_plottable_scores(results['by_leadtime'])
            
            for score_type in scores_to_plot:
                fc1_col = f'{score_type}_fc1'
                if fc1_col in results['by_leadtime'].columns:
                    create_heatmap(
                        results['by_leadtime'], 
                        config['variable'], 
                        threshold_value, 
                        output_dir, 
                        model_names,
                        score_type=score_type,
                        orog_type=orog_type,
                        config=config,
                    )
        
        # Summary plot (original multi-panel view)
        if cfg.get('create_summary', True) and 'by_leadtime' in results:
            if data is not None:
                print("\n  Creating summary plot:")
                plot_summary(data, results['by_leadtime'], config['variable'], threshold_value, output_dir, model_names, orog_type, config=config)
            else:
                print("\n  Skipping summary plot (raw data not available when loading from saved results)")
    
    print("\n✓ Step 9 complete")
