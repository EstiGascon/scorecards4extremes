"""
Diagnostic plots for hybrid_ens vs ifs_ens4hybrid ensemble comparison.

Produces a multi-panel figure per variable with:
  1. twCRPS % diff by orography (with CI bands and significance markers)
  2. fCRPS % diff by orography
  3. Brier % diff by orography
  4. quantile_score % diff by orography
  5. ens_mean_bias for both models (2t only — shows growing warm bias)
  6. extreme_spread_skill_ratio for both models
  7. twCRPS vs quantile_score divergence (scatter: one dot per lead/orog)

Run on login node — only reads small CSV files.
Output: ./results/diagnostic_plots/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT_DIR = "./results/diagnostic_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour / style ────────────────────────────────────────────────────────────
OROG_COLORS  = {'low': '#2196F3', 'mid': '#FF9800', 'high': '#9C27B0'}
OROG_MARKERS = {'low': 'o', 'mid': 's', 'high': '^'}
OROG_LABELS  = {'low': 'Low (<40 m)', 'mid': 'Mid (40–120 m)', 'high': 'High (>120 m)'}
FC1_COLOR    = '#1565C0'   # dark blue  = ifs_ens4hybrid
FC2_COLOR    = '#C62828'   # dark red   = hybrid_ens

RESULTS = {
    '2t_p99': {
        'label': '2m Temperature — warm extremes (p99)',
        'unit':  '°C',
        'dir':   './results/2t_ens_local_p99obsclim_ifs4hybrid',
        'prefix': '2t',
        'orogs': ['low', 'mid', 'high'],
    },
    '2t_p1': {
        'label': '2m Temperature — cold extremes (p1)',
        'unit':  '°C',
        'dir':   './results/2t_ens_local_p1obsclim_ifs4hybrid',
        'prefix': '2t',
        'orogs': ['low', 'mid', 'high'],
    },
    'tp24_p99': {
        'label': '24h Precipitation — heavy extremes (p99)',
        'unit':  'mm',
        'dir':   './results/tp24_ens_local_p99obsclim_ifs4hybrid',
        'prefix': 'tp24',
        'orogs': ['low', 'mid', 'high'],
    },
}


def load_data(cfg):
    dfs = {}
    for orog in cfg['orogs']:
        path = os.path.join(cfg['dir'], f"scores_by_leadtime_{cfg['prefix']}_{orog}.csv")
        if os.path.exists(path):
            dfs[orog] = pd.read_csv(path)
    return dfs


def pct_diff(df, score):
    fc1 = df[f'{score}_fc1'].values
    fc2 = df[f'{score}_fc2'].values
    return 100.0 * (fc2 - fc1) / np.abs(fc1)


def ci_pct(df, score):
    fc1 = df[f'{score}_fc1'].values
    low  = df[f'{score}_diff_ci_low'].values
    high = df[f'{score}_diff_ci_high'].values
    return 100.0 * low / np.abs(fc1), 100.0 * high / np.abs(fc1)


def sig_mask(df, score):
    col = f'{score}_is_significant'
    return df[col].values.astype(bool) if col in df.columns else np.zeros(len(df), dtype=bool)


def lead_days(df):
    return df['lead_time'].values / 24.0


# ── Plot 1: % diff per score, all orographies ─────────────────────────────────
def plot_pct_diff_panel(key, cfg, dfs):
    scores = ['twCRPS', 'fCRPS', 'Brier', 'quantile_score']
    score_labels = {
        'twCRPS':         'twCRPS % diff\n(+ = hybrid_ens worse)',
        'fCRPS':          'fCRPS % diff\n(+ = hybrid_ens worse)',
        'Brier':          'Brier score % diff\n(+ = hybrid_ens worse)',
        'quantile_score': 'Quantile score % diff\n(+ = hybrid_ens worse)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(f'hybrid_ens vs ifs_ens4hybrid\n{cfg["label"]}',
                 fontsize=13, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for ax, score in zip(axes, scores):
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        for orog, df in dfs.items():
            if f'{score}_fc1' not in df.columns:
                continue
            x  = lead_days(df)
            pd_ = pct_diff(df, score)
            sig = sig_mask(df, score)
            col = OROG_COLORS[orog]
            mk  = OROG_MARKERS[orog]

            # CI band if available
            if f'{score}_diff_ci_low' in df.columns:
                ci_lo, ci_hi = ci_pct(df, score)
                ax.fill_between(x, ci_lo, ci_hi, color=col, alpha=0.12)

            ax.plot(x, pd_, color=col, marker=mk, ms=5, lw=1.5,
                    label=OROG_LABELS[orog])
            # Significance: filled = significant
            ax.scatter(x[sig], pd_[sig], color=col, marker=mk, s=50, zorder=5)
            ax.scatter(x[~sig], pd_[~sig], color=col, marker=mk, s=50,
                       facecolors='none', linewidths=1.2, zorder=5)

        ax.set_ylabel(score_labels[score], fontsize=9)
        ax.set_title(score, fontsize=10)
        ax.grid(True, alpha=0.3)
        if score in ('Brier', 'quantile_score'):
            ax.set_xlabel('Lead time (days)', fontsize=9)

    # Legend
    handles = [mpatches.Patch(color=OROG_COLORS[o], label=OROG_LABELS[o])
               for o in dfs]
    handles += [Line2D([0], [0], marker='o', color='k', ms=6, ls='none',
                        label='significant (filled)'),
                Line2D([0], [0], marker='o', color='k', ms=6, ls='none',
                        fillstyle='none', label='not significant')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(OUT_DIR, f'pct_diff_panel_{key}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Plot 2: absolute score values for both models ─────────────────────────────
def plot_absolute_scores(key, cfg, dfs):
    scores = ['twCRPS', 'fCRPS', 'Brier', 'ens_mean_bias', 'ens_spread', 'extreme_spread_skill_ratio']
    score_labels = {
        'twCRPS':               f'twCRPS [{cfg["unit"]}]',
        'fCRPS':                f'fCRPS [{cfg["unit"]}]',
        'Brier':                'Brier score',
        'ens_mean_bias':        f'Ensemble mean bias [{cfg["unit"]}]',
        'ens_spread':           f'Ensemble spread [{cfg["unit"]}]',
        'extreme_spread_skill_ratio':'tw Spread/Skill ratio',
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle(f'Absolute scores — hybrid_ens vs ifs_ens4hybrid\n{cfg["label"]}',
                 fontsize=13, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for ax, score in zip(axes, scores):
        if score == 'ens_mean_bias':
            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        if score == 'extreme_spread_skill_ratio':
            ax.axhline(1, color='k', lw=0.8, ls='--', alpha=0.5)

        for orog, df in dfs.items():
            if f'{score}_fc1' not in df.columns:
                continue
            x = lead_days(df)
            col = OROG_COLORS[orog]
            mk  = OROG_MARKERS[orog]
            # fc1 = solid, fc2 = dashed
            ax.plot(x, df[f'{score}_fc1'].values, color=col, marker=mk,
                    ms=4, lw=1.5, ls='-')
            ax.plot(x, df[f'{score}_fc2'].values, color=col, marker=mk,
                    ms=4, lw=1.5, ls='--')

        ax.set_ylabel(score_labels.get(score, score), fontsize=9)
        ax.set_title(score, fontsize=10)
        ax.grid(True, alpha=0.3)
        if axes.tolist().index(ax) >= 3:
            ax.set_xlabel('Lead time (days)', fontsize=9)

    # Legend
    orog_handles = [mpatches.Patch(color=OROG_COLORS[o], label=OROG_LABELS[o])
                    for o in dfs]
    model_handles = [
        Line2D([0], [0], color='k', lw=1.5, ls='-',  label='ifs_ens4hybrid (fc1)'),
        Line2D([0], [0], color='k', lw=1.5, ls='--', label='hybrid_ens (fc2)'),
    ]
    fig.legend(handles=orog_handles + model_handles, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(OUT_DIR, f'absolute_scores_{key}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Plot 3: twCRPS vs quantile_score divergence scatter ───────────────────────
def plot_metric_divergence(all_data):
    """
    For each (variable, orography, lead_time): scatter twCRPS% diff vs quantile_score% diff.
    If metrics agree, points lie in Q1 or Q3.  Divergence shows up in Q2 / Q4.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('twCRPS % diff vs quantile_score % diff\n(agreement → diagonal; divergence → off-diagonal)',
                 fontsize=12, fontweight='bold')

    var_order = ['2t_p99', '2t_p1', 'tp24_p99']
    var_titles = {
        '2t_p99':  '2m Temp — warm (p99)',
        '2t_p1':   '2m Temp — cold (p1)',
        'tp24_p99':'24h Precip — heavy (p99)',
    }

    for ax, key in zip(axes, var_order):
        dfs = all_data.get(key, {})
        ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
        ax.axvline(0, color='k', lw=0.7, ls='--', alpha=0.4)

        for orog, df in dfs.items():
            if 'twCRPS_fc1' not in df.columns or 'quantile_score_fc1' not in df.columns:
                continue
            tw_p = pct_diff(df, 'twCRPS')
            qs_p = pct_diff(df, 'quantile_score')
            x    = lead_days(df)
            col  = OROG_COLORS[orog]
            mk   = OROG_MARKERS[orog]
            sc = ax.scatter(tw_p, qs_p, c=x, cmap='viridis', marker=mk,
                            s=60, edgecolors=col, linewidths=1.2,
                            label=OROG_LABELS[orog], vmin=0.5, vmax=10)

        # Shade quadrants
        xlim = ax.get_xlim() if ax.get_xlim() != (0,1) else (-15, 15)
        ylim = ax.get_ylim() if ax.get_ylim() != (0,1) else (-15, 15)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.fill_between([0, max(xlim)], 0, max(ylim), color='#FFCDD2', alpha=0.25,
                        label='both worse')
        ax.fill_between([min(xlim), 0], min(ylim), 0, color='#C8E6C9', alpha=0.25,
                        label='both better')
        ax.fill_between([0, max(xlim)], min(ylim), 0, color='#FFF9C4', alpha=0.35,
                        label='diverge (twCRPS worse, QS better)')
        ax.fill_between([min(xlim), 0], 0, max(ylim), color='#E1BEE7', alpha=0.25)

        ax.set_xlabel('twCRPS % diff (+ = hybrid worse)', fontsize=9)
        ax.set_ylabel('quantile_score % diff (+ = hybrid worse)', fontsize=9)
        ax.set_title(var_titles[key], fontsize=10)
        ax.grid(True, alpha=0.3)

        # Colorbar for lead time
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label('Lead time (days)', fontsize=8)

        # Orog legend
        handles = [Line2D([0], [0], marker=OROG_MARKERS[o], color=OROG_COLORS[o],
                          ms=7, ls='none', label=OROG_LABELS[o]) for o in dfs]
        ax.legend(handles=handles, fontsize=7, loc='upper left')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'metric_divergence_scatter.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Plot 4: bias evolution (2t only) ─────────────────────────────────────────
def plot_bias_evolution(all_data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Ensemble mean bias evolution — 2t\n(shows growing warm bias in hybrid_ens)',
                 fontsize=12, fontweight='bold')

    for ax, key, title in zip(axes,
                               ['2t_p99', '2t_p1'],
                               ['Warm extremes (p99)', 'Cold extremes (p1)']):
        dfs = all_data.get(key, {})
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        for orog, df in dfs.items():
            if 'ens_mean_bias_fc1' not in df.columns:
                continue
            x   = lead_days(df)
            col = OROG_COLORS[orog]
            mk  = OROG_MARKERS[orog]
            ax.plot(x, df['ens_mean_bias_fc1'].values, color=col, marker=mk,
                    ms=4, lw=1.8, ls='-')
            ax.plot(x, df['ens_mean_bias_fc2'].values, color=col, marker=mk,
                    ms=4, lw=1.8, ls='--')

        ax.set_xlabel('Lead time (days)', fontsize=10)
        ax.set_ylabel('Ensemble mean bias (°C)', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    orog_handles = [mpatches.Patch(color=OROG_COLORS[o], label=OROG_LABELS[o])
                    for o in ['low', 'mid', 'high']]
    model_handles = [
        Line2D([0], [0], color='k', lw=1.8, ls='-',  label='ifs_ens4hybrid (fc1)'),
        Line2D([0], [0], color='k', lw=1.8, ls='--', label='hybrid_ens (fc2)'),
    ]
    fig.legend(handles=orog_handles + model_handles, loc='lower center',
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(OUT_DIR, 'bias_evolution_2t.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Plot 5: all-variables twCRPS summary ─────────────────────────────────────
def plot_twcrps_summary(all_data):
    """Single figure with one row per variable, one column per orography."""
    keys  = ['2t_p99', '2t_p1', 'tp24_p99']
    orogs = ['low', 'mid', 'high']
    labels_row = {
        '2t_p99':  '2t warm (p99)',
        '2t_p1':   '2t cold (p1)',
        'tp24_p99':'tp24 heavy (p99)',
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    fig.suptitle('twCRPS % difference: hybrid_ens vs ifs_ens4hybrid\n(blue = hybrid better, red = hybrid worse)',
                 fontsize=13, fontweight='bold', y=1.0)

    for row_i, key in enumerate(keys):
        dfs = all_data.get(key, {})
        for col_i, orog in enumerate(orogs):
            ax = axes[row_i, col_i]
            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)

            if orog in dfs:
                df  = dfs[orog]
                x   = lead_days(df)
                pd_ = pct_diff(df, 'twCRPS')
                sig = sig_mask(df, 'twCRPS')

                # Colour each segment by sign
                for i in range(len(x)):
                    color = '#C62828' if pd_[i] > 0 else '#1565C0'
                    ax.bar(x[i], pd_[i], width=0.6, color=color, alpha=0.7)

                # CI band
                if 'twCRPS_diff_ci_low' in df.columns:
                    ci_lo, ci_hi = ci_pct(df, 'twCRPS')
                    ax.fill_between(x, ci_lo, ci_hi, color='gray', alpha=0.2,
                                    label='95% CI')

                # Significance stars
                for xi, vi, si in zip(x, pd_, sig):
                    if si:
                        ax.text(xi, vi + (0.15 if vi >= 0 else -0.4), '*',
                                ha='center', va='bottom', fontsize=9, color='k')

            ax.set_title(f'{orog.upper()} orog', fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f'{labels_row[key]}\ntwCRPS % diff', fontsize=8)
            if row_i == 2:
                ax.set_xlabel('Lead time (days)', fontsize=9)
            ax.grid(True, alpha=0.25, axis='y')

    blue_patch = mpatches.Patch(color='#1565C0', alpha=0.7, label='hybrid_ens better')
    red_patch  = mpatches.Patch(color='#C62828', alpha=0.7, label='hybrid_ens worse')
    star_line  = Line2D([0], [0], marker='*', color='k', ms=8, ls='none', label='significant')
    fig.legend(handles=[blue_patch, red_patch, star_line],
               loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out = os.path.join(OUT_DIR, 'twcrps_summary_all_variables.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Output directory: {OUT_DIR}\n')
    all_data = {}
    for key, cfg in RESULTS.items():
        print(f'Loading {key} ...')
        dfs = load_data(cfg)
        if not dfs:
            print(f'  No data found, skipping.')
            continue
        all_data[key] = dfs
        print(f'  Orographies: {list(dfs.keys())}')
        print(f'  Plotting % diff panel ...')
        plot_pct_diff_panel(key, cfg, dfs)
        print(f'  Plotting absolute scores ...')
        plot_absolute_scores(key, cfg, dfs)

    print('\nCross-variable plots ...')
    plot_metric_divergence(all_data)
    plot_bias_evolution(all_data)
    plot_twcrps_summary(all_data)
    print('\nDone. All plots saved to:', OUT_DIR)
