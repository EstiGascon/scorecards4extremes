"""
STEP 8: SAVE RESULTS
====================
Save verification scores to CSV files
"""

import pandas as pd
from pathlib import Path
from utils import format_threshold_string as _format_threshold_string


def run_step8(config, overall_scores, results, orog_type=None, season=None, threshold_value=None):
    """
    Execute Step 8: Save Results
    Returns output directory
    """
    print("\n" + "="*80)
    print("STEP 8: SAVE RESULTS")
    print("="*80)
    
    cfg = config['save']
    output_dir = Path(cfg['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variable = config['variable']
    
    # Add threshold/percentile suffix
    threshold_suffix = ""
    if threshold_value is not None:
        threshold_str = _format_threshold_string(config)
        if threshold_str:
            threshold_suffix = f"_{threshold_str}"
    
    # Add season and orography suffixes if specified
    season_suffix = f"_{season}" if season else ""
    orog_suffix = f"_{orog_type}" if orog_type else ""
    suffix = f"{threshold_suffix}{season_suffix}{orog_suffix}"
    
    print(f"\nOutput directory: {output_dir}")
    
    # Save scores by lead time
    if 'by_leadtime' in results and cfg['save_scores_csv']:
        csv_file = output_dir / f"scores_by_leadtime_{variable}{suffix}.csv"
        results['by_leadtime'].to_csv(csv_file, index=False)
        print(f"  ✓ Saved: scores_by_leadtime_{variable}{suffix}.csv")
    
    # Save overall scores
    if cfg['save_scores_csv']:
        overall_csv = output_dir / f"overall_scores_{variable}{suffix}.csv"
        pd.DataFrame([overall_scores]).to_csv(overall_csv, index=False)
        print(f"  ✓ Saved: overall_scores_{variable}{suffix}.csv")
    
    # Save forecast-observation pairs if requested
    if cfg.get('save_forecast_obs_pairs', False):
        print(f"  [Saving forecast-obs pairs not yet implemented]")
    
    print("\n✓ Step 8 complete")
    
    return output_dir


def save_observation_counts(all_results, output_dir, config):
    """
    Write a CSV summarising the number of observations per condition
    (season × orography × forecast day).
    """
    import numpy as np

    rows = []
    for r in all_results:
        orog = (r.get('orog_type') or 'all').upper()
        season = r.get('season') or 'all'
        by_lt = r.get('results', {}).get('by_leadtime')
        if by_lt is None or by_lt.empty:
            continue
        for _, lt_row in by_lt.iterrows():
            day = lt_row.get('forecast_day', '')
            lead = lt_row.get('lead_time', '')
            n = lt_row.get('n_samples', '')
            thr = lt_row.get('threshold', np.nan)
            rows.append({
                'season': season,
                'orography': orog,
                'forecast_day': day,
                'lead_time_h': lead,
                'n_observations': n,
                'mean_threshold': thr,
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    variable = config.get('variable', '')
    out_file = Path(output_dir) / f"observation_counts_{variable}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n  ✓ Saved observation counts: {out_file.name}")
