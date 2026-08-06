"""
STEP 6: CALCULATE VERIFICATION SCORES
======================================
Scores for deterministic forecasts:
  - ETS, PSS, POD, FAR
  - twMAE, twRMSE
  - bias, MAE, RMSE, correlation

Scores for ensemble forecasts (future):
  - twCRPS, Brier, CRPS
"""

import numpy as np
import pandas as pd


def calculate_contingency_table(fc_binary, obs_binary):
    """Calculate contingency table"""
    hits = np.sum((fc_binary == 1) & (obs_binary == 1))
    misses = np.sum((fc_binary == 0) & (obs_binary == 1))
    false_alarms = np.sum((fc_binary == 1) & (obs_binary == 0))
    correct_negatives = np.sum((fc_binary == 0) & (obs_binary == 0))
    return hits, misses, false_alarms, correct_negatives


def _nan_mask(forecast, observation):
    """Return boolean mask of valid (non-NaN) pairs."""
    return ~(np.isnan(forecast) | np.isnan(observation))


def calculate_ets(forecast, observation, threshold, event_type):
    """Equitable Threat Score"""
    valid = _nan_mask(forecast, observation)
    fc_v, obs_v = forecast[valid], observation[valid]
    thr_v = threshold[valid] if isinstance(threshold, np.ndarray) else threshold
    if len(fc_v) == 0:
        return np.nan
    if event_type == 'below':
        fc_binary = (fc_v <= thr_v).astype(int)
        obs_binary = (obs_v <= thr_v).astype(int)
    else:
        fc_binary = (fc_v >= thr_v).astype(int)
        obs_binary = (obs_v >= thr_v).astype(int)
    
    hits, misses, false_alarms, cn = calculate_contingency_table(fc_binary, obs_binary)
    total = len(fc_binary)
    hits_random = (hits + misses) * (hits + false_alarms) / total
    denominator = hits + misses + false_alarms - hits_random
    return (hits - hits_random) / denominator if denominator > 0 else np.nan


def calculate_pss(forecast, observation, threshold, event_type):
    """Peirce Skill Score"""
    valid = _nan_mask(forecast, observation)
    fc_v, obs_v = forecast[valid], observation[valid]
    thr_v = threshold[valid] if isinstance(threshold, np.ndarray) else threshold
    if len(fc_v) == 0:
        return np.nan
    if event_type == 'below':
        fc_binary = (fc_v <= thr_v).astype(int)
        obs_binary = (obs_v <= thr_v).astype(int)
    else:
        fc_binary = (fc_v >= thr_v).astype(int)
        obs_binary = (obs_v >= thr_v).astype(int)
    
    hits, misses, false_alarms, cn = calculate_contingency_table(fc_binary, obs_binary)
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0
    pofd = false_alarms / (false_alarms + cn) if (false_alarms + cn) > 0 else 0
    return pod - pofd


def bootstrap_confidence_interval(forecast, observation, threshold, event_type, score_func, n_bootstrap=1000, confidence=0.95, max_samples=500000):
    """Calculate bootstrap confidence intervals for a score
    
    Args:
        max_samples: Maximum number of samples to use (subsamples if needed for speed)
    """
    n = len(forecast)
    if n < 10:
        return np.nan, np.nan
    
    # Subsample if dataset is too large (for computational efficiency)
    if n > max_samples:
        print(f"    Subsampling {n:,} → {max_samples:,} for bootstrap", flush=True)
        indices = np.random.choice(n, max_samples, replace=False)
        forecast = forecast[indices]
        observation = observation[indices]
        n = max_samples
    
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        boot_forecast = forecast[indices]
        boot_obs = observation[indices]
        
        score = score_func(boot_forecast, boot_obs, threshold, event_type)
        if not np.isnan(score):
            bootstrap_scores.append(score)
    
    if len(bootstrap_scores) < n_bootstrap * 0.5:
        return np.nan, np.nan
    
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return ci_low, ci_high


def calculate_pod(forecast, observation, threshold, event_type):
    """Probability of Detection"""
    valid = _nan_mask(forecast, observation)
    fc_v, obs_v = forecast[valid], observation[valid]
    thr_v = threshold[valid] if isinstance(threshold, np.ndarray) else threshold
    if len(fc_v) == 0:
        return np.nan
    if event_type == 'below':
        fc_binary = (fc_v <= thr_v).astype(int)
        obs_binary = (obs_v <= thr_v).astype(int)
    else:
        fc_binary = (fc_v >= thr_v).astype(int)
        obs_binary = (obs_v >= thr_v).astype(int)
    
    hits, misses, _, _ = calculate_contingency_table(fc_binary, obs_binary)
    return hits / (hits + misses) if (hits + misses) > 0 else np.nan


def calculate_far(forecast, observation, threshold, event_type):
    """False Alarm Ratio"""
    valid = _nan_mask(forecast, observation)
    fc_v, obs_v = forecast[valid], observation[valid]
    thr_v = threshold[valid] if isinstance(threshold, np.ndarray) else threshold
    if len(fc_v) == 0:
        return np.nan
    if event_type == 'below':
        fc_binary = (fc_v <= thr_v).astype(int)
        obs_binary = (obs_v <= thr_v).astype(int)
    else:
        fc_binary = (fc_v >= thr_v).astype(int)
        obs_binary = (obs_v >= thr_v).astype(int)
    
    hits, _, false_alarms, _ = calculate_contingency_table(fc_binary, obs_binary)
    return false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan


def calculate_twmae(forecast, observation, threshold, event_type):
    """Threshold-weighted MAE using the chaining function v(x) = max(x, T) for upper tail
    (min for lower tail). Applied to all samples, same framework as twCRPS.
    twMAE = mean |v(fc) - v(obs)|
    """
    valid = ~(np.isnan(forecast) | np.isnan(observation))
    if isinstance(threshold, np.ndarray):
        valid &= ~np.isnan(threshold)  # keep only rows with a valid (non-NaN) threshold
    if event_type == 'below':
        fc_v = np.minimum(forecast[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
        obs_v = np.minimum(observation[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
    else:
        fc_v = np.maximum(forecast[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
        obs_v = np.maximum(observation[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
    if len(fc_v) == 0:
        return np.nan
    return float(np.mean(np.abs(fc_v - obs_v)))


def calculate_twmae_components(forecast, observation, threshold, event_type):
    """Decompose twMAE into hit / miss / false-alarm contributions.

    For the upper tail (event_type='above'):
      Hits  (obs≥T, fc≥T) : cost = |fc − obs|,  contribution = n_hits/N   × mean_cost
      Misses(obs≥T, fc<T) : cost = obs − T,       contribution = n_misses/N × mean_cost
      FAs   (obs<T, fc≥T) : cost = fc  − T,       contribution = n_FA/N    × mean_cost
      CNs   (obs<T, fc<T) : cost = 0              → zero contribution

    The three contributions sum exactly to twMAE.
    Works with scalar or per-station NumPy array threshold.
    """
    valid = ~(np.isnan(forecast) | np.isnan(observation))
    if isinstance(threshold, np.ndarray):
        valid &= ~np.isnan(threshold)
        fc_v  = forecast[valid];  obs_v = observation[valid];  T_v = threshold[valid]
    else:
        fc_v  = forecast[valid];  obs_v = observation[valid];  T_v = threshold

    N = len(obs_v)
    nan_result = {'twMAE_hits': np.nan, 'twMAE_misses': np.nan, 'twMAE_FA': np.nan}
    if N == 0:
        return nan_result

    if event_type == 'below':
        hit_m  = (obs_v <= T_v) & (fc_v <= T_v)
        miss_m = (obs_v <= T_v) & (fc_v >  T_v)
        fa_m   = (obs_v >  T_v) & (fc_v <= T_v)
        T_miss = T_v[miss_m] if isinstance(T_v, np.ndarray) else T_v
        T_fa   = T_v[fa_m]   if isinstance(T_v, np.ndarray) else T_v
        hit_cost  = float(np.mean(np.abs(fc_v[hit_m] - obs_v[hit_m]))) if hit_m.sum() > 0 else 0.0
        miss_cost = float(np.mean(T_miss - obs_v[miss_m]))              if miss_m.sum() > 0 else 0.0
        # FA cost: chaining function gives |min(fc,T) - min(obs,T)| = T - fc  (fc<=T, obs>T)
        fa_cost   = float(np.mean(T_fa - fc_v[fa_m]))                   if fa_m.sum()  > 0 else 0.0
    else:
        hit_m  = (obs_v >= T_v) & (fc_v >= T_v)
        miss_m = (obs_v >= T_v) & (fc_v <  T_v)
        fa_m   = (obs_v <  T_v) & (fc_v >= T_v)
        T_miss = T_v[miss_m] if isinstance(T_v, np.ndarray) else T_v
        T_fa   = T_v[fa_m]   if isinstance(T_v, np.ndarray) else T_v
        hit_cost  = float(np.mean(np.abs(fc_v[hit_m] - obs_v[hit_m]))) if hit_m.sum() > 0 else 0.0
        miss_cost = float(np.mean(obs_v[miss_m] - T_miss))             if miss_m.sum() > 0 else 0.0
        # FA cost: chaining function gives |max(fc,T) - max(obs,T)| = fc - T  (fc>=T, obs<T)
        fa_cost   = float(np.mean(fc_v[fa_m]   - T_fa))                if fa_m.sum()  > 0 else 0.0

    return {
        # Total budget contributions (n_cases/N × per-case cost) — sum to twMAE
        'twMAE_hits':   hit_cost  * hit_m.sum()  / N,
        'twMAE_misses': miss_cost * miss_m.sum() / N,
        'twMAE_FA':     fa_cost   * fa_m.sum()   / N,
        # Per-case conditional metrics — intuitive for heatmap comparison
        # lower = better; show model quality *given* that category occurred
        'twMAE_hit_mae':       hit_cost  if hit_m.sum()  > 0 else np.nan,  # |fc-obs| at hits
        'twMAE_miss_severity': miss_cost if miss_m.sum() > 0 else np.nan,  # how extreme missed events were
        'twMAE_fa_severity':   fa_cost   if fa_m.sum()   > 0 else np.nan,  # how extreme false alarms were
    }


def calculate_twrmse(forecast, observation, threshold, event_type):
    """Threshold-weighted RMSE using the chaining function v(x) = max(x, T) for upper tail
    (min for lower tail). Applied to all samples, same framework as twCRPS.
    twRMSE = sqrt(mean (v(fc) - v(obs))^2)
    """
    valid = ~(np.isnan(forecast) | np.isnan(observation))
    if event_type == 'below':
        fc_v = np.minimum(forecast[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
        obs_v = np.minimum(observation[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
    else:
        fc_v = np.maximum(forecast[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
        obs_v = np.maximum(observation[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
    if len(fc_v) == 0:
        return np.nan
    return float(np.sqrt(np.mean((fc_v - obs_v) ** 2)))


def calculate_all_scores(forecast, observation, threshold, event_type, selected_scores):
    """Calculate all requested scores"""
    scores = {}
    
    # Continuous scores
    if 'bias' in selected_scores:
        scores['bias'] = np.nanmean(forecast - observation)
    if 'mae' in selected_scores:
        scores['mae'] = np.nanmean(np.abs(forecast - observation))
    if 'rmse' in selected_scores:
        scores['rmse'] = np.sqrt(np.nanmean((forecast - observation) ** 2))
    if 'correlation' in selected_scores:
        valid = ~(np.isnan(forecast) | np.isnan(observation))
        if valid.sum() > 2:
            scores['correlation'] = np.corrcoef(forecast[valid], observation[valid])[0, 1]
        else:
            scores['correlation'] = np.nan
    
    # Categorical scores
    if 'ETS' in selected_scores:
        scores['ETS'] = calculate_ets(forecast, observation, threshold, event_type)
    if 'PSS' in selected_scores:
        scores['PSS'] = calculate_pss(forecast, observation, threshold, event_type)
    if 'POD' in selected_scores:
        scores['POD'] = calculate_pod(forecast, observation, threshold, event_type)
    if 'FAR' in selected_scores:
        scores['FAR'] = calculate_far(forecast, observation, threshold, event_type)
    
    # Threshold-weighted scores
    if 'twMAE' in selected_scores:
        scores['twMAE'] = calculate_twmae(forecast, observation, threshold, event_type)
        # Auto-compute hit / miss / FA decomposition alongside twMAE
        scores.update(calculate_twmae_components(forecast, observation, threshold, event_type))
    if 'twRMSE' in selected_scores:
        scores['twRMSE'] = calculate_twrmse(forecast, observation, threshold, event_type)
    
    scores['n_samples'] = len(forecast)
    
    return scores


def run_step6(config, data, threshold, event_type, model_names):
    """
    Execute Step 6: Calculate Verification Scores
    Returns overall scores and stratified results for BOTH models
    """
    print("\n" + "="*80)
    print("STEP 6: CALCULATE VERIFICATION SCORES")
    print("="*80)
    
    cfg = config['scores']
    selected_scores = cfg['deterministic']
    
    print(f"\nComparing two forecast models:")
    print(f"  Model 1: {model_names['fc1_name']}")
    print(f"  Model 2: {model_names['fc2_name']}")
    
    print(f"\nScores to calculate:")
    for score in selected_scores:
        print(f"  - {score}")
    
    # Note: overall scores will be computed as mean-of-per-leadtime scores
    # (not pooled across all data) to give equal weight to each lead time.
    bootstrap_results = {}
    
    # Stratified scores
    stratify_by = cfg.get('stratify_by', ['lead_time'])
    results = {}
    
    if 'lead_time' in stratify_by:
        print("\n" + "-"*40)
        print("Scores by Forecast Day:")
        print("-"*40)
        
        # Group data by forecast day
        # Use forecast_day from data if available (from filename), otherwise calculate from step
        # Formula: day = ((step-1)//24)+1  →  1-24h=Day1, 25-48h=Day2, 49-72h=Day3, etc.
        if 'forecast_day' not in data.columns:
            # Fallback: calculate from step using consistent formula
            data['forecast_day'] = ((data['step'] - 1) / 24).astype(int) + 1
        
        results_by_day = []
        for day in sorted(data['forecast_day'].unique()):
            day_data = data[data['forecast_day'] == day]

            # Handle per-station threshold (pd.Series aligned to data index)
            if isinstance(threshold, pd.Series):
                day_threshold_series = threshold.loc[day_data.index]
                valid_mask = day_threshold_series.notna().values
                if not valid_mask.any():
                    print(f"  Day {day}: no stations with valid threshold, skipping")
                    continue
                fc1_day = day_data['fc1_value'].values[valid_mask]
                fc2_day = day_data['fc2_value'].values[valid_mask]
                obs_day = day_data['obs_value'].values[valid_mask]
                day_threshold = day_threshold_series.values[valid_mask]
            else:
                fc1_day = day_data['fc1_value'].values
                fc2_day = day_data['fc2_value'].values
                obs_day = day_data['obs_value'].values
                day_threshold = threshold
            
            # Get the step range for this day
            steps_in_day = sorted(day_data['step'].unique())
            step_range = f"{int(min(steps_in_day))}-{int(max(steps_in_day))}h"
            
            day_scores_fc1 = calculate_all_scores(fc1_day, obs_day, day_threshold, event_type, selected_scores)
            day_scores_fc2 = calculate_all_scores(fc2_day, obs_day, day_threshold, event_type, selected_scores)

            # Count exceedances (obs crossing the threshold)
            if event_type == 'below':
                n_exceedances = int(np.sum(obs_day <= day_threshold))
            else:
                n_exceedances = int(np.sum(obs_day >= day_threshold))

            # Combine scores
            # For per-station thresholds (array), store mean for reporting
            threshold_scalar = float(np.nanmean(day_threshold)) if isinstance(day_threshold, np.ndarray) else day_threshold
            day_scores = {
                'forecast_day': day,
                'step_range': step_range,
                'lead_time': int(np.mean(steps_in_day)),  # Representative lead time for plotting
                'threshold': threshold_scalar,
                'n_exceedances': n_exceedances,
            }
            for score_name in selected_scores:
                day_scores[f'{score_name}_fc1'] = day_scores_fc1.get(score_name, np.nan)
                day_scores[f'{score_name}_fc2'] = day_scores_fc2.get(score_name, np.nan)
                day_scores[f'{score_name}_diff'] = day_scores_fc2.get(score_name, np.nan) - day_scores_fc1.get(score_name, np.nan)

            # Auto-include twMAE decomposition components (computed whenever twMAE is scored)
            for comp in ('twMAE_hits', 'twMAE_misses', 'twMAE_FA'):
                if comp in day_scores_fc1:
                    day_scores[f'{comp}_fc1']  = day_scores_fc1[comp]
                    day_scores[f'{comp}_fc2']  = day_scores_fc2[comp]
                    day_scores[f'{comp}_diff'] = day_scores_fc2[comp] - day_scores_fc1[comp]

            # Bootstrap significance for this day
            if config.get('bootstrap', {}).get('enabled', False):
                score_func_map = {
                    'ETS': calculate_ets,
                    'PSS': calculate_pss,
                    'POD': calculate_pod,
                    'FAR': calculate_far,
                    'twMAE': calculate_twmae,
                    'twRMSE': calculate_twrmse,
                    'bias': lambda fc, obs, thr, et: np.nanmean(fc - obs),
                    'mae': lambda fc, obs, thr, et: np.nanmean(np.abs(fc - obs)),
                    'rmse': lambda fc, obs, thr, et: np.sqrt(np.nanmean((fc - obs) ** 2)),
                }
                n_boot = config['bootstrap']['n_samples']
                conf   = config['bootstrap']['confidence_level']
                alpha  = 1 - conf
                n_day  = len(fc1_day)
                rng    = np.random.default_rng()

                for score_name in selected_scores:
                    score_func = score_func_map.get(score_name)
                    if score_func:
                        # Paired bootstrap of the DIFFERENCE (fc2 - fc1).
                        # Using the same resampled indices for both models removes
                        # shared obs variance, giving much higher power than
                        # comparing non-overlapping independent CIs.
                        diff_samples = []
                        indices_pool = rng.integers(0, n_day, size=(n_boot, n_day))
                        for b_idx in indices_pool:
                            s1 = score_func(fc1_day[b_idx], obs_day[b_idx], day_threshold if np.isscalar(day_threshold) else day_threshold[b_idx], event_type)
                            s2 = score_func(fc2_day[b_idx], obs_day[b_idx], day_threshold if np.isscalar(day_threshold) else day_threshold[b_idx], event_type)
                            if not (np.isnan(s1) or np.isnan(s2)):
                                diff_samples.append(s2 - s1)
                        if len(diff_samples) >= n_boot * 0.5:
                            ci_lo = np.percentile(diff_samples, 100 * alpha / 2)
                            ci_hi = np.percentile(diff_samples, 100 * (1 - alpha / 2))
                            # Significant if CI of difference does not include 0
                            is_significant = not (ci_lo <= 0.0 <= ci_hi)
                        else:
                            is_significant = False
                        day_scores[f'{score_name}_is_significant'] = is_significant
            day_scores['n_samples'] = day_scores_fc1['n_samples']
            
            results_by_day.append(day_scores)
            
            print(f"  Day {day} ({step_range}): n={day_scores['n_samples']}")
        
        results['by_leadtime'] = pd.DataFrame(results_by_day)
    
    # Combine overall scores as mean of per-leadtime scores
    by_lt_df = results.get('by_leadtime', pd.DataFrame())
    overall_scores = {
        'model1_name': model_names['fc1_name'],
        'model2_name': model_names['fc2_name']
    }
    for score_name in selected_scores:
        k1 = f'{score_name}_fc1'
        k2 = f'{score_name}_fc2'
        kd = f'{score_name}_diff'
        if k1 in by_lt_df.columns:
            overall_scores[k1] = by_lt_df[k1].mean()
        else:
            overall_scores[k1] = np.nan
        if k2 in by_lt_df.columns:
            overall_scores[k2] = by_lt_df[k2].mean()
        else:
            overall_scores[k2] = np.nan
        overall_scores[kd] = overall_scores[k2] - overall_scores[k1]
    
    overall_scores['n_samples'] = int(by_lt_df['n_samples'].sum()) if 'n_samples' in by_lt_df.columns else len(data)

    # Include twMAE decomposition components in overall scores (mean over lead days)
    for comp in ('twMAE_hits', 'twMAE_misses', 'twMAE_FA'):
        k1 = f'{comp}_fc1'; k2 = f'{comp}_fc2'; kd = f'{comp}_diff'
        if k1 in by_lt_df.columns:
            overall_scores[k1] = by_lt_df[k1].mean()
            overall_scores[k2] = by_lt_df[k2].mean()
            overall_scores[kd] = overall_scores[k2] - overall_scores[k1]

    # Propagate bootstrap significance to overall scores
    # A score is significant overall if ALL lead times agree on significance
    if config.get('bootstrap', {}).get('enabled', False):
        for score_name in selected_scores:
            sig_col = f'{score_name}_is_significant'
            if sig_col in by_lt_df.columns:
                overall_scores[sig_col] = bool(by_lt_df[sig_col].all())
    
    print("\n" + "-"*40)
    print("Overall Scores (mean of per-leadtime):")
    print("-"*40)
    for score_name in selected_scores:
        v1 = overall_scores.get(f'{score_name}_fc1', np.nan)
        v2 = overall_scores.get(f'{score_name}_fc2', np.nan)
        print(f"  {score_name}: fc1={v1:.4f}, fc2={v2:.4f}")
    
    print("\n✓ Step 6 complete")
    
    return overall_scores, results
