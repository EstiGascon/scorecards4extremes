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


def _hit_rate_false_alarm_rate(forecast, observation, threshold, event_type):
    """Return (H, F) = (hit rate = POD, false-alarm rate = POFD) from the 2x2 table.

    H and F are the two quantities the extremal-dependence indices (EDI/SEDI) are
    built from. Returns (nan, nan) if there are no valid pairs or an empty margin.
    """
    valid = _nan_mask(forecast, observation)
    fc_v, obs_v = forecast[valid], observation[valid]
    thr_v = threshold[valid] if isinstance(threshold, np.ndarray) else threshold
    if len(fc_v) == 0:
        return np.nan, np.nan
    if event_type == 'below':
        fc_binary = (fc_v <= thr_v).astype(int)
        obs_binary = (obs_v <= thr_v).astype(int)
    else:
        fc_binary = (fc_v >= thr_v).astype(int)
        obs_binary = (obs_v >= thr_v).astype(int)
    hits, misses, false_alarms, cn = calculate_contingency_table(fc_binary, obs_binary)
    H = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    F = false_alarms / (false_alarms + cn) if (false_alarms + cn) > 0 else np.nan
    return H, F


def calculate_sedi(forecast, observation, threshold, event_type):
    """Symmetric Extremal Dependence Index (Ferro & Stephenson, 2011).

        SEDI = [ln F - ln H - ln(1-F) + ln(1-H)] / [ln F + ln H + ln(1-F) + ln(1-H)]

    with H = hit rate (POD) and F = false-alarm rate (POFD). Unlike ETS/PSS, SEDI does
    NOT degenerate to a trivial value as the event base rate -> 0, so it is the
    recommended categorical score for rare/extreme events. Range (-1, 1];
    0 = no skill, 1 = perfect.

    IMPORTANT (bias caveat): EDI/SEDI are only base-rate independent when the forecast
    frequency bias B = (H_count+F_count)/(H_count+M_count) is ~ 1. The forecasts scored
    here are NOT bias-corrected, so a SEDI difference between two models reflects BOTH
    their discrimination and their frequency-bias behaviour. Read it as a joint statement,
    and cross-check with the frequency bias / base-rate columns. See docs/USER_GUIDE.md.
    """
    H, F = _hit_rate_false_alarm_rate(forecast, observation, threshold, event_type)
    if np.isnan(H) or np.isnan(F):
        return np.nan
    # Logs blow up at the margins (H or F equal to 0 or 1) -> undefined, return NaN.
    if H in (0.0, 1.0) or F in (0.0, 1.0):
        return np.nan
    num = np.log(F) - np.log(H) - np.log(1.0 - F) + np.log(1.0 - H)
    den = np.log(F) + np.log(H) + np.log(1.0 - F) + np.log(1.0 - H)
    return num / den if den != 0 else np.nan


def calculate_edi(forecast, observation, threshold, event_type):
    """Extremal Dependence Index (Ferro & Stephenson, 2011).

        EDI = [ln F - ln H] / [ln F + ln H]

    with H = hit rate (POD) and F = false-alarm rate (POFD). Non-degenerate for rare
    events like SEDI, but (unlike SEDI) it is not symmetric under swapping events with
    non-events and is easier to hedge; reported alongside SEDI as a cross-check.
    Range (-1, 1]; 0 = no skill, 1 = perfect. Same frequency-bias caveat as SEDI.
    """
    H, F = _hit_rate_false_alarm_rate(forecast, observation, threshold, event_type)
    if np.isnan(H) or np.isnan(F):
        return np.nan
    if H in (0.0, 1.0) or F in (0.0, 1.0):
        return np.nan
    num = np.log(F) - np.log(H)
    den = np.log(F) + np.log(H)
    return num / den if den != 0 else np.nan


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
    if isinstance(threshold, np.ndarray):
        valid &= ~np.isnan(threshold)  # keep only rows with a valid threshold (matches twMAE)
    if event_type == 'below':
        fc_v = np.minimum(forecast[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
        obs_v = np.minimum(observation[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
    else:
        fc_v = np.maximum(forecast[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
        obs_v = np.maximum(observation[valid], threshold[valid] if isinstance(threshold, np.ndarray) else threshold)
    if len(fc_v) == 0:
        return np.nan
    return float(np.sqrt(np.mean((fc_v - obs_v) ** 2)))


def calculate_seeps(forecast, observation, station_ids, dry_threshold=0.2, min_obs=30):
    """SEEPS — Stable Equitable Error in Probability Space (Rodwell et al. 2010).

    PRECIPITATION ONLY. Uses the validated penalty matrix from ``scores.categorical.seeps``
    (Rodwell Eq. 15), with the three categories dry / light / heavy. For each station the
    climatological dry probability p1 and the light/heavy boundary are estimated from THAT
    station's observations in the current evaluation cell. This is an *in-sample*,
    station-specific climatology (self-contained, no external files); a long external
    reference (e.g. from ``diagnostics/climatology/build_tp24_climatology.py``) would be more
    rigorous and is the recommended upgrade. Stations with p1 outside [0.1, 0.85] (Rodwell
    stability range) or with fewer than ``min_obs`` values are dropped.

    ``dry_threshold`` (default 0.2) is the dry/wet boundary in the obs units (mm for tp24).
    Negatively oriented: lower = better. Returns the mean SEEPS over retained station-date
    pairs, or NaN if no station qualifies.
    """
    import pandas as pd
    import xarray as xr
    from scores.categorical import seeps as _seeps_fn

    fc = np.asarray(forecast, dtype=float)
    ob = np.asarray(observation, dtype=float)
    st = np.asarray(station_ids)
    valid = ~(np.isnan(fc) | np.isnan(ob))
    fc, ob, st = fc[valid], ob[valid], st[valid]
    if len(ob) < min_obs:
        return np.nan

    # Per-station in-sample precip climatology: p1 (dry prob) and light/heavy boundary.
    df = pd.DataFrame({'st': st, 'ob': ob})
    p1_map, tlh_map = {}, {}
    for s, g in df.groupby('st'):
        o = g['ob'].values
        if len(o) < min_obs:
            continue
        p1 = float(np.mean(o <= dry_threshold))
        if not (0.1 <= p1 <= 0.85):          # Rodwell stability range
            continue
        p3 = (1.0 - p1) / 3.0                 # heavy-category probability
        tlh = float(np.quantile(o, 1.0 - p3))  # boundary at the (2+p1)/3 climatological quantile
        if tlh <= dry_threshold:
            continue
        p1_map[s] = p1
        tlh_map[s] = tlh
    if not p1_map:
        return np.nan

    keep = np.array([s in p1_map for s in st])
    if keep.sum() < min_obs:
        return np.nan
    fc_k, ob_k, st_k = fc[keep], ob[keep], st[keep]
    p1_arr = np.array([p1_map[s] for s in st_k])
    tlh_arr = np.array([tlh_map[s] for s in st_k])
    try:
        res = _seeps_fn(
            xr.DataArray(fc_k, dims=['s']),
            xr.DataArray(ob_k, dims=['s']),
            prob_dry=xr.DataArray(p1_arr, dims=['s']),
            light_heavy_threshold=xr.DataArray(tlh_arr, dims=['s']),
            dry_light_threshold=dry_threshold,
        )
        return float(np.asarray(res.values))
    except Exception as e:
        print(f"    SEEPS error: {e}")
        return np.nan


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
    if 'SEDI' in selected_scores:
        scores['SEDI'] = calculate_sedi(forecast, observation, threshold, event_type)
    if 'EDI' in selected_scores:
        scores['EDI'] = calculate_edi(forecast, observation, threshold, event_type)
    
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
    if 'SEEPS' in selected_scores and config.get('variable') != 'tp24':
        # SEEPS is a precipitation score (dry/light/heavy categories); meaningless otherwise.
        print(f"  NOTE: SEEPS is precipitation-only — skipped for variable '{config.get('variable')}'")
        selected_scores = [s for s in selected_scores if s != 'SEEPS']
    
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
                dates_day = (day_data['date'].values[valid_mask]
                             if 'date' in day_data.columns else None)
            else:
                fc1_day = day_data['fc1_value'].values
                fc2_day = day_data['fc2_value'].values
                obs_day = day_data['obs_value'].values
                day_threshold = threshold
                dates_day = (day_data['date'].values
                             if 'date' in day_data.columns else None)
            
            # Get the step range for this day
            steps_in_day = sorted(day_data['step'].unique())
            step_range = f"{int(min(steps_in_day))}-{int(max(steps_in_day))}h"
            
            day_scores_fc1 = calculate_all_scores(fc1_day, obs_day, day_threshold, event_type, selected_scores)
            day_scores_fc2 = calculate_all_scores(fc2_day, obs_day, day_threshold, event_type, selected_scores)

            # SEEPS (precip only) needs per-station climatology, so it is computed here from
            # the FULL day_data (with station_id), independent of the extreme-threshold mask.
            if 'SEEPS' in selected_scores and 'station_id' in day_data.columns:
                _st = day_data['station_id'].values
                day_scores_fc1['SEEPS'] = calculate_seeps(day_data['fc1_value'].values, day_data['obs_value'].values, _st)
                day_scores_fc2['SEEPS'] = calculate_seeps(day_data['fc2_value'].values, day_data['obs_value'].values, _st)

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
                    'SEDI': calculate_sedi,
                    'EDI': calculate_edi,
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

                # Block bootstrap over DATES: resample whole forecast dates (all stations
                # for a drawn date kept together) so spatial correlation between stations
                # on the same date is preserved. iid row resampling would treat all
                # station-date pairs as independent, underestimate the CI width and
                # over-flag significance. Falls back to iid rows when the date column is
                # absent or there are too few distinct dates (<5) to block-resample.
                date_blocks = None
                if dates_day is not None:
                    _uniq, _inv = np.unique(dates_day, return_inverse=True)
                    if len(_uniq) >= 5:
                        date_blocks = [np.where(_inv == k)[0] for k in range(len(_uniq))]
                n_dates = len(date_blocks) if date_blocks is not None else 0

                def _resample_idx():
                    if date_blocks is not None:
                        drawn = rng.integers(0, n_dates, n_dates)
                        return np.concatenate([date_blocks[d] for d in drawn])
                    return rng.integers(0, n_day, n_day)

                for score_name in selected_scores:
                    score_func = score_func_map.get(score_name)
                    if score_func:
                        # Paired bootstrap of the DIFFERENCE (fc2 - fc1).
                        # Using the same resampled indices for both models removes
                        # shared obs variance, giving much higher power than
                        # comparing non-overlapping independent CIs.
                        diff_samples = []
                        for _ in range(n_boot):
                            b_idx = _resample_idx()
                            thr_b = day_threshold if np.isscalar(day_threshold) else day_threshold[b_idx]
                            s1 = score_func(fc1_day[b_idx], obs_day[b_idx], thr_b, event_type)
                            s2 = score_func(fc2_day[b_idx], obs_day[b_idx], thr_b, event_type)
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
