"""
STEP 6 (ENSEMBLE): CALCULATE ENSEMBLE VERIFICATION SCORES
===========================================================
Scores for ensemble forecasts using the 'scores' library:
  - CRPS (Continuous Ranked Probability Score, fair method)
  - fCRPS (alias for fair CRPS, same as CRPS)
  - twCRPS (threshold-weighted CRPS for extremes, fair method)
  - Brier Score (for ensemble)
  - Quantile Score (at ensemble quantiles)
  - tw Quantile Score (threshold-weighted quantile score)
  - Diagonal Score (observation-climatology-based, following VTB/Quaver methodology)
  - Ensemble Mean Bias, MAE, RMSE, Spread

Uses xarray DataArrays as required by the 'scores' library.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from scores.probability import (
    crps_for_ensemble,
    tw_crps_for_ensemble,
    tail_tw_crps_for_ensemble,
    brier_score_for_ensemble,
)
from scores.continuous import (
    tw_quantile_score,
)


def _get_member_columns(df, prefix):
    """Get sorted list of member column names for a model prefix (e.g. 'fc1_member_')."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort(key=lambda c: int(c.split('_')[-1]))
    return cols


def _df_to_xarray(df, member_cols):
    """Convert DataFrame member columns + obs to xarray DataArrays.
    
    Returns:
        fcst: DataArray with dims (sample, ensemble)
        obs: DataArray with dim (sample,)
    """
    fcst_vals = df[member_cols].values  # shape: (n_samples, n_members)
    obs_vals = df['obs_value'].values   # shape: (n_samples,)
    
    fcst = xr.DataArray(fcst_vals, dims=['sample', 'ensemble'])
    obs = xr.DataArray(obs_vals, dims=['sample'])
    return fcst, obs


def _fair_crps_numpy(fcst_np, obs_np):
    """Fair (bias-corrected) CRPS computed via numpy, for per-sample threshold support.

    Implements: fCRPS = E|X - y| - (1/2) E|X - X'|
    using the sorted-array trick for efficient pairwise computation.

    Args:
        fcst_np: (n_samples, n_members) ensemble forecasts (after chaining transform)
        obs_np:  (n_samples,) observations (after chaining transform)
    Returns:
        float: mean fair CRPS across samples
    """
    n_members = fcst_np.shape[1]
    # Term 1: mean |member - obs| per sample
    term1 = np.mean(np.abs(fcst_np - obs_np[:, None]), axis=1)
    # Term 2: fair pairwise correction via sorted-array trick
    # sum_{i<j} |x_i - x_j| = sum_k (2k - M + 1) * x_{(k)}  (0-indexed k)
    fcst_sorted = np.sort(fcst_np, axis=1)
    weights = 2 * np.arange(n_members) - n_members + 1  # shape (n_members,)
    pairwise_sum = np.sum(fcst_sorted * weights[None, :], axis=1)  # per sample
    term2 = pairwise_sum / (n_members * (n_members - 1))  # = E|X - X'| / 2
    return float(np.mean(term1 - term2))


def _bss_clim_freq(percentile, event_type):
    """Climatological event frequency from threshold percentile definition.

    Since T_i = p{percentile} of the station obs climatology, by definition
    the long-run exceedance rate is percentile/100 for below events and
    (100 - percentile)/100 for above events.
    This matches the ERA5-based reference used by VTB/Quaver for BSS.
    """
    if percentile is None:
        return None
    p = float(percentile) / 100.0
    return p if event_type == 'below' else (1.0 - p)


# Tail quantile levels for threshold-weighted quantile scores (Taggart 2022, QJRMS).
# For 'above' events: probe the upper tail (α ∈ {0.90, 0.95, 0.99}).
# For 'below' events: probe the lower tail (α ∈ {0.01, 0.05, 0.10}).
# These levels are fixed; the threshold T (from config) sets WHERE the tail starts.
_TW_QS_ALPHAS = {
    'above': [0.90, 0.95, 0.99],
    'below': [0.01, 0.05, 0.10],
}


def _alpha_label(alpha):
    """Format alpha as a three-digit column label, e.g. 0.99 → 'q099', 0.01 → 'q001'."""
    return f"q{int(round(alpha * 100)):03d}"


def _alpha_from_label(label):
    """Parse alpha from a label string, e.g. 'q099' → 0.99, 'q001' → 0.01."""
    return int(label[1:]) / 100.0


def calculate_ensemble_scores(df, threshold, event_type, selected_scores, model_prefix,
                               extreme_only_basic=False, tw_qs_alphas=None):
    """Calculate ensemble verification scores for one model.
    
    Args:
        df: DataFrame with member columns (e.g. fc1_member_0, fc1_member_1, ...)
        threshold: float or numpy array (per-row, aligned to df), extreme event threshold.
            When a numpy array is supplied (local_obs_climatology), it is used for binary
            masks; the mean is used for library score functions that require a scalar.
        event_type: 'above' or 'below'
        selected_scores: list of score names to compute
        model_prefix: 'fc1' or 'fc2'
        extreme_only_basic: if True, bias/MAE/RMSE/spread are computed only on
            cases where the observation exceeds (or falls below) the threshold
            (conditional verification). Default False uses all valid samples.
    
    Returns:
        dict of {score_name: value}
    """
    # BSS reference uses the sample event rate (Murphy 1973 definition).
    # Using the definitional percentile (e.g. 0.01 for p99) gives severely negative
    # BSS when the verification period event rate differs from the climatological rate
    # (e.g. ERA5 threshold bias vs station observations, or climate non-stationarity).
    member_cols = _get_member_columns(df, f'{model_prefix}_member_')
    if not member_cols:
        return {}
    
    # Drop rows with any NaN in members or obs
    valid_cols = member_cols + ['obs_value']
    valid_mask = df[valid_cols].notna().all(axis=1)
    df_clean = df.loc[valid_mask]
    
    if len(df_clean) < 10:
        return {'n_samples': len(df_clean)}
    
    fcst, obs = _df_to_xarray(df_clean, member_cols)
    result = {'n_samples': len(df_clean)}

    # Resolve threshold: per-row array (from _obs_threshold column) or scalar
    if '_obs_threshold' in df_clean.columns:
        thr_arr = df_clean['_obs_threshold'].values  # per-row numpy array
        float_threshold = float(np.nanmean(thr_arr))
    elif isinstance(threshold, np.ndarray):
        thr_arr = threshold[valid_mask.values] if len(threshold) == len(df) else threshold
        float_threshold = float(np.nanmean(thr_arr))
    else:
        thr_arr = None
        float_threshold = float(threshold)

    def _exceeds(obs_vals):
        """Vectorised obs >= threshold comparison, supporting per-row thresholds."""
        if thr_arr is not None:
            return (obs_vals >= thr_arr) if event_type == 'above' else (obs_vals <= thr_arr)
        return (obs_vals >= float_threshold) if event_type == 'above' else (obs_vals <= float_threshold)
    
    # For basic stats, optionally restrict to extreme cases (obs exceeds threshold)
    if extreme_only_basic:
        extreme_mask = _exceeds(df_clean['obs_value'].values)
        df_extreme = df_clean.loc[df_clean.index[extreme_mask]]
        if len(df_extreme) >= 5:
            fcst_ext, obs_ext = _df_to_xarray(df_extreme, member_cols)
        else:
            fcst_ext, obs_ext = None, None
    else:
        fcst_ext, obs_ext = fcst, obs
    
    # Ensemble mean for basic stats
    if fcst_ext is not None:
        ens_mean_ext = fcst_ext.mean(dim='ensemble')
    
    # Basic continuous scores on ensemble mean (conditioned on extremes if requested)
    if 'ens_mean_bias' in selected_scores:
        result['ens_mean_bias'] = float((ens_mean_ext - obs_ext).mean()) if fcst_ext is not None else np.nan
    
    if 'ens_mean_mae' in selected_scores:
        result['ens_mean_mae'] = float(np.abs(ens_mean_ext - obs_ext).mean()) if fcst_ext is not None else np.nan
    
    if 'ens_mean_rmse' in selected_scores:
        result['ens_mean_rmse'] = float(np.sqrt(((ens_mean_ext - obs_ext) ** 2).mean())) if fcst_ext is not None else np.nan
    
    if 'ens_spread' in selected_scores:
        result['ens_spread'] = float(fcst_ext.std(dim='ensemble').mean()) if fcst_ext is not None else np.nan

    # Threshold-weighted spread-skill ratio: R = σ_extreme / RMSE_extreme
    # Both numerator and denominator are restricted to cases where obs crosses
    # the threshold, making this explicitly threshold-conditioned.
    # R = 1 is perfect; R < 1 → underdispersive; R > 1 → overdispersive.
    if 'extreme_spread_skill_ratio' in selected_scores:
        tw_mask = _exceeds(df_clean['obs_value'].values)
        df_tw = df_clean.loc[df_clean.index[tw_mask]]
        if len(df_tw) >= 5:
            tw_fcst_np = df_tw[member_cols].values  # (n_extreme, n_members)
            tw_obs_np = df_tw['obs_value'].values
            spread_tw = float(np.std(tw_fcst_np, ddof=1, axis=1).mean())
            ens_mean_tw = tw_fcst_np.mean(axis=1)
            rmse_tw = float(np.sqrt(np.mean((ens_mean_tw - tw_obs_np) ** 2)))
            result['extreme_spread_skill_ratio'] = spread_tw / rmse_tw if rmse_tw > 1e-10 else np.nan
        else:
            result['extreme_spread_skill_ratio'] = np.nan

    # CRPS (fair method — equivalent to Quaver's fcrps)
    if 'CRPS' in selected_scores or 'fCRPS' in selected_scores:
        try:
            crps_val = crps_for_ensemble(
                fcst, obs,
                ensemble_member_dim='ensemble',
                method='fair',
            )
            if 'CRPS' in selected_scores:
                result['CRPS'] = float(crps_val)
            if 'fCRPS' in selected_scores:
                result['fCRPS'] = float(crps_val)
        except Exception as e:
            print(f"    CRPS error: {e}")
            if 'CRPS' in selected_scores:
                result['CRPS'] = np.nan
            if 'fCRPS' in selected_scores:
                result['fCRPS'] = np.nan
    
    # Threshold-weighted CRPS for extremes.
    # Uses the chaining function v(x) = max(x, T) for the upper tail
    # (min for lower), which is mathematically equivalent to indicator weighting:
    # w(x) = 1 for x >= T, w(x) = 0 otherwise. This is a proper score.
    # When per-station thresholds are available, the chaining function is applied
    # row-by-row and fair CRPS is computed via _fair_crps_numpy.
    if 'twCRPS' in selected_scores:
        try:
            tail = 'upper' if event_type == 'above' else 'lower'
            if thr_arr is not None:
                # Per-station thresholds: apply chaining function per row
                fcst_np_tw = fcst.values  # (n_samples, n_members)
                obs_np_tw = obs.values    # (n_samples,)
                if event_type == 'above':
                    fcst_v = np.maximum(fcst_np_tw, thr_arr[:, None])
                    obs_v = np.maximum(obs_np_tw, thr_arr)
                else:
                    fcst_v = np.minimum(fcst_np_tw, thr_arr[:, None])
                    obs_v = np.minimum(obs_np_tw, thr_arr)
                twcrps_val = _fair_crps_numpy(fcst_v, obs_v)
            else:
                twcrps_val = float(tail_tw_crps_for_ensemble(
                    fcst, obs,
                    ensemble_member_dim='ensemble',
                    threshold=float_threshold,
                    tail=tail,
                    method='fair',
                ))
            result['twCRPS'] = float(twcrps_val)
        except Exception as e:
            print(f"    twCRPS error: {e}")
            result['twCRPS'] = np.nan
    
    # Brier score for ensemble
    # When per-station thresholds are available, exceedance probabilities and
    # binary observations are computed row-by-row using thr_arr.
    if 'Brier' in selected_scores:
        try:
            if thr_arr is not None:
                fcst_np_b = fcst.values  # (n_samples, n_members)
                obs_np_b = obs.values
                if event_type == 'above':
                    p_hat = (fcst_np_b >= thr_arr[:, None]).mean(axis=1)
                    o_ev = (obs_np_b >= thr_arr).astype(float)
                else:
                    p_hat = (fcst_np_b <= thr_arr[:, None]).mean(axis=1)
                    o_ev = (obs_np_b <= thr_arr).astype(float)
                brier_val = float(np.mean((p_hat - o_ev) ** 2))
            else:
                brier_val = float(brier_score_for_ensemble(
                    fcst, obs,
                    ensemble_member_dim='ensemble',
                    event_thresholds=[float_threshold],
                ).values)
            result['Brier'] = brier_val
        except Exception as e:
            print(f"    Brier score error: {e}")
            result['Brier'] = np.nan
    
    # Brier Skill Score: BSS = 1 - BS / BS_ref
    # BS_ref = p_c*(1-p_c) where p_c is the SAMPLE event frequency (Murphy 1973).
    # The definitional percentile rate (e.g. 0.01 for p99) can differ substantially
    # from the actual verification period rate, producing artificially negative BSS.
    if 'BSS' in selected_scores:
        try:
            obs_np_bss = obs.values
            p_c = float(np.mean(_exceeds(obs_np_bss)))
            bs_ref = p_c * (1.0 - p_c)
            if bs_ref > 1e-10:
                bs_val = result.get('Brier')
                if bs_val is None:
                    # Compute Brier if not already done, respecting per-station thresholds
                    if thr_arr is not None:
                        fcst_np_bss = fcst.values
                        obs_np_bss2 = obs.values
                        if event_type == 'above':
                            p_hat = (fcst_np_bss >= thr_arr[:, None]).mean(axis=1)
                            o_ev = (obs_np_bss2 >= thr_arr).astype(float)
                        else:
                            p_hat = (fcst_np_bss <= thr_arr[:, None]).mean(axis=1)
                            o_ev = (obs_np_bss2 <= thr_arr).astype(float)
                        bs_val = float(np.mean((p_hat - o_ev) ** 2))
                    else:
                        bv = brier_score_for_ensemble(
                            fcst, obs,
                            ensemble_member_dim='ensemble',
                            event_thresholds=[float_threshold],
                        )
                        bs_val = float(bv.values)
                result['BSS'] = float(1.0 - bs_val / bs_ref)
            else:
                result['BSS'] = np.nan  # event never or always occurs
        except Exception as e:
            print(f"    BSS error: {e}")
            result['BSS'] = np.nan
    
    # Threshold-weighted Quantile Score at multiple extreme tail quantile levels.
    # Reference: Taggart (2022, QJRMS) — "Evaluation of point forecasts for extreme events
    #   using consistent scoring functions".
    #
    # Formula: twQS_α(q̂_α, y; T) = ρ_α(v_T(y) − v_T(q̂_α))
    # where:
    #   ρ_α(u) = u·(α − 1_{u<0})          (pinball / quantile loss)
    #   v_T(x) = max(x−T, 0) for 'above'   (upper-tail chaining function)
    #   v_T(x) = max(T−x, 0) for 'below'   (lower-tail chaining function)
    #
    # α levels: [0.90, 0.95, 0.99] for 'above', [0.01, 0.05, 0.10] for 'below'.
    # The threshold T (per-station or global) is the configured extreme threshold.
    # Scores are named tw_quantile_score_q{label} (e.g. tw_quantile_score_q099).
    # A mean across levels is stored as 'tw_quantile_score' for bootstrap use.
    if 'tw_quantile_score' in selected_scores:
        # Build weight interval once (same for all α levels)
        if thr_arr is not None:
            thr_da = xr.DataArray(thr_arr, dims=['sample'])
            if event_type == 'above':
                interval_where_one = (thr_da, np.inf)
            else:
                interval_where_one = (-np.inf, thr_da)
        else:
            if event_type == 'above':
                interval_where_one = (float_threshold, np.inf)
            else:
                interval_where_one = (-np.inf, float_threshold)

        twqs_vals = []
        for alpha in (tw_qs_alphas or _TW_QS_ALPHAS)[event_type]:
            try:
                q_fcst_a = fcst.quantile(alpha, dim='ensemble')
                twqs_a = float(tw_quantile_score(
                    q_fcst_a, obs, alpha,
                    interval_where_one=interval_where_one,
                ))
            except Exception as e:
                print(f"    tw_quantile_score α={alpha:.2f} error: {e}")
                twqs_a = np.nan
            lbl = _alpha_label(alpha)
            result[f'tw_quantile_score_{lbl}'] = twqs_a
            twqs_vals.append(twqs_a)
        # Mean across all levels → used as summary and for bootstrap significance test
        valid_vals = [v for v in twqs_vals if not np.isnan(v)]
        result['tw_quantile_score'] = float(np.mean(valid_vals)) if valid_vals else np.nan

    # Diagonal Score (observation-climatology-based)
    # Following VTB xmetrics.diagonal() methodology:
    #   1. Derive observation climatology percentiles from the obs pool
    #   2. For each tau, check if obs exceeds its climatological percentile
    #   3. Compute ensemble probability of exceeding that same percentile
    #   4. Score: obs_ev * (p <= 1-tau) * tau + (1-obs_ev) * (p > 1-tau) * (1-tau)
    if 'diagonal_score' in selected_scores:
        try:
            result['diagonal_score'] = float(_compute_diagonal_score_clim(fcst, obs))
        except Exception as e:
            print(f"    diagonal_score error: {e}")
            result['diagonal_score'] = np.nan

    return result


def _compute_diagonal_score_clim(fcst, obs, ncat=20):
    """Diagonal score using observation climatology percentiles.
    
    Follows VTB xmetrics.diagonal() methodology.
    Uses pooled observation values as the climatology source.
    
    Args:
        fcst: xr.DataArray (sample, ensemble)
        obs: xr.DataArray (sample,)
        ncat: number of categories (VTB default=20 → 19 tau levels: 5%,10%,...,95%)
    
    Returns:
        float: mean diagonal score
    """
    obs_np = obs.values
    fcst_np = fcst.values  # (n_samples, n_members)
    n_samples, n_members = fcst_np.shape
    
    # Valid mask
    valid = ~np.isnan(obs_np) & ~np.any(np.isnan(fcst_np), axis=1)
    if valid.sum() < 10:
        return np.nan
    obs_v = obs_np[valid]
    fcst_v = fcst_np[valid]
    
    # Observation climatology percentiles (tau = 1/ncat, 2/ncat, ..., (ncat-1)/ncat)
    taus = np.arange(1, ncat, dtype=float) / ncat  # e.g. [0.05, 0.10, ..., 0.95] for ncat=20
    obs_clim_pctls = np.percentile(obs_v, taus * 100)  # shape: (ncat-1,)
    
    # For each tau level, compute the score contribution
    ds_total = 0.0
    n_valid_taus = 0
    
    for k, tau in enumerate(taus):
        threshold_k = obs_clim_pctls[k]
        
        # Check if threshold is non-distinct (all same value → skip)
        if np.all(obs_v == threshold_k):
            continue
        
        # obs_ev: did the observation exceed the climatological percentile?
        obs_ev = (obs_v > threshold_k).astype(float)  # (n_valid,)
        
        # p: ensemble probability of exceeding the same percentile
        # Using obs climatology as both obs and forecast climatology reference
        # (same as VTB when forecast_climatology=None → uses observation_climatology)
        p = (fcst_v > threshold_k).sum(axis=1).astype(float) / n_members  # (n_valid,)
        
        # Score formula from VTB:
        # dst = obs_ev * (p <= (1-tau)) * tau + (1-obs_ev) * (p > (1-tau)) * (1-tau)
        dst = obs_ev * (p <= (1.0 - tau)) * tau + (1.0 - obs_ev) * (p > (1.0 - tau)) * (1.0 - tau)
        
        ds_total += dst.sum()
        n_valid_taus += 1
    
    if n_valid_taus == 0:
        return np.nan
    
    # Average: sum over taus, then divide by n_valid_taus and n_valid_samples
    return ds_total / (n_valid_taus * valid.sum())


def _compute_single_score(score_name, fcst, obs, threshold, event_type, tw_qs_alphas=None):
    """Compute a single ensemble score. Returns float or np.nan."""
    # Resolve scalar threshold for library functions
    float_thr = float(np.nanmean(threshold)) if isinstance(threshold, np.ndarray) else float(threshold)
    def _exceeds(obs_vals):
        if isinstance(threshold, np.ndarray):
            return (obs_vals >= threshold) if event_type == 'above' else (obs_vals <= threshold)
        return (obs_vals >= float_thr) if event_type == 'above' else (obs_vals <= float_thr)
    try:
        if score_name == 'CRPS':
            return float(crps_for_ensemble(fcst, obs, ensemble_member_dim='ensemble', method='fair'))
        elif score_name == 'twCRPS':
            tail = 'upper' if event_type == 'above' else 'lower'
            if isinstance(threshold, np.ndarray):
                fcst_np = fcst.values
                obs_np = obs.values
                if event_type == 'above':
                    fcst_v = np.maximum(fcst_np, threshold[:, None])
                    obs_v = np.maximum(obs_np, threshold)
                else:
                    fcst_v = np.minimum(fcst_np, threshold[:, None])
                    obs_v = np.minimum(obs_np, threshold)
                return _fair_crps_numpy(fcst_v, obs_v)
            return float(tail_tw_crps_for_ensemble(
                fcst, obs, ensemble_member_dim='ensemble',
                threshold=float_thr, tail=tail, method='fair'))
        elif score_name == 'Brier':
            if isinstance(threshold, np.ndarray):
                fcst_np = fcst.values
                obs_np = obs.values
                if event_type == 'above':
                    p_hat = (fcst_np >= threshold[:, None]).mean(axis=1)
                    o_ev = (obs_np >= threshold).astype(float)
                else:
                    p_hat = (fcst_np <= threshold[:, None]).mean(axis=1)
                    o_ev = (obs_np <= threshold).astype(float)
                return float(np.mean((p_hat - o_ev) ** 2))
            return float(brier_score_for_ensemble(
                fcst, obs, ensemble_member_dim='ensemble',
                event_thresholds=[float_thr]).values)
        elif score_name == 'BSS':
            obs_np = obs.values
            p_c = float(np.mean(_exceeds(obs_np)))
            bs_ref = p_c * (1.0 - p_c)
            if bs_ref < 1e-10:
                return np.nan
            bs_val = _compute_single_score('Brier', fcst, obs, threshold, event_type)
            if np.isnan(bs_val):
                return np.nan
            return float(1.0 - bs_val / bs_ref)
        elif score_name == 'tw_quantile_score' or score_name.startswith('tw_quantile_score_q'):
            # For the mean 'tw_quantile_score', use the most extreme α level as representative.
            # For per-level names (tw_quantile_score_q090 etc.), parse the exact α.
            if score_name == 'tw_quantile_score':
                alpha = (tw_qs_alphas or _TW_QS_ALPHAS)[event_type][-1]
            else:
                alpha = _alpha_from_label(score_name.split('_')[-1])
            q_fcst = fcst.quantile(alpha, dim='ensemble')
            if isinstance(threshold, np.ndarray):
                thr_da = xr.DataArray(threshold, dims=['sample'])
                if event_type == 'above':
                    interval_where_one = (thr_da, np.inf)
                else:
                    interval_where_one = (-np.inf, thr_da)
            else:
                if event_type == 'above':
                    interval_where_one = (float_thr, np.inf)
                else:
                    interval_where_one = (-np.inf, float_thr)
            return float(tw_quantile_score(q_fcst, obs, alpha, interval_where_one=interval_where_one))
        elif score_name == 'fCRPS':
            return float(crps_for_ensemble(fcst, obs, ensemble_member_dim='ensemble', method='fair'))
        elif score_name == 'diagonal_score':
            return float(_compute_diagonal_score_clim(fcst, obs))
        elif score_name == 'extreme_spread_skill_ratio':
            obs_np = obs.values
            fcst_np = fcst.values
            mask = _exceeds(obs_np)
            obs_ext = obs_np[mask]
            fcst_ext = fcst_np[mask]
            if len(obs_ext) < 5:
                return np.nan
            spread = float(np.std(fcst_ext, ddof=1, axis=1).mean())
            ens_mean = fcst_ext.mean(axis=1)
            rmse = float(np.sqrt(np.mean((ens_mean - obs_ext) ** 2)))
            return spread / rmse if rmse > 1e-10 else np.nan
    except Exception:
        return np.nan
    return np.nan


def _per_case_score_diff(score_name, fc1_np, fc2_np, obs_np, thr_np, event_type, tw_qs_alphas=None):
    """Compute per-case score difference d_i = score2_i - score1_i.

    For proper scoring rules that decompose case-by-case (CRPS, twCRPS, Brier,
    quantile_score, ens_mean_*), this avoids threshold resampling in bootstrap
    and is both faster and more statistically correct.
    Returns a 1-D array of shape (n,). Returns None if not supported.
    """
    n, m = fc1_np.shape

    if score_name in ('CRPS', 'fCRPS'):
        # Fair CRPS per case: E|X-y| - (1/2)*E|X-X'|
        def _crps_per_case(fc, obs):
            term1 = np.abs(fc - obs[:, None]).mean(axis=1)
            s = np.sort(fc, axis=1)
            w = 2 * np.arange(m) - m + 1
            term2 = (s * w[None, :]).sum(axis=1) / (m * (m - 1))
            return term1 - term2
        return _crps_per_case(fc2_np, obs_np) - _crps_per_case(fc1_np, obs_np)

    elif score_name == 'twCRPS':
        # Fair tail-weighted CRPS per case using chaining function
        def _twcrps_per_case(fc, obs, thr):
            if event_type == 'above':
                fc_v = np.maximum(fc, thr[:, None])
                obs_v = np.maximum(obs, thr)
            else:
                fc_v = np.minimum(fc, thr[:, None])
                obs_v = np.minimum(obs, thr)
            term1 = np.abs(fc_v - obs_v[:, None]).mean(axis=1)
            s = np.sort(fc_v, axis=1)
            w = 2 * np.arange(m) - m + 1
            term2 = (s * w[None, :]).sum(axis=1) / (m * (m - 1))
            return term1 - term2
        return _twcrps_per_case(fc2_np, obs_np, thr_np) - _twcrps_per_case(fc1_np, obs_np, thr_np)

    elif score_name == 'Brier':
        def _brier_per_case(fc, obs, thr):
            if event_type == 'above':
                p_hat = (fc >= thr[:, None]).mean(axis=1)
                o_ev = (obs >= thr).astype(float)
            else:
                p_hat = (fc <= thr[:, None]).mean(axis=1)
                o_ev = (obs <= thr).astype(float)
            return (p_hat - o_ev) ** 2
        return _brier_per_case(fc2_np, obs_np, thr_np) - _brier_per_case(fc1_np, obs_np, thr_np)

    elif score_name == 'tw_quantile_score' or score_name.startswith('tw_quantile_score_q'):
        # Threshold-weighted pinball loss per case (Taggart 2022):
        #   twQS_α_i = ρ_α(v_T(y_i) − v_T(q̂_{α,i}))
        # where v_T is the tail chaining function and ρ_α is the pinball loss.
        # For 'tw_quantile_score' (mean), use the most extreme α as representative.
        if score_name == 'tw_quantile_score':
            alpha = (tw_qs_alphas or _TW_QS_ALPHAS)[event_type][-1]
        else:
            alpha = _alpha_from_label(score_name.split('_')[-1])

        def _twqs_per_case(fc, obs, thr):
            q_hat = np.quantile(fc, alpha, axis=1)
            if event_type == 'above':
                obs_v = np.maximum(obs - thr, 0.0)
                q_v   = np.maximum(q_hat - thr, 0.0)
                # For 'above': v_T(x)=max(x-T,0) → larger = warmer = more extreme.
                # err > 0 means obs more extreme than q̂ (miss) → penalise with alpha (≈1).
                eff_alpha = alpha
            else:
                obs_v = np.maximum(thr - obs, 0.0)
                q_v   = np.maximum(thr - q_hat, 0.0)
                # For 'below': v_T(x)=max(T-x,0) → larger = colder = more extreme.
                # err > 0 means obs more extreme than q̂ (miss) → must still penalise heavily.
                # The chaining inverts polarity vs the standard QS, so we need eff_alpha = 1-alpha
                # (e.g. alpha=0.01 for q001 → eff_alpha=0.99 to give large miss penalty).
                eff_alpha = 1.0 - alpha
            err = obs_v - q_v
            return np.where(err >= 0, eff_alpha * err, (eff_alpha - 1.0) * err)

        return _twqs_per_case(fc2_np, obs_np, thr_np) - _twqs_per_case(fc1_np, obs_np, thr_np)

    elif score_name == 'ens_mean_bias':
        diff = (fc2_np.mean(axis=1) - obs_np) - (fc1_np.mean(axis=1) - obs_np)
        return diff  # = fc2_mean - fc1_mean

    elif score_name == 'ens_mean_mae':
        return np.abs(fc2_np.mean(axis=1) - obs_np) - np.abs(fc1_np.mean(axis=1) - obs_np)

    elif score_name == 'ens_mean_rmse':
        return (fc2_np.mean(axis=1) - obs_np) ** 2 - (fc1_np.mean(axis=1) - obs_np) ** 2

    # Scores that don't decompose per-case (BSS, extreme_spread_skill_ratio, ens_spread)
    return None


def bootstrap_paired_significance(df, threshold, event_type, score_name,
                                  n_bootstrap=1000, confidence=0.95, max_samples=200000,
                                  tw_qs_alphas=None):
    """Test if the difference between fc1 and fc2 is significant using paired bootstrap.

    Bootstraps the per-case score DIFFERENCES d_i = score2_i - score1_i and computes
    the CI of the mean difference. This is the standard paired one-sample bootstrap test:
    both models are evaluated on the exact same cases, per case, avoiding any threshold
    resampling issues and making the CI directly comparable to the observed mean diff.

    Falls back to full-score resampling for scores that don't decompose per case
    (BSS, extreme_spread_skill_ratio, ens_spread).

    Returns (is_significant, diff_ci_low, diff_ci_high).
    """
    fc1_member_cols = _get_member_columns(df, 'fc1_member_')
    fc2_member_cols = _get_member_columns(df, 'fc2_member_')
    valid_cols = fc1_member_cols + fc2_member_cols + ['obs_value']
    valid_mask = df[valid_cols].notna().all(axis=1)
    df_clean = df.loc[valid_mask].reset_index(drop=True)

    n = len(df_clean)
    if n < 20:
        return False, np.nan, np.nan

    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        df_clean = df_clean.iloc[idx].reset_index(drop=True)
        n = max_samples

    fc1_np = df_clean[fc1_member_cols].values
    fc2_np = df_clean[fc2_member_cols].values
    obs_np = df_clean['obs_value'].values

    # Per-row thresholds: use stored column (set by _process_single_day for local_obs_climatology)
    if '_obs_threshold' in df_clean.columns:
        thr_np = df_clean['_obs_threshold'].values
    elif isinstance(threshold, np.ndarray):
        thr_np = threshold
    else:
        thr_np = np.full(n, float(threshold))

    alpha = 1 - confidence

    # --- Per-case difference path (preferred) ---
    d_i = _per_case_score_diff(score_name, fc1_np, fc2_np, obs_np, thr_np, event_type, tw_qs_alphas)
    if d_i is not None:
        valid = np.isfinite(d_i)
        d_i = d_i[valid]
        if len(d_i) < 20:
            return False, np.nan, np.nan
        # Bootstrap the mean of d_i
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            boot_means[b] = d_i[np.random.randint(0, len(d_i), len(d_i))].mean()
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        is_significant = not (ci_low <= 0.0 <= ci_high)
        return is_significant, ci_low, ci_high

    # --- Fallback: resample full data and recompute score (BSS, spread, etc.) ---
    scalar_threshold = float(np.nanmean(thr_np))
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        obs_b = xr.DataArray(obs_np[idx], dims=['sample'])
        fcst1_b = xr.DataArray(fc1_np[idx], dims=['sample', 'ensemble'])
        fcst2_b = xr.DataArray(fc2_np[idx], dims=['sample', 'ensemble'])
        boot_thr = thr_np[idx]

        val1 = _compute_single_score(score_name, fcst1_b, obs_b, boot_thr, event_type, tw_qs_alphas)
        val2 = _compute_single_score(score_name, fcst2_b, obs_b, boot_thr, event_type, tw_qs_alphas)

        if not (np.isnan(val1) or np.isnan(val2)):
            boot_diffs.append(val2 - val1)

    if len(boot_diffs) < n_bootstrap * 0.3:
        return False, np.nan, np.nan

    ci_low = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    is_significant = not (ci_low <= 0.0 <= ci_high)
    return is_significant, ci_low, ci_high


def _process_single_day(day, day_data, threshold, event_type, all_scores, selected_scores,
                        threshold_method, threshold_percentile,
                        bootstrap_enabled, n_boot, confidence, extreme_only_basic=False,
                        tw_qs_alphas=None):
    """Process scores and bootstrap for a single forecast day (used by parallel executor)."""
    import pandas as pd
    steps_in_day = sorted(day_data['step'].unique())
    step_range = f"{int(min(steps_in_day))}-{int(max(steps_in_day))}h"

    day_threshold = threshold
    if threshold_method == 'dataset_climatology':
        day_threshold = np.percentile(day_data['obs_value'].dropna(), threshold_percentile)
    elif isinstance(threshold, pd.Series):
        # Per-station threshold from local_obs_climatology — subset to rows with valid values
        day_thr_series = threshold.loc[day_data.index]
        valid_thr = day_thr_series.notna()
        if not valid_thr.any():
            return None
        day_data = day_data.loc[valid_thr].copy()
        day_threshold = day_thr_series.loc[valid_thr].values  # numpy array
        # Store per-row thresholds as a column so calculate_ensemble_scores can use them
        day_data['_obs_threshold'] = day_threshold

    # Count exceedances using per-row thresholds (or scalar)
    obs_vals = day_data['obs_value'].values
    if isinstance(day_threshold, np.ndarray):
        n_exceedances = int(np.sum(obs_vals >= day_threshold) if event_type == 'above'
                           else np.sum(obs_vals <= day_threshold))
        threshold_scalar = float(np.nanmean(day_threshold))
    else:
        n_exceedances = int(np.sum(obs_vals >= day_threshold) if event_type == 'above'
                           else np.sum(obs_vals <= day_threshold))
        threshold_scalar = float(day_threshold)

    day_fc1 = calculate_ensemble_scores(day_data, day_threshold, event_type, all_scores, 'fc1',
                                        extreme_only_basic=extreme_only_basic,
                                        tw_qs_alphas=tw_qs_alphas)
    day_fc2 = calculate_ensemble_scores(day_data, day_threshold, event_type, all_scores, 'fc2',
                                        extreme_only_basic=extreme_only_basic,
                                        tw_qs_alphas=tw_qs_alphas)

    day_scores = {
        'forecast_day': day,
        'step_range': step_range,
        'lead_time': int(np.mean(steps_in_day)),
        'threshold': threshold_scalar,
        'n_exceedances': n_exceedances,
    }

    for score_name in all_scores:
        v1 = day_fc1.get(score_name, np.nan)
        v2 = day_fc2.get(score_name, np.nan)
        day_scores[f'{score_name}_fc1'] = v1
        day_scores[f'{score_name}_fc2'] = v2
        day_scores[f'{score_name}_diff'] = v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan

    # Collect per-level tw_quantile_score keys produced by calculate_ensemble_scores.
    # These keys (e.g. tw_quantile_score_q090) are not in all_scores but are returned
    # by calculate_ensemble_scores when 'tw_quantile_score' is requested.
    if 'tw_quantile_score' in all_scores:
        for _key in sorted(k for k in day_fc1 if k.startswith('tw_quantile_score_q')):
            v1 = day_fc1.get(_key, np.nan)
            v2 = day_fc2.get(_key, np.nan)
            day_scores[f'{_key}_fc1'] = v1
            day_scores[f'{_key}_fc2'] = v2
            day_scores[f'{_key}_diff'] = (
                v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan
            )

    if bootstrap_enabled:
        for score_name in selected_scores:
            is_sig, diff_ci_low, diff_ci_high = bootstrap_paired_significance(
                day_data, day_threshold, event_type, score_name, n_boot, confidence,
                tw_qs_alphas=tw_qs_alphas)
            day_scores[f'{score_name}_is_significant'] = is_sig
            day_scores[f'{score_name}_diff_ci_low'] = diff_ci_low
            day_scores[f'{score_name}_diff_ci_high'] = diff_ci_high

    day_scores['n_samples'] = day_fc1.get('n_samples', 0)
    return day_scores


def score_single_day_file(config, day, day_data, threshold, event_type, model_names):
    """Score a single forecast-day DataFrame (from one per-day parquet file).

    Called by the streaming run.py path that loads one day file at a time.
    Returns the same day_scores dict that _process_single_day returns, or None.
    """
    cfg = config.get('scores', {})
    selected_scores = cfg.get('ensemble', ['CRPS', 'twCRPS', 'Brier'])
    extra_scores = ['ens_mean_bias', 'ens_mean_mae', 'ens_mean_rmse', 'ens_spread',
                    'extreme_spread_skill_ratio']
    all_scores = selected_scores + extra_scores

    bootstrap_cfg = config.get('bootstrap', {})
    bootstrap_enabled = bootstrap_cfg.get('enabled', False)
    n_boot = bootstrap_cfg.get('n_samples', 100)
    confidence = bootstrap_cfg.get('confidence_level', 0.95)

    extreme_only_basic = cfg.get('extreme_only_basic_scores', False)
    tw_qs_alphas = cfg.get('tw_qs_alphas')

    _thr_cfg = config.get('threshold', {})
    threshold_method = _thr_cfg.get('method', 'fixed')
    threshold_percentile = (
        _thr_cfg.get('local_obs_climatology', {}).get('percentile') or
        _thr_cfg.get('dataset_climatology', {}).get('percentile') or
        _thr_cfg.get('station_climatology', {}).get('percentile') or
        99
    )

    return _process_single_day(
        day, day_data, threshold, event_type,
        all_scores, selected_scores,
        threshold_method, threshold_percentile,
        bootstrap_enabled, n_boot, confidence, extreme_only_basic,
        tw_qs_alphas=tw_qs_alphas,
    )


def aggregate_day_results(day_results, model_names):
    """Build (overall_scores, results) from a list of per-day score dicts.

    This is the counterpart of run_step6_ensemble for the streaming path:
    run_step6_ensemble builds day_results internally; here we receive them
    already computed (one per day file) and just aggregate.
    """
    if not day_results:
        return {
            'model1_name': model_names['fc1_name'],
            'model2_name': model_names['fc2_name'],
            'n_samples': 0,
        }, {'by_leadtime': pd.DataFrame()}

    day_results_sorted = sorted(day_results, key=lambda x: x['forecast_day'])
    results = {'by_leadtime': pd.DataFrame(day_results_sorted)}
    df_days = results['by_leadtime']

    # Detect score names from columns ending with '_fc1'
    score_names = [c[:-4] for c in df_days.columns
                   if c.endswith('_fc1') and not c.startswith('n_')]

    overall_scores = {
        'model1_name': model_names['fc1_name'],
        'model2_name': model_names['fc2_name'],
    }
    for score_name in score_names:
        col_fc1 = f'{score_name}_fc1'
        col_fc2 = f'{score_name}_fc2'
        if col_fc1 in df_days.columns:
            overall_scores[col_fc1] = df_days[col_fc1].mean()
            overall_scores[col_fc2] = df_days[col_fc2].mean()
            overall_scores[f'{score_name}_diff'] = (
                overall_scores[col_fc2] - overall_scores[col_fc1]
            )
        else:
            overall_scores[col_fc1] = np.nan
            overall_scores[col_fc2] = np.nan
            overall_scores[f'{score_name}_diff'] = np.nan

    overall_scores['n_samples'] = (
        int(df_days['n_samples'].sum()) if 'n_samples' in df_days.columns else 0
    )

    return overall_scores, results


def run_step6_ensemble(config, data, threshold, event_type, model_names):
    """
    Execute Step 6 for ensemble verification.
    Returns (overall_scores, results) in the same format as det_scores.run_step6.
    """
    print("\n" + "="*80)
    print("STEP 6: CALCULATE ENSEMBLE VERIFICATION SCORES")
    print("="*80)
    
    cfg = config.get('scores', {})
    selected_scores = cfg.get('ensemble', ['CRPS', 'twCRPS', 'Brier'])
    
    # Also compute basic ensemble mean statistics
    extra_scores = ['ens_mean_bias', 'ens_mean_mae', 'ens_mean_rmse', 'ens_spread', 'extreme_spread_skill_ratio']
    all_scores = selected_scores + extra_scores
    
    print(f"\nComparing two ensemble forecast models:")
    print(f"  Model 1: {model_names['fc1_name']}")
    print(f"  Model 2: {model_names['fc2_name']}")
    import pandas as pd
    if isinstance(threshold, pd.Series):
        thr_display = f"per-station (mean {float(threshold.mean()):.4f})"
    else:
        thr_display = f"{float(threshold):.4f}"
    print(f"  Threshold: {thr_display} ({event_type})")
    print(f"\nEnsemble scores to calculate:")
    for s in all_scores:
        print(f"  - {s}")
    
    # Bootstrap config
    bootstrap_cfg = config.get('bootstrap', {})
    bootstrap_enabled = bootstrap_cfg.get('enabled', False)
    n_boot = bootstrap_cfg.get('n_samples', 100)
    confidence = bootstrap_cfg.get('confidence_level', 0.95)
    
    # Conditional verification option: restrict bias/MAE/RMSE/spread to extreme cases
    extreme_only_basic = cfg.get('extreme_only_basic_scores', False)
    tw_qs_alphas = cfg.get('tw_qs_alphas')
    if extreme_only_basic:
        print(f"  (bias/MAE/RMSE/spread computed on extreme cases only: obs {event_type} threshold)")
    
    # Skip overall scores — only per-day (heatmap box) scores are needed
    print("\n  Skipping overall scores (only per-day scores needed for heatmap)")
    scores_fc1 = {}
    scores_fc2 = {}
    
    # Scores by forecast day (parallelized across days)
    print("\n" + "-"*40)
    print("Scores by Forecast Day (parallel):")
    print("-"*40)
    
    if 'forecast_day' not in data.columns:
        data = data.copy()
        data['forecast_day'] = ((data['step'] - 1) / 24).astype(int) + 1
    
    threshold_method = config['threshold']['method']
    # Resolve threshold percentile from whichever method is configured.
    # Used for BSS reference: p_c = percentile/100 by climatological definition.
    _thr_cfg = config['threshold']
    threshold_percentile = (
        _thr_cfg.get('local_obs_climatology', {}).get('percentile') or
        _thr_cfg.get('dataset_climatology', {}).get('percentile') or
        _thr_cfg.get('station_climatology', {}).get('percentile') or
        99
    )
    
    days = sorted(data['forecast_day'].unique())
    # Cap at 4 workers: each worker receives a copy of ~600 MB day_data, so
    # spawning more than 4 at once can spike memory by several GB.
    n_workers = min(4, len(days), max(1, os.cpu_count() or 1))
    print(f"\n  Processing {len(days)} forecast days using {n_workers} parallel workers...")
    
    results_by_day = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for day in days:
                day_data = data[data['forecast_day'] == day].copy()
                future = executor.submit(
                    _process_single_day,
                    day, day_data, threshold, event_type,
                    all_scores, selected_scores,
                    threshold_method, threshold_percentile,
                    bootstrap_enabled, n_boot, confidence, extreme_only_basic,
                    tw_qs_alphas
                )
                futures[future] = day
            
            for future in as_completed(futures):
                day = futures[future]
                try:
                    day_scores = future.result()
                    if day_scores is None:
                        print(f"  Day {day}: skipped (no valid thresholds)")
                        continue
                    results_by_day.append(day_scores)
                    print(f"  Day {day} ({day_scores['step_range']}): n={day_scores['n_samples']:,} ✓")
                except Exception as e:
                    print(f"  Day {day}: ERROR - {e}")
    except Exception as e:
        # Fallback to sequential processing
        print(f"  Parallel processing failed ({e}), falling back to sequential...")
        results_by_day = []
        for day in days:
            day_data = data[data['forecast_day'] == day].copy()
            day_scores = _process_single_day(
                day, day_data, threshold, event_type,
                all_scores, selected_scores,
                threshold_method, threshold_percentile,
                bootstrap_enabled, n_boot, confidence, extreme_only_basic,
                tw_qs_alphas
            )
            if day_scores is None:
                print(f"  Day {day}: skipped (no valid thresholds)")
                continue
            results_by_day.append(day_scores)
            print(f"  Day {day} ({day_scores['step_range']}): n={day_scores['n_samples']:,}")
    
    results_by_day.sort(key=lambda x: x['forecast_day'])
    results = {'by_leadtime': pd.DataFrame(results_by_day)}
    
    # Build overall_scores from per-day averages (no separate overall computation)
    overall_scores = {
        'model1_name': model_names['fc1_name'],
        'model2_name': model_names['fc2_name'],
    }
    df_days = results['by_leadtime']
    for score_name in all_scores:
        col_fc1 = f'{score_name}_fc1'
        col_fc2 = f'{score_name}_fc2'
        if col_fc1 in df_days.columns:
            overall_scores[col_fc1] = df_days[col_fc1].mean()
            overall_scores[col_fc2] = df_days[col_fc2].mean()
            overall_scores[f'{score_name}_diff'] = overall_scores[col_fc2] - overall_scores[col_fc1]
        else:
            overall_scores[col_fc1] = np.nan
            overall_scores[col_fc2] = np.nan
            overall_scores[f'{score_name}_diff'] = np.nan
    
    overall_scores['n_samples'] = int(df_days['n_samples'].sum()) if 'n_samples' in df_days.columns else 0
    
    print("\n✓ Step 6 (ensemble) complete")
    return overall_scores, results
