# Code Review Guide: local_obs_climatology Method

## Purpose of This Document

This document is intended to help a reviewer audit the correctness of the
`local_obs_climatology` scoring pipeline in `scorecards4extremes`. For each pipeline
step it describes: what the code is supposed to do, where that code lives, what to
check, and what bugs to look for.

---

## Pipeline Overview

```
Parquet files (sub-daily)
    │
    ▼  filter.run_step4()           ← filter.py
Step 4: coastal + QC + season + terrain filtering
    │
    ▼  threshold.run_step5()        ← threshold.py  _compute_local_obs_climatology_threshold()
Step 5: per-station threshold via nearest-neighbour clim match
    │
    ▼  run._aggregate_to_daily_mean()   ← run.py
Step 5b: aggregate sub-daily → daily means; realign threshold Series
    │
    ▼  det_scores.run_step6()       ← det_scores.py   (deterministic)
    ▼  ens_scores.run_step6_ensemble()  ← ens_scores.py  (ensemble)
Step 6: compute scores on daily-mean data with per-station thresholds
    │
    ▼  plot.run_step9()             ← plot.py
Step 9: heatmap of relative % difference fc2 vs fc1
```

The method is activated in the YAML config:
```yaml
threshold:
  method: local_obs_climatology
  local_obs_climatology:
    path: /path/to/clim/files
    parameter: 2t
    percentile: 99
    window_days: 1
    n_years: 20
    first_year: 2005
    last_year: 2024
    min_availability_pct: 65
    max_match_dist: 0.1
```

---

## Step 4 — Filtering (`filter.py :: run_step4`)

### What it should do
Apply filters in this exact order, all **before** the threshold is computed:

1. Coastal filter: `lsm > coastal_lsm_threshold` (default 0.9)
2. Variable-specific QC (2t: −60 to +60°C applied to obs, fc1, fc2)
3. Season filter: by **init date month** from `data['date']` (format YYYYMMDD)
4. Orography filter: `sdfor` in `[lo, hi)` per terrain category

### What to check in the code
- Filters are applied in the order above, not after aggregation.
- The `lsm` column is present in the parquet (added during extraction via auxiliary fields).
- Season filtering reads `data['date']`, not `data['valid_time']` — init date, not valid date.
- The orography loop reloads data fresh per terrain type (`filter.run_step4` is called
  once per `orog_type` inside the loop in `run.py`).

### Common bugs to look for
- ❌ Coastal filter omitted → n_samples ~5× too large for flat terrain (~80% of flat
  stations are coastal).
- ❌ Filters applied after aggregation → terrain/season boundaries cross between sub-daily
  steps, contaminating group means.
- ❌ Season filter uses valid_time instead of date → different rows excluded at long leads.
- ❌ QC applied only to obs but not to fc1/fc2 → forecast outliers remain.

---

## Step 5 — Threshold Computation (`threshold.py :: _compute_local_obs_climatology_threshold`)

### What it should do
For each row in the filtered data, find the nearest climatology station (within
`max_match_dist` degrees) and look up the `q{percentile}` value for the **valid-date
month** (= init date + step hours). Return a `pd.Series` aligned to the input index,
with `NaN` for unmatched stations.

### What to check in the code
- **Valid-date month, not init-date month.** The code derives:
  ```python
  valid_dates = init_dates + pd.to_timedelta(data['step'], unit='h')
  months = valid_dates.dt.month
  ```
  Check that `data['step']` contains hours (not days or seconds).
- **KDTree query uses lat/lon, not station_id.** Station IDs in the parquet are
  sequential integers (`S168` etc.), not WMO IDs — they cannot be used for matching.
- **NaN threshold = excluded entirely.** After matching, `threshold.notna()` rows only
  enter scoring. Verify this by checking that `n_samples` in the CSV equals the number
  of rows with non-NaN thresholds after Step 5b.
- The returned Series must share the same index as the filtered DataFrame passed in.
  A misaligned index causes silently wrong threshold assignment.

### Common bugs to look for
- ❌ Using init-date month → wrong seasonal climatology at long lead times (day 7+).
- ❌ Using station_id for matching → most stations get NaN (IDs don't match WMO IDs).
- ❌ Threshold Series index not preserved after filtering → misaligned assignment in Step 5b.
- ❌ `max_match_dist` too large → distant stations matched with wrong climate.

---

## Step 5b — Daily Aggregation (`run.py :: _aggregate_to_daily_mean`)

### What it should do
Only triggered when `threshold.method == 'local_obs_climatology'` AND
`lead_time_frequency < 24`. Groups by `(lat, lon, date, forecast_day)` and takes the
mean of all sub-daily steps for obs, fc1, fc2 (and ensemble members). Realigns the
threshold Series to the aggregated index.

### Why this step is needed
The local obs climatology is built from **daily-mean** obs. Scoring sub-daily values
against a daily-mean percentile threshold is inconsistent — the forecast distribution
being scored must match the distribution the climatological percentile was derived from.

### What to check in the code
- **Called AFTER** `run_step5` (threshold computed on sub-daily data, then aggregated).
- **Threshold propagated via `.first()`** per group — all sub-daily steps for a given
  (lat, lon, date, forecast_day) share the same threshold value, so `.first()` is correct.
  Verify: `df2['_thr'].groupby(group_cols).nunique().max() == 1`.
- **Canonical step assignment** after aggregation:
  ```python
  agg['step'] = forecast_day * 24 - (24 - lead_time_frequency) // 2
  ```
  Without this, mean step varies between terrain types (complex terrain may be missing
  step 24), causing `det_scores.run_step6` to produce different `lead_time` values per
  orog type and confusing the heatmap x-axis.
- Result: N sub-daily rows → N/4 daily rows for 6-hourly data.

### Common bugs to look for
- ❌ Aggregation called BEFORE `run_step5` → threshold computed on already-aggregated
  data; valid-date month calculation is wrong (mean step ≠ any real step).
- ❌ Threshold realigned using `.mean()` instead of `.first()` → numerically identical
  here (same value repeated) but fragile if threshold ever varies within a day.
- ❌ Step 5b skipped when `lead_time_frequency == 24` → correct, no aggregation needed.
- ❌ Step 5b triggered for non-`local_obs_climatology` methods → would aggregate before
  threshold computation for `fixed` method, which is wrong.

---

## Step 6 — Score Computation

### Data state entering Step 6
One row per `(lat, lon, date, forecast_day)`. The `_obs_threshold` column (or a
`pd.Series` passed as `threshold`) holds T_i per row. NaN-threshold rows were excluded
during the `pd.Series.notna()` filter in `run_step6`.

### Categorical Scores: ETS, PSS, POD, FAR (`det_scores.py :: calculate_ets`, `calculate_pss`, etc.)

**What they should do:** Binarise each row element-wise against its own T_i, then pool
all rows for a given forecast day into one contingency table.

**Code path in `det_scores.run_step6`:**
```python
# When threshold is a pd.Series (local_obs_climatology):
day_threshold_series = threshold.loc[day_data.index]
valid_mask = day_threshold_series.notna().values
fc1_day = day_data['fc1_value'].values[valid_mask]
obs_day  = day_data['obs_value'].values[valid_mask]
day_threshold = day_threshold_series.values[valid_mask]  # numpy array
```
Then `calculate_ets(fc1_day, obs_day, day_threshold, event_type)` receives a
numpy array for `threshold`.

**Inside `calculate_ets`:**
```python
valid = _nan_mask(forecast, observation)   # filters NaN fc/obs only
fc_v, obs_v = forecast[valid], observation[valid]
# BUG RISK: if threshold is a numpy array, it is NOT indexed by valid here
fc_binary = (fc_v >= threshold).astype(int)
```

**Known latent bug:** When `threshold` is a numpy array, the `valid` mask is applied to
`fc_v` and `obs_v` but NOT to `threshold`. If any NaN fc/obs rows survived past the
threshold filter in `run_step6`, this would cause a shape mismatch or silent misalignment.
In practice this does not occur (no NaN fc/obs after filtering), but the code is fragile.

**What to check:**
- After the `valid_mask` filter in `run_step6`, confirm `fc1_day` and `day_threshold`
  have the same length and no NaN values before passing to score functions.
- The contingency table pools ALL rows for the day — not per-station tables then averaged.
  Verify there is no `groupby('station_id')` inside the score functions.
- `event_type` is `'above'` for warm/wind extremes, `'below'` for cold. Check that `<=`
  vs `>=` is consistent across ETS, PSS, POD, FAR, twMAE.

**Correct fix for the latent bug** (if NaN fc/obs could ever appear):
```python
# In calculate_ets, calculate_pss, calculate_pod, calculate_far:
valid = _nan_mask(forecast, observation)
fc_v, obs_v = forecast[valid], observation[valid]
thr_v = threshold[valid] if isinstance(threshold, np.ndarray) else threshold
fc_binary = (fc_v >= thr_v).astype(int)
obs_binary = (obs_v >= thr_v).astype(int)
```

### twMAE and twRMSE (`det_scores.py :: calculate_twmae`, `calculate_twrmse`)

**What they should do:** Apply the chaining function to ALL rows (not just extremes),
then compute MAE/RMSE on the transformed values. This is the proper scoring rule
consistent with twCRPS (Taggart 2022, QJRMS).

```
# Upper tail (above T_i):
fc_v  = max(fc_i,  T_i)
obs_v = max(obs_i, T_i)
twMAE  = mean_i |fc_v_i - obs_v_i|
twRMSE = sqrt( mean_i (fc_v_i - obs_v_i)^2 )
```

Contribution per case:
| fc vs T | obs vs T | Contribution |
|---------|---------|-------------|
| fc ≥ T, obs ≥ T | both clipped | \|fc − obs\| |
| fc < T, obs ≥ T (miss) | fc clipped to T | \|T − obs\| |
| fc ≥ T, obs < T (false alarm) | obs clipped to T | \|fc − T\| |
| fc < T, obs < T (non-event) | both clipped | 0 |

**Scale implication:** twMAE ≈ (extreme fraction) × conditional MAE. For p99 this is
~1/50th the value of "MAE restricted to extreme rows". The relative % difference shown
in the heatmap is still meaningful.

**What to check:**
- `np.maximum` (not `np.where` or masking) is used — both arrays transformed together.
- Per-station array threshold is handled: `threshold[valid]` or broadcast correctly.
- NOT a conditional MAE (no `mask = observation >= threshold` subsetting before computing).

**What a bug looks like:** If the code uses `mask = obs >= T; return mean|fc[mask] - obs[mask]|`,
that is conditional MAE, not twMAE. The values will be ~50× larger and false alarms
will be invisible.

### twCRPS (`ens_scores.py` around line 214)

**What it should do:** Apply chaining function to all (sample, member) pairs element-wise
using per-row T_i, then compute fair CRPS via `_fair_crps_numpy`.

```python
# Correct per-station path (thr_arr is 1D, shape n_samples):
fcst_v = np.maximum(fcst_np, thr_arr[:, None])   # shape (n_samples, n_members)
obs_v  = np.maximum(obs_np,  thr_arr)             # shape (n_samples,)
result = _fair_crps_numpy(fcst_v, obs_v)
```

**What to check:**
- `thr_arr` is 1D (n_samples,), not 2D — check it's not accidentally keeping the
  `pd.Series` index or being squeezed to a scalar.
- `thr_arr[:, None]` broadcasts threshold across members correctly.
- The `thr_arr is not None` branch is taken when `threshold` is a `pd.Series` or
  numpy array. Check the condition: `if thr_arr is not None:`.
- The scalar fallback (`tail_tw_crps_for_ensemble`) is only used when `thr_arr is None`
  (i.e. for `fixed` or `dataset_climatology` methods).

**Where `thr_arr` comes from:** In `ens_scores._process_single_day`:
```python
elif isinstance(threshold, pd.Series):
    day_thr_series = threshold.loc[day_data.index]
    ...
    day_data['_obs_threshold'] = day_threshold  # stored as column
```
Then in `calculate_ensemble_scores`:
```python
if '_obs_threshold' in df_clean.columns:
    thr_arr = df_clean['_obs_threshold'].values
```
Check the column is not dropped between `_process_single_day` and `calculate_ensemble_scores`.

### Brier Score (`ens_scores.py`)

**What it should do:**
```python
p_hat = (fcst_np >= thr_arr[:, None]).mean(axis=1)   # exceedance probability per case
o_ev  = (obs_np  >= thr_arr).astype(float)            # binary obs per case
BS    = mean((p_hat - o_ev)**2)
```

**What to check:**
- Comparison `>=` vs `<=` matches `event_type` ('above' → `>=`, 'below' → `<=`).
- `thr_arr[:, None]` used for member comparison (not scalar broadcast).
- BSS reference probability uses **sample event frequency** `p_c = mean(o_ev)`, not the
  definitional percentile rate (e.g. 0.01 for p99). Using the definitional rate gives
  artificially negative BSS when the verification period rate differs from climatology.

---

## Heatmap Display (`plot.py :: create_heatmap`)

The cell value shown is the **relative % change of fc2 versus fc1**:

| Score type | Formula | Blue = better |
|-----------|---------|--------------|
| Error metrics (twMAE, twRMSE, CRPS, twCRPS, MAE, RMSE, Brier) | `(fc2 − fc1) / fc1 × 100` | Negative % (fc2 lower error) |
| Skill scores (ETS, PSS, POD) | `(fc2 − fc1) / (1 − fc1) × 100` | Positive % (fc2 higher skill) |
| FAR | `(fc2 − fc1) / fc1 × 100` | Negative % (fc2 fewer false alarms) |

Skill score denominator `(1 − fc1)` represents the remaining room for improvement from
fc1's baseline — going from 0.9 to 0.95 ETS is a larger relative gain than 0.1 to 0.15.

Statistical significance dots: shown when the 95% CI of the paired bootstrap distribution
of `(score_fc2 − score_fc1)` does not include zero. Bootstrap is paired (same resampled
indices for both models) to remove shared obs variance.

**What to check in `plot.py`:**
- Score is in `_ERROR_SCORES` set → uses relative formula, blue=negative.
- Score is in `_HIGHER_IS_BETTER_SCORES` set → uses skill-score formula, blue=positive.
- `_get_color_limit(score_type)` returns the correct ±% range for the colorbar.
- Significance column name follows pattern `{score_name}_is_significant`.

---

## Key Invariants (Things That Must Be True)

These can be checked programmatically to validate a run:

1. **`n_samples` in CSV = number of non-NaN threshold rows after daily aggregation.**
   ```python
   assert len(agg[agg['_thr'].notna()]) == csv_row['n_samples']
   ```

2. **`n_exceedances / n_samples ≈ percentile/100` (within ~20%).**
   For p99 expect ~1% exceedance rate. Large deviations indicate wrong month used
   for threshold lookup or wrong event_type direction.

3. **`0 ≤ ETS ≤ 1` for a reasonable model; PSS ∈ [−1, 1].**
   ETS < 0 means worse than random chance — possible but unusual for day 1.

4. **`twMAE ≈ (n_exceedances / n_samples) × conditional_MAE` (roughly).**
   A large deviation indicates the chaining function is not being applied correctly
   (e.g. conditional masking used instead).

5. **Season-stratified n_samples sum to approximately the all-season n_samples.**
   DJF + MAM + JJA + SON ≈ annual (allowing for matched-station differences per month).

---

## File Reference

| Code | File | Key function |
|------|------|-------------|
| Filter pipeline | `filter.py` | `run_step4()` |
| Threshold computation | `threshold.py` | `_compute_local_obs_climatology_threshold()` |
| Daily aggregation | `run.py` | `_aggregate_to_daily_mean()` |
| Deterministic scores | `det_scores.py` | `calculate_ets()`, `calculate_twmae()`, `run_step6()` |
| Ensemble scores | `ens_scores.py` | `_process_single_day()`, `calculate_ensemble_scores()`, `_fair_crps_numpy()` |
| Bootstrap | `bootstrap.py` | `run_step7()` |
| Heatmap | `plot.py` | `create_heatmap()`, `_get_color_limit()` |
| Orchestration | `run.py` | `main()` — Step 5b trigger logic around lines 557 and 847 |
