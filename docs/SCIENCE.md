# Scorecards4Extremes — Scientific Methodology

**Version:** May 2026

---

## Table of Contents

1. [Motivation and Scope](#1-motivation-and-scope)
2. [Event Definition and Thresholds](#2-event-definition-and-thresholds)
3. [Deterministic Verification Scores](#3-deterministic-verification-scores)
4. [Ensemble Verification Scores](#4-ensemble-verification-scores)
5. [Threshold-Weighted Scores — Mathematical Framework](#5-threshold-weighted-scores--mathematical-framework)
6. [Threshold-Weighted Quantile Score](#6-threshold-weighted-quantile-score)
7. [Lapse-Rate Correction](#7-lapse-rate-correction)
8. [Orography and Coastal Stratification](#8-orography-and-coastal-stratification)
9. [Seasonal Stratification](#9-seasonal-stratification)
10. [Bootstrap Significance Testing](#10-bootstrap-significance-testing)
11. [References](#11-references)

---

## 1. Motivation and Scope

Standard aggregate verification metrics (MAE, RMSE, CRPS, etc.) are dominated by the large population of non-extreme cases. For forecast applications focused on extreme events — cold spells, heat waves, heavy precipitation, high winds — this makes it difficult to detect differences in model performance at the tails of the distribution.

**Scorecards4Extremes** addresses this by using **threshold-weighted** (TW) score families, introduced by Taggart (2022), alongside classical categorical scores (ETS, PSS). All metrics are computed separately for each season × terrain type stratification, providing granular insight into model behaviour.

The headline output is a **two-model comparison scorecard**: for every (lead time, season, orography class) combination, each score shows the **relative percentage improvement** of model 1 over model 2:

$$\text{Skill} = \frac{S_2 - S_1}{S_2} \times 100\%$$

where $S_1$ and $S_2$ are the scores of model 1 and model 2 respectively (for negatively oriented scores such as MAE, a positive result means model 1 is better).

---

## 2. Event Definition and Thresholds

### Binary extreme event

An **extreme event** occurs when the observed value $y$ exceeds a climatological threshold $T$:

$$\text{event} = \begin{cases} 1 & \text{if } y > T \quad (\text{above threshold: warm/wet/windy}) \\ 1 & \text{if } y < T \quad (\text{below threshold: cold}) \\ 0 & \text{otherwise} \end{cases}$$

### Per-station seasonal climatological threshold

The recommended approach is `local_obs_climatology`: each station $s$ is assigned its own threshold $T_s^{(m)}$ for calendar month $m$, computed from an independent historical obs dataset (typically 15–20 years). This removes the influence of local climatological differences between stations and between seasons.

Formally, for station $s$ and month $m$:

$$T_s^{(m)} = Q_p\!\left(\{y_{s,t} : \text{month}(t) = m, \, t \notin \text{verification period}\}\right)$$

where $Q_p$ denotes the $p$-th sample quantile and the climatology window excludes the verification period to avoid in-sample bias.

### Alternative threshold methods

| Method | Definition | When to use |
|--------|-----------|-------------|
| `fixed` | $T = c$ (constant) | Physical warning level (e.g. 30 mm/24h) |
| `dataset_climatology` | $Q_p$ of all obs in the current dataset | Quick exploratory runs |
| `station_climatology` | Per-station $Q_p$ from STVL (1980–2009) | When STVL access is available |
| `model_percentile` | $Q_p$ of model 1 forecast distribution | Model-relative event definition |

---

## 3. Deterministic Verification Scores

### Contingency table

Given $N$ forecast–observation pairs and threshold $T$, the 2×2 contingency table is:

|  | Obs event | Obs no-event |
|--|-----------|--------------|
| **Fc event** | $H$ (hits) | $F$ (false alarms) |
| **Fc no-event** | $M$ (misses) | $Z$ (correct negatives) |

$N = H + F + M + Z$

### Probability of Detection

$$\text{POD} = \frac{H}{H + M}$$

Range: [0, 1]. Perfect: 1.

### False Alarm Ratio

$$\text{FAR} = \frac{F}{H + F}$$

Range: [0, 1]. Perfect: 0.

### Critical Success Index (Threat Score)

$$\text{CSI} = \frac{H}{H + F + M}$$

Ignores correct negatives; not equitable with respect to climatology.

### Equitable Threat Score

$$\text{ETS} = \frac{H - H_{\text{random}}}{H + F + M - H_{\text{random}}}$$

where the expected number of random hits is

$$H_{\text{random}} = \frac{(H + M)(H + F)}{N}$$

Range: $(-1/3, 1]$. 0 = no skill above random. Perfect: 1.

### Peirce Skill Score (Hanssen–Kuipers discriminant)

$$\text{PSS} = \text{POD} - \text{POFD} = \frac{H}{H+M} - \frac{F}{F+Z}$$

where POFD is the probability of false detection. Range: [−1, 1]. 0 = no skill. Perfect: 1.

### Threshold-weighted MAE

$$\text{twMAE} = \frac{1}{N} \sum_{i=1}^{N} w(y_i, \hat{y}_i) \cdot |y_i - \hat{y}_i|$$

where the weight $w(y_i, \hat{y}_i) = 1$ if $\max(y_i, \hat{y}_i) \geq T$ (i.e. at least one of obs or forecast exceeds the threshold), and 0 otherwise. This concentrates the error on cases where either the observation or the forecast qualifies as an extreme event.

### Decomposition of twMAE

The total twMAE decomposes exactly into three additive components (Taggart 2022):

$$\text{twMAE} = \frac{H \cdot \overline{e}_H + M \cdot \overline{e}_M + F \cdot \overline{e}_F}{N}$$

- **Hit contribution** ($H \cdot \overline{e}_H$): error when both obs and fc exceed $T$
- **Miss contribution** ($M \cdot \overline{e}_M$): error when obs exceeds $T$ but fc does not
- **False alarm contribution** ($F \cdot \overline{e}_F$): error when fc exceeds $T$ but obs does not

### Threshold-weighted RMSE

$$\text{twRMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} w(y_i, \hat{y}_i) \cdot (y_i - \hat{y}_i)^2}$$

with the same weight function as twMAE.

---

## 4. Ensemble Verification Scores

### Continuous Ranked Probability Score (CRPS)

For an ensemble of $m$ members $\{x_k\}_{k=1}^{m}$:

$$\text{CRPS}(\hat{F}, y) = \int_{-\infty}^{\infty} \left[\hat{F}(x) - \mathbf{1}(y \leq x)\right]^2 dx$$

where $\hat{F}(x) = \frac{1}{m}\sum_{k=1}^{m} \mathbf{1}(x_k \leq x)$ is the empirical CDF of the ensemble.

Equivalently, for a finite ensemble:

$$\text{CRPS} = \frac{1}{m}\sum_{k=1}^{m} |x_k - y| - \frac{1}{2m^2}\sum_{j=1}^{m}\sum_{k=1}^{m} |x_j - x_k|$$

### Fair CRPS (bias-corrected for finite ensembles)

The standard CRPS is biased when evaluating small ensembles. The **Fair CRPS** (Ferro 2014) corrects for this:

$$\text{fCRPS} = \frac{1}{m}\sum_{k=1}^{m} |x_k - y| - \frac{1}{2m(m-1)}\sum_{j \neq k} |x_j - x_k|$$

This is the score computed by the `scores` library's `crps_for_ensemble` function.

### Brier Score

For a binary extreme event $o \in \{0, 1\}$ with forecast probability $\hat{p}$:

$$\text{BS} = (\hat{p} - o)^2$$

where $\hat{p} = \frac{1}{m}\sum_{k=1}^{m} \mathbf{1}(x_k > T)$ is the ensemble event frequency.

Lower is better. Range: [0, 1].

### Brier Skill Score

$$\text{BSS} = 1 - \frac{\overline{\text{BS}}}{\overline{\text{BS}}_{\text{clim}}}$$

where $\overline{\text{BS}}_{\text{clim}} = \bar{o}(1 - \bar{o})$ is the Brier score of a climatological forecast that always predicts the observed event frequency $\bar{o}$. Range: (−∞, 1]. 1 = perfect, 0 = no skill over climatology.

### Extreme Spread/Skill Ratio

A measure of ensemble reliability restricted to extreme cases:

$$\text{extreme SSR} = \frac{\overline{\sigma^2_{\text{extreme}}}}{\overline{e^2_{\text{extreme}}}}$$

where $\sigma^2$ is the ensemble variance and $e^2$ the squared error of the ensemble mean, both averaged only over cases where the observation exceeds $T$. A value of 1 indicates a well-calibrated spread for extreme events; < 1 indicates under-dispersion.

---

## 5. Threshold-Weighted Scores — Mathematical Framework

The TW score framework (Taggart 2022, QJRMS) transforms any proper scoring rule into a **threshold-weighted** version by applying a **chaining function** $v_T$.

### Chaining function

For **above-threshold events** (warm/wet/windy):

$$v_T^+(x) = \max(x - T,\; 0) = (x - T)_+$$

For **below-threshold events** (cold):

$$v_T^-(x) = \max(T - x,\; 0) = (T - x)_+$$

This function is zero for values on the "safe" side of the threshold and grows linearly as the value penetrates into the tail.

### Threshold-weighted CRPS

$$\text{twCRPS}(\hat{F}, y) = \int_{-\infty}^{\infty} \left[v_T(\hat{F}(x)) - v_T(\mathbf{1}(y \leq x))\right]^2 dx$$

For the above-threshold case (event_type = above):

$$\text{twCRPS}(\hat{F}, y) = \int_{T}^{\infty} \left[\hat{F}(x) - \mathbf{1}(y \leq x)\right]^2 dx$$

This can be written in the familiar CRPS energy form as:

$$\text{twCRPS} = \mathbb{E}_{\hat{F}}[v_T(X) - v_T(y)] - \frac{1}{2}\mathbb{E}_{\hat{F},\hat{F}'}[v_T(X) - v_T(X')]$$

where $X, X'$ are independent draws from $\hat{F}$.

For a finite ensemble of $m$ members $\{x_k\}$:

$$\text{twCRPS} = \frac{1}{m}\sum_{k=1}^{m} |v_T(x_k) - v_T(y)| - \frac{1}{2m(m-1)}\sum_{j \neq k} |v_T(x_j) - v_T(x_k)|$$

The chaining-function transformation ensures that:
1. The score remains a **proper scoring rule** — a perfect forecast still minimises the score.
2. Cases far below the threshold (non-events) contribute **zero** to the score.
3. Cases in the extreme tail receive **larger weight** (through the growing $v_T$).

---

## 6. Threshold-Weighted Quantile Score

### Standard quantile score

The quantile score at level $\alpha \in (0,1)$ is:

$$\text{QS}_\alpha(\hat{q}_\alpha, y) = \rho_\alpha(y - \hat{q}_\alpha) = (y - \hat{q}_\alpha)\left[\alpha - \mathbf{1}(y < \hat{q}_\alpha)\right]$$

where $\hat{q}_\alpha$ is the $\alpha$-quantile forecast.

### Threshold-weighted quantile score

The TW version replaces $y$ and $\hat{q}_\alpha$ with their chain-transformed counterparts:

$$\text{twQS}_\alpha = \rho_\alpha\!\left(v_T(y) - v_T(\hat{q}_\alpha)\right)$$

This concentrates the score on the forecast of the tail — values close to $T$ from the "safe" side contribute zero while values deep in the tail receive proportionally higher weight.

### Multi-level evaluation

Scorecards4Extremes computes twQS at multiple levels simultaneously:

**Above-threshold events (warm extremes, heavy precipitation, strong winds):**

$$\alpha \in \{0.90,\; 0.95,\; 0.99\}$$

**Below-threshold events (cold extremes):**

$$\alpha \in \{0.01,\; 0.05,\; 0.10\}$$

The ensemble quantile forecast $\hat{q}_\alpha$ is estimated from the member values using linear interpolation.

### Relationship to CRPS

The CRPS equals the integral of the quantile score over all levels:

$$\text{CRPS} = 2 \int_0^1 \text{QS}_\alpha \, d\alpha$$

The multi-level twQS provides a finer-grained view of where in the tail one model outperforms the other.

---

## 7. Lapse-Rate Correction

Model grid cells covering complex terrain represent a smoothed elevation that differs from the true station elevation. For 2m temperature, this elevation mismatch introduces a systematic bias.

### Correction formula

$$T_{\text{corrected}} = T_{\text{model}} + \Gamma \cdot (z_{\text{obs}} - z_{\text{model}})$$

where:
- $T_{\text{model}}$ is the 2m temperature from the GRIB field (K)
- $z_{\text{obs}}$ is the station elevation (m a.s.l., from obs metadata)
- $z_{\text{model}}$ is the model orography height at the nearest gridpoint (m a.s.l., from GRIB)
- $\Gamma = -0.0065$ K/m is the standard environmental lapse rate (ICAO atmosphere)

For **ensemble** mode, the correction is applied **per member** before any scoring.

### Notes

- The correction removes the smooth-orography bias but not the representativity error due to sub-grid variability (addressed partially through terrain stratification).
- The correction is only applied for `variable: "2t"` when `lapse_rate_correction: true` is set.
- For cold extremes in complex terrain, the uncorrected forecast is systematically too warm (model surface is too low), so the correction shifts forecasts to colder values.

---

## 8. Orography and Coastal Stratification

### Sub-grid Standard Deviation of Orography (sdfor)

The `sdfor` field measures the terrain roughness within a model grid cell. It is a standard ECMWF surface field available in the operational system. Interpolated to each station's location, it quantifies how representative the model topography is for that station.

Classification:

| Class | sdfor (m) | Typical locations |
|-------|-----------|-------------------|
| flat | 0–40 | Lowland Europe, plains |
| hilly | 40–120 | Foothills, gentle terrain |
| complex | > 120 | Alps, Pyrenees, Scandinavian mountains |

Stratification is important because:
- Lapse-rate correction reduces but does not eliminate orography-related biases.
- Ensemble spread characteristics differ systematically between terrain classes.
- Cold pool formation and thermal inversions in complex terrain create verification challenges distinct from flat regions.

### Coastal station removal

Coastal stations receive partial sea-surface forcing and are subject to advection from sea to land (and vice versa). Their representativity for grid-box values is poor. Stations with land-sea mask (LSM) value < 0.9 are removed by default.

---

## 9. Seasonal Stratification

All scores are computed separately per season. The seasons used are:

| Season | Months | Physical motivation |
|--------|--------|-------------------|
| DJF | Dec–Feb | Northern European winter; cold extremes, polar intrusions |
| MAM | Mar–May | Spring transition; late cold snaps, early heat waves |
| JJA | Jun–Aug | Summer; heat waves, convective precipitation |
| SON | Sep–Nov | Autumn transition; cyclone-driven extremes |
| ASO | Aug–Oct | Active Atlantic–Mediterranean season; heavy precipitation |

The threshold $T_s^{(m)}$ uses the **calendar month** of the valid time, ensuring the climatological baseline matches the seasonal context of each verification case.

---

## 10. Bootstrap Significance Testing

### Procedure

Model differences are tested for statistical significance using the **percentile bootstrap** (Efron & Tibshirani 1993) with paired block resampling:

1. Pool forecast–observation pairs as units indexed by *date*.
2. Draw $B$ bootstrap samples (default: $B = 1000$) by resampling with replacement *across dates* (block size = 1 day).
3. For each bootstrap sample, recompute the score difference $\Delta S^{(b)} = S_2^{(b)} - S_1^{(b)}$.
4. Form the $(1 − \alpha)$ confidence interval as the $[\alpha/2,\; 1 - \alpha/2]$ percentiles of $\{\Delta S^{(b)}\}$.

### Statistical significance

A result is considered **statistically significant** at the 95% level when the confidence interval does not contain zero. In the heatmap scorecard, significant cells are marked with a dot (or cross-hatching) to distinguish them from possibly spurious differences.

### Notes

- Block resampling over dates preserves within-station spatial correlation (all stations for a given date are resampled together).
- For ensemble scores, bootstrap is applied to the date axis only; member resampling is not performed.
- `n_samples: 200` is recommended for ensemble jobs to reduce compute time; `n_samples: 1000` for deterministic jobs or publication-quality results.

---

## 11. References

- **Taggart, R. J.** (2022). Evaluation of point forecasts for extreme events using consistent scoring functions. *Quarterly Journal of the Royal Meteorological Society*, 148(743), 306–327. https://doi.org/10.1002/qj.4206

- **Ferro, C. A. T.** (2014). Fair scores for ensemble forecasts. *Quarterly Journal of the Royal Meteorological Society*, 140(683), 1917–1923. https://doi.org/10.1002/qj.2270

- **Gneiting, T. & Raftery, A. E.** (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.

- **Mason, S. J.** (2003). Binary events. In I. T. Jolliffe & D. B. Stephenson (Eds.), *Forecast Verification: A Practitioner's Guide in Atmospheric Science*. Wiley.

- **Efron, B. & Tibshirani, R. J.** (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

- **ICAO.** (1993). *Manual of the ICAO Standard Atmosphere*, 3rd ed. Doc 7488. International Civil Aviation Organization.
