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

> **Degeneracy for rare events.** ETS, PSS, POD and CSI all tend to trivial values as the event
> base rate $\to 0$ (Stephenson et al. 2008; Ferro & Stephenson 2011), i.e. exactly the p99–p99.9
> regime this tool targets — two systems of equal underlying skill score progressively *lower*
> ETS/PSS as the threshold is pushed further into the tail. The extremal-dependence indices below
> were designed to remove this.

### Extremal Dependence Index (EDI) and Symmetric EDI (SEDI)

With the hit rate $H = \text{POD} = \tfrac{H}{H+M}$ and the false-alarm rate $F = \text{POFD} = \tfrac{F}{F+Z}$:

$$\text{EDI} = \frac{\ln F - \ln H}{\ln F + \ln H}, \qquad
\text{SEDI} = \frac{\ln F - \ln H - \ln(1-F) + \ln(1-H)}{\ln F + \ln H + \ln(1-F) + \ln(1-H)}.$$

Both are **non-degenerate** as the base rate $\to 0$ (asymptotically independent of base rate),
so SEDI is the recommended categorical score for extremes. SEDI is additionally symmetric under
exchanging events with non-events and is harder to hedge than EDI. Range $(-1, 1]$; $0$ = no skill,
$1$ = perfect. They are undefined at the margins ($H$ or $F$ equal to $0$ or $1$) and return `NaN`
there — this does not occur in practice for the large multi-thousand-sample cells of the scorecard.

> **Frequency-bias caveat (important here).** EDI/SEDI are base-rate independent only when the
> forecast frequency bias $B = (H{+}F)/(H{+}M) \approx 1$. **The forecasts in this tool are not
> bias-corrected**, so a SEDI/EDI *difference* between two models conflates differences in their
> discrimination with differences in their frequency bias. Interpret it as a joint statement and
> cross-check against the base-rate / frequency-bias information. Reference: Ferro & Stephenson (2011).

### Threshold-weighted MAE

twMAE is the **chaining-function** (consistent) scoring function of Taggart (2022), i.e. the
deterministic / single-valued analogue of the twCRPS. The chaining function $v_T$ is applied to
**both** the forecast and the observation before the absolute error is taken:

$$\text{twMAE} = \frac{1}{N} \sum_{i=1}^{N} \left| v_T(\hat{y}_i) - v_T(y_i) \right|, \qquad
v_T(x) = \begin{cases} \max(x, T) & \text{above-threshold (warm/wet/windy)} \\ \min(x, T) & \text{below-threshold (cold)} \end{cases}$$

Equivalently $v_T(x)=(x-T)_+ + T$, so for the upper tail $\left|v_T(\hat y)-v_T(y)\right| = \left|(\hat y-T)_+ - (y-T)_+\right|$ (the additive constant $T$ cancels).

> **This is *not* the indicator-weighted MAE** $\frac{1}{N}\sum \mathbf{1}\{\max(y,\hat y)\ge T\}\,|y-\hat y|$.
> The two agree on hits but differ on misses and false alarms, and only the chaining version
> above is a *consistent* (non-hedgeable) scoring function for the threshold-weighted median
> (Taggart 2022). Because $v_T$ is applied to the forecast as well as the observation, false
> alarms (forecast beyond $T$, observation not) are penalised — this is what makes twMAE
> impossible to improve by over-forecasting (see `docs/TWMAE_HEDGING.md`).

### Decomposition of twMAE

The total twMAE decomposes exactly into three additive components (Taggart 2022):

$$\text{twMAE} = \frac{H \cdot \overline{e}_H + M \cdot \overline{e}_M + F \cdot \overline{e}_F}{N}$$

- **Hit contribution** ($H \cdot \overline{e}_H$): error when both obs and fc exceed $T$
- **Miss contribution** ($M \cdot \overline{e}_M$): error when obs exceeds $T$ but fc does not
- **False alarm contribution** ($F \cdot \overline{e}_F$): error when fc exceeds $T$ but obs does not

### Threshold-weighted RMSE

Same chaining construction as twMAE, with the squared error:

$$\text{twRMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( v_T(\hat{y}_i) - v_T(y_i) \right)^2}, \qquad v_T \text{ as above.}$$

Whereas twMAE targets the (threshold-weighted) **median**, twRMSE targets the **mean**; reporting
both is informative because their divergence reflects skewness/bias in the tail.

### SEEPS (precipitation only)

The Stable Equitable Error in Probability Space (Rodwell et al. 2010) is ECMWF's headline score
for 24-h precipitation. Each forecast and observation is put into one of three categories:
*dry* ($\le 0.2$ mm), *light* and *heavy*. The light/heavy boundary is set per station so that
light:heavy occur 2:1 in climatology, i.e. at the $(2+p_1)/3$ climatological quantile, where
$p_1$ is the station's dry probability. The penalty matrix (Rodwell Eq. 15) depends only on
$p_1$; the implementation uses `scores.categorical.seeps`.

SEEPS is negatively oriented (0 = perfect) and equitable: any no-skill forecast scores ≈ 1.

Implementation choices:
- **All cases, not only extremes.** SEEPS is computed over every station-date in the cell. It
  complements the tail scores rather than being one.
- **In-sample, per-station climatology.** $p_1$ and the light/heavy boundary come from the same
  observations that are being verified. This is self-contained, but a long external climatology
  would be more rigorous.
- **Stability range.** Stations with $p_1 \notin [0.1, 0.85]$ are dropped (Rodwell's stability
  range), as are stations with fewer than 30 observations.
- **tp24 only.** It is skipped, with a printed note, for any other variable.

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

$$\text{BS}_{\text{fair}} = (\hat{p} - o)^2 - \frac{\hat{p}(1-\hat{p})}{m-1}$$

where $\hat{p} = \frac{1}{m}\sum_{k=1}^{m} \mathbf{1}(x_k \ge T)$ is the ensemble event frequency.
The second term is the **fair** correction (Ferro 2014): the plain score $(\hat{p}-o)^2$ carries a
sampling penalty $p(1-p)/m$ for a finite ensemble, which the correction removes, so the expected
score is that of the underlying probability and ensembles of different size are comparable.
Equivalently, $\text{BS}_{\text{fair}}$ is the probability that two distinct members both disagree with
the observation, so it is never negative. BSS is computed from the fair BS.

Lower is better. Range: [0, 1].

### Ensemble spread

$$\text{spread} = \sqrt{\overline{\sigma^2}}, \qquad \sigma_i^2 = \frac{1}{m-1}\sum_{k=1}^{m}(x_{ik}-\bar{x}_i)^2$$

the square root of the case-averaged **unbiased** member variance (Leutbecher & Palmer 2008;
Leutbecher 2009; Fortin et al. 2014). ECMWF's guidance (Leutbecher 2009, §2.5) quantifies the
spread–error relationship with region- and forecast-averaged *variances* and writes the consistency
condition as $\tfrac{M}{M-1}\,\overline{s^2_{\text{biased}}} = \tfrac{M}{M+1}\,\overline{e^2}$, noting that the
size factors are negligible for $M \approx 50$ but *"should be accounted for"* for $M \le 20$ — as for the
8-member ensembles verified here. Averaging variances rather than standard deviations makes `ens_spread`
directly comparable with `ens_mean_rmse`; for a perfectly calibrated $m$-member ensemble
$\text{RMSE} \approx \sqrt{(m+1)/m}\;\text{spread}$.

The skill side is the plain RMSE of the ensemble mean (`ens_mean_rmse`), the conventional quantity.
Its expectation, $(1 + 1/m)\,\sigma^2$, includes the sampling error of an $m$-member mean, so a
size-corrected version is also provided (Leutbecher & Palmer 2008):

$$\text{RMSE}_{\text{fair}} = \sqrt{\overline{e^2} - \overline{s^2}/m}$$

with $s^2$ the unbiased member variance; it estimates the error of the infinite-ensemble mean and is
the right choice when ensembles of different size are compared.

### Brier Skill Score

$$\text{BSS} = 1 - \frac{\overline{\text{BS}}}{\overline{\text{BS}}_{\text{clim}}}$$

where $\overline{\text{BS}}_{\text{clim}} = \bar{o}(1 - \bar{o})$ is the Brier score of a climatological forecast that always predicts the observed event frequency $\bar{o}$. Range: (−∞, 1]. 1 = perfect, 0 = no skill over climatology.

### CRPS Skill Score

$$\text{CRPSS} = 1 - \frac{\overline{\text{CRPS}}_{\text{fc}}}{\overline{\text{CRPS}}_{\text{clim}}}$$

The numerator is the fair CRPS of the ensemble. The reference is the **sample climatology**: the
empirical distribution of all observations pooled in the cell. Its mean CRPS against those same
observations is half the Gini mean difference, $\tfrac{1}{2}\,\overline{|y_i - y_j|}$, computed in
$O(n \log n)$. Range $(-\infty, 1]$; 1 = perfect, 0 = no better than climatology. Because the
reference is in-sample, it is slightly harder to beat than an independent climatology.

### Brier reliability

The reliability term of Murphy's (1973) Brier-score decomposition, for the event "crossing $T$":

$$\text{REL} = \frac{1}{N} \sum_{k=1}^{K} n_k \left(\bar{p}_k - \bar{o}_k\right)^2$$

Here $\bar{p}_k$ is the mean forecast probability in bin $k$ ($K = 10$ equal-width bins), $\bar{o}_k$
is the observed event frequency in that bin, and $n_k$ is the number of cases in the bin. 0 = perfectly
reliable; lower is better. This term isolates **calibration**, which the threshold-weighted accuracy
scores (twCRPS, Brier) mix with resolution. With few events per bin it is noisy, so read it alongside
the reliability diagram (`ens_calibration_*.png` from `diagnostics/ensemble/diagnose_ens_extremes.py`).

### Extreme Spread/Skill Ratio

A measure of ensemble dispersion restricted to extreme cases, defined as the size-corrected ratio of
the RMS ensemble spread to the RMSE of the ensemble mean:

$$\text{extreme SSR} = \sqrt{\frac{m+1}{m}}\;\frac{\sqrt{\overline{\sigma^2_{\text{extreme}}}}}{\sqrt{\overline{e^2_{\text{extreme}}}}}
= \sqrt{\frac{m+1}{m}}\;\frac{\sqrt{\overline{\sigma^2_{\text{extreme}}}}}{\text{RMSE}_{\text{extreme}}}$$

where $\sigma^2$ is the per-case ensemble variance (sample variance, `ddof=1`) and $e^2$ the squared
error of the ensemble mean, both averaged only over cases where the observation crosses $T$.
$\approx 1$ indicates calibrated spread; $< 1$ under-dispersion; $> 1$ over-dispersion.

> **Two caveats.** (1) The factor $\sqrt{(m+1)/m}$ (Leutbecher 2009; Fortin et al. 2014) removes the
> finite-ensemble effect — without it a perfectly calibrated $m$-member ensemble would score
> $\sqrt{m/(m+1)}$ (0.94 for $m=8$) — but the ratio remains a dispersion diagnostic, not a
> strictly-proper score. The same correction defines the all-cases `spread_skill_ratio`. (2) It is conditioned on the
> **observation** being extreme, which introduces the usual spread–skill *conditioning bias*
> (selecting on the outcome). Treat it as a tail-dispersion diagnostic, not as a calibration score;
> for a proper calibration assessment use the twCRPS / reliability tools.

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
- The correction is only applied for `variable: "2t"` when `lapse_rate_correction: true` is set. Setting it to `false` disables the correction in both deterministic and ensemble modes.
- For cold extremes in complex terrain, the uncorrected forecast is systematically too warm (model surface is too low), so the correction shifts forecasts to colder values.

### Quality control of the correction

To guard against unrealistic corrections caused by bad station metadata (e.g. a missing or grossly wrong reported elevation), stations are **dropped** when either:

- $|\Gamma \cdot (z_{\text{obs}} - z_{\text{model}})| > 50$ K (correction magnitude), or
- $|z_{\text{obs}} - z_{\text{model}}| > 10000$ m (elevation mismatch).

This same cap is applied consistently across all extraction backends (local, MARS/STVL, and quaver) and for both deterministic and ensemble forecasts, so the retained station population is identical regardless of how the data was extracted.

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

- Block resampling over dates preserves the **spatial correlation between stations on the same date** (all stations for a given date are resampled together). Without it, correlated stations would be counted as independent, the confidence intervals would be too narrow, and too many cells would appear significant.
- This applies to both the `local` and `quaver` backends. A cell with fewer than 5 distinct dates falls back to case-level resampling.
- The block is a single day, so **temporal** autocorrelation between consecutive days is not accounted for. For persistent regimes (e.g. multi-day heatwaves) the intervals can still be somewhat too narrow.
- For ensemble scores, bootstrap is applied to the date axis only; member resampling is not performed.
- `n_samples: 200` is recommended for ensemble jobs to reduce compute time; `n_samples: 1000` for deterministic jobs or publication-quality results.

---

## 11. References

- **Taggart, R. J.** (2022). Evaluation of point forecasts for extreme events using consistent scoring functions. *Quarterly Journal of the Royal Meteorological Society*, 148(743), 306–327. https://doi.org/10.1002/qj.4206

- **Ferro, C. A. T.** (2014). Fair scores for ensemble forecasts. *Quarterly Journal of the Royal Meteorological Society*, 140(683), 1917–1923. https://doi.org/10.1002/qj.2270

- **Gneiting, T. & Raftery, A. E.** (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.

- **Mason, S. J.** (2003). Binary events. In I. T. Jolliffe & D. B. Stephenson (Eds.), *Forecast Verification: A Practitioner's Guide in Atmospheric Science*. Wiley.

- **Ferro, C. A. T. & Stephenson, D. B.** (2011). Extremal Dependence Indices: Improved verification measures for deterministic forecasts of rare binary events. *Weather and Forecasting*, 26(5), 699–713. https://doi.org/10.1175/WAF-D-10-05030.1

- **Stephenson, D. B., Casati, B., Ferro, C. A. T. & Wilson, C. A.** (2008). The extreme dependency score: a non-vanishing measure for forecasts of rare events. *Meteorological Applications*, 15(1), 41–50. https://doi.org/10.1002/met.53

- **Rodwell, M. J., Richardson, D. S., Hewson, T. D. & Haiden, T.** (2010). A new equitable score suitable for verifying precipitation in numerical weather prediction. *Quarterly Journal of the Royal Meteorological Society*, 136(650), 1344–1363. https://doi.org/10.1002/qj.656

- **Murphy, A. H.** (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12(4), 595–600.

- **Leutbecher, M. & Palmer, T. N.** (2008). Ensemble forecasting. *Journal of Computational Physics*, 227(7), 3515–3539. https://doi.org/10.1016/j.jcp.2007.02.014

- **Leutbecher, M.** (2009). Diagnosis of ensemble forecasting systems. *ECMWF Seminar on Diagnosis of Forecasting and Data Assimilation Systems*, 7–10 September 2009, 235–266. https://www.ecmwf.int/en/elibrary/78745-diagnosis-ensemble-forecasting-systems

- **Fortin, V., Abaza, M., Anctil, F. & Turcotte, R.** (2014). Why should ensemble spread match the RMSE of the ensemble mean? *Journal of Hydrometeorology*, 15(4), 1708–1713. https://doi.org/10.1175/JHM-D-14-0008.1

- **Efron, B. & Tibshirani, R. J.** (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

- **ICAO.** (1993). *Manual of the ICAO Standard Atmosphere*, 3rd ed. Doc 7488. International Civil Aviation Organization.
