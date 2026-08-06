# Scorecards4Extremes — User Guide

**Version:** May 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Configuration File Reference](#3-configuration-file-reference)
4. [Pipeline Steps in Detail](#4-pipeline-steps-in-detail)
5. [Supported Variables](#5-supported-variables)
6. [Threshold Methods](#6-threshold-methods)
7. [Verification Scores](#7-verification-scores)
8. [Filters and Quality Control](#8-filters-and-quality-control)
9. [Output Files and Directory Structure](#9-output-files-and-directory-structure)
10. [Ensemble Mode](#10-ensemble-mode)
11. [Obs Climatology Builder](#11-obs-climatology-builder)
12. [Diagnostic and Analysis Tools](#12-diagnostic-and-analysis-tools)
13. [Observation Data Format — `.gpt` / Geopoints](#13-observation-data-format--gpt--geopoints)
14. [Batch Submission on ECMWF HPC](#14-batch-submission-on-ecmwf-hpc)

> **Case-study workflow**: see [CASE_STUDIES.md](CASE_STUDIES.md) for `find_case_studies.py` and `plot_case_study.py`.

---

## 1. Overview

**Scorecards4Extremes** is a verification toolkit designed specifically for extreme weather events. It compares two forecast models (model 1 vs model 2) against synoptic observations, computing skill scores that focus on the extreme tails of the forecast distribution.

### Key design principles

- **Two-model comparison** — all scores are expressed as the relative difference between model 1 and model 2, shown as percentage improvement/degradation.
- **Tail-focused metrics** — threshold-weighted scores (twMAE, twRMSE, twCRPS, twQS) concentrate the evaluation on the extreme events that exceed a user-defined climatological threshold.
- **Per-station thresholds** — the recommended threshold approach (`local_obs_climatology`) assigns a different climatological percentile to each station, accounting for local climatological variability.
- **Season × orography stratification** — results are broken down by season (DJF, MAM, JJA, SON, ASO, …) and by terrain complexity (flat/hilly/complex), making it easy to identify where model differences are most significant.
- **Bootstrap significance** — all score differences are accompanied by 95% bootstrap confidence intervals.

### Supported modes

| Mode | Use case | Key scores |
|------|----------|------------|
| `deterministic` | Single-valued forecast vs obs | ETS, PSS, POD, FAR, twMAE, twRMSE, bias, MAE, RMSE |
| `ensemble` | Probabilistic ensemble forecast | twCRPS, fCRPS, Brier, BSS, twQS (multi-level), extreme_spread_skill_ratio |

### Supported variables

| Variable | Description | Unit |
|----------|-------------|------|
| `2t` | 2 m temperature | °C (after K→°C conversion + optional lapse-rate correction) |
| `10ff` | 10 m wind speed | m/s (from U/V components) |
| `tp24` | 24 h total precipitation | mm (after m→mm conversion and optional accumulation) |

---

## 2. Repository Structure

```
scorecards4extremes/
├── run.py                          ← Main pipeline entry point
├── submit_job.sh                   ← SLURM batch submission script
│
├── read_data.py                    ← Step 1: Read GRIB/GPT files
├── preprocess.py                   ← Step 2: Unit conversion, lapse-rate, accumulation
├── extract_points.py               ← Step 3: Interpolate to station locations (deterministic)
├── extract_points_ensemble.py      ← Step 3: Interpolate to station locations (ensemble)
├── filter.py                       ← Step 4: Season, orography, QC filters
├── threshold.py                    ← Step 5: Compute event threshold
├── det_scores.py                   ← Step 6: Deterministic verification scores
├── ens_scores.py                   ← Step 6: Ensemble verification scores
├── bootstrap.py                    ← Step 7: Bootstrap confidence intervals
├── save.py                         ← Step 8: Write results to CSV
├── plot.py                         ← Step 9: Draw heatmap scorecards
│
├── diagnostics/                    ← Standalone diagnostic and visualisation tools
│   ├── diagnose_extremes.py        ← Comprehensive single-condition diagnostic (22 plots)
│   ├── diagnose_det_extremes_simple.py ← Quick 3-panel deterministic diagnostic
│   ├── diagnose_twcrps_simple.py   ← Quick 3-panel ensemble diagnostic
│   ├── plot_qq_extremes.py         ← Q-Q plot with warm + cold zoom panels
│   ├── plot_qq.py                  ← Full-range Q-Q plot
│   └── plot_station_diagnostics.py ← Per-station spatial maps (mode auto-detected)
│
├── case_studies/                   ← Case-study identification and visualisation
│   ├── find_case_studies.py        ← Rank dates by model-performance gap
│   ├── plot_case_study.py          ← Visualise a specific date (3 figures)
│   ├── case_study_utils.py         ← Shared threshold/classification utilities
│   ├── submit_find_cases.sh        ← SLURM wrapper for find_case_studies.py
│   └── submit_plot_case.sh         ← SLURM wrapper for plot_case_study.py
│
├── obs_clim_local/                 ← Obs climatology builder (2t, 10ff, tp)
│   ├── obsclim.py                  ← Builds per-station percentile files from STVL
│   └── run.sh                      ← SLURM job for climatology build
│
├── obs_clim_local_Italy/           ← Italy-specific tp obs climatology builder
│   ├── obsclim.py
│   └── run.sh
│
├── configs/                         ← Experiment config templates
│   ├── deterministic/               ← Deterministic mode configs
│   └── ensemble/                    ← Ensemble mode configs
├── config_example.yaml             ← Clean starter config (no ECMWF dependencies)
├── scripts/                        ← Auxiliary / one-off helper scripts
│   ├── extract_tp24_obs.py         ← tp24 obs builder (ECMWF: vtb required)
│   ├── setup_ecmwf.sh              ← HPC environment setup helper
│   └── ...                         ← Other batch/utility scripts
├── submit_job.sh                   ← SLURM submission wrapper
└── docs/                           ← This documentation folder
```

---

## 3. Configuration File Reference

Every experiment is defined by a single YAML file. Below is a fully-annotated template.

```yaml
# ── Basic settings ─────────────────────────────────────────────────────────
variable: "2t"              # tp24 | 2t | 10ff
start_date: "2025-08-01"    # inclusive, YYYY-MM-DD
end_date:   "2026-02-28"    # inclusive, YYYY-MM-DD
mode: "deterministic"       # deterministic | ensemble
forecast_days: [1, 2, 3, 4, 5]    # used in deterministic mode
steps: [24, 48, 72, ...]          # used in ensemble mode (hours)
lead_time_frequency: 24    # sub-daily interval (hours) for deterministic 2t
skip_extraction_if_exists: true   # skip step 3 if parquet files already exist
skip_scoring_if_exists:    false  # skip step 6 if score CSV already exists

# ── STEP 1: Read data ───────────────────────────────────────────────────────
read_data:
  forecast_model1:
    name: "model_A"           # label used in filenames and plots
    source: "local_grib"      # local_grib | mars | quaver   (mars/quaver: ECMWF only)
    unit_conversion_factor: 1000.0   # applied to GRIB values (e.g. m→mm for tp)
    local_grib:
      path: "/path/to/model_A/grib"
    # For source: "mars" instead, provide a `mars:` block (see §4 Step 1 — Direct retrieval)

  forecast_model2:
    name: "model_B"
    source: "local_grib"
    unit_conversion_factor: 1000.0
    local_grib:
      path: "/path/to/model_B/grib"

  observation_source: "local_gpt"   # local_gpt | stvl | quaver   (stvl/quaver: ECMWF only)
  local_gpt:
    path: "/path/to/obs/gpt_files"
  # For observation_source: "stvl" instead, provide a `stvl:` block (see §4 Step 1)

# ── STEP 2: Pre-process ─────────────────────────────────────────────────────
preprocess:
  wind_speed_from_components: false  # true for 10ff (u,v → speed)
  lapse_rate_correction: true        # true for 2t only
  lapse_rate: -0.0065                # K/m (environmental lapse rate)
  precipitation_accumulation_hours: null  # 24 for tp24; null otherwise

# ── STEP 3: Extract points ─────────────────────────────────────────────────
extract_points:
  output_path: "/perm/user/extracted_points/my_experiment"
  save_format: "pandas"   # only "pandas" (parquet) currently supported
  area: "europe"          # bounding box filter applied during extraction

# ── Auxiliary fields (for lapse-rate correction and terrain classification) ─
auxiliary_fields:
  model1:
    lsm_path:  "/path/to/lsm_model1.grib"
    orog_path: "/path/to/orog_model1.grib"
  model2:
    lsm_path:  "/path/to/lsm_model2.grib"
    orog_path: "/path/to/orog_model2.grib"
  sdfor_path:  "/path/to/sdfor.grib"   # sub-grid standard deviation of orography

# ── STEP 4: Filter data ────────────────────────────────────────────────────
filter:
  lead_times: null       # null = use all; or list of hours e.g. [24, 48, 72]
  season: ["DJF", "JJA"]  # DJF | MAM | JJA | SON | ASO (Aug-Sep-Oct) | null
  orography_type: ["flat", "hilly", "complex"]
  orography_ranges:
    flat:    [0, 40]
    hilly:   [40, 120]
    complex: [120, 3000]
  remove_coastal_stations: true
  coastal_lsm_threshold: 0.9     # stations with lsm < 0.9 are considered coastal
  remove_outliers: false
  outlier_threshold_std: 5.0
  min_valid_temperature: -60.0   # 2t QC bounds (°C)
  max_valid_temperature:  60.0
  max_valid_precipitation: 800.0 # tp24 QC ceiling (mm)

# ── STEP 5: Threshold ──────────────────────────────────────────────────────
threshold:
  method: "local_obs_climatology"   # See section 6 for all methods
  event_type: "above"               # above | below
  local_obs_climatology:
    path: "/home/user/scorecards4extremes/obs_clim_local"
    parameter: "tp"
    percentile: 99
    window_days: 1
    n_years: 20
    first_year: 2005
    last_year: 2024
    min_availability_pct: 65

# ── STEP 6: Scores ─────────────────────────────────────────────────────────
# -- Deterministic --
scores:
  deterministic:
    - "ETS"
    - "PSS"
    - "POD"
    - "FAR"
    - "twMAE"
    - "twRMSE"
    - "bias"
    - "mae"
    - "rmse"
    - "correlation"
  stratify_by:
    - "lead_time"

# -- Ensemble --
scores:
  ensemble:
    - "twCRPS"
    - "fCRPS"
    - "Brier"
    - "BSS"
    - "tw_quantile_score"
    - "tw_quantile_score_q001"   # extreme tail (cold below / warm above)
    - "tw_quantile_score_q005"
    - "tw_quantile_score_q010"
    - "tw_quantile_score_q090"
    - "tw_quantile_score_q095"
    - "tw_quantile_score_q099"
    - "extreme_spread_skill_ratio"
  stratify_by:
    - "lead_time"

# ── STEP 7: Bootstrap ─────────────────────────────────────────────────────
bootstrap:
  enabled: true
  n_samples: 1000      # 200–500 for fast runs, 1000 for publication quality
  confidence_level: 0.95

# ── STEP 8: Save ──────────────────────────────────────────────────────────
save:
  output_directory: "/perm/user/results/my_experiment"
  save_scores_csv: true
  save_forecast_obs_pairs: false

# ── STEP 9: Plot ──────────────────────────────────────────────────────────
plot:
  enabled: true            # master switch (false = skip all plotting)
  heatmap_style: "smooth"  # "smooth" or "normal"
  create_summary: true     # also draw the multi-panel summary figure
  dpi: 300                 # output resolution (dots per inch)
  format: "png"            # png | pdf | svg
  # forecast_days: [1, 3, 5]  # optional: restrict heatmaps to these lead days
```

---

## 4. Pipeline Steps in Detail

### Step 1 — Read Data

Reads GRIB forecast files and `.gpt` observation files from local disk (or from the Quaver/STVL backend). For `local_grib` source, files must be named:

- 2t: `2t_YYYYMMDD.grib`
- tp24: `tp24_YYYYMMDD.grib`
- 10ff: `10u_YYYYMMDD.grib` and `10v_YYYYMMDD.grib` (U and V components separately)

For `local_gpt` observations:
- 2t: `2t_obs_YYYYMMDD00.geo` or `2t_YYYYMMDD.gpt`
- tp24: `tp24_obs_YYYYMMDD00.geo` or `tp24_YYYYMMDD.gpt`
- 10ff: `10ff_obs_YYYYMMDD00.geo` or `10ff_YYYYMMDD.gpt`

#### Direct retrieval — `source: "mars"` and `observation_source: "stvl"` *(ECMWF only)*

Instead of pointing at pre-retrieved files, a config can ask the tool to fetch the
data it needs itself, so no external retrieval script is required. This runs inline
as the first thing in Step 1 (on a compute node where the `mars` CLI and the
ECMWF-internal `vtb` package are available), then the pipeline proceeds exactly as
for local files. Only the steps the config actually needs are retrieved
(derived from `forecast_days` × `lead_time_frequency`), and retrieval is idempotent
— existing files are skipped, partial files from a failed retrieval are deleted.

**Forecasts (`source: "mars"`)** — replace the `local_grib` block with a `mars` block:

```yaml
forecast_model1:
  name: "j5vo"
  source: "mars"
  unit_conversion_factor: 1.0
  mars:
    class:  rd            # rd (research) | od (IFS oper) | ai (AIFS)
    type:   fc            # fc for deterministic; ignored in ensemble mode (cf+pf auto)
    stream: oper          # oper | enfo   (auto enfo when mode: ensemble)
    expver: "j5vo"
    levtype: sfc          # default sfc
    time:   "00"          # base cycle, default 00
    database: fdb         # optional (research/fdb data)
    grid: "0.25/0.25"     # optional regrid
    base_path: "/ec/vol/destine/continuous_evaluation/2mtemp/forecast/raw"
```

The data is stored in a folder **derived from the MARS identity keys**:
`{base_path}/{class}_{stream}_{expver}` (e.g. `.../raw/rd_oper_j5vo`,
`.../raw/od_enfo_0001`). Because the folder name is built from the same keys used to
retrieve, **the folder can never disagree with the expver retrieved** — this removes
the long-standing footgun of a hand-named folder not matching its contents. `param` is
derived from `variable` (`2t`→`2t`, `10ff`→`10u`+`10v`, `tp24`→`tp`); files are written
with the standard names the extractor reads (`2t_YYYYMMDD.grib`, etc.). For
`mode: ensemble` the control and 50 perturbed (`pf`) members are retrieved
together. `base_path` must not be under `$HOME` (small quota) — use `/ec/vol/...`,
`$SCRATCH`, or `$HPCPERM`.

> **ENS control after IFS Cycle 50r1 (12 May 2026):** for the *operational* IFS
> (`class: od`), the ensemble control forecast is no longer archived as
> `stream=enfo, type=cf` — from 12 May 2026 it lives in `stream=oper, type=fc`.
> The retriever handles this automatically **per day** (so a date range spanning
> the boundary works): operational `od` control uses `oper/fc` on/after 12 May 2026
> and `enfo/cf` before; AIFS (`class: ai`) and research (`class: rd`) keep `enfo/cf`;
> perturbed members are always `enfo/pf`. The ensemble extractor treats a `type=fc`
> control field as member 0, same as `cf`.

**Observations (`observation_source: "stvl"`)** — replace `local_gpt` with:

```yaml
observation_source: "stvl"
stvl:
  sources: ["synop"]
  base_path: "/ec/vol/destine/continuous_evaluation/2mtemp/obs/raw"
```

Observations are pulled from STVL (`vtb.media.stvl_retrieve`) for every forecast valid
time and written as Metview geopoints `{variable}_obs_YYYYMMDDHH.geo`. They are written
into an **identity-derived sub-folder** of `base_path`, named `stvl_{sorted_sources}`
(e.g. `.../obs/raw/stvl_synop`, `.../obs/raw/stvl_hdobs_synop`). This mirrors the
identity-folder scheme used for MARS forecasts and, crucially, **keeps STVL obs separate
from any pre-staged `local_gpt` observations that may already live directly under
`base_path`**. Without the sub-folder the two write the same `{variable}_obs_*.geo`
names, so with `skip_extraction_if_exists`/skip-if-present logic a `stvl` run would
silently reuse whichever obs happened to be there first (this was a real bug where a
`mars`+`stvl` run produced byte-identical results to a `local_gpt` run). The STVL
parameter is derived from `variable` (`tp24` uses `tp` with a 24 h period).

> **One retrieval call per valid time.** STVL obs are fetched with a *separate*
> `stvl_retrieve()` call per valid time (`date=[single_vdt]`) rather than by passing the
> whole date list at once. Batching a date list into one call makes `vtb` build one
> `Fieldset` per date and then run them through `Fieldset.aligned_fieldsets()`, which
> cross-matches stations by `[stationID, lat, lon]` proximity *across dates* and merges
> them — subtly changing the station population versus single-date calls. Retrieving one
> date at a time avoids that cross-date merging and keeps the obs set identical to the
> reference `local_gpt` / MARS+STVL path. (The same one-date-per-call scheme is applied
> in the in-memory `quaver_extract` backend, so all three retrieval paths agree.)

> **No special submission needed.** `vtb` is a compiled extension tied to the ECMWF
> `python3` module and is not importable from the project's `.venv`. The tool handles
> this automatically: the STVL step runs in-process if `vtb` is importable, otherwise it
> is run once in a subprocess under a vtb-capable Python (resolved via `module load
> python3`, or `$S4E_VTB_PYTHON` if set). So a `stvl` config runs unchanged via
> `sbatch submit_job.sh` — no need to switch the whole pipeline off the `.venv`.
> Forecast (`mars`) retrieval only needs the `mars` CLI and never `vtb`.

> External (non-ECMWF) users cannot use `mars`/`stvl` (no MARS/STVL access) — keep
> `local_grib` / `local_gpt` and pre-retrieve, or use `dataset_climatology`. The tool
> raises a clear error if `mars`/`vtb` are unavailable.

### Step 2 — Pre-process

- **Lapse-rate correction** (2t only): adjusts forecast 2t values to station elevation using `correction = lapse_rate × (station_height − model_height)`. Applied per member in ensemble mode. Enabled with `lapse_rate_correction: true`; setting it `false` disables the correction consistently in both deterministic and ensemble modes. Stations with an unrealistic correction (> 50 °C) or elevation mismatch (> 10000 m) are dropped as a quality safeguard, identically across all extraction backends.
- **Unit conversion**: `unit_conversion_factor` multiplied into GRIB values (e.g. 1000.0 converts m → mm for precipitation).
- **24 h accumulation** (tp24): `tp[step] − tp[step−24]` applied when `precipitation_accumulation_hours: 24`.
- **Wind speed** (10ff): computed as `sqrt(u² + v²)` per member.

### Step 3 — Extract Points

Interpolates gridded forecast fields to observation station lat/lon coordinates using nearest-gridpoint. Outputs one **parquet file per forecast day** (e.g. `2t_model1_vs_model2_day1.parquet`).

Each parquet file contains columns:
- `date`, `step`, `lat`, `lon`, `height`, `sdfor` — metadata
- `obs_value` — observed value
- `fc1_value`, `fc2_value` — deterministic forecast values
- `fc1_member_0 … fc1_member_50`, `fc2_member_0 … fc2_member_50` — ensemble members
- `lsm` — land-sea mask value at station (for coastal filter)

The `skip_extraction_if_exists: true` flag skips this step if the parquet files already exist, saving significant time on reruns.

For ensemble mode, intermediate per-date files are cached in a `_tmp/` subfolder. If a job is killed and resubmitted, already-extracted dates are automatically skipped.

### Step 4 — Filter Data

Applied to the loaded parquet data:

1. **Date range** — rows outside `[start_date, end_date]` are removed.
2. **Lead time** — if `lead_times` is specified, only those steps are kept.
3. **Season** — filter by month. Supported seasons: `DJF` (Dec–Feb), `MAM` (Mar–May), `JJA` (Jun–Aug), `SON` (Sep–Nov), `ASO` (Aug–Oct).
4. **Orography** — stations classified by `sdfor` (sub-grid standard deviation of orography): flat (0–40 m), hilly (40–120 m), complex (>120 m).
5. **Coastal removal** — stations with `lsm < coastal_lsm_threshold` are removed.
6. **QC filters** — temperature bounds, precipitation ceiling, outlier sigma clipping.

### Step 5 — Threshold

See [Section 6](#6-threshold-methods).

### Step 6 — Scores

See [Section 7](#7-verification-scores).

### Step 7 — Bootstrap

Generates 95% confidence intervals for score differences using block bootstrap resampling over dates × stations. The result for each score is stored as `score_diff`, `ci_lower`, `ci_upper` columns in the output CSV. Cells where the CI does not include zero are considered statistically significant.

### Step 8 — Save

Writes CSVs to `save.output_directory`. One CSV per `(season, orography_type)` combination, named e.g.:
```
scores_DJF_flat.csv
scores_DJF_hilly.csv
scores_DJF_complex.csv
scores_JJA_flat.csv
...
```

> **Give every experiment its own `output_directory`.** The per-score CSV names encode
> the threshold (via `_format_threshold_string()`), but the **heatmap PNGs and
> `observation_counts` files do not** — they are keyed only by `(season, orography)`. So
> two configs that differ *only* in threshold (e.g. a `fixed35` run and a `p99` run) but
> share one `output_directory` will silently **overwrite each other's heatmaps and count
> files** while leaving both sets of CSVs intact — an easy way to end up with a heatmap
> that doesn't match its CSV. Point each config at a dedicated folder (encode the model
> pair and threshold in the path, e.g. `results/2t_local_fixed35_ifs_oper_aifs1.0_oper`).

### Step 9 — Plot

Draws heatmap scorecards from the scores computed in Step 6. The colouring shows the **relative % difference** of model 1 vs model 2 for every (lead time × season × orography) cell: positive values (green) mean model 1 is better; the colourscale is symmetric (default ±20%, per-score).

#### Configuration options

All options live under the `plot:` block. Only these keys are read — any others are ignored.

| Key | Values | Default | Effect |
|-----|--------|---------|--------|
| `enabled` | `true` / `false` | `true` | Master switch. `false` skips Step 9 entirely. |
| `heatmap_style` | `"smooth"` / `"normal"` | `"normal"` | `smooth` = interpolated heatmaps **plus** a combined 4-score panel figure; `normal` = discrete blocky heatmaps, one per score. Use `smooth` for publication-style figures. |
| `create_summary` | `true` / `false` | `true` | Additionally draw the multi-panel diagnostic summary figure. Only produced for single-condition runs, and skipped automatically when re-plotting from saved CSVs (raw pairs aren't reloaded). |
| `dpi` | integer | `300` | Output resolution in dots-per-inch. Lower (e.g. `150`) for quick previews, higher for print. |
| `format` | `"png"` / `"pdf"` / `"svg"` | `"png"` | Output file type. Use `pdf`/`svg` for scalable vector figures. |
| `forecast_days` | list of ints, e.g. `[1, 3, 5]` | all scored days | Restrict the heatmap columns to these lead days. Omit to show every day that was scored. |

#### What gets produced

With `heatmap_style: "smooth"` a typical run writes, per season:

| File | Content |
|------|---------|
| `heatmap_smooth_panel_<var>_<m1>_vs_<m2>_<season>.<fmt>` | 4-panel summary scorecard (the headline figure) |
| `heatmap_smooth_<score>_<var>_<m1>_vs_<m2>_<season>.<fmt>` | one per-score heatmap for each available score |

The 4-panel headline figure contains:

| Panel | Score | Interpretation |
|-------|-------|----------------|
| A | twCRPS (ensemble) / twMAE (deterministic) | Overall threshold-weighted error |
| B | TW Spread/Skill Ratio (ensemble) / twRMSE (deterministic) | Reliability / spread |
| C | twQS at mid-tail level (q95 / q05) | Tail calibration at moderate extreme |
| D | twQS at extreme-tail level (q99 / q01) | Tail calibration at the extreme |

Positive values (green) indicate model 1 is better than model 2. The colorscale is symmetric ±20%.

#### Re-plotting without re-scoring

To tweak plot styling/format without recomputing scores, set `skip_scoring_if_exists: true` at the top level of the config and rerun. If the score CSVs already exist in the `save.output_directory`, Steps 4–8 are skipped and only Step 9 runs against the saved results (the `create_summary` figure is skipped in this mode).

**Example** — regenerate only the plots after changing appearance (e.g. switch to PDF at higher resolution):

```yaml
# in your config.yaml
skip_extraction_if_exists: true   # keep existing parquet files
skip_scoring_if_exists:    true   # keep existing score CSVs → only Step 9 runs

plot:
  enabled: true
  heatmap_style: "smooth"
  format: "pdf"                    # was "png"
  dpi: 300
```

```bash
# then just rerun the pipeline on the same config
python run.py config.yaml
```

Because both `skip_*_if_exists` flags are `true`, extraction and scoring are skipped and the run only redraws the figures from the saved CSVs — fast, and safe to repeat while iterating on `format`, `dpi`, `heatmap_style`, or `forecast_days`.

---

## 5. Supported Variables

### 2m Temperature (`2t`)

- GRIB parameter: `2t` (shortName)
- Observations: synop 2m temperature (°C after K→°C)
- Pre-processing: Kelvin→Celsius conversion always applied; lapse-rate correction strongly recommended for complex terrain
- Typical threshold: p1 (cold extremes) or p99 (warm extremes) from obs climatology
- `lead_time_frequency: 6` gives 6-hourly steps; results aggregated to daily means before scoring

### 24h Precipitation (`tp24`)

- GRIB parameter: `tp` (total precipitation, accumulated from T+0)
- Observations: 24h accumulated precipitation from synop/hdobs (mm)
- Pre-processing: m→mm conversion (`unit_conversion_factor: 1000`); 24h de-accumulation (`precipitation_accumulation_hours: 24`)
- Typical threshold: p99 (heavy precipitation)
- `lead_time_frequency: 24` (daily)

### 10m Wind Speed (`10ff`)

- GRIB parameters: `10u`, `10v` (U and V components)
- Wind speed computed as `sqrt(u² + v²)`
- Observations: 10m instantaneous wind speed (m/s)
- Typical threshold: p98 (strong winds)
- `wind_speed_from_components: true` required in `preprocess`

---

## 6. Threshold Methods

All methods are specified under `threshold.method` in the config.

### `fixed`

A single global value applied uniformly to all stations and dates.

```yaml
threshold:
  method: "fixed"
  fixed:
    value: -10.0
    event_type: "below"
```

Best for: WMO warning thresholds, physical process studies.
Caveat: ignores geographic and seasonal climatological variability.

### `dataset_climatology`

The Nth percentile of the observations in the current verification dataset (in-sample).

```yaml
threshold:
  method: "dataset_climatology"
  event_type: "above"
  dataset_climatology:
    percentile: 99
    use_filtered_data: true
```

Best for: quick exploratory runs where no external climatology is available.
Caveat: threshold changes if the dataset period or station list changes; not reproducible across experiments.

### `local_obs_climatology` *(recommended; requires pre-built climatology files)*

Per-station percentile from an **independent historical obs climatology**, pre-computed by `obs_clim_local/obsclim.py`.

> **ECMWF users**: the builder (`obs_clim_local/obsclim.py`) retrieves observations from the ECMWF STVL database. Pre-built files for Europe covering 2005–2024 are included in the repository at `obs_clim_local/`.
>
> **External users**: the obs climatology builder requires STVL (ECMWF-internal). If you do not have STVL access you cannot generate new climatology files. For your first runs, use `dataset_climatology` instead (see above) — it requires no external files and produces comparable results for a single well-defined verification period.

Stations are matched by nearest lat/lon (≤ 0.1° tolerance).

```yaml
threshold:
  method: "local_obs_climatology"
  event_type: "below"
  local_obs_climatology:
    path: "/path/to/scorecards4extremes/obs_clim_local"
    parameter: "2t"      # 2t | tp | 10si
    percentile: 1        # 1 for cold extremes, 99 for warm/wet
    window_days: 1
    n_years: 20
    first_year: 2005
    last_year: 2024
    min_availability_pct: 65
```

This method uses **monthly** climatology files (one per calendar month) allowing the threshold to vary seasonally. Produces the most physically consistent and stable results.

### `model_percentile`

Percentile of the model 1 forecast distribution (useful for model-relative thresholds).

### `station_climatology` *(ECMWF only — quaver backend)*

Retrieves per-station percentiles from the STVL ECMWF observation climatology database (1980–2009 baseline). Requires the `quaver` backend and ECMWF network access. Not available to external users; use `dataset_climatology` or `local_obs_climatology` instead.

---

## 7. Verification Scores

### Deterministic scores

| Score | Config key | Description |
|-------|-----------|-------------|
| Equitable Threat Score | `ETS` | Skill relative to random chance; 0 = no skill, 1 = perfect |
| Peirce Skill Score | `PSS` | POD − POFD; range −1 to 1 |
| Probability of Detection | `POD` | Fraction of observed events that were forecast |
| False Alarm Ratio | `FAR` | Fraction of forecast events that were false alarms |
| Threshold-weighted MAE | `twMAE` | MAE weighted to cases where the observation **or** the forecast exceeds the threshold (hits, misses, and false alarms) |
| Threshold-weighted RMSE | `twRMSE` | RMSE with the same threshold-exceedance weighting as twMAE |
| Bias | `bias` | Mean forecast − observation (all cases) |
| Mean Absolute Error | `mae` | MAE over all cases |
| Root Mean Square Error | `rmse` | RMSE over all cases |
| Pearson Correlation | `correlation` | Pearson correlation coefficient between forecast and observation |

### Ensemble scores

| Score | Config key | Description |
|-------|-----------|-------------|
| Fair CRPS | `fCRPS` | Bias-corrected CRPS; lower = better |
| Threshold-weighted CRPS | `twCRPS` | CRPS with chaining function focused on extremes |
| Brier Score | `Brier` | Probability score for the binary extreme event |
| Brier Skill Score | `BSS` | Brier score relative to climatological baseline |
| TW Quantile Score (mean) | `tw_quantile_score` | Mean twQS across all tail levels |
| TW Quantile Score — level | `tw_quantile_score_q001` etc. | twQS at a specific quantile level (see below) |
| Extreme Spread/Skill Ratio | `extreme_spread_skill_ratio` | Ensemble spread vs RMSE in the extreme tail |

#### Threshold-weighted Quantile Score levels

For **cold extremes** (`event_type: below`):

| Key | Alpha | Description |
|-----|-------|-------------|
| `tw_quantile_score_q001` | 0.01 | Deepest 1% of the cold tail |
| `tw_quantile_score_q005` | 0.05 | 5th percentile level |
| `tw_quantile_score_q010` | 0.10 | 10th percentile level |

For **warm/wet extremes** (`event_type: above`):

| Key | Alpha | Description |
|-----|-------|-------------|
| `tw_quantile_score_q090` | 0.90 | 90th percentile level |
| `tw_quantile_score_q095` | 0.95 | 95th percentile level |
| `tw_quantile_score_q099` | 0.99 | Deepest 99% of the warm tail |

---

## 8. Filters and Quality Control

### Season

Rows are filtered by the month of the **initialisation date** (`data['date']`), not the valid date. A Day-7 forecast issued in late November therefore falls in SON, even though its valid time is in early December.

| Season | Init-date months |
|--------|------------------|
| DJF | Dec, Jan, Feb |
| MAM | Mar, Apr, May |
| JJA | Jun, Jul, Aug |
| SON | Sep, Oct, Nov |
| ASO | Aug, Sep, Oct |

### Orography (terrain complexity)

Classified by `sdfor` — the **sub-grid standard deviation of orography** (in metres). This field is interpolated from the `sdfor_path` GRIB file to station locations during extraction.

| Class | sdfor range | Terrain |
|-------|------------|---------|
| `flat` | 0–40 m | Plains, low-lying coastal areas |
| `hilly` | 40–120 m | Rolling hills, gentle slopes |
| `complex` | 120–3000 m | Alpine, mountainous terrain |

### Coastal station removal

Stations with `lsm < coastal_lsm_threshold` (default 0.9) are considered coastal and removed. The LSM value is interpolated from each model's own land-sea mask field to avoid model-dependency.

### Quality control

- **Temperature**: obs outside `[min_valid_temperature, max_valid_temperature]` (°C) are removed.
- **Precipitation**: obs above `max_valid_precipitation` (mm) are removed.
- **Outlier sigma clipping**: when `remove_outliers: true`, pairs where `|obs − mean| > outlier_threshold_std × std` are removed.

---

## 9. Output Files and Directory Structure

```
/perm/user/results/my_experiment/
├── scores_by_leadtime_2t_DJF_flat.csv                 ← per-lead-time scores + CIs
├── overall_scores_2t_DJF_flat.csv                     ← period-aggregated scores
├── scores_by_leadtime_2t_DJF_complex.csv
├── ...
├── heatmap_smooth_twMAE_2t_modelA_vs_modelB_DJF.png   ← one heatmap per score
├── heatmap_smooth_twCRPS_2t_modelA_vs_modelB_DJF.png
├── ...
└── heatmap_smooth_panel_2t_modelA_vs_modelB_DJF.png   ← 4-panel summary scorecard
```

Heatmap filenames follow `heatmap_smooth[_<score>]_<variable>_<model1>_vs_<model2>_<season>.<format>`
(the `_smooth` segment is dropped when `heatmap_style: "normal"`; `<format>` follows `plot.format`).
Terrain classes are column groups **within** each figure, not separate files.

### CSV format

Each score CSV contains one row per forecast lead time with columns:

```
lead_time | fc1_ETS | fc2_ETS | ETS_diff | ETS_ci_lower | ETS_ci_upper | ...
```

`diff` = (model1 − model2) / model2 × 100 %  (positive = model1 better)

---

## 10. Ensemble Mode

Set `mode: "ensemble"` and list `steps` instead of `forecast_days`.

### Config differences from deterministic

```yaml
mode: "ensemble"
steps: [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

ensemble:
  n_members: 50
  include_control: true   # include control member (member 0) as member 51
```

### Extraction differences

- Ensemble extraction is handled by `extract_points_ensemble.py`
- Per-date files are saved in a `_tmp/` subfolder during extraction; on completion they are merged into per-day parquet files
- Each parquet row contains `fc1_member_0 … fc1_member_50` and `fc2_member_0 … fc2_member_50` columns
- Lapse-rate correction is applied **per member**

### Scoring differences

- Sub-daily steps (e.g. 6-hourly) are **aggregated to daily means** before scoring when `method: local_obs_climatology` is active
- Worker processes are capped at 4 to avoid memory spikes during parallel bootstrap
- Bootstrap `n_samples: 200` is recommended for large ensemble jobs (reduces run time ~5×)

---

## 11. Obs Climatology Builder

The `local_obs_climatology` threshold method requires pre-built climatology files.

> **ECMWF users**: pre-built files for Europe (2t, tp, 10ff, 2005–2024) are
> already included in the repository under `obs_clim_local/`. You only need
> to run the builder if you want a different parameter, period, or domain.
>
> **External users**: the obs climatology builder (`obs_clim_local/obsclim.py`)
> retrieves observations from the STVL ECMWF internal database and will not
> work outside ECMWF. For external workflows, use `dataset_climatology` as
> the threshold method — it computes the percentile directly from the
> observations in your verification dataset without any external files.

### Standard Europe climatology (`obs_clim_local/`)

Builds **monthly** per-station percentile files for 2t, tp, and 10ff over a configurable multi-year period from STVL synop/hdobs data.

```bash
# Edit obs_clim_local/obsclim.py to set:
#   param      = '2t'  # or 'tp', '10si'
#   fyear      = 2005
#   lyear      = 2024
#   crit       = 0.65  # minimum availability fraction

# Submit as SLURM job:
sbatch obs_clim_local/run.sh
```

Output files are named:
```
clim_2t_1_01_20years_2005_2024_65   # month=01, 20 years, 65% availability
clim_2t_1_02_20years_2005_2024_65   # month=02
...
clim_2t_1_12_20years_2005_2024_65   # month=12
```

### Italy tp annual climatology (`obs_clim_local_Italy/`)

Builds a **whole-year** (non-seasonal) tp climatology for stations within Italy (lat 36–47°N, lon 6–19°E). All 12 monthly output files are identical (same annual pool).

```bash
sbatch obs_clim_local_Italy/run.sh
```

---

## 12. Diagnostic and Analysis Tools

### `diagnostics/diagnose_extremes.py` — Comprehensive single-condition diagnostics

Produces **22 diagnostic plots** for a specific (day, season, orography, threshold) combination. Reads from the extracted parquet files; the main pipeline must have completed through Step 3 first.

```bash
python diagnostics/diagnose_extremes.py \
  --config configs/deterministic/config_2t_local_p1obsclim.yaml \
  --day 3 --threshold-pct 1 --season DJF --orog complex

# Or with a fixed threshold value:
python diagnostics/diagnose_extremes.py \
  --config configs/deterministic/config_tp24_local_p99obs.yaml \
  --day 3 --threshold-value 30.0 --season JJA --orog flat

# Submit as SLURM job (64 GB, 4 CPU, 4 h):
sbatch submit_diagnose.sh --config configs/deterministic/config_2t_local_p1obsclim.yaml \
  --day 3 --threshold-pct 1 --season DJF --orog complex
```

**Plots generated:**

| # | Name | Content |
|---|------|---------|
| 1 | Skill Score Comparison | POD, FAR, CSI for both models |
| 2 | ETS and PSS Comparison | Bar charts per lead time |
| 3 | Skill Score Evolution | Scores vs threshold sweep (p1–p99) |
| 4 | Error Distribution | Histogram, boxplot, scatter, stats table |
| 5 | Frequency Bias Evolution | Forecast vs observed event frequency |
| 6 | Empirical Distributions | Obs and forecast PDFs; tail highlighted |
| 7 | Q-Q Plots | Full-range Q-Q with tail segments coloured |
| 8 | Contingency Table | 2×2 table for both models with counts |
| 9 | Conditional Error Analysis | MAE/MSE for extreme vs all cases |
| 10 | twMAE Decomposition | Hit / miss / false-alarm contributions |
| 11 | twMAE Percentile Decomposition | twMAE broken down across threshold sweep |
| 12 | twMAE Component Fractions | 100% stacked bars — dominant failure mode |
| 13 | Extreme Intensity Scatter | fc vs obs for hits — intensity bias |
| 14 | Miss & FA Severity | How extreme were missed events and false alarms? |
| 15 | Conditional Bias & Noise | Systematic vs random error on extreme cases |
| 16 | twMAE Skill Score | Relative to obs-based reference (analogous to BSS) |
| 17 | Error Depth Profile | Error binned by exceedance magnitude (hits only) |
| 18 | Summary Scorecard Table | All components + auto-generated narrative |
| 19 | Count Evolution | Absolute hits/misses/FAs across threshold sweep |
| 20 | Count Difference | Δhits/Δmisses/ΔFAs between models across sweep |
| 21 | Detection Profile | Normalised 100% stacked bars — fraction of sample |
| 22 | Conditional Bias Decomposed | Splits Plot 15's conditional bias into **real events** (`fc − obs \| obs ≥ T`, hits ∪ misses — negative = under-prediction of true extremes) vs **false alarms** (`fc − obs \| fc ≥ T, obs < T` — always positive), so the two opposite-signed error sources are not blended into one misleading average |

Output goes to `{save.output_directory}/day{N}_pct{P}_{SEASON}_{OROG}/`.

### `diagnostics/diagnose_det_extremes_simple.py` — Quick deterministic diagnostic

A lightweight 3-row figure for a fast visual check; does not require the full score CSV.

```bash
python diagnostics/diagnose_det_extremes_simple.py \
  --config configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml \
  --orog flat --days 1 3 --season JJA
```

**Rows produced:**
- **Row 1** — KDE of `(fc − T)` and `(obs − T)` for both models when obs exceeded T (0 = exactly at threshold; left = missed extreme).
- **Row 2** — Box plots of `(fc − obs)` error during extreme events (0 = perfect).
- **Row 3** — Stacked bar chart of hits / misses / false alarms, consistent with the heatmap CSV.

Pools data across the requested forecast days. Output PNG saved to `{save.output_directory}/`.

### `diagnostics/diagnose_twcrps_simple.py` — Quick ensemble diagnostic

Ensemble counterpart of `diagnostics/diagnose_det_extremes_simple.py`; replicates the pipeline's 6-hourly → daily aggregation internally.

```bash
python diagnostics/diagnose_twcrps_simple.py \
  --config configs/ensemble/config_2t_ens_local_p99obsclim_aifsvsifs.yaml \
  --orog low --day 5
```

**Rows produced:**
- **Row 1** — KDE of obs (green), model 1 ensemble median (red), model 2 ensemble median (blue), each normalised as `(temperature − threshold T)`.
- **Row 2** — Box plots of `(ensemble_median − obs)` during extreme events.
- **Row 3** — Bar chart of twCRPS for each model, annotated with percentage difference.

Output PNG saved to `{save.output_directory}/`.

### `diagnostics/plot_qq_extremes.py` — Q-Q plot (warm + cold + cold-zoom panels)

```bash
python diagnostics/plot_qq_extremes.py \
  --config configs/deterministic/config_2t_local_p1obsclim.yaml \
  --day 3 --season DJF --orog complex
```

Three-panel figure:
- **Panel A**: Warm extremes (p90–p99.9), dots every 1%, × markers every 0.1%
- **Panel B**: Cold extremes overview (p0.1–p10), same marker scheme
- **Panel C**: Cold extremes zoom — connected line from p10 to p0.1, every 1% down to p1 then every 0.1%

### `diagnostics/plot_qq.py` — General Q-Q plot

```bash
python diagnostics/plot_qq.py --config configs/deterministic/config_tp24_precipitation.yaml \
  --season DJF MAM JJA SON --orog flat mid high
```

### `diagnostics/plot_station_diagnostics.py` — Per-station spatial maps

Takes the **extracted points directory** as a positional argument (not a config file). Forecast mode (deterministic vs ensemble) is auto-detected from the parquet column names. Produces scatter maps on a European map with station dots coloured by bias, error, or threshold-exceedance class.

```bash
# Ensemble parquet in /extracted_points/2t_ens:
python diagnostics/plot_station_diagnostics.py ./extracted_points/2t_ens \
  --threshold 30 --event-type above --variable 2t \
  --forecast-day 3 --model1-name ifs_ens --model2-name aifs_ens \
  --output-dir ./plots/station_diagnostics/2t_warm

# Deterministic:
python diagnostics/plot_station_diagnostics.py ./extracted_points/2t \
  --threshold -5 --event-type below --variable 2t \
  --forecast-day 3 --model1-name ifs_oper --model2-name iekm \
  --output-dir ./plots/station_diagnostics/2t_cold
```

### Case-study tools (`case_studies/`)

Two-step workflow for identifying and visualising specific dates where one model clearly outperforms the other. See [docs/CASE_STUDIES.md](CASE_STUDIES.md) for full details.

```bash
# Step 1 — rank all dates by model-performance gap:
python case_studies/find_case_studies.py \
  --config configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml \
  --days 3 5 7 --season JJA --top-n 20

# Step 2 — visualise the worst date:
python case_studies/plot_case_study.py \
  --config configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml \
  --date 20250718 --day 5 --season JJA
```

---

## 13. Observation Data Format — `.gpt` / Geopoints

Observation files must be in **Geopoints** format (file extension `.gpt` or
`.geo`). Geopoints is a plain-text format originally defined by Metview/ECMWF,
but it is straightforward to produce from any tabular observation dataset.

### File structure

A Geopoints file looks like this (the `#GEO` header is mandatory):

```
#GEO
FORMAT XYV
DATE 20240101
TIME 000000
#DATA
lat       lon       height   date      time  value
48.2000   16.3667   198.0    20240101  0     -3.5
51.4667   -0.4500    25.0    20240101  0      2.1
47.8000    2.5000   152.0    20240101  0      0.8
...
```

- `lat`, `lon`: station coordinates in decimal degrees
- `height`: station elevation in metres above sea level
- `date`: YYYYMMDD
- `time`: HHMMSS (use 000000 for 00 UTC)
- `value`: the observed variable value in the units expected by the pipeline
  (°C for 2t, mm for tp24, m/s for 10ff)

### Naming convention

The pipeline (`read_data.py`) looks for files named:

| Variable | Pattern 1 | Pattern 2 |
|----------|-----------|-----------|
| `2t` | `2t_YYYYMMDD.gpt` | `2t_obs_YYYYMMDD00.geo` |
| `tp24` | `tp24_YYYYMMDD.gpt` | `tp24_obs_YYYYMMDD00.geo` |
| `10ff` | `10ff_YYYYMMDD.gpt` | `10ff_obs_YYYYMMDD00.geo` |

All files for the verification period must be in the directory specified by
`read_data.local_gpt.path`.

### Converting your observations to Geopoints format

If your observations are in a different format (CSV, NetCDF, etc.), convert
them with a short Python script before running the pipeline:

```python
import pandas as pd

# Example: convert a CSV with columns lat, lon, elev, date, value
df = pd.read_csv("my_obs.csv")

for date, group in df.groupby("date"):
    lines = ["#GEO", "FORMAT XYV", f"DATE {date}", "TIME 000000", "#DATA",
             "lat       lon       height   date      time  value"]
    for _, row in group.iterrows():
        lines.append(
            f"{row.lat:.4f}  {row.lon:.4f}  {row.elev:.1f}  "
            f"{date}  0  {row.value:.2f}"
        )
    out_file = f"2t_{date}.gpt"
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
```

### Where to obtain synoptic observations

- **SYNOP data** is exchanged under WMO agreements and is available from
  national meteorological services and global archives such as
  [NCEI ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)
  or [OGIMET](https://www.ogimet.com).
- **ERA5 station-level diagnostics** or **SEVIRI AMVs** from the
  [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu)
  can supplement surface obs for some applications.
- At ECMWF, observations are available from STVL (the internal observation
  server) via the `quaver` backend or `vtb.media.stvl_retrieve()`.

---

## 14. Batch Submission on ECMWF HPC

### Submitting a job

```bash
cd /path/to/scorecards4extremes
sbatch submit_job.sh config_my_experiment.yaml
```

### Monitoring

```bash
squeue -u $USER                    # check queue status
tail -f scorecards_<JOBID>.out     # live output
tail -f scorecards_<JOBID>.err     # live errors
```

### Cancelling

```bash
scancel <JOBID>
```

### Restarting after wall-time kill

The pipeline supports **incremental restart** without re-doing already completed work:

- **Extraction** (`skip_extraction_if_exists: false`): Per-date progress is cached in `_tmp/`. On resubmission, already-extracted dates are skipped automatically.
- **Scoring** (`skip_scoring_if_exists: true`): If score CSVs already exist, the scoring step is skipped entirely.

### Specialist submission scripts

| Script | Purpose | Resources |
|--------|---------|----------|
| `submit_job.sh` | Main pipeline (extract + score + plot) | 128 GB, 12 CPU, 48 h |
| `submit_qq.sh` | Q-Q plot tool (`diagnostics/plot_qq.py`) | 64 GB, 1 CPU, 1 h |
| `submit_diagnose.sh` | `diagnostics/diagnose_extremes.py` (simple wrapper, passes all args through) | 32 GB, 1 CPU, 2 h |
| `scripts/submit_diagnose_job.sh` | `diagnostics/diagnose_extremes.py` (race-condition-safe multi-job helper) | 64 GB, 4 CPU, 4 h |
| `scripts/submit_extraction.sh` | tp24 obs extraction from STVL | 64 GB, 4 CPU, 12 h |
| `case_studies/submit_find_cases.sh` | `find_case_studies.py` via `--export` env vars | 64 GB, 1 CPU, 4 h |
| `case_studies/submit_plot_case.sh` | `plot_case_study.py` via `--export` env vars | 32 GB, 1 CPU, 1 h |
