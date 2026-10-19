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
| `deterministic` | Single-valued forecast vs obs | SEDI, EDI, ETS, PSS, POD, FAR, twMAE (+ components), twRMSE, bias, MAE, RMSE, SEEPS (tp24) |
| `ensemble` | Probabilistic ensemble forecast | twCRPS, fCRPS, CRPSS, Brier, BSS, reliability, twQS (multi-level), extreme_spread_skill_ratio, ens_spread (all finite-ensemble corrected) |

### Supported variables

| Variable | Description | Unit |
|----------|-------------|------|
| `2t` | 2 m temperature (daily **mean**) | °C (after K→°C conversion + optional lapse-rate correction) |
| `mx2t` | Daily **maximum** 2 m temperature (derived from 6-hourly `2t`) | °C — same as `2t`; warm-extreme (p99) verification |
| `mn2t` | Daily **minimum** 2 m temperature (derived from 6-hourly `2t`) | °C — same as `2t`; cold-extreme (p1) verification |
| `10ff` | 10 m wind speed | m/s (from U/V components) |
| `mx10ff` | Daily **maximum** 10 m wind speed (derived from 6-hourly `10ff`) | m/s — same as `10ff`; strong-wind (p99) verification |
| `tp24` | 24 h total precipitation | mm (after m→mm conversion and optional accumulation) |
| `aod500`, `pm2p5`, `pm10`, `go3`, `no2`, `so2`, `co`, `no` | CAMS atmospheric composition (AOD / surface chemistry vs AERONET/AirNow) | see [§5 CAMS variables](#5-supported-variables) — different config schema (`source: "cams_netcdf"`), deterministic mode only |

---

## 2. Repository Structure

```
scorecards4extremes/
├── run.py                          ← Main pipeline entry point (thin wrapper over src/run.py)
├── config_example.yaml             ← Clean starter config (no ECMWF dependencies)
├── requirements.txt                ← Python dependencies
├── README.md                       ← Project overview + repository layout
│
├── src/                            ← Core pipeline modules (the scorecards tool)
│   ├── run.py                      ← 9-step workflow runner (main())
│   ├── read_data.py                ← Step 1: Read GRIB/GPT files
│   ├── preprocess.py               ← Step 2: Unit conversion, lapse-rate, accumulation
│   ├── extract_points.py           ← Step 3: Interpolate to station locations (deterministic)
│   ├── extract_points_ensemble.py  ← Step 3: Interpolate to station locations (ensemble)
│   ├── filter.py                   ← Step 4: Season, orography, QC filters
│   ├── threshold.py                ← Step 5: Compute event threshold
│   ├── det_scores.py               ← Step 6: Deterministic verification scores
│   ├── ens_scores.py               ← Step 6: Ensemble verification scores
│   ├── save.py                     ← Step 8: Write results to CSV
│   ├── plot.py                     ← Step 9: Draw heatmap scorecards
│   ├── cams_extract.py             ← Step 3 backend: raw CAMS NetCDF extraction
│   ├── quaver_extract.py           ← Step 3 backend: quaver extraction
│   ├── quaver_backend.py           ← quaver retrieval helper
│   ├── quaver_compute_backend.py   ← quaver compute helper
│   ├── mars_retrieve.py            ← MARS/STVL retrieval helper
│   ├── plot_paper.py               ← Publication-quality figure helper
│   ├── utils.py                    ← Shared utilities (steps, geometry, IO)
│   └── season_utils.py             ← Season masking helpers
│
├── diagnostics/                    ← Standalone diagnostic and visualisation tools
│   ├── diagnose_extremes.py        ← Comprehensive single-condition diagnostic (22 plots)
│   ├── diagnose_det_extremes_simple.py ← Quick 3-panel deterministic diagnostic
│   ├── diagnose_twcrps_simple.py   ← Quick 3-panel ensemble diagnostic
│   ├── diagnose_ens_extremes.py    ← Comprehensive ensemble diagnostic (twCRPS vs quantile score)
│   ├── plot_qq_extremes.py         ← Q-Q plot with warm + cold zoom panels
│   ├── plot_qq.py                  ← Full-range Q-Q plot
│   └── plot_station_diagnostics.py ← Per-station spatial maps (mode auto-detected)
│
├── analysis/                       ← One-off analysis and figure scripts
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
├── configs/                        ← Experiment config templates
│   ├── deterministic/              ← Deterministic mode configs
│   ├── ensemble/                   ← Ensemble mode configs
│   └── cams/                       ← CAMS composition configs
│
├── scripts/                        ← SLURM submission + auxiliary helper scripts
│   ├── submit_job.sh               ← Main pipeline SLURM wrapper
│   ├── submit_job_quaver.sh        ← Pipeline wrapper with vtb/quaver env
│   ├── submit_qq.sh                ← Q-Q plot SLURM wrapper
│   ├── submit_diagnose.sh          ← diagnose_extremes.py SLURM wrapper
│   ├── check_mx2t_availability.sh  ← MARS mx2t availability check
│   ├── chunk_extraction.py         ← Split an extraction into date chunks / merge them back
│   ├── watch_and_score.sh          ← Waits for chunks, merges, scores (+ extra configs, post step)
│   ├── check_extracted.py          ← Guard: is a config's merged extraction complete?
│   ├── plot_combined_regions.py    ← One scorecard with NH-orography + tropics columns
│   ├── plot_ocean_twcrps_panels.py ← Multi-experiment panel per variable (--score twCRPS|Brier|…)
│   ├── plot_ocean_ranking_summary.py ← Experiment ranking summary plots (--score …)
│   ├── make_ocean_ens_configs.py   ← Example generator: many experiments × regions × variables
│   └── ...                         ← Other batch/utility helpers
│
└── docs/                           ← This documentation folder
```

---

## 3. Configuration File Reference

Every experiment is defined by a single YAML file. Below is a fully-annotated template.

```yaml
# ── Basic settings ─────────────────────────────────────────────────────────
variable: "2t"              # tp24 | 2t | mx2t | mn2t | 10ff | mx10ff | aod500 | pm2p5 | pm10 | go3 | no2 | so2 | co | no
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
  area: "europe"          # europe | nh_extratropics | tropics | nh_extratropics_tropics | [N, W, S, E]
                          # bounding box applied during extraction (null = no box; see filter.area)

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
  area: null             # optional named region selected AFTER extraction (requires
                         # extract_points.area: null) — lets one extraction be scored per region (§8)
  lead_times: null       # null = use all; or list of hours e.g. [24, 48, 72]
  season: ["DJF", "JJA"]  # DJF | MAM | JJA | SON | ASO (Aug-Sep-Oct) | null
  orography_type: ["flat", "hilly", "complex"]
  orography_ranges:
    flat:    [0, 40]
    hilly:   [40, 120]
    complex: [120, 3000]
  remove_coastal_stations: true
  coastal_lsm_threshold: 0.9     # stations with lsm < 0.9 are considered coastal
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
    - "tw_quantile_score_q001"   # cold: α=0.001 (deep tail, below column syntax)
    - "tw_quantile_score_q0005"  # cold: α=0.005
    - "tw_quantile_score_q099"   # warm: α=0.99
    - "tw_quantile_score_q0995"  # warm: α=0.995
    - "tw_quantile_score_q0999"  # warm: α=0.999
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
derived from `variable` (`2t`/`mx2t`/`mn2t`→`2t`, `10ff`→`10u`+`10v`, `tp24`→`tp`);
files are written with the standard names the extractor reads (`2t_YYYYMMDD.grib`,
etc.). `mx2t`/`mn2t` retrieve ordinary `2t` (they are NOT native archived fields) and
differ only in the daily aggregation applied at scoring time. For
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
> `sbatch scripts/submit_job.sh` — no need to switch the whole pipeline off the `.venv`.
> Forecast (`mars`) retrieval only needs the `mars` CLI and never `vtb`.

> External (non-ECMWF) users cannot use `mars`/`stvl` (no MARS/STVL access) — keep
> `local_grib` / `local_gpt` and pre-retrieve, or use `dataset_climatology`. The tool
> raises a clear error if `mars`/`vtb` are unavailable.

#### Direct extraction — `source: "cams_netcdf"` (CAMS atmospheric composition)

CAMS variables (`aod500`, `pm2p5`, `pm10`, `go3`, `no2`, `so2`, `co`, `no`) are read
straight from raw per-site model/observation NetCDF files — no GRIB, no gridded
interpolation, and no `backend:` key needed (auto-selected, see below). Both
`forecast_model1`/`forecast_model2` use `source: "cams_netcdf"`, and
`observation_source` is `"cams_netcdf"` too:

```yaml
variable: "pm2p5"
read_data:
  forecast_model1:
    name: "icki_airnow"          # defaults expid to this name; override with cams_netcdf.expid
    source: "cams_netcdf"
    cams_netcdf:
      path: "/path/to/icki_airnow"       # dir containing <expid>_<YYYYMMDD>_<run_hour>.nc
  forecast_model2:
    name: "oper_airnow"
    source: "cams_netcdf"
    cams_netcdf:
      path: "/path/to/oper_airnow"
  observation_source: "cams_netcdf"
  cams_netcdf:
    obs_path: "/path/to/obs_airnow"      # AirNow (surface chemistry) or AERONET (aod500) NetCDF
```

Model/observation directory paths, names and (optional) `expid`/`run_hour` overrides
are entirely config-driven — nothing is hardcoded to any specific experiment name.
The pipeline auto-selects the `cams_extract` backend the moment `variable` is a
known CAMS species and no pre-extracted parquet already exists (governed by
`skip_extraction_if_exists`, same as any other backend); no explicit `backend:` key
is required. See [§5 CAMS variables](#5-supported-variables) for the species table
and caveats (deterministic only, no orography/lapse-rate correction, no
`local_obs_climatology`/`station_climatology` threshold support).

### Step 2 — Pre-process

- **Lapse-rate correction** (2t only): adjusts forecast 2t values to station elevation using `correction = lapse_rate × (station_height − model_height)`. Applied per member in ensemble mode. Enabled with `lapse_rate_correction: true`; setting it `false` disables the correction consistently in both deterministic and ensemble modes. Stations with an unrealistic correction (> 50 °C) or elevation mismatch (> 10000 m) are dropped as a quality safeguard, identically across all extraction backends.
- **Unit conversion**: `unit_conversion_factor` multiplied into GRIB values (e.g. 1000.0 converts m → mm for precipitation).
- **24 h accumulation** (tp24): `tp[step] − tp[step−24]` applied when `precipitation_accumulation_hours: 24`.
- **Wind speed** (10ff): computed as `sqrt(u² + v²)` per member.

### Step 3 — Extract Points

Interpolates gridded forecast fields to observation station lat/lon coordinates using nearest-gridpoint. Outputs one **parquet file per forecast day** (e.g. `2t_model1_vs_model2_day1.parquet`).

> **CAMS variables** use a different, fused Step 1+3 backend (`cams_extract.py`) that reads
> raw per-site model/obs NetCDF directly (no gridded interpolation needed — see
> [§5 CAMS variables](#5-supported-variables)). It writes the same parquet schema minus
> `sdfor`/`lsm` (no orography data for these species), so Steps 4-9 are unaffected.

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
6. **QC filters** — temperature bounds and precipitation ceiling (physical-validity caps).

### Step 5 — Threshold

See [Section 6](#6-threshold-methods).

### Step 6 — Scores

See [Section 7](#7-verification-scores).

### Step 7 — Bootstrap

Generates 95% confidence intervals for score differences using a paired **date-block** bootstrap: whole initialisation dates are resampled with replacement, keeping all stations of a drawn date together, so same-day spatial correlation between stations does not inflate the effective sample size. This applies to both backends (`local` and `quaver`); if a cell has fewer than 5 distinct dates it falls back to case-level resampling. It runs inside Step 6, not as a separate module. The result for each score is stored as `score_diff`, `ci_lower`, `ci_upper` columns in the output CSV. Cells where the CI does not include zero are considered statistically significant.

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
| `color_limits` | mapping, e.g. `{twMAE: 20}` or `{default: 20}` | built-in per score (e.g. twMAE ±50%) | Symmetric colourbar range (±%) of the % difference heatmaps, per score name; `default` applies to every score not listed. Use a narrower range when differences are small. |

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

### Daily Max / Min 2m Temperature (`mx2t` / `mn2t`)

`mx2t` and `mn2t` verify the **daily maximum** and **daily minimum** 2 m
temperature. They are **not** native archived fields (many models, e.g. AIFS,
do not store `mx2t`/`mn2t`) — they are **derived from the ordinary 6-hourly
`2t`**: for every forecast day the max (`mx2t`) or min (`mn2t`) of the sub-daily
2t values is taken, on **both** the forecast and the observations, at scoring
time.

Because they are physically `2t`, everything upstream of the daily
aggregation is identical to `2t` — retrieval (`param=2t`), GRIB/obs file IO,
K→°C conversion, lapse-rate correction and 2t quality control all treat
`mx2t`/`mn2t` exactly as `2t` (via `utils.physical_variable`). Only two things
differ, both driven by the `variable:` field:

- **Daily aggregation** (`utils.daily_aggregation_method`): `mean` for `2t`,
  `max` for `mx2t`, `min` for `mn2t`.
- **Climatology reference**: the `threshold.local_obs_climatology.parameter`
  must be set to `mx2t`/`mn2t` so the per-station percentiles are read from the
  matching `clim_mx2t_*` / `clim_mn2t_*` files (daily-max / daily-min obs
  percentiles), not the daily-mean `clim_2t_*` files.

| `variable` | Aggregation | Clim `parameter` | Typical `event_type` / `percentile` |
|-----------|-------------|------------------|-------------------------------------|
| `2t`   | daily mean | `2t`   | `above`/99 (warm) or `below`/1 (cold) |
| `mx2t` | daily max  | `mx2t` | `above`/99 (warm max extremes) |
| `mn2t` | daily min  | `mn2t` | `below`/1 (cold min extremes) |

> **Requires sub-daily samples.** Because the daily max/min is taken over the
> sub-daily 2t values, deterministic `mx2t`/`mn2t` configs **must** use
> `lead_time_frequency: 6` (so each day has the 06/12/18/24 h samples to
> aggregate); `lead_time_frequency: 24` would give one value per day and make
> the max/min meaningless. Ensemble configs already retrieve 6-hourly `steps`,
> so no change is needed there.

Build the `clim_mx2t_*` / `clim_mn2t_*` climatology files with
`obs_clim_local/run_mx2tmn2t.sh` (daily max/min of 6-hourly STVL 2t, consistent
with the forecast aggregation) before running a `local_obs_climatology`
threshold. MARS availability of the underlying 2t can be checked with
`check_mx2t_availability.sh`.

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

### Daily Max 10m Wind Speed (`mx10ff`)

`mx10ff` verifies the **daily maximum** of the 6-hourly 10 m wind speed, built the
same way as `mx2t`: it is physically `10ff` (same U/V GRIB, observations, wind QC and
extraction, via `utils.physical_variable`), and for every forecast day the **max** of
the sub-daily values is taken on both forecast and observations at scoring time.

- Requires `lead_time_frequency: 6` (the 06/12/18/24 h samples are what is maximised).
- Set `threshold.local_obs_climatology.parameter: "mx10ff"` so the per-station
  percentiles come from the daily-max climatology (`clim_mx10ff_*`), not the
  daily-mean `clim_10ff_*` files. Build them with `obs_clim_local/run_mx10ff.sh`
  (daily max of 6-hourly STVL 10ff, 2005-2024, 65% availability).
- An existing `10ff` extraction can be reused: extraction files are named after
  `variable`, so link the `10ff_*_day{d}.parquet` files as `mx10ff_*_day{d}.parquet`
  in a separate `output_path` instead of re-extracting.
- Station ids in climatology files may be non-numeric (e.g. `_1961389`); thresholds
  are matched to observations by latitude/longitude, so this is harmless.

### CAMS Atmospheric Composition Variables

Read via the raw-NetCDF `cams_extract` backend (`source: "cams_netcdf"`, see
[Step 1](#4-pipeline-steps-in-detail)) — no GRIB/gridded interpolation involved.

| Variable | Description | Obs network | Unit | Typical threshold |
|----------|-------------|-------------|------|--------------------|
| `aod500` | Aerosol optical depth at 500 nm | AERONET | dimensionless | fixed (e.g. 0.2–0.5) |
| `pm2p5` | PM2.5 surface concentration | AirNow | µg/m³ | fixed 35 (US EPA "Unhealthy for Sensitive Groups" 24h) |
| `pm10` | PM10 surface concentration | AirNow | µg/m³ | fixed |
| `go3` | Surface ozone | AirNow | ppb | fixed 70 (US EPA "Unhealthy for Sensitive Groups" 8h O3) |
| `no2` | Surface NO2 | AirNow | ppb | fixed |
| `so2` | Surface SO2 | AirNow | ppb | fixed |
| `co` | Surface CO | AirNow | ppm | fixed |
| `no` | Surface NO | AirNow | ppb | fixed |

Notes and caveats:

- **Unit conversion is automatic** inside `cams_extract.py` (kg/m³→µg/m³ for
  PM species; kg/kg→ppb/ppm for gases via the molar-mass ratio) — no
  `unit_conversion_factor` needed in the config.
- **Deterministic mode only** — there is no ensemble Step-3 path for CAMS species.
- **No orography/coastal filtering or lapse-rate correction**: the per-site
  NetCDF has no `sdfor`/`lsm`/height columns, so `filter.orography_type` and
  `remove_coastal_stations` are self-skipped and `lapse_rate_correction` has no
  effect for these variables.
- **Threshold methods**: only `fixed`, `dataset_climatology` and
  `model_percentile` apply — `station_climatology` and `local_obs_climatology`
  require pre-built per-station climatology files that don't exist for CAMS
  species (see [§6 Threshold Methods](#6-threshold-methods)).
- **Area filtering** still works via `filter.area` (predefined boxes: `europe`,
  `nh_extratropics`, `tropics`) since it's a plain lat/lon subset.
- Add a new species by registering it in `cams_extract.SPECIES` (model/obs
  variable names + unit-conversion kind) — no other code changes needed.

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
| Equitable Threat Score | `ETS` | Skill relative to random chance; 0 = no skill, 1 = perfect. **Degenerates toward 0 for rare events — prefer SEDI for the tail.** |
| Peirce Skill Score | `PSS` | POD − POFD; range −1 to 1. Also degenerates for rare events. |
| Symmetric Extremal Dependence Index | `SEDI` | **Recommended categorical score for extremes.** Non-degenerate as the event gets rarer (Ferro & Stephenson 2011); 0 = no skill, 1 = perfect. See bias caveat below. |
| Extremal Dependence Index | `EDI` | Non-degenerate like SEDI but not symmetric and easier to hedge; reported as a cross-check. Same bias caveat. |
| Probability of Detection | `POD` | Fraction of observed events that were forecast |
| False Alarm Ratio | `FAR` | Fraction of forecast events that were false alarms |
| Threshold-weighted MAE | `twMAE` | Chaining-function score (Taggart 2022): `mean\|max(fc,T) − max(obs,T)\|` for the upper tail (`min` for the lower). The deterministic analogue of twCRPS; penalises hits, misses **and** false alarms. **Not** the indicator-weighted `1{max≥T}·\|fc−obs\|` — see SCIENCE.md / `docs/TWMAE_HEDGING.md`. |
| Threshold-weighted RMSE | `twRMSE` | Same chaining construction as twMAE with the squared error; targets the mean rather than the median. |
| Bias | `bias` | Mean forecast − observation (all cases) |
| Mean Absolute Error | `mae` | MAE over all cases |
| Root Mean Square Error | `rmse` | RMSE over all cases |
| Pearson Correlation | `correlation` | Pearson correlation coefficient between forecast and observation |
| SEEPS | `SEEPS` | **tp24 only** (skipped with a note for other variables). Stable Equitable Error in Probability Space (Rodwell et al. 2010), ECMWF's headline precipitation score; dry/light/heavy categories, lower = better, ≈1 for a no-skill forecast. Computed over **all** cases (not threshold-masked), using a per-station in-sample climatology; stations with dry probability outside [0.1, 0.85] are dropped. |

> **SEDI/EDI frequency-bias caveat.** EDI and SEDI are base-rate independent *only* when the
> forecast frequency bias ≈ 1. The forecasts in this tool are **not bias-corrected**, so a
> SEDI/EDI difference between two models reflects both their discrimination and their frequency
> bias — read it as a joint statement and cross-check the base-rate columns. SEDI/EDI are
> undefined (reported as blank/`NaN`) in the degenerate case where a cell has no misses or no
> false alarms; this does not occur for the large cells of a normal scorecard run. Example config
> block:
>
> ```yaml
> scores:
>   deterministic: ["SEDI", "EDI", "ETS", "PSS", "twMAE", "twRMSE", "bias"]
> ```

Requesting `twMAE` automatically adds six extra columns per condition, decomposing the score into
its hit / miss / false-alarm contributions (these three sum exactly to `twMAE`):

| Column | Meaning |
|--------|---------|
| `twMAE_hits` | Share of total twMAE from hits = (n_hits / N) × mean\|fc − obs\| on hits |
| `twMAE_misses` | Share of total twMAE from misses = (n_misses / N) × mean(obs − T) on misses |
| `twMAE_FA` | Share of total twMAE from false alarms = (n_FA / N) × mean(fc − T) on false alarms |
| `twMAE_hit_mae` | Per-case accuracy on hits only: mean\|fc − obs\|, **not** weighted by how often hits occur |
| `twMAE_miss_severity` | Per-case severity of missed events: mean(obs − T) among misses |
| `twMAE_fa_severity` | Per-case severity of false alarms: mean(fc − T) among false alarms |

The `_hits`/`_misses`/`_FA` fields are "budget shares" (affected by both event frequency and
per-case error); the `_hit_mae`/`_miss_severity`/`_fa_severity` fields isolate per-case accuracy
from frequency and are usually more informative for comparing two models' error character.

### Ensemble scores

| Score | Config key | Description |
|-------|-----------|-------------|
| Fair CRPS | `fCRPS` | Bias-corrected CRPS; lower = better |
| Threshold-weighted CRPS | `twCRPS` | CRPS with chaining function focused on extremes |
| Brier Score | `Brier` | **Fair** probability score for the binary extreme event: (p̂ − o)² − p̂(1 − p̂)/(M − 1), so ensembles of different size are comparable |
| Brier Skill Score | `BSS` | Brier score relative to climatological baseline |
| CRPS Skill Score | `CRPSS` | `1 − CRPS_fc / CRPS_clim`, with the sample climatology of the cell's pooled observations as reference; 1 = perfect, 0 = no better than climatology |
| Brier reliability | `reliability` | Reliability term of the Brier-score decomposition (Murphy 1973) for P(crossing T), 10 probability bins; 0 = perfectly reliable, lower = better. A pure calibration measure |
| TW Quantile Score (mean) | `tw_quantile_score` | Mean twQS across all tail levels |
| TW Quantile Score — level | `tw_quantile_score_q001` etc. | twQS at a specific quantile level (see below) |
| Extreme Spread/Skill Ratio | `extreme_spread_skill_ratio` | Size-corrected √((M+1)/M) · √(mean unbiased variance) / RMSE of the ensemble mean (1 = perfect for any M), **always** restricted to cases where obs crosses the threshold (unconditional on the `extreme_only_basic_scores` flag below). Because it selects on the observed outcome it carries a conditioning bias: treat it as a tail-dispersion diagnostic, not a calibration score |
| Ensemble Mean Bias | `ens_mean_bias` | Mean(ensemble mean − obs) |
| Ensemble Mean MAE | `ens_mean_mae` | MAE of the ensemble mean vs obs |
| Ensemble Mean RMSE | `ens_mean_rmse` | RMSE of the ensemble mean vs obs — the standard "skill" side of the spread–skill relationship (not size-corrected) |
| Ensemble Mean RMSE, size-corrected | `ens_mean_rmse_fair` | √(MSE − s²/M): the ensemble-mean error expected for an infinite ensemble (Leutbecher & Palmer 2008); use it to compare ensembles of different size |
| Spread/Skill Ratio | `spread_skill_ratio` | √((M+1)/M) · `ens_spread` / `ens_mean_rmse` on the same cases (all cases): 1 = perfect for any M, < 1 under-, > 1 over-dispersive |
| Ensemble Spread | `ens_spread` | √(mean over cases of the unbiased member variance, ÷(M−1)) — the standard spread–skill definition, directly comparable with `ens_mean_rmse` |

**Fair and unbiased ensemble scores (finite-ensemble corrections).** With small ensembles (e.g. 8
members) the plain estimators are biased, so the tool uses:

- `fCRPS`, `twCRPS`: **fair** CRPS (Ferro 2014) — the pairwise term uses M(M−1); the bootstrap and the
  hit/miss/FA and accuracy/spread decompositions use the same fair form.
- `Brier` (and therefore `BSS`): **fair** Brier score (p̂ − o)² − p̂(1 − p̂)/(M − 1) for both fixed and
  per-station thresholds. *(Per-station thresholds used the plain score before 2026-10-15.)*
- `ens_spread`: √(mean over cases of the **unbiased** member variance, ÷(M−1)) — the standard
  spread–skill definition (Leutbecher 2009, ECMWF; Fortin et al. 2014), directly comparable with
  `ens_mean_rmse`. *(Before 2026-10-15 it was the mean of per-case standard deviations ÷M, about 6.5 %
  low for 8 members.)* For a perfectly calibrated M-member ensemble, RMSE ≈ √((M+1)/M) · spread.
- `ens_mean_rmse` (the skill/accuracy side) is the plain RMSE of the ensemble mean, as in ECMWF
  spread–error plots; its expected square is (1 + 1/M)σ², so it is inflated for small ensembles.
  `ens_mean_rmse_fair` = √(MSE − s²/M) removes that, for comparisons across ensemble sizes.
- `spread_skill_ratio` (all cases) and `extreme_spread_skill_ratio` (cases where the observation
  reaches T): **size-corrected** √((M+1)/M) · spread / RMSE, so a statistically consistent ensemble
  scores 1 for any M (all backends). *(Before 2026-10-15 the extreme ratio had no size factor, so a
  perfect 8-member ensemble read ≈ 0.94; the general ratio is new.)*
- `twCRPS_spread`: the fair dispersion term of twCRPS (members chained at T), not a standard deviation.

`reliability` remains the reliability term of the plain Brier decomposition (a calibration diagnostic).

`ens_mean_bias`/`ens_mean_mae`/`ens_mean_rmse`/`ens_spread` are **always computed automatically**
for ensemble mode — they don't need to be (and can't be) listed under `scores.ensemble`.

By default these four scores (and only these four — `extreme_spread_skill_ratio` is always
tail-conditioned regardless) are computed over **all** samples, not just the extreme tail. To
restrict them to cases where the observation crosses the threshold (conditional verification),
set:

```yaml
scores:
  ensemble: [...]
  extreme_only_basic_scores: true   # default false = computed over all samples
```

#### Count-based hit / miss / false-alarm fractions (no magnitude)

`count_hits_pct`, `count_misses_pct`, `count_FA_pct` are **always computed automatically** for
ensemble mode (like `ens_mean_bias` etc. — no config change needed). They answer "how many cases
fall in each category?", with **zero magnitude weighting** — the exact complement of the
magnitude-weighted `twCRPS_hits`/`twCRPS_misses`/`twCRPS_FA` decomposition above. The ensemble
**mean** is used as the single representative forecast value (same basis as `ens_mean_bias`) to
get a binary "extreme predicted?" decision, compared against the binary observed outcome:

| Column | Meaning |
|--------|---------|
| `count_hits_pct` | % of all cases where obs crossed T **and** the ensemble mean also crossed T |
| `count_misses_pct` | % of all cases where obs crossed T but the ensemble mean did **not** |
| `count_FA_pct` | % of all cases where the ensemble mean crossed T but obs did **not** |

These three (plus the unreported correct-negative %) sum to 100% of the sample. Comparing them
against `twCRPS_hits`/`twCRPS_misses`/`twCRPS_FA` on the same condition tells you whether a
magnitude-weighted difference is driven by *how often* each category occurs or by *how large*
the error is when it occurs — e.g. for tp24 heavy precip, miss *counts* are usually a small
fraction of the sample, yet `twCRPS_misses` still dominates the twCRPS budget because each missed
extreme carries a very large chained-CRPS cost.

#### twCRPS decomposition

Requesting `twCRPS` automatically adds five extra columns per condition, decomposing the score
in two complementary ways (the probabilistic analogue of the deterministic `twMAE` breakdown).
Because the forecast is a distribution rather than a single value, the split uses the ensemble
CDF `F` and the integral form `twCRPS = ∫_T^∞ [F(z) − 1{y ≤ z}]² dz`.

**Hit / miss / false-alarm contributions** (these three sum exactly to `twCRPS`):

| Column | Meaning |
|--------|---------|
| `twCRPS_misses` | Underforecast of observed extremes: ensemble probability mass sitting **below** a real extreme (obs ≥ T), i.e. the ensemble is too weak. Contribution `∫_T^y F(z)² dz`. |
| `twCRPS_hits` | Residual cost at observed extremes from ensemble mass **above** the observation (event captured, upper distribution imperfect). Contribution `∫_y^∞ (1−F(z))² dz` for obs ≥ T. |
| `twCRPS_FA` | False alarms: ensemble tail probability beyond the threshold when **no** extreme occurred (obs < T). Contribution `∫_T^∞ (1−F(z))² dz`. |

**Accuracy / spread split** (kernel form of the fair CRPS, `twCRPS = accuracy − spread`):

| Column | Meaning |
|--------|---------|
| `twCRPS_accuracy` | Ensemble-mean placement term `mean_m \|v(x_m) − v(y)\|` with chaining `v(u)=max(u,T)`; lower = better. |
| `twCRPS_spread` | Fair dispersion term `(1/(2M(M−1))) ΣΣ \|v(x_m) − v(x_m')\|`. A larger spread term **lowers** twCRPS, so on heatmaps it is coloured like `ens_spread` (higher → better for the typically under-dispersive NWP case). |

The hit/miss/FA columns isolate *where* a model's tail error comes from (under-prediction of real
extremes vs false alarms); the accuracy/spread columns show *whether* a twCRPS difference is
driven by ensemble-mean placement or by dispersion. All five are computed for both per-station
(`local_obs_climatology`) and scalar thresholds and are plotted automatically as heatmaps.

On the `twCRPS_hits`/`twCRPS_misses`/`twCRPS_FA` heatmaps, each cell shows an extra small teal
annotation in the top-left corner: that component's **% share of the total twCRPS** for that
cell (the three shares sum to 100%). This makes the dominant failure mode visible on a single
heatmap — for heavy-tailed variables like `tp24`, `twCRPS_misses` typically carries ~90%+ of the
budget (under-dispersive ensembles rarely overshoot genuine extremes), so `twCRPS` differences
between models mostly track `twCRPS_misses`, while `twCRPS_hits`/`twCRPS_FA` differences are much
smaller in absolute terms even if their own % colour looks large.

#### Threshold-weighted Quantile Score levels

The α levels are chosen **at or beyond** the extreme threshold's own percentile — evaluating a
quantile forecast at α=0.90 when the event is only "extreme" beyond the 99th percentile mixes in
a much less extreme, less relevant quantile level. Default (`_TW_QS_ALPHAS` in
`src/ens_scores.py`) probes progressively deeper into the tail, not toward the distribution's
centre:

For **cold extremes** (`event_type: below`):

| Key | Alpha | Description |
|-----|-------|-------------|
| `tw_quantile_score_q001` | 0.01 | 1st percentile level (matches a p1 threshold) |
| `tw_quantile_score_q0005` | 0.005 | deeper: 0.5th percentile |
| `tw_quantile_score_q0001` | 0.001 | deepest: 0.1th percentile |

For **warm/wet extremes** (`event_type: above`):

| Key | Alpha | Description |
|-----|-------|-------------|
| `tw_quantile_score_q099` | 0.99 | 99th percentile level (matches a p99 threshold) |
| `tw_quantile_score_q0995` | 0.995 | deeper: 99.5th percentile |
| `tw_quantile_score_q0999` | 0.999 | deepest: 99.9th percentile |

Column-name format: alphas that are exact multiples of 0.01 use the legacy 2-decimal code
(`q090`, `q099`, `q001`, ...); finer alphas get a 4-digit code (`q0995`, `q0001`, ...) that cannot
collide with any 2-decimal code. Configs can still override the default via `scores.tw_qs_alphas`
(e.g. the 10ff p98 configs use `above: [0.90, 0.95, 0.98]` to stay within their own p98 threshold)
— those keep their existing `q090`/`q095`/`q098` columns until deliberately migrated.

> **Migration note (2026-09):** the default alphas changed from `[0.90, 0.95, 0.99]` /
> `[0.01, 0.05, 0.10]` to the deeper-tail levels above. Configs that list the old column names
> explicitly (`tw_quantile_score_q090`/`q095`, `tw_quantile_score_q005`/`q010`) keep producing
> those exact columns until the config is edited and rescored — `src/plot.py` still recognises
> both the legacy and current column names so old results keep plotting correctly.

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

Stations with `lsm < coastal_lsm_threshold` (default 0.9) are considered coastal and removed. With the
`quaver_extract` backend a single common land-sea mask (IFS static analysis) is interpolated to the
stations, so both models are filtered identically.

> **Ensemble runs before 2026-10-15:** the streaming ensemble scoring path did not apply
> `remove_coastal_stations` (nor `filter.area`, below) — ensemble scorecards kept coastal stations even
> when configured to remove them. Both filters are now applied in the deterministic and the ensemble
> path; re-score older ensemble results (no re-extraction needed) to make them consistent.

### Region (`filter.area`)

`extract_points.area` limits the **extraction** to a box. `filter.area` selects a region **after**
extraction, from data extracted over a larger box — it only acts when `extract_points.area` is `null`
(otherwise the data are already cut and the setting is ignored with a warning). This allows one
expensive extraction to be scored for several regions:

| Named area | Box [N, W, S, E] |
|------------|------------------|
| `europe` | 68, −15, 27, 50 |
| `france` | 51.5, −5.5, 42, 8.5 (filter only) |
| `nh_extratropics` | 90, −180, 20, 180 |
| `tropics` | 20, −180, −20, 180 |
| `nh_extratropics_tropics` | 90, −180, −20, 180 (extract once, then split) |

Pattern: extract with `extract_points.area: nh_extratropics_tropics`, then score two configs that point
at the same `output_path` with `extract_points.area: null` and `filter.area: nh_extratropics` (with an
orography split) / `filter.area: tropics` (e.g. `orography_type: null`). `scripts/plot_combined_regions.py`
draws both as one scorecard (§12). Note that tropical regions have far fewer stations with a usable
20-year obs climatology, so `local_obs_climatology` thresholds rest on a thinner sample there.

### Quality control

- **Temperature**: obs outside `[min_valid_temperature, max_valid_temperature]` (°C) are removed.
- **Precipitation**: obs above `max_valid_precipitation` (mm) are removed.

> **Note:** Gaussian mean±N·std outlier ("sigma clipping") removal has been removed from the pipeline. It is statistically invalid for skewed fields such as `tp24`, where it strips the genuine extreme tail being verified. Only the physical-validity caps above are applied.

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

`{score}_diff` in the CSV is a plain **absolute** difference: `fc2 − fc1` (model2 minus model1),
for every score including twMAE/twRMSE — it is not a percentage.

The **heatmap color/annotation** shown in the PNGs is computed separately at plot time (not
stored in the CSV) and is a percentage, normalized differently depending on the score type —
and the color convention flips between the two groups:

- Error-type scores (`twMAE`, `twRMSE`, `mae`, `rmse`, `SEEPS`, the twMAE decomposition columns,
  `CRPS`, `twCRPS`, the `twCRPS_hits`/`_misses`/`_FA`/`_accuracy` decomposition columns,
  `count_misses_pct`, `count_FA_pct`, `tw_quantile_score`, `diagonal_score`, `ens_mean_mae`,
  `ens_mean_rmse`) and bounded error scores with perfect = 0 (`Brier`, `reliability`, `FAR`):
  `pct_diff = (fc2 − fc1) / fc1 × 100`.
  Lower is better, so **negative = model2 (fc2) better, shown blue; positive = model1 (fc1)
  better, shown red.**
- Signed, zero-crossing scores (`bias`, `ens_mean_bias`): a relative % of a quantity whose target
  is 0 is meaningless (it explodes near zero bias and flips sign when fc1 < 0), so these show the
  **reduction in |bias|**: `pct_diff = (|fc1| − |fc2|) / max(|fc1|, |fc2|) × 100`, bounded to
  ±100. **Positive = model2 has the smaller |bias|, shown blue.**
- Spread–skill ratios (`spread_skill_ratio`, `extreme_spread_skill_ratio`), perfect = 1: shown as
  **closeness to 1**, the same transform applied to (ratio − 1). **Positive = model2 is closer to
  1, shown blue** — so too much spread is not rewarded.
- "Higher-is-better" scores without a perfect value of 1 (`ens_spread`,
  `twCRPS_spread`, `count_hits_pct`):
  `pct_diff = (fc2 − fc1) / fc1 × 100`, **positive = model2 better, shown blue.**
- Skill scores with perfect = 1 (`ETS`, `PSS`, `SEDI`, `EDI`, `POD`, `BSS`, `CRPSS`,
  `correlation`): `pct_diff = (fc2 − fc1) / (1 − fc1) × 100` (normalised by the remaining
  headroom to a perfect score). **Positive = model2 (fc2) better, shown blue; negative =
  model1 (fc1) better, shown red.**

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

### Comparing ensembles of different sizes / missing control members

Model 1 and model 2 do **not** need the same number of members. All ensemble
scores (twCRPS, Brier, ens_mean_bias, etc.) are computed **independently per
model** from whatever `fc{1,2}_member_*` columns that model actually has, so
e.g. a 51-member ensemble (control + 50 perturbed) can be compared directly
against a 50-member one (perturbed only, no archived control) — nothing
needs to be padded or truncated to match.

For ensembles made only of perturbed members (e.g. an 8-member `rd` experiment with no control
anywhere), the simplest setup is `ensemble: {n_members: 8, include_control: false}` with
`number: "1/to/8"` for both models — no control is requested at all.

When a model has no archived control member (common for research/`rd`
experiments), configure how to handle it per model under its `mars:` (method
2) or `quaver:` (method 3) block with `control_source`:

```yaml
read_data:
  forecast_model2:
    name: "j1l9_ens"
    source: "mars"          # or "quaver" for method 3 (same control_source keys)
    mars:
      class: rd
      expver: "j1l9"
      # ... other mars keys ...
      control_source:
        mode: "omit"        # this model simply has no control member (default
                             # if control_source is unset AND retrieval fails:
                             # falls back to perturbed-only automatically with
                             # a [WARN] print — set 'omit' to skip the attempt
                             # entirely and avoid that per-date retrieval cost)
```

Or substitute a **different experiment's deterministic run** as the control
(e.g. a sibling research experiment that WAS run deterministically):

```yaml
      control_source:
        mode: "external"
        class: rd            # MARS identity of the substitute control run
        expver: "j1l8"        # (required: class + expver)
        type: fc              # optional, default "fc"
        stream: oper           # optional (method 2 only omits if unset;
                                # method 3/quaver defaults to "oper")
        # model / grid / database: optional, inherited from this model's own
        # mars/quaver block if not given here
```

`mode: "own"` (or omitting `control_source` entirely) is the default: control
is retrieved from the SAME experiment as the perturbed members, exactly as
before. This currently applies to methods 2 (`mars`) and 3 (`quaver`); method
1 (`local_grib`) already tolerates a pre-staged GRIB file with no control
field (falls back to perturbed-only automatically) but does not support
substituting an external control file.

### Extraction differences

- Ensemble extraction is handled by `src/extract_points_ensemble.py`
- Per-date files are saved in a `_tmp/` subfolder during extraction; on completion they are merged into per-day parquet files
- Each parquet row contains `fc1_member_0 … fc1_member_50` and `fc2_member_0 … fc2_member_50` columns by default — but the two counts can differ (see above) if one model has no control member
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

### `diagnostics/deterministic/diagnose_extremes.py` — Comprehensive single-condition diagnostics

Produces **22 diagnostic plots** for a specific (day, season, orography, threshold) combination. Reads from the extracted parquet files; the main pipeline must have completed through Step 3 first.

```bash
python diagnostics/deterministic/diagnose_extremes.py \
  --config configs/deterministic/config_2t_local_p1obsclim.yaml \
  --day 3 --threshold-pct 1 --season DJF --orog complex

# Or with a fixed threshold value:
python diagnostics/deterministic/diagnose_extremes.py \
  --config configs/deterministic/config_tp24_local_p99obs.yaml \
  --day 3 --threshold-value 30.0 --season JJA --orog flat

# Submit as SLURM job (64 GB, 4 CPU, 4 h):
sbatch scripts/submit_diagnose.sh --config configs/deterministic/config_2t_local_p1obsclim.yaml \
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

### `diagnostics/deterministic/diagnose_det_extremes_simple.py` — Quick deterministic diagnostic

A lightweight 3-row figure for a fast visual check; does not require the full score CSV.

```bash
python diagnostics/deterministic/diagnose_det_extremes_simple.py \
  --config configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml \
  --orog flat --days 1 3 --season JJA
```

**Rows produced:**
- **Row 1** — KDE of `(fc − T)` and `(obs − T)` for both models when obs exceeded T (0 = exactly at threshold; left = missed extreme).
- **Row 2** — Box plots of `(fc − obs)` error during extreme events (0 = perfect).
- **Row 3** — Stacked bar chart of hits / misses / false alarms, consistent with the heatmap CSV.

Pools data across the requested forecast days. Output PNG saved to `{save.output_directory}/`.

### `diagnostics/ensemble/diagnose_twcrps_simple.py` — Quick ensemble diagnostic

Ensemble counterpart of `diagnostics/deterministic/diagnose_det_extremes_simple.py`; replicates the pipeline's 6-hourly → daily aggregation internally.

```bash
python diagnostics/ensemble/diagnose_twcrps_simple.py \
  --config configs/ensemble/config_2t_ens_local_p99obsclim_aifsvsifs.yaml \
  --orog low --day 5
```

**Rows produced:**
- **Row 1** — KDE of obs (green), model 1 ensemble median (red), model 2 ensemble median (blue), each normalised as `(temperature − threshold T)`.
- **Row 2** — Box plots of `(ensemble_median − obs)` during extreme events.
- **Row 3** — Bar chart of twCRPS for each model, annotated with percentage difference.

Output PNG saved to `{save.output_directory}/`.

### `diagnostics/ensemble/diagnose_ens_extremes.py` — Comprehensive ensemble diagnostics (twCRPS vs quantile score)

Probabilistic counterpart of `diagnostics/deterministic/diagnose_extremes.py`. Built specifically to explain **why the threshold-weighted CRPS (`twCRPS`) and the threshold-weighted quantile score (`tw_quantile_score`) can rank two ensembles differently**. It reproduces the pipeline's *exact* ensemble scoring path (Step 5 threshold on sub-daily data → Step 5b daily aggregation → `ens_scores.calculate_ensemble_scores`), so every number matches the heatmap CSV for the corresponding `(season, orography, forecast_day)` cell. The multi-GB per-day parquet is read in chunks to stay within memory.

```bash
python diagnostics/ensemble/diagnose_ens_extremes.py \
  --config configs/ensemble/config_2t_ens_local_p99obsclim_aifsvsifs.yaml \
  --day 5 --season JJA --orog low
```

Arguments: `--config` (required), `--day` (default 5), `--season` (default JJA), `--orog` (default low), `--output-dir` (default `case_study_output/ens_diagnostics/<config stem>`). The `event_type` (warm/cold) is taken from the config.

**Why twCRPS and the quantile score can disagree even though both use the same tail:**
The identity `twCRPS = 2·∫₀¹ twQS(α) dα` (Gneiting & Ranjan, 2011) means twCRPS is the
continuous integral of the pinball loss over *every* quantile level α, while the reported
`tw_quantile_score` is the mean of only 3 discrete levels (α = 0.99/0.995/0.999 by default,
i.e. at-or-beyond the extreme threshold's own percentile — see "Threshold-weighted Quantile
Score levels" above). If a model's tail performance is not uniform across α — e.g. better
mid-tail but worse at the single deepest quantile — the two `twQS(α)` curves cross, and the
full-integral score (twCRPS) and the 3-point summary (twQS) can pick different winners.

**Figure 1 — `ens_score_attribution_*.png`** (six panels):
1. Forecast vs observed CDF in the tail — shows exactly what `twCRPS = ∫[F(z)−1{obs≤z}]²dz` integrates over (shaded region, z ≥ T).
2. Quantile-coverage curve — is each tail quantile `q̂_α` calibrated (curve below the 1:1 line at α → that quantile is forecast too low)?
3. `twQS(α)` swept across most of the [0,1] quantile axis (α∈[0.5,0.999] for warm / [0.001,0.5] for cold), with the 3 reported levels marked — the shared quantity behind both scores. `twCRPS = 2·∫₀¹twQS(α)dα` holds over the *full* [0,1] domain; the plotted partial-domain area is an illustrative check only (the boxed numbers state whether its winner agrees with the exact twCRPS). A disagreement between twCRPS and twQS means the two models' `twQS(α)` curves cross somewhere in α.
4. twCRPS split by daily outcome at the threshold (hit / miss / false alarm; edge colour = model).
5. Head-to-head `%` difference, one row per score (twCRPS, twQS mean, twQS at each level) — a sign flip down the list means the scores disagree.
6. Auto-generated attribution summary: states the `twCRPS = 2·∫twQS dα` identity, the winner of each score, the coverage read-out, the spread-skill ratio, and bootstrap-significant differences.

**Figure 2 — `ens_calibration_*.png`** (three panels): rank (Talagrand) histogram (U-shape → under-dispersive), spread–skill diagram (above 1:1 → under-dispersive/over-confident), and reliability diagram for `P(exceed T)` annotated with BSS.

> Memory note: the ensemble scoring path deliberately does **not** apply the coastal (`lsm`) filter and thresholds *before* daily aggregation — this tool matches that exactly. Running it as a plain command on a login node is fine (chunked reads keep it under the memory cap).

### `diagnostics/qq/plot_qq_extremes.py` — Q-Q plot (warm + cold + cold-zoom panels)

```bash
python diagnostics/qq/plot_qq_extremes.py \
  --config configs/deterministic/config_2t_local_p1obsclim.yaml \
  --day 3 --season DJF --orog complex
```

Three-panel figure:
- **Panel A**: Warm extremes (p90–p99.9), dots every 1%, × markers every 0.1%
- **Panel B**: Cold extremes overview (p0.1–p10), same marker scheme
- **Panel C**: Cold extremes zoom — connected line from p10 to p0.1, every 1% down to p1 then every 0.1%

### `diagnostics/qq/plot_qq.py` — General Q-Q plot

```bash
python diagnostics/qq/plot_qq.py --config configs/deterministic/config_tp24_precipitation.yaml \
  --season DJF MAM JJA SON --orog flat mid high
```

### `diagnostics/stations/plot_station_diagnostics.py` — Per-station spatial maps

Takes the **extracted points directory** as a positional argument (not a config file). Forecast mode (deterministic vs ensemble) is auto-detected from the parquet column names. Produces scatter maps on a European map with station dots coloured by bias, error, or threshold-exceedance class.

```bash
# Ensemble parquet in /extracted_points/2t_ens:
python diagnostics/stations/plot_station_diagnostics.py ./extracted_points/2t_ens \
  --threshold 30 --event-type above --variable 2t \
  --forecast-day 3 --model1-name ifs_ens --model2-name aifs_ens \
  --output-dir ./plots/station_diagnostics/2t_warm

# Deterministic:
python diagnostics/stations/plot_station_diagnostics.py ./extracted_points/2t \
  --threshold -5 --event-type below --variable 2t \
  --forecast-day 3 --model1-name ifs_oper --model2-name iekm \
  --output-dir ./plots/station_diagnostics/2t_cold
```

### `scripts/plot_combined_regions.py` — NH-orography + tropics in one scorecard

```bash
python scripts/plot_combined_regions.py <nh_config.yaml> <tropics_config.yaml> <output_dir>
```

Reads the `scores_by_leadtime` CSVs of an NH-extratropics run (orography split) and a tropics run
(unsplit) that share one extraction (§8, *Region*), and redraws every score as one heatmap with columns
NH FLAT | NH HILLY | NH COMPLEX | TROPICS through the normal plotting code (same colours, limits and
significance boxes).

### `scripts/plot_ocean_twcrps_panels.py` and `scripts/plot_ocean_ranking_summary.py` — multi-experiment views

```bash
python scripts/plot_ocean_twcrps_panels.py   [--score twCRPS|Brier|fCRPS|…] [output_dir]
python scripts/plot_ocean_ranking_summary.py [--score twCRPS|Brier|…]       [output_dir]
```

Written for the ocean-coupling ensemble study (Control vs Ocean A/B, GT, T; configs from
`scripts/make_ocean_ens_configs.py`), but the pattern generalises:
- **Panels**: one figure per variable with the four experiment scorecards side by side
  (NH FLAT | HILLY | COMPLEX | TROPICS × day 1–10), a fixed −30…+30 % colour bar, blue = experiment better
  than the reference, boxed = significant. Lower-is-better scores only.
- **Ranking summary**: (1) significantly better/worse cell counts, mean % change and mean rank per
  experiment; (2) mean % change per variable × experiment with the better/worse counts in each cell.
  Significance is always *vs the reference*; experiments are not tested against each other.

Default output: `plots/ocean_ens_FebAug2026/`.

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

The pipeline (`src/read_data.py`) looks for files named:

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
sbatch scripts/submit_job.sh config_my_experiment.yaml
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

### Chunked extraction and automatic merge + scoring

`quaver_extract` processes dates serially, so long periods are split into date chunks that run as
parallel jobs, then merged and scored automatically:

```bash
python scripts/chunk_extraction.py split configs/.../my_config.yaml --n-chunks 4 --submit
sbatch scripts/watch_and_score.sh configs/.../my_config.yaml 4 [extra_config.yaml ...]
```

- Chunk configs go to `configs/_chunks/<config name>/` (git-ignored), each with `extraction_only: true`
  and its own `<output_path>_chunkN`; `job_ids.txt` lists their jobs.
- `watch_and_score.sh` waits for the chunk jobs, resubmits incomplete chunks (they resume from their
  per-date `_tmp` cache), then submits one 128 GB job that merges the chunks
  (`chunk_extraction.py merge`) and scores the base config.
- **Extra configs** (optional arguments) are further scoring runs over the *same* merged extraction —
  e.g. a `filter.area: tropics` config next to an NH one. They are submitted with
  `--dependency=afterany:<merge+score job>` and `S4E_REQUIRE_EXTRACTED=1`.
- **`S4E_POST_SCRIPT=<batch script>`** (exported to the watcher) is submitted after all scoring jobs,
  e.g. a combined-region plot.
- **Guard:** with `S4E_REQUIRE_EXTRACTED=1`, `submit_job_quaver.sh` runs `scripts/check_extracted.py`
  first and refuses to start (exit 3, no auto-retry) if any expected `*_day<N>.parquet` is missing —
  otherwise `run.py` would silently re-extract from scratch.
- **A model whose archive changes during the period** (e.g. a research expver until some date, then the
  operational stream): write the chunk configs by hand so each chunk carries the right
  `read_data.forecast_modelN.quaver` identity and `unit_conversion_factor` (e.g. tp in m → ×1000 vs
  kg m⁻² → ×1), keeping the same model `name` so the chunks merge. Example:
  `configs/deterministic/config_*_aifsv2vsjbxy_FebAug2026*.yaml`.

> **Exit code 139 after "Extraction complete":** `quaver_extract` jobs often segfault while Python is
> shutting down, *after* writing all day files. SLURM then reports FAILED and `submit_job_quaver.sh`
> auto-resubmits; the resubmitted job finds the files and exits cleanly within a minute. Check the log
> for "extraction complete" before treating such a job as failed.

### Specialist submission scripts

| Script | Purpose | Resources |
|--------|---------|----------|
| `scripts/submit_job.sh` | Main pipeline (extract + score + plot) | 128 GB, 12 CPU, 48 h |
| `scripts/submit_job_quaver.sh` | Same, with the vtb/quaver environment; honours `S4E_MERGE_CHUNKS`, `S4E_REQUIRE_EXTRACTED`, auto-retry | 128 GB, 12 CPU, 48 h |
| `scripts/watch_and_score.sh` | Chunk watcher → merge + score (+ extra configs, `S4E_POST_SCRIPT`) | 1 GB, 1 CPU, 48 h |
| `scripts/submit_qq.sh` | Q-Q plot tool (`diagnostics/qq/plot_qq.py`) | 64 GB, 1 CPU, 1 h |
| `scripts/submit_diagnose.sh` | `diagnostics/deterministic/diagnose_extremes.py` (simple wrapper, passes all args through) | 32 GB, 1 CPU, 2 h |
| `scripts/submit_diagnose_job.sh` | `diagnostics/deterministic/diagnose_extremes.py` (race-condition-safe multi-job helper) | 64 GB, 4 CPU, 4 h |
| `scripts/submit_extraction.sh` | tp24 obs extraction from STVL | 64 GB, 4 CPU, 12 h |
| `case_studies/submit_find_cases.sh` | `find_case_studies.py` via `--export` env vars | 64 GB, 1 CPU, 4 h |
| `case_studies/submit_plot_case.sh` | `plot_case_study.py` via `--export` env vars | 32 GB, 1 CPU, 1 h |
