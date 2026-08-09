# Scorecards4Extremes — Case-Study Tools

**Version:** June 2026

The case-study workflow is a two-step process: first identify the dates where one model performs substantially better or worse than the other, then visualise a specific date in detail.

All tools in this section read from the **extracted parquet files** produced by `run.py` (Step 3). The main pipeline must have run through at least that step before using these tools.

---

## Table of Contents

1. [Overview](#1-overview)
2. [`find_case_studies.py` — Identify dates](#2-find_case_studiespy--identify-dates)
3. [`plot_case_study.py` — Visualise a date](#3-plot_case_studypy--visualise-a-date)
4. [Batch Submission](#4-batch-submission)
5. [Worked Example](#5-worked-example)

---

## 1. Overview

The workflow answers: *"On which dates did model 1 (or model 2) perform substantially worse, and can we see why?"*

```
run.py → extracted parquet files
                │
                ▼
  find_case_studies.py
    Ranks every date × forecast-day by composite performance gap
    Outputs: case_study_ranking_*.csv + case_study_summary_*.txt
                │
                ▼
  plot_case_study.py --date YYYYMMDD --day N
    Visualises a single date with 3 figures:
      Fig 1: hit/miss/FA station maps (4-panel)
      Fig 2: observed and forecast value maps (3-panel)
      Fig 3: raw GRIB field + obs overlay (2-panel)
```

Both tools use the same threshold method configured in the YAML (including `local_obs_climatology`) so results are consistent with the heatmap scorecards.

---

## 2. `find_case_studies.py` — Identify dates

Loops over every date × forecast-day in the parquet files. For each combination it:

1. Applies per-station thresholds (same method as the main pipeline).
2. Classifies each station as hit / miss / false alarm / correct negative for both models.
3. Computes a rich set of per-date metrics (FA count, miss count, twMAE, POD, FAR, ETS, spatial concentration of errors by sub-region).
4. Assigns a **composite ranking score** (high = model 1 much worse; low = model 2 much worse; near zero = similar).
5. Classifies the case type (e.g. `M1_FALSE_ALARM`, `M2_MISS_COUNT`).

### Outputs

| File | Content |
|------|---------|
| `case_study_ranking_<name>_day<N>.csv` | One row per date with all metrics and composite rank |
| `case_study_summary_<name>.txt` | Human-readable top-N worst cases per day |

### Usage

```bash
python case_studies/find_case_studies.py --config <yaml> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | *(required)* | YAML config file |
| `--days N [N ...]` | all found | Forecast days to analyse |
| `--top-n N` | 20 | Number of worst cases to print in the summary |
| `--output-dir DIR` | `./case_study_output` | Where to save CSV and text files |
| `--min-stations N` | 50 | Minimum stations per date to include |
| `--min-concentration F` | none | Minimum fraction of FAs/misses in one sub-region (0–1) |
| `--season DJF\|MAM\|JJA\|SON` | all | Filter to a single season |
| `--orog low\|mid\|high` | all | Filter to an orography class |
| `--no-ensemble-prob` | off | Use ensemble mean instead of P > 0.5 for event classification |

### Example

```bash
python case_studies/find_case_studies.py \
  --config configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml \
  --days 3 5 7 --season JJA --orog flat \
  --top-n 20 --output-dir ./case_study_output/2t_JJA_flat
```

---

## 3. `plot_case_study.py` — Visualise a date

Produces three figures for a single date identified by `find_case_studies.py`.

### Outputs

| Figure | File suffix | Content |
|--------|-------------|---------|
| 1 | `.png` | 4-panel: hit/miss/FA maps for model 1 and model 2, exceedance bar chart, score table |
| 2 | `_values.png` | 3-panel: observed values, model 1 values, model 2 values (shared colour scale) |
| 3 | `_overlay.png` | 2-panel: raw GRIB field at 0.25° with obs dots overlaid, one panel per model |

Figure 3 reads **raw GRIB files** from the paths in the config (`read_data.forecast_model1.local_grib.path`). It requires `cartopy` and `scipy`.

### Usage

```bash
python case_studies/plot_case_study.py \
  --config <yaml> --date YYYYMMDD --day N [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | *(required)* | YAML config file |
| `--date YYYYMMDD` | *(required)* | Initialisation date |
| `--day N` | *(required)* | Forecast day integer |
| `--output-dir DIR` | `./case_study_output/<config_stem>` | Output directory |
| `--output PATH` | auto | Full PNG path for Figure 1 (overrides `--output-dir`) |
| `--season DJF\|MAM\|JJA\|SON` | all | Apply seasonal filter when loading data |
| `--orog low\|mid\|high` | all | Apply orography filter when loading data |
| `--title TEXT` | auto | Additional title string |

### Predefined zoom regions

The map panels use one of the following named bounding boxes (passed via `--title` or hard-coded when adapting the script):

| Name | Lon range | Lat range |
|------|-----------|-----------|
| `europe` | −25 to 40 | 35 to 72 |
| `germany` | 5 to 16 | 47 to 56 |
| `uk` | −11 to 3 | 49 to 61 |
| `central_europe` | 2 to 26 | 43 to 59 |

### Example

```bash
python case_studies/plot_case_study.py \
  --config configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml \
  --date 20250718 --day 5 --season JJA \
  --output-dir ./case_study_output/2t_JJA_flat
```

---

## 4. Batch Submission

Both tools have SLURM wrappers in `case_studies/`. Parameters are passed via `--export` environment variables (compatible with ECMWF's `ecsbatch` wrapper).

### `case_studies/submit_find_cases.sh`

Resources: 64 GB, 1 CPU, 4 h.

```bash
sbatch --export=CONFIG=configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml,\
DAYS="3 5 7",SEASON=JJA,OROG=flat,TOP=30 \
  case_studies/submit_find_cases.sh
```

**Supported `--export` variables:**

| Variable | Argument | Example |
|----------|----------|---------|
| `CONFIG` | `--config` *(required)* | `configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml` |
| `DAYS` | `--days` | `"3 5 7"` |
| `SEASON` | `--season` | `JJA` |
| `OROG` | `--orog` | `flat` |
| `TOP` | `--top-n` | `30` |
| `OUTPUT` | `--output-dir` | `./case_study_output/myrun` |

### `case_studies/submit_plot_case.sh`

Resources: 32 GB, 1 CPU, 1 h.

```bash
sbatch --export=CONFIG=configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml,\
DATE=20250718,DAY=5,SEASON=JJA \
  case_studies/submit_plot_case.sh
```

**Supported `--export` variables:**

| Variable | Argument | Example |
|----------|----------|---------|
| `CONFIG` | `--config` *(required)* | `configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml` |
| `DATE` | `--date` *(required)* | `20250718` |
| `DAY` | `--day` *(required)* | `5` |
| `OROG` | `--orog` | `flat` |
| `SEASON` | `--season` | `JJA` |
| `OUTPUT` | `--output` | `./plots/my_case.png` |
| `TITLE` | `--title` | `"JJA warm extreme"` |

---

## 5. Worked Example

Full end-to-end example for 2 m temperature warm extremes, AIFS vs IFS, JJA season.

### Prerequisites

The main pipeline must have completed extraction (`run.py`, Step 3) for the config:

```bash
sbatch scripts/submit_job.sh configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml
```

### Step 1 — Find worst dates

```bash
sbatch --export=\
CONFIG=configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml,\
DAYS="3 5 7",SEASON=JJA,OROG=flat,TOP=20 \
  case_studies/submit_find_cases.sh
```

Check output:

```bash
cat case_study_output/case_study_summary_aifs_vs_ifs_2t_p99.txt
# → Lists top 20 dates where the models diverge most in JJA flat terrain
```

### Step 2 — Visualise the worst date

Suppose the summary identifies 20250718 day-5 as the most striking case:

```bash
sbatch --export=\
CONFIG=configs/deterministic/config_2t_local_p99obsclim_aifs_ifs_single.yaml,\
DATE=20250718,DAY=5,SEASON=JJA \
  case_studies/submit_plot_case.sh
```

Three PNGs appear in `./case_study_output/config_2t_local_p99obsclim_aifs_ifs_single/`:
- `20250718_day5.png` — hit/miss/FA maps and score table
- `20250718_day5_values.png` — obs and forecast value maps
- `20250718_day5_overlay.png` — raw GRIB fields with obs dots

### Step 3 — Iterate

Use `--min-concentration 0.5` in `find_case_studies.py` to restrict to dates where errors are geographically concentrated in one sub-region — these are typically the most interpretable case studies.
