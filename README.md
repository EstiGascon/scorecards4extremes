# Scorecards for Extremes

A verification toolkit for extreme weather events. It compares two NWP forecast
models against surface observations, computing skill scores that focus on the
extreme tails of the forecast distribution, and produces heatmap scorecards and
Q-Q diagnostic plots.

## Features

- **Two-model comparison** — all scores express the relative improvement/degradation of model 1 vs model 2
- **Tail-focused metrics** — threshold-weighted scores (twMAE, twRMSE, twCRPS, twQS) concentrate the evaluation on extreme cases
- **Three supported variables** — 2 m temperature (`2t`), 24 h precipitation (`tp24`), 10 m wind speed (`10ff`)
- **Deterministic and ensemble** modes
- **Bootstrap significance testing** — 95% confidence intervals on all score differences
- **Season × terrain stratification** — results broken down by season (DJF/MAM/JJA/SON) and terrain complexity (flat/hilly/complex)

## Quick start

```bash
# 1. Install (see docs/INSTALL.md for system dependencies and Metview)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install metview

# 2. Copy and edit a config template
cp config_example.yaml my_run.yaml
# edit my_run.yaml — set start_date, end_date, data paths, output paths

# 3. Run
python run.py my_run.yaml
```

For a full step-by-step walkthrough, see [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Documentation

| Document | Content |
|----------|---------|
| [docs/INSTALL.md](docs/INSTALL.md) | System dependencies, virtual environment, Metview, ECMWF-only features |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Step-by-step first run in 5 steps |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Full reference — config schema, pipeline steps, scores, threshold methods, diagnostics |
| [docs/SCIENCE.md](docs/SCIENCE.md) | Mathematical definitions of all scores |
| [docs/COMPUTING.md](docs/COMPUTING.md) | Hardware requirements, SLURM config, runtime estimates |
| [docs/CASE_STUDIES.md](docs/CASE_STUDIES.md) | Case-study identification and visualisation tools |

## How it works

```
config.yaml
    │
    ▼
python run.py my_config.yaml
    │
    ├─ Step 1  src/read_data.py      Read GRIB forecasts + .gpt observations
    ├─ Step 2  src/preprocess.py     Unit conversion, lapse-rate, accumulation
    ├─ Step 3  src/extract_points.py Interpolate to observation stations (Metview)
    ├─ Step 4  src/filter.py         Season, terrain, QC filters
    ├─ Step 5  src/threshold.py      Compute event threshold
    ├─ Step 6  src/det_scores.py /   Verification scores (ETS, PSS, twMAE, …)
    │          src/ens_scores.py
    ├─ Step 7  src/bootstrap.py      95% confidence intervals
    ├─ Step 8  src/save.py           Write CSV results
    └─ Step 9  src/plot.py           Draw heatmap scorecards
```

## Repository layout

```
run.py              Entry point — run `python run.py <config.yaml>`
config_example.yaml Annotated config template
requirements.txt

src/                Core pipeline modules (the scorecards tool itself)
configs/            Ready-to-use config files (deterministic/, ensemble/, cams/)
scripts/            SLURM submission + helper scripts (submit_job.sh, …)
diagnostics/        Q-Q and extreme-event diagnostic plots
analysis/           One-off analysis and figure scripts
case_studies/       Case-study identification and visualisation
docs/               Documentation (INSTALL, USER_GUIDE, SCIENCE, …)
tests/              Unit tests
```

## Requirements

Python ≥ 3.10. See [docs/INSTALL.md](docs/INSTALL.md) for the full dependency list and
system libraries. Key third-party packages: `metview`, `cfgrib`, `scores`,
`xarray`, `pandas`, `matplotlib`, `Cartopy`.

> **Note**: the `quaver`/STVL backend and the obs climatology builder require
> ECMWF-internal services and will not work on external systems. The standard
> `local_grib` / `local_gpt` workflow has no ECMWF dependencies beyond Metview.

