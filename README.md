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
# 1. Install (see INSTALL.md for system dependencies and Metview)
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
| [INSTALL.md](INSTALL.md) | System dependencies, virtual environment, Metview, ECMWF-only features |
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
    ├─ Step 1  read_data.py          Read GRIB forecasts + .gpt observations
    ├─ Step 2  preprocess.py         Unit conversion, lapse-rate, accumulation
    ├─ Step 3  extract_points.py     Interpolate to observation stations (Metview)
    ├─ Step 4  filter.py             Season, terrain, QC filters
    ├─ Step 5  threshold.py          Compute event threshold
    ├─ Step 6  det_scores.py /       Verification scores (ETS, PSS, twMAE, …)
    │          ens_scores.py
    ├─ Step 7  bootstrap.py          95% confidence intervals
    ├─ Step 8  save.py               Write CSV results
    └─ Step 9  plot.py               Draw heatmap scorecards
```

## Requirements

Python ≥ 3.10. See [INSTALL.md](INSTALL.md) for the full dependency list and
system libraries. Key third-party packages: `metview`, `cfgrib`, `scores`,
`xarray`, `pandas`, `matplotlib`, `Cartopy`.

> **Note**: the `quaver`/STVL backend and the obs climatology builder require
> ECMWF-internal services and will not work on external systems. The standard
> `local_grib` / `local_gpt` workflow has no ECMWF dependencies beyond Metview.

