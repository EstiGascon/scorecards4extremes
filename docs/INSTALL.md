# Installation Guide

This guide covers installation on a standard Linux system. An ECMWF HPC
environment is **not** required for the core `local_grib` / `local_gpt`
workflow, but some features depend on ECMWF-internal services — see
[ECMWF-only features](#ecmwf-only-features) below.

---

## Requirements

- **Python ≥ 3.10**
- **ecCodes** system library (for GRIB reading)
- **PROJ ≥ 9** and **GEOS** libraries (for Cartopy / Shapely)
- **Metview** binary (for point extraction, Step 3)

---

## Step 1 — Install system libraries

### Debian / Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y \
    libeccodes-dev \
    libproj-dev \
    libgeos-dev \
    build-essential
```

### Red Hat / CentOS / Rocky Linux

```bash
sudo dnf install -y eccodes-devel proj-devel geos-devel gcc
```

### macOS (Homebrew)

```bash
brew install eccodes proj geos
```

---

## Step 2 — Install Metview

Metview is required for Step 3 (point extraction from GRIB files). The
recommended path for non-ECMWF systems is conda-forge:

```bash
conda create -n s4e python=3.11
conda activate s4e
conda install -c conda-forge metview
```

Alternatively, follow the official installation guide:
<https://metview.readthedocs.io/en/latest/install.html>

> **ECMWF HPC**: Metview is available as a module:
> `module load ecmwf-toolbox/new`
> The Python bindings are then installed into the project venv (see Step 4).

---

## Step 3 — Create a virtual environment

If you installed Metview via conda-forge (Step 2), use the conda environment
directly and skip to Step 4. If you installed Metview system-wide:

```bash
cd /path/to/scorecards4extremes
python3 -m venv .venv
source .venv/bin/activate
```

---

## Step 4 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Then install the Metview Python bindings:

```bash
pip install metview
```

Verify the installation:

```bash
python -c "import metview; print('Metview OK')"
python -c "import scores; print('scores OK')"
python -c "import cfgrib; print('cfgrib OK')"
```

If the `metview` import fails, check that the Metview binary is in `PATH`
and that `METVIEW_PYTHON_START_TIMEOUT` is set:

```bash
export METVIEW_PYTHON_START_TIMEOUT=30
```

---

## Step 5 — (ECMWF HPC only) Update the SLURM submission scripts

Before submitting batch jobs, update `scripts/submit_job.sh` and
`scripts/submit_diagnose.sh` to point to your own clone of the repository:

1. Change the `#SBATCH --chdir=` line to the actual path where you checked
   out the repository.
2. Change the `source ... .venv/bin/activate` line to the correct path to
   your venv.

```bash
# Example — edit these two lines in scripts/submit_job.sh:
#SBATCH --chdir=/home/<your_username>/scorecards4extremes
source /home/<your_username>/scorecards4extremes/.venv/bin/activate
```

---

## Quick check

After installation, verify the main entry point loads without error:

```bash
python -c "import run; print('Pipeline imports OK')"
```

> **Note**: this will print a warning if `metview` is not importable, but
> will not crash — extraction (Step 3) requires Metview; all other steps do
> not.

---

## ECMWF-only features

The following features will **not** work outside ECMWF because they depend on
internal services or packages:

| Feature | Requires | Alternative for external users |
|---------|----------|-------------------------------|
| `source: quaver` forecast data | Quaver/MARS ECMWF service | Use `source: local_grib` with GRIB files obtained via any means (e.g. CDS, own NWP model) |
| `observation_source: quaver` | STVL ECMWF observation database | Use `observation_source: local_gpt` with Geopoints files converted from your own obs data |
| `threshold.method: local_obs_climatology` | Pre-built climatology files from STVL (obs climatology builder requires STVL access) | Use `threshold.method: dataset_climatology` — computes the percentile directly from the observations in your verification dataset; no external files needed |
| `threshold.method: station_climatology` | STVL (1980–2009 baseline) | Use `dataset_climatology` or `fixed` |
| `scripts/extract_tp24_obs.py` obs builder | `vtb` (ECMWF-internal Python package) | Not available externally; obtain tp24 obs via ERA5 CDS or national networks in Geopoints format |

For the standard external workflow use:
```yaml
read_data:
  forecast_model1:
    source: local_grib
    ...
  forecast_model2:
    source: local_grib
    ...
  observation_source: local_gpt

threshold:
  method: dataset_climatology
  ...
```
