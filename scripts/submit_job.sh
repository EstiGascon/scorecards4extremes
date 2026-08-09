#!/bin/bash
#SBATCH --job-name=scorecards_extract
#SBATCH --output=scorecards_%j.out
#SBATCH --error=scorecards_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --chdir=/home/moeg/scorecards4extremes

# ==============================================================================
# Batch submission script for scorecards4extremes on ECMWF
# ==============================================================================
# Usage:
#   sbatch scripts/submit_job.sh config_tp24_test.yaml
#
# ==============================================================================


echo "=========================================="
echo "Scorecards4Extremes Batch Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "Memory requested: 64GB"
echo ""

# Get config file from command line argument or SLURM array
CONFIG=${1:?'Usage: sbatch submit_job.sh <config_file.yaml>'}

# Minimal setup (avoid verbose output that might buffer)
export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p $TMPDIR
export METVIEW_PYTHON_START_TIMEOUT=120

# Set all env vars that ecmwf-toolbox/new module provides, explicitly.
# module load may fail silently on compute nodes due to module conflicts
# (ecmwf-toolbox conflicts with fdb, eccodes, atlas etc. that may be pre-loaded).
# Setting these directly ensures metview (a shell script) can find its components.
_TB="/usr/local/apps/ecmwf-toolbox/2026.04.0.0/GNU/8.5"
export PATH="${_TB}/bin:$PATH"
export ECMWF_TOOLBOX_DIR="${_TB}"
export ECCODES_DIR="${_TB}"
export ECCODES_PYTHON_USE_FINDLIBS="1"
export FINDLIBS_DISABLE_PACKAGE="yes"
export MAGICS_DIR="${_TB}"
export MAGPLUS_HOME="${_TB}"
export MAGPLUS_DEV="OFF"
export MAGPLUS_INFO="OFF"
export METVIEW_DIR="${_TB}"
export FDB5_DIR="${_TB}"
export ODC_DIR="${_TB}"

# Activate the project virtualenv — provides metview Python bindings (pip installed),
# scores, xarray, and all other packages.
source "/home/moeg/scorecards4extremes/.venv/bin/activate"

# vtb (ECMWF-internal toolbox) — needed by mars_retrieve.py for STVL observation
# retrieval (observation_source: stvl) and quaver forecast retrieval. The vtb
# extension is built for a specific Python; `module load quaver/3.6.4` is broken
# on Atos (missing ecmwf-toolbox dependency), so instead we point mars_retrieve's
# STVL subprocess worker directly at this venv (which already has vtb's runtime
# deps installed: python-magic, jsonschema, psycopg[binary]) via S4E_VTB_PYTHON,
# and put the matching vtb build on PYTHONPATH so the subprocess (which inherits
# this environment) can import it.
export S4E_VTB_PYTHON="/home/moeg/scorecards4extremes/.venv/bin/python3"
export PYTHONPATH="/usr/local/apps/vtb/1.3.3/lib/python3.12/site-packages:${PYTHONPATH}"

echo ""
echo "=========================================="
echo "Running: python run.py $CONFIG"
echo "=========================================="
echo ""

# Run the extraction (unbuffered Python output)
python -u run.py "$CONFIG"

exit_code=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

exit $exit_code
