#!/bin/bash
#SBATCH --job-name=scorecards_extract
#SBATCH --output=scorecards_%j.out
#SBATCH --error=scorecards_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --chdir=/path/to/scorecards4extremes

# ==============================================================================
# Batch submission script for scorecards4extremes on ECMWF
# ==============================================================================
# Usage:
#   sbatch submit_job.sh config_tp24_test.yaml
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
export METVIEW_PYTHON_START_TIMEOUT=30

# Load required modules (provides the metview binary in PATH)
module load ecmwf-toolbox/new

# Activate the project virtualenv — provides metview Python bindings (pip installed),
# scores, xarray, and all other packages.
source "$(dirname "$0")/.venv/bin/activate"

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
