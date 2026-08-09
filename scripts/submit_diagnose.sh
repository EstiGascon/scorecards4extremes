#!/bin/bash
#SBATCH --job-name=diagnose_extremes
#SBATCH --output=diagnose_%j.out
#SBATCH --error=diagnose_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/path/to/scorecards4extremes

# This script lives in scripts/; the venv is one level up at the repo root.
source "$(dirname "$0")/../.venv/bin/activate"

echo "Job ID: $SLURM_JOB_ID  |  Node: $HOSTNAME  |  Start: $(date)"
echo "Args: $@"
echo ""

python3 -u diagnostics/diagnose_extremes.py "$@"

exit_code=$?
echo ""
echo "Finished with exit code: $exit_code  |  End: $(date)"
exit $exit_code
