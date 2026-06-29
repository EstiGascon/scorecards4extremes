#!/bin/bash
#SBATCH --job-name=qq_extremes
#SBATCH --output=qq_extremes_%j.out
#SBATCH --error=qq_extremes_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=

# ==============================================================================
# Batch submission script for plot_qq_extremes.py on ECMWF
# ==============================================================================
# Write desired args (one line) to .qq_extremes_args, then: sbatch submit_qq_extremes.sh
#
# Example:
#   echo "--config config_2t_local_p1obsclim_aifs_ifs_single.yaml --day 7 --season DJF --orog flat" > .qq_extremes_args
#   sbatch submit_qq_extremes.sh
# ==============================================================================

ARGS_FILE=".qq_extremes_args"

if [[ -f "$ARGS_FILE" ]]; then
    QQ_ARGS=$(cat "$ARGS_FILE")
else
    echo "ERROR: $ARGS_FILE not found. Write args to that file before submitting."
    exit 1
fi

echo "==========================================="
echo "Q-Q Extremes Plot Job"
echo "==========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $HOSTNAME"
echo "Start  : $(date)"
echo "Args   : $QQ_ARGS"
echo ""

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"
export METVIEW_PYTHON_START_TIMEOUT=30

module load python3

echo ""
echo "=========================================="
echo "Running: python3 plot_qq_extremes.py $QQ_ARGS"
echo "=========================================="
echo ""

python3 -u plot_qq_extremes.py $QQ_ARGS

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished  |  exit code: $exit_code"
echo "End: $(date)"
echo "=========================================="

exit $exit_code
