#!/bin/bash
#SBATCH --job-name=bias_orog
#SBATCH --output=bias_orog_%j.out
#SBATCH --error=bias_orog_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=

# ==============================================================================
# Batch submission for analyse_bias_orog.py
# ==============================================================================
# Usage:
#   sbatch --export=CONFIG=<yaml>,SEASON=DJF,OROG=complex,OUTPUT=<dir> \
#       submit_bias_orog.sh
#
# Required:
#   CONFIG   — YAML config file path
#
# Optional:
#   SEASON   — DJF | MAM | JJA | SON (default: all)
#   OROG     — flat | hilly | complex (default: all)
#   OUTPUT   — Output directory (default: results/bias_<season>_<orog>)
# ==============================================================================

echo "==========================================="
echo "Bias / Distribution Analysis Job"
echo "==========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $HOSTNAME"
echo "Start   : $(date)"
echo "Config  : ${CONFIG:-NOT SET}"
echo "Season  : ${SEASON:-all}"
echo "Orog    : ${OROG:-all}"
echo ""

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: CONFIG env variable not set."
    exit 1
fi

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"

PYTHON=./.venv/bin/python

ARGS="--config $CONFIG"
[[ -n "$SEASON" ]] && ARGS="$ARGS --season $SEASON"
[[ -n "$OROG"   ]] && ARGS="$ARGS --orog $OROG"
[[ -n "$OUTPUT" ]] && ARGS="$ARGS --output-dir $OUTPUT"

echo "Running: $PYTHON -u analysis/analyse_bias_orog.py $ARGS"
echo ""

$PYTHON -u analysis/analyse_bias_orog.py $ARGS

exit_code=$?
echo ""
echo "==========================================="
echo "Done  |  exit code: $exit_code  |  $(date)"
echo "==========================================="
exit $exit_code
