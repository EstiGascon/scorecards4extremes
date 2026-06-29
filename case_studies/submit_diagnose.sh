#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --output=diagnose_%j.out
#SBATCH --error=diagnose_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/moeg/scorecards4extremes

# ==============================================================================
# Batch submission for diagnose_extremes.py
# ==============================================================================
# Usage:
#   sbatch --export=CONFIG=<yaml>,DAY=3,SEASON=DJF,OROG=flat,PCT=99,OUTPUT=<dir> \
#       case_studies/submit_diagnose.sh
#
# Required:
#   CONFIG   — YAML config file path
#
# Optional:
#   DAY      — Forecast day (default: 3)
#   SEASON   — DJF | MAM | JJA | SON (default: all)
#   OROG     — flat | low | mid | hilly | high | complex (default: all)
#   PCT      — Threshold percentile, e.g. 99 or 1 (default: 99)
#   OUTPUT   — Output directory (default: results dir from config)
# ==============================================================================

echo "==========================================="
echo "Diagnose Extremes Job"
echo "==========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $HOSTNAME"
echo "Start   : $(date)"
echo "Config  : ${CONFIG:-NOT SET}"
echo ""

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: CONFIG env variable not set."
    echo "Submit with: sbatch --export=CONFIG=my_config.yaml ... submit_diagnose.sh"
    exit 1
fi

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"

PYTHON=/home/moeg/scorecards4extremes/.venv/bin/python

ARGS="--config $CONFIG"
[[ -n "$DAY"    ]] && ARGS="$ARGS --day $DAY"
[[ -n "$SEASON" ]] && ARGS="$ARGS --season $SEASON"
[[ -n "$OROG"   ]] && ARGS="$ARGS --orog $OROG"
[[ -n "$PCT"    ]] && ARGS="$ARGS --threshold-pct $PCT"
[[ -n "$OUTPUT" ]] && ARGS="$ARGS --output-dir $OUTPUT"

echo "Running: $PYTHON -u diagnose_extremes.py $ARGS"
echo ""

$PYTHON -u diagnose_extremes.py $ARGS

exit_code=$?
echo ""
echo "==========================================="
echo "Done  |  exit code: $exit_code  |  $(date)"
echo "==========================================="
exit $exit_code
