#!/bin/bash
#SBATCH --job-name=find_cases
#SBATCH --output=find_cases_%j.out
#SBATCH --error=find_cases_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/moeg/scorecards4extremes

# ==============================================================================
# Batch submission for find_case_studies.py
# ==============================================================================
# Usage:
#   sbatch --export=CONFIG=<yaml>,DAYS="1 3 5 7",SEASON=DJF,OROG=low,TOP=30 \
#       case_studies/submit_find_cases.sh
#
# Required:
#   CONFIG   — YAML config file path
#
# Optional:
#   DAYS     — Space-separated forecast days (default: all found)
#   SEASON   — DJF | MAM | JJA | SON (default: all)
#   OROG     — low | mid | high (default: all)
#   TOP      — Top N cases to print (default: 20)
#   OUTPUT   — Output directory (default: ./case_study_output/<config_name>)
# ==============================================================================

echo "==========================================="
echo "Case Study Finder Job"
echo "==========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $HOSTNAME"
echo "Start   : $(date)"
echo "Config  : ${CONFIG:-NOT SET}"
echo ""

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: CONFIG env variable not set."
    echo "Submit with: sbatch --export=CONFIG=my_config.yaml ... submit_find_cases.sh"
    exit 1
fi

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"

PYTHON=/home/moeg/scorecards4extremes/.venv/bin/python

# Build argument list
ARGS="--config $CONFIG"
[[ -n "$DAYS"   ]] && ARGS="$ARGS --days $DAYS"
[[ -n "$SEASON" ]] && ARGS="$ARGS --season $SEASON"
[[ -n "$OROG"   ]] && ARGS="$ARGS --orog $OROG"
[[ -n "$TOP"    ]] && ARGS="$ARGS --top-n $TOP"
[[ -n "$OUTPUT" ]] && ARGS="$ARGS --output-dir $OUTPUT"

echo "Running: $PYTHON -u case_studies/find_case_studies.py $ARGS"
echo ""

$PYTHON -u case_studies/find_case_studies.py $ARGS

exit_code=$?
echo ""
echo "==========================================="
echo "Done  |  exit code: $exit_code  |  $(date)"
echo "==========================================="
exit $exit_code
