#!/bin/bash
#SBATCH --job-name=plot_case
#SBATCH --output=plot_case_%j.out
#SBATCH --error=plot_case_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/moeg/scorecards4extremes

# ==============================================================================
# Batch submission for plot_case_study.py
# ==============================================================================
# Usage:
#   sbatch --export=CONFIG=<yaml>,DATE=YYYYMMDD,DAY=3,OROG=low,OUTPUT=<dir> \
#       case_studies/submit_plot_case.sh
#
# Required:
#   CONFIG  — YAML config file path
#   DATE    — YYYYMMDD
#   DAY     — Forecast day integer
#
# Optional:
#   OROG    — low | mid | high (default: none = all)
#   SEASON  — DJF | MAM | JJA | SON (default: all)
#   OUTPUT  — Output directory (default: ./case_study_output/<config_stem>)
#   TITLE   — Optional plot title override
# ==============================================================================

echo "==========================================="
echo "Case Study Plot Job"
echo "==========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $HOSTNAME"
echo "Start   : $(date)"
echo "Config  : ${CONFIG:-NOT SET}"
echo "Date    : ${DATE:-NOT SET}"
echo "Day     : ${DAY:-NOT SET}"
echo ""

if [[ -z "$CONFIG" || -z "$DATE" || -z "$DAY" ]]; then
    echo "ERROR: CONFIG, DATE and DAY must all be set."
    exit 1
fi

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"

PYTHON=/home/moeg/scorecards4extremes/.venv/bin/python

ARGS="--config $CONFIG --date $DATE --day $DAY"
[[ -n "$OROG"   ]] && ARGS="$ARGS --orog $OROG"
[[ -n "$SEASON" ]] && ARGS="$ARGS --season $SEASON"
[[ -n "$OUTPUT" ]] && ARGS="$ARGS --output $OUTPUT"
[[ -n "$TITLE"  ]] && ARGS="$ARGS --title '$TITLE'"

echo "Running: $PYTHON -u case_studies/plot_case_study.py $ARGS"
echo ""

$PYTHON -u case_studies/plot_case_study.py $ARGS

exit_code=$?
echo ""
echo "==========================================="
echo "Done  |  exit code: $exit_code  |  $(date)"
echo "==========================================="
exit $exit_code
