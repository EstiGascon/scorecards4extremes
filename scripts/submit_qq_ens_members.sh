#!/bin/bash
#SBATCH --job-name=qq_plot_ens_mem
#SBATCH --output=qq_plot_ens_mem_%j.out
#SBATCH --error=qq_plot_ens_mem_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Batch submission script for plot_qq_ens_members.py  (ECMWF / ecsbatch compatible)
# ==============================================================================
#
# Pooling every member of one model across a full verification period easily
# exceeds the login-node memory cap (confirmed: even ONE model's ~51 columns
# OOM-kills at ~18GB). Run it on a compute node with reserved memory instead.
#
# ECMWF's sbatch wrapper does not pass positional arguments to the script, so
# options are passed via --export environment variables:
#
#   sbatch --export=CONFIG=configs/ensemble/config_2t_ens_local_p99obsclim_aifsvsifs.yaml,MODEL="fc1 fc2",OROG="low high" \
#       scripts/submit_qq_ens_members.sh
#
# Supported environment variables (all optional except CONFIG):
#   CONFIG       - YAML config file path                 (required)
#   MODEL        - "fc1" | "fc2" | "fc1 fc2"             (default: both)
#   SEASON       - Season(s): "DJF" or "DJF MAM JJA SON"
#   OROG         - Orography: "low" or "low high"
#   LEAD_TIME    - Lead time(s) in hours: "24" or "24 48 72"
#   START_DATE   - Override start date: "YYYY-MM-DD"
#   END_DATE     - Override end date:   "YYYY-MM-DD"
#   THRESHOLD    - Threshold value override (float)
#   OUTPUT_DIR   - Output directory for PNGs
#   DPI          - Image resolution (default: 150)
# ==============================================================================

set -u

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/moeg/scorecards4extremes}"
if [[ ! -f "$REPO_ROOT/diagnostics/qq/plot_qq_ens_members.py" ]]; then
    REPO_ROOT="/home/moeg/scorecards4extremes"
fi
cd "$REPO_ROOT" || exit 1

echo "==========================================="
echo "Q-Q Plot Job (ENSEMBLE, within-model members)"
echo "==========================================="
echo "Job ID : ${SLURM_JOB_ID:-<none>}"
echo "Node   : $HOSTNAME"
echo "Repo   : $REPO_ROOT"
echo "Start  : $(date)"
echo ""

if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG environment variable is not set."
    echo "Submit with: sbatch --export=CONFIG=<your_config.yaml>[,MODEL=\"fc1 fc2\",...] scripts/submit_qq_ens_members.sh"
    exit 1
fi

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"
export METVIEW_PYTHON_START_TIMEOUT=30

source "$REPO_ROOT/.venv/bin/activate"

ARGS="--config $CONFIG"

[[ -n "${MODEL:-}" ]]       && ARGS="$ARGS --model $MODEL"
[[ -n "${SEASON:-}" ]]      && ARGS="$ARGS --season $SEASON"
[[ -n "${OROG:-}" ]]        && ARGS="$ARGS --orog $OROG"
[[ -n "${LEAD_TIME:-}" ]]   && ARGS="$ARGS --lead-time $LEAD_TIME"
[[ -n "${START_DATE:-}" ]]  && ARGS="$ARGS --start-date $START_DATE"
[[ -n "${END_DATE:-}" ]]    && ARGS="$ARGS --end-date $END_DATE"
[[ -n "${THRESHOLD:-}" ]]   && ARGS="$ARGS --threshold $THRESHOLD"
[[ -n "${OUTPUT_DIR:-}" ]]  && ARGS="$ARGS --output-dir $OUTPUT_DIR"
[[ -n "${DPI:-}" ]]         && ARGS="$ARGS --dpi $DPI"

echo "==========================================="
echo "Running: python3 -u diagnostics/qq/plot_qq_ens_members.py $ARGS"
echo "==========================================="
echo ""

python3 -u diagnostics/qq/plot_qq_ens_members.py $ARGS
exit_code=$?

echo ""
echo "==========================================="
echo "Job finished  |  exit code: $exit_code"
echo "End: $(date)"
echo "==========================================="

exit $exit_code
