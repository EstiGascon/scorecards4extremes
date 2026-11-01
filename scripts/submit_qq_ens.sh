#!/bin/bash
#SBATCH --job-name=qq_plot_ens
#SBATCH --output=qq_plot_ens_%j.out
#SBATCH --error=qq_plot_ens_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Batch submission script for plot_qq_ens.py  (ECMWF / ecsbatch compatible)
# ==============================================================================
#
# The ensemble Q-Q tool can pool every member of a large ensemble across the
# whole verification period, which easily exceeds the login-node memory cap.
# Run it on a compute node with reserved memory instead.
#
# ECMWF's sbatch wrapper does not pass positional arguments to the script, so
# options are passed via --export environment variables:
#
#   # Full ensemble, low + high orography:
#   sbatch --export=CONFIG=configs/ensemble/config_2t_ens_local_p99obsclim_aifsvsifs.yaml,MEMBER=full,OROG="low high" \
#       scripts/submit_qq_ens.sh
#
#   # Control member only:
#   sbatch --export=CONFIG=<cfg>,MEMBER=control,OROG="low high" scripts/submit_qq_ens.sh
#
#   # A single perturbed member, day-3 only:
#   sbatch --export=CONFIG=<cfg>,MEMBER=1,LEAD_TIME=72 scripts/submit_qq_ens.sh
#
# Supported environment variables (all optional except CONFIG):
#   CONFIG       - YAML config file path                 (required)
#   MEMBER       - "full" | "control" | integer 0-50     (default: full)
#   SEASON       - Season(s): "DJF" or "DJF MAM JJA SON"
#   OROG         - Orography: "low" or "low high"
#   LEAD_TIME    - Lead time(s) in hours: "24" or "24 48 72"
#   START_DATE   - Override start date: "YYYY-MM-DD"
#   END_DATE     - Override end date:   "YYYY-MM-DD"
#   THRESHOLD    - Threshold value override (float)
#   N_QUANTILES  - Number of quantile points (default: 200)
#   OUTPUT_DIR   - Output directory for PNGs
#   DPI          - Image resolution (default: 150)
#   NO_LOOP      - Set to "1" to produce a single combined plot
# ==============================================================================

set -u

# Repo root. SLURM copies the batch script to a spool dir, so BASH_SOURCE is
# unreliable here; use the submit directory (sbatch is run from the repo root),
# with an absolute fallback.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/moeg/scorecards4extremes}"
if [[ ! -f "$REPO_ROOT/diagnostics/qq/plot_qq_ens.py" ]]; then
    REPO_ROOT="/home/moeg/scorecards4extremes"
fi
cd "$REPO_ROOT" || exit 1

echo "==========================================="
echo "Q-Q Plot Job (ENSEMBLE)"
echo "==========================================="
echo "Job ID : ${SLURM_JOB_ID:-<none>}"
echo "Node   : $HOSTNAME"
echo "Repo   : $REPO_ROOT"
echo "Start  : $(date)"
echo ""

if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG environment variable is not set."
    echo "Submit with: sbatch --export=CONFIG=<your_config.yaml>[,MEMBER=full,...] scripts/submit_qq_ens.sh"
    exit 1
fi

# -- Minimal ECMWF environment -----------------------------------------------
export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"
export METVIEW_PYTHON_START_TIMEOUT=30

# The tool needs pyarrow/pandas/matplotlib from the project venv.
source "$REPO_ROOT/.venv/bin/activate"

# -- Build argument list from environment variables --------------------------
ARGS="--config $CONFIG"

[[ -n "${MEMBER:-}" ]]      && ARGS="$ARGS --member $MEMBER"
[[ -n "${SEASON:-}" ]]      && ARGS="$ARGS --season $SEASON"
[[ -n "${OROG:-}" ]]        && ARGS="$ARGS --orog $OROG"
[[ -n "${LEAD_TIME:-}" ]]   && ARGS="$ARGS --lead-time $LEAD_TIME"
[[ -n "${START_DATE:-}" ]]  && ARGS="$ARGS --start-date $START_DATE"
[[ -n "${END_DATE:-}" ]]    && ARGS="$ARGS --end-date $END_DATE"
[[ -n "${THRESHOLD:-}" ]]   && ARGS="$ARGS --threshold $THRESHOLD"
[[ -n "${N_QUANTILES:-}" ]] && ARGS="$ARGS --n-quantiles $N_QUANTILES"
[[ -n "${OUTPUT_DIR:-}" ]]  && ARGS="$ARGS --output-dir $OUTPUT_DIR"
[[ -n "${DPI:-}" ]]         && ARGS="$ARGS --dpi $DPI"
[[ "${NO_LOOP:-}" == "1" ]] && ARGS="$ARGS --no-loop"

echo "==========================================="
echo "Running: python3 -u diagnostics/qq/plot_qq_ens.py $ARGS"
echo "==========================================="
echo ""

python3 -u diagnostics/qq/plot_qq_ens.py $ARGS
exit_code=$?

echo ""
echo "==========================================="
echo "Job finished  |  exit code: $exit_code"
echo "End: $(date)"
echo "==========================================="

exit $exit_code
