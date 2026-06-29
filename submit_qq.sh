#!/bin/bash
#SBATCH --job-name=qq_plot
#SBATCH --output=qq_plot_%j.out
#SBATCH --error=qq_plot_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Batch submission script for plot_qq.py  (ECMWF / ecsbatch compatible)
# ==============================================================================
#
# ECMWF's sbatch wrapper does not pass extra positional arguments to the script.
# Pass options via --export environment variables instead:
#
#   # All data combined:
#   sbatch --export=CONFIG=config_tp24_precipitation.yaml submit_qq.sh
#
#   # Specific season + orography:
#   sbatch --export=CONFIG=config_tp24_precipitation.yaml,SEASON=DJF,OROG=low \
#       submit_qq.sh
#
#   # Multiple seasons and orography types (space-separated inside quotes):
#   sbatch --export=CONFIG=config_tp24_precipitation.yaml,SEASON="DJF MAM JJA SON",OROG="low mid high" \
#       submit_qq.sh
#
#   # Day-3 only, custom output directory:
#   sbatch --export=CONFIG=config_tp24_precipitation.yaml,LEAD_TIME=72,OUTPUT_DIR=./plots/qq \
#       submit_qq.sh
#
# Supported environment variables (all optional except CONFIG):
#   CONFIG       - YAML config file path          (required)
#   SEASON       - Season(s): "DJF" or "DJF MAM JJA SON"
#   OROG         - Orography: "low" or "low mid high"
#   LEAD_TIME    - Lead time(s) in hours: "24" or "24 48 72"
#   START_DATE   - Override start date: "YYYY-MM-DD"
#   END_DATE     - Override end date:   "YYYY-MM-DD"
#   THRESHOLD    - Threshold value override (float)
#   N_QUANTILES  - Number of quantile points (default: 200)
#   OUTPUT_DIR   - Output directory for PNGs
#   DPI          - Image resolution (default: 150)
#   NO_LOOP      - Set to "1" to produce a single combined plot
# ==============================================================================

echo "==========================================="
echo "Q-Q Plot Job"
echo "==========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $HOSTNAME"
echo "Start  : $(date)"
echo ""

# -- Validate required variable ----------------------------------------------
if [[ -z "$CONFIG" ]]; then
    echo "ERROR: CONFIG environment variable is not set."
    echo "Submit with: sbatch --export=CONFIG=<your_config.yaml>[,...] submit_qq.sh"
    exit 1
fi

# -- Minimal ECMWF environment -----------------------------------------------
export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p "$TMPDIR"
export METVIEW_PYTHON_START_TIMEOUT=30

# Uncomment if modules are not loaded automatically:
# module load ecmwf-toolbox/new
# module load python3

# -- Build argument list from environment variables --------------------------
ARGS="--config $CONFIG"

[[ -n "$SEASON" ]]      && ARGS="$ARGS --season $SEASON"
[[ -n "$OROG" ]]        && ARGS="$ARGS --orog $OROG"
[[ -n "$LEAD_TIME" ]]   && ARGS="$ARGS --lead-time $LEAD_TIME"
[[ -n "$START_DATE" ]]  && ARGS="$ARGS --start-date $START_DATE"
[[ -n "$END_DATE" ]]    && ARGS="$ARGS --end-date $END_DATE"
[[ -n "$THRESHOLD" ]]   && ARGS="$ARGS --threshold $THRESHOLD"
[[ -n "$N_QUANTILES" ]] && ARGS="$ARGS --n-quantiles $N_QUANTILES"
[[ -n "$OUTPUT_DIR" ]]  && ARGS="$ARGS --output-dir $OUTPUT_DIR"
[[ -n "$DPI" ]]         && ARGS="$ARGS --dpi $DPI"
[[ "$NO_LOOP" == "1" ]] && ARGS="$ARGS --no-loop"

# -- Run the tool ------------------------------------------------------------
echo "==========================================="
echo "Running: python3 -u diagnostics/plot_qq.py $ARGS"
echo "==========================================="
echo ""

python3 -u diagnostics/plot_qq.py $ARGS

exit_code=$?

echo ""
echo "==========================================="
echo "Job finished  |  exit code: $exit_code"
echo "End: $(date)"
echo "==========================================="

exit $exit_code
