#!/bin/bash
#SBATCH --job-name=mars_2t_aifs_ens
#SBATCH --output=mars_2t_aifs_ens_%j.out
#SBATCH --error=mars_2t_aifs_ens_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve AIFS-ENS 2-metre temperature from MARS
# Period  : 2025-07-02 to 2026-05-11
# Steps   : 0, 6, 12, ..., 192 h  (every 6 h)
# Members : control (cf) + perturbed (pf, 1–50)
# Output  : one GRIB file per day, all steps and members combined
# ==============================================================================
# Usage (interactive):  bash retrieve_2t_aifs_ens.sh
# Usage (batch):        sbatch retrieve_2t_aifs_ens.sh
# ==============================================================================

OUTPUT_DIR="/ec/vol/destine/continuous_evaluation/2mtemp/forecast_ENS/raw/aifs_ens"

mkdir -p "$OUTPUT_DIR"

STEPS="0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"

START_DATE="20250702"
END_DATE="20260511"

echo "=============================================="
echo "MARS 2T AIFS-ENS retrieval"
echo "Period : $START_DATE – $END_DATE"
echo "Steps  : $STEPS"
echo "Output : $OUTPUT_DIR"
echo "Started: $(date)"
echo "=============================================="

current="$START_DATE"
n_ok=0
n_skip=0
n_err=0

while [[ "$current" -le "$END_DATE" ]]; do

    out="$OUTPUT_DIR/2t_aifs_ens_${current}.grib"

    echo "[$current] AIFS-ENS – retrieving..."
    if true; then
        reqfile=$(mktemp /tmp/mars_aifs_2t_XXXXXX.req)
        cat > "$reqfile" << MARSEOF
retrieve,
  class    = ai,
  type     = cf,
  stream   = enfo,
  expver   = 1,
  model    = aifs-ens,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = 2t,
  target   = "${out}"
retrieve,
  class    = ai,
  type     = pf,
  stream   = enfo,
  expver   = 1,
  model    = aifs-ens,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  number   = 1/to/50,
  levtype  = sfc,
  param    = 2t,
  target   = "${out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        exit_code=${PIPESTATUS[0]}
        rm -f "$reqfile"
        if [[ $exit_code -eq 0 ]]; then
            echo "[$current] AIFS-ENS – OK"
            (( n_ok++ ))
        else
            echo "[$current] AIFS-ENS – FAILED (exit $exit_code)"
            (( n_err++ ))
            rm -f "$out"
        fi
    fi

    current=$(date -d "$current + 1 day" +"%Y%m%d")
done

echo ""
echo "=============================================="
echo "Done: $(date)"
echo "  OK      : $n_ok"
echo "  Skipped : $n_skip"
echo "  Failed  : $n_err"
echo "  Output  : $OUTPUT_DIR"
echo "=============================================="
