#!/bin/bash
#SBATCH --job-name=mars_2t_ens_hybrid_pf
#SBATCH --output=mars_2t_ens_hybrid_pf_%j.out
#SBATCH --error=mars_2t_ens_hybrid_pf_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve IFS-ENS and Hybrid-ENS (pf only) 2-metre temperature from MARS
# Period  : 2025-10-20 to 2026-03-31
# Steps   : 0, 12, 24, ..., 240 h  (every 12 h)
# Members : perturbed only (pf, 1–50) for Hybrid; cf + pf for IFS
# Output  : one GRIB file per model per day, all steps and members combined
# ==============================================================================
# Usage (interactive):  bash retrieve_2t_ens_hybrid_pf.sh
# Usage (batch):        sbatch retrieve_2t_ens_hybrid_pf.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/2mtemp/forecast_ENS/raw/"
HYBRID_DIR="$OUTPUT_BASE/hybrid_ens"

mkdir -p "$HYBRID_DIR"

# 12-hourly steps from 0 to 240 h
STEPS="0/12/24/36/48/60/72/84/96/108/120/132/144/156/168/180/192/204/216/228/240"

START_DATE="20251020"
END_DATE="20260331"

echo "=============================================="
echo "MARS 2T ENS (hybrid, pf only) retrieval"
echo "Period : $START_DATE – $END_DATE"
echo "Steps  : $STEPS"
echo "Output : $OUTPUT_BASE"
echo "Started: $(date)"
echo "=============================================="

current="$START_DATE"
n_ok=0
n_skip=0
n_err=0

while [[ "$current" -le "$END_DATE" ]]; do

    # -------------------------------------------------------------- Hybrid-ENS (pf only)
    hybrid_out="$HYBRID_DIR/2t_hybrid_ens_${current}.grib"

    if [[ -f "$hybrid_out" ]]; then
        echo "[$current] Hybrid-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] Hybrid-ENS – retrieving (pf only)..."
        reqfile=$(mktemp /tmp/mars_hybrid_2t_XXXXXX.req)
        cat > "$reqfile" << MARSEOF
retrieve,
  class    = rd,
  type     = pf,
  stream   = enfo,
  expver   = iy2u,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  number   = 1/to/50,
  levtype  = sfc,
  param    = 2t,
  target   = "${hybrid_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        exit_code=${PIPESTATUS[0]}
        rm -f "$reqfile"
        if [[ $exit_code -eq 0 ]]; then
            echo "[$current] Hybrid-ENS – OK"
            (( n_ok++ ))
        else
            echo "[$current] Hybrid-ENS – FAILED (exit $exit_code)"
            (( n_err++ ))
            rm -f "$hybrid_out"  # remove incomplete file
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
echo "  Output  : $OUTPUT_BASE"
echo "=============================================="
