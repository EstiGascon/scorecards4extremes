#!/bin/bash
#SBATCH --job-name=mars_10ff_ens_hybrid_pf
#SBATCH --output=mars_10ff_ens_hybrid_pf_%j.out
#SBATCH --error=mars_10ff_ens_hybrid_pf_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve IFS-ENS and Hybrid-ENS (pf only) 10m wind components (10u, 10v) from MARS
# Period  : 2025-10-20 to 2026-03-31
# Steps   : 0, 12, 24, ..., 240 h  (every 12 h)
# Members : perturbed only (pf, 1–50) for Hybrid; cf + pf for IFS
# Output  : separate GRIB files for 10u and 10v per model per day
#           Wind speed (10ff = sqrt(10u² + 10v²)) is computed at extraction time
# ==============================================================================
# Usage (interactive):  bash retrieve_10ff_ens_hybrid_pf.sh
# Usage (batch):        sbatch retrieve_10ff_ens_hybrid_pf.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/10ff/forecast_ENS/raw/"
HYBRID_DIR="$OUTPUT_BASE/hybrid_ens"

mkdir -p "$HYBRID_DIR"

# 12-hourly steps from 0 to 240 h
STEPS="0/12/24/36/48/60/72/84/96/108/120/132/144/156/168/180/192/204/216/228/240"

START_DATE="20251020"
END_DATE="20260331"

echo "=============================================="
echo "MARS 10U/10V ENS (hybrid, pf only) retrieval"
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
    hybrid_u_out="$HYBRID_DIR/10u_hybrid_ens_${current}.grib"
    hybrid_v_out="$HYBRID_DIR/10v_hybrid_ens_${current}.grib"

    if [[ -f "$hybrid_u_out" && -f "$hybrid_v_out" ]]; then
        echo "[$current] Hybrid-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] Hybrid-ENS – retrieving 10u (pf only)..."
        reqfile=$(mktemp /tmp/mars_hybrid_10u_XXXXXX.req)
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
  param    = 10u,
  target   = "${hybrid_u_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        u_exit=${PIPESTATUS[0]}
        rm -f "$reqfile"

        echo "[$current] Hybrid-ENS – retrieving 10v (pf only)..."
        reqfile=$(mktemp /tmp/mars_hybrid_10v_XXXXXX.req)
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
  param    = 10v,
  target   = "${hybrid_v_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        v_exit=${PIPESTATUS[0]}
        rm -f "$reqfile"

        if [[ $u_exit -eq 0 && $v_exit -eq 0 ]]; then
            echo "[$current] Hybrid-ENS – OK (10u + 10v)"
            (( n_ok++ ))
        else
            echo "[$current] Hybrid-ENS – FAILED (10u=$u_exit, 10v=$v_exit)"
            (( n_err++ ))
            [[ $u_exit -ne 0 ]] && rm -f "$hybrid_u_out"
            [[ $v_exit -ne 0 ]] && rm -f "$hybrid_v_out"
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
