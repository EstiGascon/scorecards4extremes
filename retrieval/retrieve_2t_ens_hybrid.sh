#!/bin/bash
#SBATCH --job-name=mars_2t_ens_hybrid
#SBATCH --output=mars_2t_ens_hybrid_%j.out
#SBATCH --error=mars_2t_ens_hybrid_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve IFS-ENS and Hybrid-ENS 2-metre temperature from MARS
# Period  : 2025-12-01 to 2026-02-28
# Steps   : 0, 6, 12, 18, 24, ..., 240 h  (every 6 h)
# Members : control (cf) + perturbed (pf, 1–50)
# Output  : one GRIB file per model per day, all steps and members combined
# ==============================================================================
# Usage (interactive):  bash retrieve_2t_ens_hybrid.sh
# Usage (batch):        sbatch retrieve_2t_ens_hybrid.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/2mtemp/forecast_ENS/raw/"
IFS_DIR="$OUTPUT_BASE/ifs_ens4hybrid"
HYBRID_DIR="$OUTPUT_BASE/hybrid_ens"

mkdir -p "$IFS_DIR" "$HYBRID_DIR"

# 12-hourly steps from 0 to 240 h
STEPS="0/12/24/36/48/60/72/84/96/108/120/132/144/156/168/180/192/204/216/228/240"

START_DATE="20251020"
END_DATE="20260331"

echo "=============================================="
echo "MARS 2T ENS (hybrid) retrieval"
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

    # ------------------------------------------------------------------ IFS-ENS
    ifs_out="$IFS_DIR/2t_ifs_ens_${current}.grib"

    if [[ -f "$ifs_out" ]]; then
        echo "[$current] IFS-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] IFS-ENS – retrieving..."
        reqfile=$(mktemp /tmp/mars_ifs_2t_XXXXXX.req)
        cat > "$reqfile" << MARSEOF
retrieve,
  class    = od,
  type     = cf,
  stream   = enfo,
  expver   = 1,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = 2t,
  target   = "${ifs_out}"
retrieve,
  class    = od,
  type     = pf,
  stream   = enfo,
  expver   = 1,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  number   = 1/to/50,
  levtype  = sfc,
  param    = 2t,
  target   = "${ifs_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        exit_code=${PIPESTATUS[0]}
        rm -f "$reqfile"
        if [[ $exit_code -eq 0 ]]; then
            echo "[$current] IFS-ENS – OK"
            (( n_ok++ ))
        else
            echo "[$current] IFS-ENS – FAILED (exit $exit_code)"
            (( n_err++ ))
            rm -f "$ifs_out"   # remove incomplete file
        fi
    fi

    # -------------------------------------------------------------- Hybrid-ENS
    hybrid_out="$HYBRID_DIR/2t_hybrid_ens_${current}.grib"

    if [[ -f "$hybrid_out" ]]; then
        echo "[$current] Hybrid-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] Hybrid-ENS – retrieving..."
        reqfile=$(mktemp /tmp/mars_hybrid_2t_XXXXXX.req)
        cat > "$reqfile" << MARSEOF
retrieve,
  class    = rd,
  type     = cf,
  stream   = enfo,
  expver   = iy2u,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = 2t,
  target   = "${hybrid_out}"
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
