#!/bin/bash
#SBATCH --job-name=mars_tp_ens_hybrid
#SBATCH --output=mars_tp_ens_hybrid_%j.out
#SBATCH --error=mars_tp_ens_hybrid_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve IFS-ENS and Hybrid-ENS total precipitation from MARS
# Period  : 2025-10-20 to 2026-03-31
# Steps   : 0, 24, 48, ..., 240 h  (every 24 h)
# Members : control (cf) + perturbed (pf, 1–50)
# Output  : one GRIB file per model per day, all steps and members combined
# ==============================================================================
# Usage (interactive):  bash retrieve_tp_ens_hybrid.sh
# Usage (batch):        sbatch retrieve_tp_ens_hybrid.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/precip/forecast_ENS/raw/"
IFS_DIR="$OUTPUT_BASE/ifs_ens4hybrid"
HYBRID_DIR="$OUTPUT_BASE/hybrid_ens"

mkdir -p "$IFS_DIR" "$HYBRID_DIR"

STEPS="0/24/48/72/96/120/144/168/192/216/240"

START_DATE="20251020"
END_DATE="20260331"

echo "=============================================="
echo "MARS TP ENS (hybrid) retrieval"
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
    ifs_out="$IFS_DIR/tp_ifs_ens_${current}.grib"

    if [[ -f "$ifs_out" ]]; then
        echo "[$current] IFS-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] IFS-ENS – retrieving..."
        reqfile=$(mktemp /tmp/mars_ifs_XXXXXX.req)
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
  param    = tp,
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
  param    = tp,
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
    hybrid_out="$HYBRID_DIR/tp_hybrid_ens_${current}.grib"

    if [[ -f "$hybrid_out" ]]; then
        echo "[$current] Hybrid-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] Hybrid-ENS – retrieving..."
        reqfile=$(mktemp /tmp/mars_hybrid_XXXXXX.req)
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
  param    = tp,
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
  param    = tp,
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
