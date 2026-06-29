#!/bin/bash
#SBATCH --job-name=mars_10ff_ens
#SBATCH --output=mars_10ff_ens_%j.out
#SBATCH --error=mars_10ff_ens_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve IFS-ENS 10m wind components (10u, 10v) from MARS
# Period  : 2026-02-15 to 2026-03-31
# Steps   : 0, 12, 24, ..., 240 h  (every 12 h)
# Members : control (cf) + perturbed (pf, 1–50)
# Output  : separate GRIB files for 10u and 10v per day
#           Wind speed (10ff = sqrt(10u² + 10v²)) is computed at extraction time
# ==============================================================================
# Usage (interactive):  bash retrieve_10ff_ens_hybrid.sh
# Usage (batch):        sbatch retrieve_10ff_ens_hybrid.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/10ff/forecast_ENS/raw/"
IFS_DIR="$OUTPUT_BASE/ifs_ens4hybrid"

mkdir -p "$IFS_DIR"

# 12-hourly steps from 0 to 240 h
STEPS="0/12/24/36/48/60/72/84/96/108/120/132/144/156/168/180/192/204/216/228/240"

START_DATE="20260215"
END_DATE="20260331"

echo "=============================================="
echo "MARS 10U/10V IFS-ENS retrieval"
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
    ifs_u_out="$IFS_DIR/10u_ifs_ens_${current}.grib"
    ifs_v_out="$IFS_DIR/10v_ifs_ens_${current}.grib"

    if [[ -f "$ifs_u_out" && -f "$ifs_v_out" ]]; then
        echo "[$current] IFS-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] IFS-ENS – retrieving 10u..."
        reqfile=$(mktemp /tmp/mars_ifs_10u_XXXXXX.req)
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
  param    = 10u,
  target   = "${ifs_u_out}"
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
  param    = 10u,
  target   = "${ifs_u_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        u_exit=${PIPESTATUS[0]}
        rm -f "$reqfile"

        echo "[$current] IFS-ENS – retrieving 10v..."
        reqfile=$(mktemp /tmp/mars_ifs_10v_XXXXXX.req)
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
  param    = 10v,
  target   = "${ifs_v_out}"
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
  param    = 10v,
  target   = "${ifs_v_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        v_exit=${PIPESTATUS[0]}
        rm -f "$reqfile"

        if [[ $u_exit -eq 0 && $v_exit -eq 0 ]]; then
            echo "[$current] IFS-ENS – OK (10u + 10v)"
            (( n_ok++ ))
        else
            echo "[$current] IFS-ENS – FAILED (10u=$u_exit, 10v=$v_exit)"
            (( n_err++ ))
            [[ $u_exit -ne 0 ]] && rm -f "$ifs_u_out"
            [[ $v_exit -ne 0 ]] && rm -f "$ifs_v_out"
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
