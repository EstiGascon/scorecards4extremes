#!/bin/bash
#SBATCH --job-name=mars_10ff_ens
#SBATCH --output=mars_10ff_ens_%j.out
#SBATCH --error=mars_10ff_ens_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve IFS-ENS and AIFS-ENS 10m wind components (10u, 10v) from MARS
# Period  : 2025-12-01 to 2026-02-28
# Steps   : 0, 6, 12, 18, 24, ..., 240 h  (every 6 h)
# Members : control (cf) + perturbed (pf, 1–50)
# Output  : separate GRIB files for 10u and 10v per model per day
#           Wind speed (10ff = sqrt(10u² + 10v²)) is computed at extraction time
# ==============================================================================
# Usage (interactive):  bash retrieve_10ff_ens.sh
# Usage (batch):        sbatch retrieve_10ff_ens.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/10ff/forecast_ENS/raw/"
IFS_DIR="$OUTPUT_BASE/ifs_ens"
AIFS_DIR="$OUTPUT_BASE/aifs_ens"

mkdir -p "$IFS_DIR" "$AIFS_DIR"

# 6-hourly steps from 0 to 240 h
STEPS="0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"

START_DATE="20251201"
END_DATE="20260228"

echo "=============================================="
echo "MARS 10U/10V ENS retrieval"
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

    # ----------------------------------------------------------------- AIFS-ENS
    aifs_u_out="$AIFS_DIR/10u_aifs_ens_${current}.grib"
    aifs_v_out="$AIFS_DIR/10v_aifs_ens_${current}.grib"

    if [[ -f "$aifs_u_out" && -f "$aifs_v_out" ]]; then
        echo "[$current] AIFS-ENS – already exists, skipping"
        (( n_skip++ ))
    else
        echo "[$current] AIFS-ENS – retrieving 10u..."
        reqfile=$(mktemp /tmp/mars_aifs_10u_XXXXXX.req)
        cat > "$reqfile" << MARSEOF
retrieve,
  class    = ai,
  type     = cf,
  stream   = enfo,
  expver   = 1,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = 10u,
  target   = "${aifs_u_out}"
retrieve,
  class    = ai,
  type     = pf,
  stream   = enfo,
  expver   = 1,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  number   = 1/to/50,
  levtype  = sfc,
  param    = 10u,
  target   = "${aifs_u_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        u_exit=${PIPESTATUS[0]}
        rm -f "$reqfile"

        echo "[$current] AIFS-ENS – retrieving 10v..."
        reqfile=$(mktemp /tmp/mars_aifs_10v_XXXXXX.req)
        cat > "$reqfile" << MARSEOF
retrieve,
  class    = ai,
  type     = cf,
  stream   = enfo,
  expver   = 1,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = 10v,
  target   = "${aifs_v_out}"
retrieve,
  class    = ai,
  type     = pf,
  stream   = enfo,
  expver   = 1,
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  number   = 1/to/50,
  levtype  = sfc,
  param    = 10v,
  target   = "${aifs_v_out}"
MARSEOF
        mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
        v_exit=${PIPESTATUS[0]}
        rm -f "$reqfile"

        if [[ $u_exit -eq 0 && $v_exit -eq 0 ]]; then
            echo "[$current] AIFS-ENS – OK (10u + 10v)"
            (( n_ok++ ))
        else
            echo "[$current] AIFS-ENS – FAILED (10u=$u_exit, 10v=$v_exit)"
            (( n_err++ ))
            [[ $u_exit -ne 0 ]] && rm -f "$aifs_u_out"
            [[ $v_exit -ne 0 ]] && rm -f "$aifs_v_out"
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
