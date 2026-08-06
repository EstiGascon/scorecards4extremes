#!/bin/bash
# ==============================================================================
# Check MARS availability of daily-max 2m temperature params (mx2t / mx2t6 /
# mx2t24) for AIFS-single, AIFS-ENS and DestinE (expver=iekm).
# Uses the lightweight `mars list` verb (metadata only, no data download).
# ==============================================================================
set -u

DATE="20260301"          # a date inside all three experiments' periods
TIME="00:00:00"
STEP="24"                # single step; enough to prove existence
OUTDIR="/home/moeg/scorecards4extremes/mx2t_check"
mkdir -p "$OUTDIR"

run_list () {
    local label="$1"; shift
    local param="$1"; shift
    local keys="$1"     # extra identity keys, comma-terminated per line
    local req; req=$(mktemp /tmp/mars_mx2t_XXXXXX.req)
    {
        echo "list,"
        echo "$keys"
        echo "  date    = ${DATE},"
        echo "  time    = ${TIME},"
        echo "  step    = ${STEP},"
        echo "  levtype = sfc,"
        echo "  param   = ${param},"
        echo "  output  = cost"
    } > "$req"
    echo "==================================================================="
    echo ">>> ${label} | param=${param}"
    echo "-------------------------------------------------------------------"
    cat "$req"
    echo "-------------------------------------------------------------------"
    mars "$req" 2>&1 | grep -Ei "grib|field|number of|no data|not found|ERROR|WARN|bytes|cost" \
        || echo "   (no matching summary lines — see nothing = likely no data)"
    rm -f "$req"
    echo
}

# ---- AIFS-single (deterministic) : class=ai, stream=oper, type=fc, model=aifs-single
AIFS_SINGLE_KEYS="  class   = ai,
  type    = fc,
  stream  = oper,
  expver  = 0001,
  model   = aifs-single,"

# ---- AIFS-ENS (control member) : class=ai, stream=enfo, type=cf, model=aifs-ens
AIFS_ENS_KEYS="  class   = ai,
  type    = cf,
  stream  = enfo,
  expver  = 0001,
  model   = aifs-ens,"

# ---- DestinE iekm : class=rd, stream=oper, type=fc, expver=iekm
IEKM_KEYS="  class   = rd,
  type    = fc,
  stream  = oper,
  expver  = iekm,"

for param in mx2t mx2t6 mx2t24; do
    run_list "AIFS-single" "$param" "$AIFS_SINGLE_KEYS"
    run_list "AIFS-ENS"    "$param" "$AIFS_ENS_KEYS"
    run_list "DestinE-iekm" "$param" "$IEKM_KEYS"
done

echo "Done."
