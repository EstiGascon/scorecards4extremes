#!/bin/bash
#SBATCH --job-name=mars_tp_det_rd
#SBATCH --output=mars_tp_det_rd_%j.out
#SBATCH --error=mars_tp_det_rd_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve total precipitation (deterministic) for rd experiments
# Experiments : iyr0, j3d0, j5zr, j6zg  (class=rd, type=fc, time=00)
# Period      : 2023-01-01 to 2026-06-30
# Steps       : 24, 48, ..., 240 h  (every 24 h, days 1–10)
#
# Note: param 228 is the ECMWF accumulated total precipitation (m).
#       The pipeline derives 24 h accumulations by differencing consecutive
#       steps when precipitation_accumulation_hours: 24 is set.
#       Step 0 is NOT stored in FDB for these experiments (tp=0 at init;
#       first 24 h accumulation = step 24 − 0 = step 24, so it is not needed).
#
# Output      : one GRIB file per experiment per day
#               /ec/vol/destine/continuous_evaluation/precip/forecast/raw/{expver}/
# ==============================================================================
# Usage (interactive):  bash retrieve_tp_det_rd.sh
# Usage (batch):        sbatch retrieve_tp_det_rd.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/precip/forecast/raw"

# Set this to the FDB config path provided by the experiment owner:
# export FDB5_CONFIG=/ec/res4/scratch/<owner>/fdb/config.yaml

EXPERIMENTS="j3d0 j5zr j6zg"

STEPS="24/48/72/96/120/144/168/192/216/240"

START_DATE="20230101"
END_DATE="20260630"

echo "=============================================="
echo "MARS TP DET (rd) retrieval"
echo "Experiments : $EXPERIMENTS"
echo "Period      : $START_DATE – $END_DATE"
echo "Steps       : $STEPS"
echo "Started     : $(date)"
echo "=============================================="

total_ok=0
total_skip=0
total_err=0

for EXP in $EXPERIMENTS; do

    OUT_DIR="${OUTPUT_BASE}/${EXP}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "────────────────────────────────────────"
    echo "Experiment: $EXP  →  $OUT_DIR"
    echo "────────────────────────────────────────"

    n_ok=0
    n_skip=0
    n_err=0

    current="$START_DATE"
    while [[ "$current" -le "$END_DATE" ]]; do

        outfile="${OUT_DIR}/tp_${EXP}_${current}.grib"

        if [[ -f "$outfile" ]]; then
            echo "[$EXP $current] already exists, skipping"
            (( n_skip++ ))
        else
            echo "[$EXP $current] retrieving..."
            reqfile=$(mktemp /tmp/mars_tp_det_${EXP}_XXXXXX.req)
            cat > "$reqfile" << MARSEOF
retrieve,
  class    = rd,
  type     = fc,
  expver   = ${EXP},
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = tp,
  database = fdb,
  target   = "${outfile}"
MARSEOF
            mars "$reqfile" 2>&1 | grep -E "retrieved|WARN|ERROR|No errors"
            exit_code=${PIPESTATUS[0]}
            rm -f "$reqfile"
            if [[ $exit_code -eq 0 ]]; then
                echo "[$EXP $current] OK"
                (( n_ok++ ))
            else
                echo "[$EXP $current] FAILED (exit $exit_code)"
                (( n_err++ ))
                rm -f "$outfile"
            fi
        fi

        current=$(date -d "$current + 1 day" +"%Y%m%d")
    done

    echo "[$EXP] done — OK: $n_ok  Skipped: $n_skip  Failed: $n_err"
    (( total_ok   += n_ok   ))
    (( total_skip += n_skip ))
    (( total_err  += n_err  ))

done

echo ""
echo "=============================================="
echo "All experiments done: $(date)"
echo "  OK      : $total_ok"
echo "  Skipped : $total_skip"
echo "  Failed  : $total_err"
echo "  Output  : $OUTPUT_BASE"
echo "=============================================="
