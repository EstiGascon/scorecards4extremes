#!/bin/bash
#SBATCH --job-name=mars_2t_j5vo
#SBATCH --output=mars_2t_j5vo_%j.out
#SBATCH --error=mars_2t_j5vo_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve 2 m temperature (deterministic) from MARS/FDB for j5vo
# Experiments : j5vo  (class=rd, type=fc, time=00)
# Period      : 2023-01-01 to 2026-06-30
# Steps       : 0, 6, 12, ..., 240 h  (every 6 h, days 0–10)
# Output      : one GRIB file per experiment per day
#               /ec/vol/destine/continuous_evaluation/2mtemp/forecast/raw/{expver}/
# ==============================================================================
# Usage (interactive):  bash retrieve_2t_det_j5vo.sh
# Usage (batch):        sbatch retrieve_2t_det_j5vo.sh
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/2mtemp/forecast/raw"

EXPERIMENTS="j5vo"

STEPS="0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"

START_DATE="20230101"
END_DATE="20260630"

echo "=============================================="
echo "MARS 2T DET (j5vo) retrieval"
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

        outfile="${OUT_DIR}/2t_${EXP}_${current}.grib"

        if [[ -f "$outfile" ]]; then
            echo "[$EXP $current] already exists, skipping"
            (( n_skip++ ))
        else
            echo "[$EXP $current] retrieving..."
            reqfile=$(mktemp /tmp/mars_2t_det_${EXP}_XXXXXX.req)
            cat > "$reqfile" << MARSEOF
retrieve,
  class    = rd,
  type     = fc,
  expver   = ${EXP},
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = 2t,
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
