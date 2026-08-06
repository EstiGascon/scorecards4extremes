#!/bin/bash
#SBATCH --job-name=mars_10uv_new_exps
#SBATCH --output=mars_10uv_new_exps_%j.out
#SBATCH --error=mars_10uv_new_exps_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ==============================================================================
# Retrieve 10 m U and V wind components (deterministic) for new rd experiments
# Experiments : j6uz, j78d, j78e, j7ba, j7bd, j7bc  (class=rd, type=fc, time=00)
# Period      : 2023-01-01 to 2023-06-30
# Steps       : 0, 6, 12, ..., 240 h  (every 6 h, days 0–10)
# Output      : /ec/vol/destine/continuous_evaluation/10wind_speed/forecast/raw/{expver}/
#               Files named: 10uv_{YYYYMMDD}.grib  (10u + 10v combined)
# ==============================================================================

OUTPUT_BASE="/ec/vol/destine/continuous_evaluation/10wind_speed/forecast/raw"

EXPERIMENTS="j6uz j78d j78e j7ba j7bd j7bc"

STEPS="0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"

START_DATE="20230101"
END_DATE="20230630"

echo "=============================================="
echo "MARS 10UV DET (new exps) retrieval"
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

        outfile="${OUT_DIR}/10uv_${current}.grib"

        if [[ -f "$outfile" ]]; then
            echo "[$EXP $current] already exists, skipping"
            (( n_skip++ ))
        else
            echo "[$EXP $current] retrieving..."
            reqfile=$(mktemp /tmp/mars_10uv_new_${EXP}_XXXXXX.req)
            cat > "$reqfile" << MARSEOF
retrieve,
  class    = rd,
  type     = fc,
  expver   = ${EXP},
  date     = ${current},
  time     = 00:00:00,
  step     = ${STEPS},
  levtype  = sfc,
  param    = 10u/10v,
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
