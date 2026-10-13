#!/bin/bash
#SBATCH --job-name=sync_scorecards
#SBATCH --output=sync_scorecards_%j.out
#SBATCH --error=sync_scorecards_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/moeg/scorecards4extremes

# Periodically mirror the canonical full-year experiment result folders into
# $SCRATCH/scorecards so they are all present by the deadline without manual
# action. Runs one rsync pass every 30 min; re-submits itself before the 48h
# wall limit; stops after DEADLINE. rsync is idempotent, so pending experiments
# are picked up automatically as their scoring jobs finish.

set -u
DEST="${SCRATCH:?SCRATCH unset}/scorecards"
DEADLINE="2026-09-29"                 # stop re-submitting after this date
INTERVAL=1800                          # seconds between passes
MAX_RUNTIME=$(( 46 * 3600 ))           # resubmit before the 48h wall limit
SELF="scripts/sync_scorecards.sh"

# Canonical obs-climatology full-year set: 4 vars x ens/det x Europe/NH-extratropics.
RUNS=(
  10ff_method3_quaverextract_p98obsclim_ifsvsaifs_fullyear_allseasons_withcoastal
  10ff_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_allseasons_withcoastal
  mn2t_method3_quaverextract_p1obsclim_ifsvsaifs_fullyear
  mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear
  tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear
  tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_regrid025
  mn2t_method3_quaverextract_p1obsclim_ifsvsaifs_fullyear_nhextrop
  mx2t_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_nhextrop
  tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_nhextrop
  tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_regrid025_nhextrop
  10ff_method3_quaverextract_p98obsclim_ifsvsaifs_fullyear_allseasons_withcoastal_nhextrop
  10ff_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_allseasons_withcoastal_nhextrop
  mn2t_ens_quaver_p1obsclim_aifsvsifs_fullyear
  mx2t_ens_quaver_p99obsclim_aifsvsifs_fullyear
  tp24_ens_local_p99obsclim_aifsvsifs_fullyear
  tp24_ens_local_p99obsclim_aifsvsifs_native_fullyear
  10ff_ens_quaver_p98obsclim_aifsvsifs_fullyear_allseasons_withcoastal
  10ff_ens_quaver_p99obsclim_aifsvsifs_fullyear_allseasons_withcoastal
  mn2t_ens_quaver_p1obsclim_aifsvsifs_fullyear_nhextrop
  mx2t_ens_quaver_p99obsclim_aifsvsifs_fullyear_nhextrop
  tp24_ens_local_p99obsclim_aifsvsifs_fullyear_nhextrop
  tp24_ens_local_p99obsclim_aifsvsifs_native_fullyear_nhextrop
  10ff_ens_quaver_p98obsclim_aifsvsifs_fullyear_allseasons_withcoastal_nhextrop
  10ff_ens_quaver_p99obsclim_aifsvsifs_fullyear_allseasons_withcoastal_nhextrop
)

mkdir -p "$DEST"

one_pass() {
  local copied=0 pending=0
  for r in "${RUNS[@]}"; do
    if [ -d "results/$r" ] && ls "results/$r"/*.png >/dev/null 2>&1; then
      rsync -a "results/$r" "$DEST/"
      copied=$((copied+1))
    else
      pending=$((pending+1))
    fi
  done
  echo "$(date '+%F %T')  pass done: $copied present, $pending pending  -> $DEST"
}

START=$(date +%s)
while true; do
  one_pass
  # Re-submit successor before hitting the wall limit, unless past the deadline.
  if [ $(( $(date +%s) - START )) -ge "$MAX_RUNTIME" ]; then
    if [ "$(date +%F)" \< "$DEADLINE" ]; then
      echo "$(date '+%F %T')  approaching wall limit, resubmitting $SELF"
      sbatch "$SELF"
    else
      echo "$(date '+%F %T')  past deadline $DEADLINE, not resubmitting"
    fi
    exit 0
  fi
  # Stop entirely once the deadline has passed.
  if [ "$(date +%F)" \> "$DEADLINE" ]; then
    echo "$(date '+%F %T')  past deadline, final pass complete"
    exit 0
  fi
  sleep "$INTERVAL"
done
