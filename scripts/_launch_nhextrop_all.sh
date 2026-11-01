#!/bin/bash
# Launch all 10 NH-extratropics full-year experiments, fully self-contained
# (no cross-job dependencies). Deterministic = single sbatch each.
# Ensemble = chunked extraction + watcher (auto-merge + auto-score).
set -u
cd "$(dirname "$0")/.." || exit 1
TRACK="configs/_nhextrop_launch_jobids.txt"
: > "$TRACK"
N=10   # chunks per ensemble run

DET=(
  configs/deterministic/config_mn2t_method3_quaverextract_p1obsclim_ifsvsaifs_fullyear_nhextrop.yaml
  configs/deterministic/config_tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_nhextrop.yaml
  configs/deterministic/config_tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_regrid025_nhextrop.yaml
  configs/deterministic/config_10ff_method3_quaverextract_p98obsclim_ifsvsaifs_fullyear_allseasons_withcoastal_nhextrop.yaml
  configs/deterministic/config_10ff_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_allseasons_withcoastal_nhextrop.yaml
)
ENS=(
  configs/ensemble/config_mn2t_ens_quaver_p1obsclim_aifsvsifs_fullyear_nhextrop.yaml
  configs/ensemble/config_tp24_ens_local_p99obsclim_aifsvsifs_native_fullyear_nhextrop.yaml
  configs/ensemble/config_tp24_ens_local_p99obsclim_aifsvsifs_fullyear_nhextrop.yaml
  configs/ensemble/config_10ff_ens_quaver_p98obsclim_aifsvsifs_fullyear_allseasons_withcoastal_nhextrop.yaml
  configs/ensemble/config_10ff_ens_quaver_p99obsclim_aifsvsifs_fullyear_allseasons_withcoastal_nhextrop.yaml
)

echo "=========== DETERMINISTIC (single sbatch each) ==========="
for cfg in "${DET[@]}"; do
  out=$(sbatch scripts/submit_job_quaver.sh "$cfg")
  jid=$(echo "$out" | awk '{print $NF}')
  echo "DET  $jid  $cfg" | tee -a "$TRACK"
done

echo "=========== ENSEMBLE (chunk split + watcher) ==========="
for cfg in "${ENS[@]}"; do
  echo "--- split $cfg (N=$N) ---"
  python scripts/chunk_extraction.py split "$cfg" --n-chunks "$N" --submit
  wout=$(sbatch scripts/watch_and_score.sh "$cfg" "$N")
  wjid=$(echo "$wout" | awk '{print $NF}')
  echo "ENS-WATCHER  $wjid  $cfg (N=$N)" | tee -a "$TRACK"
done

echo
echo "=========== TRACKING FILE: $TRACK ==========="
cat "$TRACK"
