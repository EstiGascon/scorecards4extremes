#!/bin/bash
# Replot deterministic configs without re-extracting or re-scoring.
# Creates temporary configs with skip flags forced to true, submits, then removes temp files after submission.

set -e
cd "$(dirname "$0")"

CONFIGS=(
    "config_10ff_local_p98obsclim_aifs_ifs_single.yaml"
    "config_2t_local_p99obsclim_aifs_ifs_single.yaml"
    "config_2t_local_p1obsclim_aifs_ifs_single.yaml"
    "config_tp24_local_p99obsclim_destine50r1.yaml"
    "config_2t_local_p1obspool_destine50r1.yaml"
    "config_2t_local_p1obsclim_destine50r1.yaml"
)

source .venv/bin/activate

for cfg in "${CONFIGS[@]}"; do
    tmp="_replot_${cfg}"
    python3 - "$cfg" "$tmp" <<'EOF'
import sys, yaml
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
c['skip_extraction_if_exists'] = True
c['skip_scoring_if_exists'] = True
with open(sys.argv[2], 'w') as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True)
EOF
    jid=$(sbatch submit_job.sh "$tmp" | awk '{print $4}')
    echo "Submitted $cfg → job $jid (tmp: $tmp)"
done

echo ""
echo "Temp configs will be cleaned up after jobs finish."
echo "To remove them: rm _replot_config_*.yaml"
