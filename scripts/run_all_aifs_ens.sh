#!/bin/bash
# Run all AIFS vs IFS ENS scoring and plotting pipelines
# Configs: 2t warm, 10ff, tp24
# (2t cold is run separately)

PYTHON=/usr/local/apps/python3/3.11.10-01/bin/python3
LOGDIR=/home/moeg/scorecards4extremes/results
cd /home/moeg/scorecards4extremes

echo "======================================"
echo "AIFS vs IFS ENS batch run"
echo "Started: $(date)"
echo "======================================"

for CONFIG in config_2t_ensemble.yaml config_10ff_ensemble.yaml config_tp24_ensemble.yaml; do
    echo ""
    echo "======================================"
    echo "Running: $CONFIG"
    echo "Started: $(date)"
    echo "======================================"
    $PYTHON -u run.py "$CONFIG"
    EXIT=$?
    echo ""
    echo "Finished $CONFIG with exit code $EXIT at $(date)"
done

echo ""
echo "======================================"
echo "All done: $(date)"
echo "======================================"
