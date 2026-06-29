#!/bin/bash
#SBATCH --job-name=10ff_topup_score
#SBATCH --output=scorecards_%j.out
#SBATCH --error=scorecards_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --chdir=

# ==============================================================================
# 10ff ensemble top-up + consolidation + scoring
#
# Step A: Extract the 33 missing dates (2026-02-26 to 2026-03-29) into _tmp/
#         using the top-up config (skip_scoring_if_exists: true).
# Step B: Consolidate all _tmp/ per-date parquets into final per-day parquets
#         (127 original + 33 new = 160 dates total).
# Step C: Score the full dataset using the main config.
# ==============================================================================

echo "=========================================="
echo "10ff Ensemble Top-Up + Consolidation + Score"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p $TMPDIR
export METVIEW_PYTHON_START_TIMEOUT=30

module load ecmwf-toolbox/new
module load python3

# ---------------------------------------------------------------------------
# Step A: Extract missing dates into _tmp/
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Step A: Extracting missing dates (2026-02-26 to 2026-03-29)"
echo "=========================================="
python3 -u run.py config_10ff_ens_local_p99obsclim_ifs4hybrid_topup.yaml
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "ERROR: Top-up extraction failed (exit $exit_code)"
    exit $exit_code
fi

# ---------------------------------------------------------------------------
# Step B: Consolidate all _tmp/ parquets into final per-day files
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step B: Consolidating _tmp/ into final parquets"
echo "=========================================="
python3 -u consolidate_10ff_tmp.py
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "ERROR: Consolidation failed (exit $exit_code)"
    exit $exit_code
fi

# ---------------------------------------------------------------------------
# Step C: Score full dataset with main config
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step C: Scoring full dataset"
echo "=========================================="
python3 -u run.py config_10ff_ens_local_p99obsclim_ifs4hybrid.yaml
exit_code=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

exit $exit_code
