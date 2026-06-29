#!/bin/bash
#SBATCH --job-name=extract_tp24_obs
#SBATCH --output=/ec/res4/scratch/$USER/obs_climatology_new/extract_%j.out
#SBATCH --error=/ec/res4/scratch/$USER/obs_climatology_new/extract_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/path/to/scorecards4extremes

# ==============================================================================
# Extract tp24 observations from STVL (1990-2025)
# ==============================================================================
# Usage:
#   sbatch submit_extraction.sh              # Full extraction
#   sbatch submit_extraction.sh --test       # Test with one month
# ==============================================================================

echo "=========================================="
echo "tp24 Observation Extraction from STVL"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

module load ecmwf-toolbox/new
module load python3

mkdir -p /ec/res4/scratch/$USER/obs_climatology_new/raw

python3 scripts/extract_tp24_obs.py "$@"

echo ""
echo "End time: $(date)"
echo "=========================================="
