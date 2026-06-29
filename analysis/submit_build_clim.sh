#!/bin/bash
#SBATCH --job-name=build_tp24_clim
#SBATCH --output=/ec/res4/scratch/$USER/obs_climatology_new/build_%j.out
#SBATCH --error=/ec/res4/scratch/$USER/obs_climatology_new/build_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=

echo "=========================================="
echo "tp24 Climatology Build"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

module load ecmwf-toolbox/new
module load python3

python3 analysis/build_tp24_climatology.py

echo ""
echo "End time: $(date)"
echo "=========================================="
