#!/bin/bash
#SBATCH --job-name=compute_extreme_pcts
#SBATCH --output=compute_extreme_pcts_%j.out
#SBATCH --error=compute_extreme_pcts_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=

export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p $TMPDIR
module load ecmwf-toolbox/new
module load python3

source ./.venv/bin/activate

echo "Start: $(date)"
python3 -u compute_extreme_pcts.py
echo "End: $(date)"
