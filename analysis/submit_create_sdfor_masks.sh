#!/bin/bash
#SBATCH --job-name=sdfor_o2560
#SBATCH --output=sdfor_o2560_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1

module load ecmwf-toolbox/new
module load python3

cd "$(dirname "$0")/.." 2>/dev/null || cd /path/to/scorecards4extremes
python3 -u analysis/create_sdfor_o2560_masks.py
