#!/bin/bash
#SBATCH --job-name=clim_compare_plot
#SBATCH --output=/ec/res4/scratch/$USER/obs_climatology_new/clim_compare_plot_%j.log
#SBATCH --error=/ec/res4/scratch/$USER/obs_climatology_new/clim_compare_plot_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module load ecmwf-toolbox/new
module load python3

cd "$(dirname "$0")/.." 2>/dev/null || cd /path/to/scorecards4extremes
python3 plot_clim_station_comparison.py
