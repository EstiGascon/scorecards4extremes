#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=obsclim_mx10ff
#SBATCH --output=/home/moeg/scorecards4extremes/obs_clim_local/obsclim_mx10ff_%j.out
#SBATCH --error=/home/moeg/scorecards4extremes/obs_clim_local/obsclim_mx10ff_%j.err

# Observation climatology of the DAILY MAXIMUM 10 m wind speed (mx10ff), built from
# the daily max of the 6-hourly STVL 10ff observations (consistent with how mx10ff
# forecasts are scored). 1-day window, 2005-2024, 65% min availability, all months.
# Produces clim_mx10ff_1_MM_20years_2005_2024_65 / climmean_mx10ff_*.

set -u
DIR=/home/moeg/scorecards4extremes/obs_clim_local
module load ecmwf-toolbox
export METVIEW_PYTHON_START_TIMEOUT=60

fail=0
for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
    python3 "$DIR/obsclim.py" mx10ff 0.65 2005 2024 $mm "$DIR" > "$DIR/out_mx10ff_$mm" 2>&1
    f="$DIR/clim_mx10ff_1_${mm}_20years_2005_2024_65"
    if [[ -s "$f" ]]; then echo "month $mm OK"; else echo "month $mm FAILED (see out_mx10ff_$mm)"; fail=1; fi
done
exit $fail
