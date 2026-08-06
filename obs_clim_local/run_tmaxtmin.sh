#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=obsclim_tmaxtmin
#SBATCH --output=obsclim_tmaxtmin_%j.out
#SBATCH --error=obsclim_tmaxtmin_%j.err

# ============================================================
# (Re)generate the OBSERVATION climatology for daily 2m maximum
# (tmax) and 2m minimum (tmin) temperature — all 12 months.
# 1-day window, 20-year window 2005-2024, 65% min availability.
# Overwrites existing clim_tmax_* / clim_tmin_* / climmean_* files.
# ============================================================

DIR=/home/moeg/scorecards4extremes/obs_clim_local

# ---- 2-metre maximum temperature (tmax) ----
for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
    python3 "$DIR/obsclim.py" tmax 0.65 2005 2024 $mm "$DIR" > "$DIR/out_tmax_$mm" 2>&1
done

# ---- 2-metre minimum temperature (tmin) ----
for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
    python3 "$DIR/obsclim.py" tmin 0.65 2005 2024 $mm "$DIR" > "$DIR/out_tmin_$mm" 2>&1
done

echo "tmax/tmin obs climatology generation finished."
