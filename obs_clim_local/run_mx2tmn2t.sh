#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=obsclim_mx2tmn2t
#SBATCH --output=obsclim_mx2tmn2t_%j.out
#SBATCH --error=obsclim_mx2tmn2t_%j.err

# ============================================================
# (Re)generate the OBSERVATION climatology for daily 2m maximum
# (mx2t) and 2m minimum (mn2t) temperature — all 12 months.
#
# IMPORTANT: unlike the existing clim_tmax_*/clim_tmin_* (which use the
# STVL station-reported daily max/min), these are built from the DAILY
# MAX/MIN of the 6-hourly STVL 2t observations, so they are fully
# consistent with how mx2t/mn2t forecasts are scored in this pipeline
# (daily max/min of 6-hourly 2t).
#
# 1-day window, 20-year window 2005-2024, 65% min availability.
# Produces clim_mx2t_* / clim_mn2t_* / climmean_mx2t_* / climmean_mn2t_*.
# ============================================================

DIR=/home/moeg/scorecards4extremes/obs_clim_local

# ---- 2-metre daily maximum temperature from 6-hourly 2t (mx2t) ----
for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
    python3 "$DIR/obsclim.py" mx2t 0.65 2005 2024 $mm "$DIR" > "$DIR/out_mx2t_$mm" 2>&1
done

# ---- 2-metre daily minimum temperature from 6-hourly 2t (mn2t) ----
for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
    python3 "$DIR/obsclim.py" mn2t 0.65 2005 2024 $mm "$DIR" > "$DIR/out_mn2t_$mm" 2>&1
done

echo "mx2t/mn2t obs climatology generation finished."
