#!/bin/bash
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=obsclim_Italy

# ============================================================
# Italy whole-year tp climatology
# 24h precipitation, 5 years 2020–2024, 50% min availability
# Italy bbox: lat 36–47N, lon 6–19E
# Writes 12 identical monthly files for threshold.py compatibility
# ============================================================

CLIMDIR="/home/moeg/scorecards4extremes/obs_clim_local_Italy"

python3 ${CLIMDIR}/obsclim.py ${CLIMDIR} > ${CLIMDIR}/out_tp_allyear_50_with_p995_p999 2>&1

