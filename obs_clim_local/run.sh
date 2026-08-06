#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=obsclim

# ============================================================
# Total precipitation (tp) — 1-day accumulation, 65% min avail
# ============================================================
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 01 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_01 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 02 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_02 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 03 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_03 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 04 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_04 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 05 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_05 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 06 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_06 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 07 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_07 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 08 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_08 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 09 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_09 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 10 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_10 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 11 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_11 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 12 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_12 2>&1

# ============================================================
# 10m wind speed (10ff) — 1-day mean, 65% min avail
# ============================================================
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 01 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_01 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 02 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_02 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 03 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_03 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 04 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_04 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 05 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_05 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 06 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_06 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 07 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_07 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 08 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_08 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 09 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_09 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 10 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_10 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 11 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_11 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 10ff 0.65 2005 2024 12 /home/moeg/scorecards4extremes/obs_clim_local > out_10ff_12 2>&1

# ============================================================
# 2-metre temperature (2t) — 1-day mean, 65% min avail
# ============================================================
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 01 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_01 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 02 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_02 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 03 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_03 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 04 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_04 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 05 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_05 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 06 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_06 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 07 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_07 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 08 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_08 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 09 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_09 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 10 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_10 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 11 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_11 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py 2t 0.65 2005 2024 12 /home/moeg/scorecards4extremes/obs_clim_local > out_2t_12 2>&1

# ============================================================
# 2-metre maximum temperature (tmax) — 1-day max, 65% min avail
# ============================================================
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 01 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_01 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 02 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_02 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 03 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_03 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 04 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_04 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 05 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_05 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 06 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_06 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 07 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_07 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 08 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_08 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 09 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_09 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 10 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_10 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 11 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_11 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmax 0.65 2005 2024 12 /home/moeg/scorecards4extremes/obs_clim_local > out_tmax_12 2>&1

# ============================================================
# 2-metre minimum temperature (tmin) — 1-day min, 65% min avail
# ============================================================
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 01 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_01 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 02 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_02 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 03 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_03 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 04 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_04 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 05 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_05 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 06 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_06 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 07 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_07 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 08 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_08 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 09 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_09 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 10 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_10 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 11 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_11 2>&1
python3 /home/moeg/scorecards4extremes/obs_clim_local/obsclim.py tmin 0.65 2005 2024 12 /home/moeg/scorecards4extremes/obs_clim_local > out_tmin_12 2>&1



