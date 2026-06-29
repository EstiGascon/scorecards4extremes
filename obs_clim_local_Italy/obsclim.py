"""Observation climatology for Italy — 24h total precipitation (tp).

Differences from the generic obs_clim_local/obsclim.py:
  - Only tp, only 1-day window.
  - Stations filtered to Italy bounding box (lat 36–47 N, lon 6–19 E).
  - 10-year period: 2016–2025.
  - Minimum data availability: 75%.
  - Whole-year climatology: all daily tp values (all months pooled together)
    are used to compute percentiles — no seasonal distinction.
    Twelve identical GEO NCOLS output files are written (one per calendar month)
    so that the existing threshold.py month-based lookup works without changes.

Usage (called from run.sh):
    python3 obsclim.py <climdir>
"""

import numpy
import xarray
import metview
import sys
import time
from datetime import datetime, timedelta

from vtb.media import stvl
from vtb.tools import aligned_geodfs

start_time = time.time()

if len(sys.argv) < 2:
    print("Usage: python3 obsclim.py <climdir>")
    sys.exit(1)

climdir = sys.argv[1]

# ── Fixed settings ────────────────────────────────────────────────────────────
param      = 'tp'
pname      = 'total precipitation'
obs_period = 24          # hours between tp observations
period     = 24          # retrieval sampling period (hours)
vw         = 1           # verification window in days (24h accumulation)
fyear      = 2020
lyear      = 2024
crit       = 0.50        # minimum data availability fraction (50%)
n_years    = lyear - fyear + 1

# Italy bounding box
LAT_MIN, LAT_MAX =  36.0,  47.0
LON_MIN, LON_MAX =   6.0,  19.0


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def arr_sum(xarr, listOfTimes):
    try:
        return xarr.sel(date=listOfTimes).sum('date', skipna=False).values
    except KeyError:
        return numpy.full_like(xarr.isel(date=0).values, numpy.nan, dtype=float)


# ── Build list of ALL daily dates in the climatology period ──────────────────
print(f"Building date list for {fyear}–{lyear} (all months)...")
datesToRetrieve = []
d   = datetime(fyear, 1, 1, 0) + timedelta(hours=obs_period)   # first valid time = 24h after start
end = datetime(lyear, 12, 31, 0) + timedelta(hours=obs_period)
while d <= end:
    datesToRetrieve.append(d)
    d += timedelta(hours=period)

print(f"  Total dates to retrieve: {len(datesToRetrieve)}")

# ── Retrieve observations from STVL in chunks of 100 ────────────────────────
print("Retrieving OBS from STVL database (all months)...")
observations = []
for ldates in list(chunks(datesToRetrieve, 100)):
    for obs in stvl.retrieve_to_geodfs(
            table='observation',
            sources=['synop', 'hdobs'],
            parameter=param,
            period=obs_period * 3600,
            reference_datetimes=ldates,
    ):
        obs.drop_duplicates()
        observations.append(obs)

print(f"  Retrieved {len(observations)} geopoint frames.")

# ── Align all GeoDataFrames to a common station set ──────────────────────────
print("Aligning GeoDataFrames...")
aligned_observations = aligned_geodfs(observations, max_cluster_size=0.1)

lats_all = aligned_observations[0]['latitude'].to_numpy().astype(float)
lons_all = aligned_observations[0]['longitude'].to_numpy().astype(float)
all_elevations    = numpy.array([o['elevation'] for o in aligned_observations])
first_valid_index = numpy.isnan(all_elevations).argmin(axis=0)
heights_all       = [all_elevations[i, j] for j, i in enumerate(first_valid_index)]
stnids_all        = aligned_observations[0]['stnid'].to_list()

temp = time.time() - start_time
print(f"Time after aligning: {int(temp//3600):d}h {int((temp%3600)//60):d}m {int(temp%60):d}s")

# ── Filter to Italy bounding box ─────────────────────────────────────────────
print(f"Filtering to Italy bbox lat=[{LAT_MIN},{LAT_MAX}] lon=[{LON_MIN},{LON_MAX}]...")
italy_mask = (
    (lats_all >= LAT_MIN) & (lats_all <= LAT_MAX) &
    (lons_all >= LON_MIN) & (lons_all <= LON_MAX)
)
italy_idx = numpy.where(italy_mask)[0].tolist()

lats    = lats_all[italy_idx]
lons    = lons_all[italy_idx]
heights = [heights_all[i] for i in italy_idx]
stnids  = [stnids_all[i]  for i in italy_idx]
print(f"  {len(stnids)} stations inside Italy bbox (out of {len(stnids_all)} total).")

# ── Build xarray for Italy stations only ─────────────────────────────────────
obs_array   = numpy.array([geo["value_0"][italy_idx] for geo in aligned_observations])
dates_array = numpy.array([geo.geodf_header['date'] for geo in aligned_observations])
obs_xarray  = xarray.DataArray(obs_array, (('date', dates_array), ('stnid', stnids)))

# ── Compute whole-year 1-day tp climatology ──────────────────────────────────
print("Computing whole-year 1-day tp climatology (pooling all months)...")
clim = []
for dt in datesToRetrieve:
    clim.append(arr_sum(obs_xarray, [dt]))

clim_array        = xarray.DataArray(clim, dims=('date', 'stnid'))
population        = clim_array.notnull().sum('date')
where_min_reached = population > (crit * len(clim))
clim_masked       = clim_array.where(where_min_reached)

q_int   = [q / 100 for q in range(0, 101)]
q_extra = [0.995, 0.999]           # 99.5th and 99.9th percentiles
q_all   = q_int + q_extra
perc = clim_masked.quantile(
    q_all, dim='date', interpolation='nearest'
)
mean = clim_masked.mean('date')

quan = {'q' + str(q): numpy.round(perc.isel(quantile=q).values, 1)
        for q in range(0, 101)}
quan['q99p5'] = numpy.round(perc.isel(quantile=101).values, 1)   # 99.5th pct
quan['q99p9'] = numpy.round(perc.isel(quantile=102).values, 1)   # 99.9th pct

print(f"  Stations with >= {int(crit*100)}% availability: "
      f"{int(where_min_reached.sum().values)} / {len(stnids)}")

# ── Write 12 identical monthly files (for threshold.py compatibility) ────────
# threshold.py looks up the clim file by month-of-valid-date, so we write the
# same whole-year percentile data under each of the 12 monthly filenames.
print("Writing output files (12 identical monthly files)...")
for mm in range(1, 13):
    mm_str   = f"{mm:02d}"
    base     = (f"{climdir}/clim_{param}_{vw}_{mm_str}_{n_years}years"
                f"_{fyear}_{lyear}_{int(crit * 100)}")
    basem    = (f"{climdir}/climmean_{param}_{vw}_{mm_str}_{n_years}years"
                f"_{fyear}_{lyear}_{int(crit * 100)}")
    date_val = int(f"{lyear}{mm_str}01")

    # Percentile file
    geo_perc = metview.create_geo(
        type='ncols',
        latitudes=lats,
        longitudes=lons,
        date=date_val,
        height=heights,
        stnids=stnids,
        **quan,
    )
    geo_perc = metview.remove_missing_values(geo_perc)
    geo_perc = metview.set_metadata(geo_perc, {
        'Parameter ':    f' Annual climatology for {pname} (Italy)',
        'Period ':       f' All months, {fyear}–{lyear}',
        'Time window ':  f' {vw} day(s)',
    })
    metview.write(base, geo_perc)

    # Mean file
    geo_mean = metview.create_geo(
        type='ncols',
        latitudes=lats,
        longitudes=lons,
        date=date_val,
        height=heights,
        stnids=stnids,
        values=numpy.round(mean.values, 1),
    )
    geo_mean = metview.remove_missing_values(geo_mean)
    geo_mean = metview.set_metadata(geo_mean, {
        'Parameter ':    f' Annual mean climatology for {pname} (Italy)',
        'Period ':       f' All months, {fyear}–{lyear}',
        'Time window ':  f' {vw} day(s)',
    })
    metview.write(basem, geo_mean)

    print(f"  Written: {base}")

temp = time.time() - start_time
print(f"\nProgram finished. Total time: "
      f"{int(temp//3600):d}h {int((temp%3600)//60):d}m {int(temp%60):d}s")


