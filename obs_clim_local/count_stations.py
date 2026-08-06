"""
Diagnostic: compare how many stations are available for 2t (mean), tmax and tmin
straight from the STVL database, before and after the 65% availability filter.

Usage:
    python3 count_stations.py <mm> [fyear] [lyear]
e.g.
    python3 count_stations.py 07 2005 2024
"""
import sys
import numpy
import xarray
from datetime import datetime, timedelta

from vtb.media import stvl
from vtb.tools import aligned_geodfs

mm = sys.argv[1]
fyear = int(sys.argv[2]) if len(sys.argv) > 2 else 2005
lyear = int(sys.argv[3]) if len(sys.argv) > 3 else 2024
crit = 0.65

# parameter config (mirrors obsclim.py)
PARAMS = {
    '2t':   {'obs_period': 0,  'period': 6},
    'tmax': {'obs_period': 24, 'period': 24},
    'tmin': {'obs_period': 24, 'period': 24},
}
vw = 1  # 1-day window (the only meaningful one for tmax/tmin)


def listOfDatesPerMonth(month, y1, y2, maxlength):
    out = []
    for year in range(y1, y2 + 1):
        mday = datetime(year, month, 1, 0)
        d = mday - timedelta(days=round(maxlength / 2))
        start = d - timedelta(days=round(maxlength / 2))
        end = d + timedelta(days=round(maxlength / 2))
        while start.month == month or end.month == month:
            out.append(d)
            d += timedelta(days=1)
            start = d - timedelta(days=round(maxlength / 2))
            end = d + timedelta(days=round(maxlength / 2))
    return out


def listOfDatesInPeriod(date, period, nn):
    out = []
    dx = date
    for j in range(nn * (24 // period)):
        out.append(dx + timedelta(hours=period))
        dx += timedelta(hours=period)
    return out


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def arr_agg(xarr, listOfTimes, how):
    try:
        sel = xarr.sel(date=listOfTimes)
        if how == 'mean':
            return sel.mean("date", skipna=False).values
        if how == 'min':
            return sel.min("date", skipna=False).values
        if how == 'max':
            return sel.max("date", skipna=False).values
    except KeyError:
        return numpy.full_like(xarr.isel(date=0).values, numpy.nan, dtype=float)


def analyse(param):
    cfg = PARAMS[param]
    obs_period = cfg['obs_period']
    period = cfg['period']
    maxlength = vw

    # dates to retrieve
    listOfListsOfDates = []
    for d in listOfDatesPerMonth(int(mm), fyear, lyear, maxlength):
        if (d + timedelta(days=round(maxlength / 2))).month == int(mm):
            listOfListsOfDates.append(listOfDatesInPeriod(d, period, maxlength))
    datesToRetrieve = sorted(set([d for dd in listOfListsOfDates for d in dd]))

    observations = []
    for ldates in list(chunks(datesToRetrieve, 100)):
        for obs in stvl.retrieve_to_geodfs(
                table='observation',
                sources=['synop', 'hdobs'],
                parameter=param, period=obs_period * 3600,
                reference_datetimes=[d for d in ldates]):
            obs.drop_duplicates()
            observations.append(obs)

    aligned_observations = aligned_geodfs(observations, max_cluster_size=0.1)
    stnids_array = aligned_observations[0]['stnid'].to_list()
    n_network = len(stnids_array)  # stations in the aligned network

    obs_array = numpy.array([geo["value_0"] for geo in aligned_observations])
    dates_array = numpy.array([geo.geodf_header['date'] for geo in aligned_observations])
    obs_xarray = xarray.DataArray(obs_array, (('date', dates_array), ('stnid', stnids_array)))

    # build daily-aggregated climate sample
    how = {'2t': 'mean', 'tmax': 'max', 'tmin': 'min'}[param]
    clim = []
    for dd in listOfListsOfDates:
        clim.append(arr_agg(obs_xarray, dd, how))
    clim_array = xarray.DataArray(clim, dims=('date', 'stnid'))

    population = clim_array.notnull().sum('date')
    n_any = int((population > 0).sum())                       # >=1 valid daily value
    n_pass = int((population > (crit * len(clim))).sum())     # passes 65% availability
    return n_network, n_any, n_pass, len(clim)


print(f"Month {mm}, period {fyear}-{lyear}, window {vw} day, crit {int(crit*100)}%")
print(f"{'param':6} {'network':>9} {'>=1 obs':>9} {'>=65%':>9} {'sampleN':>9}")
results = {}
for p in ('2t', 'tmax', 'tmin'):
    n_network, n_any, n_pass, sampleN = analyse(p)
    results[p] = (n_network, n_any, n_pass)
    print(f"{p:6} {n_network:>9} {n_any:>9} {n_pass:>9} {sampleN:>9}")

base = results['2t'][2]
print()
print("Usable (>=65%) relative to 2t mean:")
for p in ('tmax', 'tmin'):
    print(f"  {p}: {results[p][2]} ({100*results[p][2]/base:.0f}% of 2t)")
