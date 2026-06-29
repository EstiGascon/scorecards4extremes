import numpy
import xarray
import pandas
import sys
import calendar
import metview
import time
import json

#from ecmp.media import stvl
#from ecmp.tools import aligned_geodfs
from vtb.media import stvl
from vtb.tools import aligned_geodfs
#from ecmp.BasePy.Memory import Memory
from datetime import datetime, timedelta

start_time = time.time()

param = sys.argv[1]
critList = json.loads(sys.argv[2])
fyear = sys.argv[3]
lyear = sys.argv[4]
mm = sys.argv[5]
climdir = sys.argv[6]


def listOfDatesPerMonth(month,y1,y2,maxlength):
    # month -> month of interest
    # y1 -> first year of the climate window
    # y2 -> last year of the climate window
    # maxlength -> length in days of the longest time window (this is used
    # to retrieve all the observations at once) 
    out = []
    for year in range(y1,y2+1):
        mday = datetime(year,month,1,0)
        d = mday - timedelta(days=round(maxlength/2))
        start = d - timedelta(days=round(maxlength/2))
        end = d + timedelta(days=round(maxlength/2))
        while start.month == month or end.month == month:
            out.append(d)
            d += timedelta(days=1)
            start = d - timedelta(days=round(maxlength/2))
            end = d + timedelta(days=round(maxlength/2))
    return(out)

def listOfDatesInPeriod(date,period,nn):
    out = []
    dx = date
    for j in range(nn*(24//period)):
        out.append(dx+timedelta(hours=period))
        dx += timedelta(hours=period)
    return out

def chunks(l,n):
    # Yield successive n-sized chunks from l. 
    #looping till length l
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def arr_mean(xarr,listOfTimes):
    '''
    Computes the mean of xarray for listOfTimes period
    It specifically handles the situation when there is no data for a given time/times by creating an array of NaN
    :param xarr: xarray
    :param listOfTimes: list
    :return: mean of xarr for the period of time given by listofTimes
    '''
    try:
        array = xarr.sel(date=listOfTimes).mean("date",skipna=False).values
    except KeyError:
        array = numpy.full_like(xarr.isel(date=0).values,numpy.nan,dtype=float)

    return array


def arr_sum(xarr, listOfTimes):
    try:
        array = xarr.sel(date=listOfTimes).sum("date", skipna=False).values
    except KeyError:
        array = numpy.full_like(xarr.isel(date=0).values, numpy.nan, dtype=float)

    return array


def arr_min(xarr, listOfTimes):
    try:
        array = xarr.sel(date=listOfTimes).min("date", skipna=False).values
    except KeyError:
        array = numpy.full_like(xarr.isel(date=0).values, numpy.nan, dtype=float)

    return array


def arr_max(xarr, listOfTimes):
    try:
        array = xarr.sel(date=listOfTimes).max("date", skipna=False).values
    except KeyError:
        array = numpy.full_like(xarr.isel(date=0).values, numpy.nan, dtype=float)

    return array

if param == 'tp':
    pname = 'total precipitation'
    obs_period = 24 # hours
    # verifWindowList = [1,3,5,7,10,14,15] # days
    verifWindowList = [1,3,5,7,10,14] # days
    period = 24
if param == '2t':
    pname = '2-metre mean temperature'
    obs_period = 0 # hours
    verifWindowList = [1,3,5,7,10,14,15] # days
    period = 6
if param == 'tmax':
    pname = '2-metre maximum temperature'
    obs_period = 24 # hours
    verifWindowList = [1] # days
    period = 24
if param == 'tmin':
    pname = '2-metre minimum temperature'
    obs_period = 24 # hours
    verifWindowList = [1] # days
    period = 24
if param == '10ff':
    pname = 'mean 10-metre wind speed'
    obs_period = 0 # hours
    verifWindowList = [1,3,5,7,10,14,15] # days
    period = 6

maxlength = max(verifWindowList)    
# list of the dates to be retrieved
listOfListsOfDates = []
for d in listOfDatesPerMonth(int(mm),int(fyear),int(lyear),maxlength):
    if (d+timedelta(days=round(maxlength/2))).month == int(mm):
        listOfListsOfDates.append(listOfDatesInPeriod(d,period,maxlength))
    
datesToRetrieve = sorted(set([d for dd in listOfListsOfDates for d in dd]))

# print('datesToRetrieve = ', datesToRetrieve)

# observations from STVL as pandas.GeoDataFrames ("GeoDataFrames")
print("Retrieving OBS from STVL database.")
observations = []
for ldates in list(chunks(datesToRetrieve,100)):
    for obs in stvl.retrieve_to_geodfs(
                    table = 'observation',
                    sources = ['synop','hdobs'],
                    parameter=param,period=obs_period*3600,
                    reference_datetimes=[d for d in ldates]
        ):
        # remove duplicated observations
        obs.drop_duplicates() 
        # append the obtained observations to the fieldset
        observations.append(obs)

# align GeoDataFrames in the observations list so that they all contain
# the same stations on the same row
# the alignment is made primarily on the station id but also checking their geographic
# coordinates are close enough (as stations' metadata sometimes change over time)
# the "close enough" is defined by the parameter max_cluster_size
aligned_observations = aligned_geodfs(observations, max_cluster_size=0.1)

lats = aligned_observations[0]['latitude'].to_numpy().astype(float)
lons = aligned_observations[0]['longitude'].to_numpy().astype(float)
# Extracting station elevations
all_elevations = numpy.array([o['elevation'] for o in aligned_observations])
first_valid_index = numpy.isnan(all_elevations).argmin(axis=0)
heights = [all_elevations[i,j] for j,i in enumerate(first_valid_index)]

temp = time.time() - start_time
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('Time after equalising:')
print('%d:%d:%d' %(hours,minutes,seconds))

# creates an xarray with all the observations
obs_array = numpy.array([geo["value_0"] for geo in aligned_observations])

dates_array = numpy.array([geo.geodf_header['date'] for geo in aligned_observations])
stnids_array = aligned_observations[0]['stnid'].to_list()
obs_xarray = xarray.DataArray(obs_array,(('date',dates_array),('stnid',stnids_array)))

######################################################################################################
# OBSERVATION CLIMATE
#
# Climatology (mean and percentiles) for different aggregation times (days) in verifWindowList
# valid from fyear to lyear (e.g. 2000-2019) in a month-long period 
#
# IMPORTANT: Climate sample contains all the data for which the middle of the time window
#            belongs to the month of interest, e.g. for 7-day 2t climatology for September
#            contains all 7-day windows from 28 Aug/06UTC -4 Sep/00UTC to 26 Sep/06UTC-3 Oct/00UTC.
#
######################################################################################################

for vw in verifWindowList:
    print(vw)

    listOfListsOfDates = []
    for d in listOfDatesPerMonth(int(mm),int(fyear),int(lyear),maxlength):
        if (d+timedelta(days=round(vw/2))).month == int(mm):
            listOfListsOfDates.append(listOfDatesInPeriod(d,period,vw))

    clim = []
    for dd in listOfListsOfDates:
        print('dd = ', dd)
        if (param == '2t') or (param == '10ff'):
            arr = arr_mean(obs_xarray,dd)
        elif param == 'tp':
            arr = arr_sum(obs_xarray,dd)
        elif param == 'tmin':
            arr = arr_min(obs_xarray,dd)
        elif param == 'tmax':
            arr = arr_max(obs_xarray,dd)
        clim.append(arr)

    # for crit in critList:
    for crit in [0.65]:
        ofile = climdir + "/clim_" + param + "_" + str(vw) + "_" + mm + "_" + str(
            int(lyear) - int(fyear) + 1) + "years_" + fyear + "_" + lyear + "_" + str(int(crit * 100))
        ofilem = climdir + "/climmean_" + param + "_" + str(vw) + "_" + mm + "_" + str(
            int(lyear) - int(fyear) + 1) + "years_" + fyear + "_" + lyear + "_" + str(int(crit * 100))

        # mask out all stations with less than crit % availability in the period
        # NaN s used when there is no observed value for a given date/time on a given station
        clim_array = xarray.DataArray(clim,dims=('date','stnid'))
        population = clim_array.notnull().sum('date') # this gives the number of valid obs at each station
        where_minpopul_reached = population > (crit*len(clim))

        # for stations where there was less than crit % of valid observations we set them all as missing
        clim_array_masked = clim_array.where(where_minpopul_reached)
        perc = clim_array_masked.quantile([q/100 for q in range(0,101)],dim='date',interpolation='nearest')
        mean = clim_array_masked.mean('date')

        if (param == '2t') or (param == 'tmax') or (param == 'tmin'):
            perc = perc - 273.15 # Kelvin to Celsius
            mean = mean - 273.15

        # clim precentile file
        quan = {'q'+str(q) : numpy.round(perc.sel(quantile=q/100).values,1) for q in range(0,101)}

        geo_perc = metview.create_geo(
                                      type ='ncols',
                                      latitudes = lats,
                                      longitudes = lons,
                                      date   = int(lyear+mm+'01'),
                                      height =  heights,
                                      stnids = stnids_array,
                                      **quan)
        geo_perc = metview.remove_missing_values(geo_perc)
        geo_perc = metview.set_metadata(geo_perc,{'Parameter ' : ' Monthly climatology for ' + pname,
                                                  'Period ' : ' ' + calendar.month_name[int(mm)] + ", " + fyear + "-" + lyear,
                                                  'Time window ' : ' ' + str(vw) + ' day(s)'})
        metview.write(ofile,geo_perc)

        # clim mean file
        geo_mean = metview.create_geo(type ='ncols',
                                      latitudes = lats,
                                      longitudes = lons,
                                      date   = int(lyear+mm+'01'),
                                      height =  heights,
                                      stnids = stnids_array,
                                      values = numpy.round(mean.values,1))
        geo_mean = metview.remove_missing_values(geo_mean)
        geo_mean = metview.set_metadata(geo_mean,{'Parameter ' : ' Monthly mean climatology for ' + pname,
                                                  'Period ' : ' ' + calendar.month_name[int(mm)] + ', ' + fyear + '-' + lyear,
                                                  'Time window ' : ' ' + str(vw) + ' day(s)'})
        metview.write(ofilem,geo_mean)

print('Program finished')

# print(Memory())
# temp = time.time() - start_time
# hours = temp//3600
# temp = temp - 3600*hours
# minutes = temp//60
# seconds = temp - 60*minutes
# print('%d:%d:%d' %(hours,minutes,seconds))
#print("--- %s seconds ---" % (time.time() - start_time))
