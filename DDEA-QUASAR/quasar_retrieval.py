# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:56:40 2014

@author: deokwoo
"""
from __future__ import division # To forace float point division
import numpy as np
import datetime as dt
import shlex
import subprocess
import time
from multiprocessing import Pool
import datetime as dt

import mytool as mt
import retrieve_weather as rw
from data_tools import *
from shared_constants import *

"""
################################################################################
# Parsing BMS sensor data 
# Data samples are regularized for specified times with timelet
################################################################################
# Check start time and end time of sensors

def time_range_check(input_files, ANS_START_T, ANS_END_T, TIMELET_INV):
    est_num_time_slot = int((dtime_to_unix([ANS_END_T])-dtime_to_unix([ANS_START_T]))/TIMELET_INV.seconds)
    t_range_min = dt.datetime(2000, 1, 1, 0, 0, 0)
    t_range_max = dt.datetime(2100, 1, 1, 0, 0, 0)
    purge_list = []
    input_file_to_be_included = []
    for filename in input_files:
        try:
            data = mt.loadObjectBinary(filename)
            if (len(data["ts"]) == 0) or (len(data["value"]) == 0):
                print 'Not enough data included: the number of samples is zero'
                purge_list.append(filename)
            elif (len(data["ts"]) < est_num_time_slot) or (len(data["value"]) < est_num_time_slot):
                print 'Not enough data included: the number of samples is less than est_num_time_slot'
                purge_list.append(filename)
            else:
                input_file_to_be_included.append(filename)
                if data["ts"][0][0] > t_range_min:
                    t_range_min = data["ts"][0][0]
                if data["ts"][0][0] < t_range_max:
                    t_range_max = data["ts"][-1][0]
        except Exception as e:
            print 'error in loading ', filename, ' ... Skipped', e

    if ANS_START_T > t_range_min:
        t_range_min = ANS_START_T
    if ANS_END_T < t_range_max:
        t_range_max = ANS_END_T
    return t_range_min, t_range_max, input_file_to_be_included
"""

###############################################################################
# Retrive weather data from internet for the specified periods 
###############################################################################
def get_weather_timelet(data_dict,t_slots, TIMELET_INV, USE_WEATHER_DATA_BIN=True):
    print '------------------------------------'
    print 'Retrieving weather data... '
    print '------------------------------------'
    t_start=t_slots[0]
    t_end=t_slots[-1]
    print 'start time:', t_start, ' ~ end time:',t_end
    # Date iteration given start time and end-time
    # Iterate for each day for all weather data types
    for date_idx,date in enumerate(daterange(t_start, t_end, inclusive=True)):
        print date.strftime("%Y-%m-%d")
        temp=date.strftime("%Y,%m,%d").rsplit(',')
        if USE_WEATHER_DATA_BIN:
            #filename = WEATHER_DIR + "VTT_%04d_%02d_%02d.bin"%(int(temp[0]), int(temp[1]), int(temp[2]))
            filename = WEATHER_DIR + "%04d_%02d_%02d.bin"%(int(temp[0]), int(temp[1]), int(temp[2]))
            data_day = mt.loadObjectBinaryFast(filename)
        else:
            data_day=rw.retrieve_data('VTT', int(temp[0]), int(temp[1]), int(temp[2]), view='d')
        # split the data into t
        data_day=data_day.split('\n')
        # Iterate for each time index(h_idx) of a day  for all weather data types
        for h_idx,hour_sample in enumerate(data_day):
            hour_samples=hour_sample.split(',')
            # Initialize weather data lists of dictionary
            # The first row is always the list of weather data types
            if (h_idx==0) and (date_idx==0):
                sensor_name_list=hour_sample.split(',')
                sensor_name_list = [sensor_name.replace('/','-') for sensor_name in sensor_name_list]
                for sample_idx,each_sample in enumerate(hour_samples):
                    sensor_name=sensor_name_list[sample_idx]
                    sensor_read=[[] for i in range(len(t_slots))]
                    stime_read=[[] for i in range(len(t_slots))] # Creat the list of lists for minute index
                    utc_t=[];val=[]
                    #data_dict.update({sensor_name:sensor_read})
                    #data_dict.update({sensor_name:zip(mtime_read,sensor_read)})
                    data_dict.update({sensor_name:[stime_read,sensor_read,[utc_t,val]]})
            elif h_idx>0:
                ################################################################
                # 'DateUTC' is the one
                sample_DateUTC=hour_samples[sensor_name_list.index('DateUTC')]
                # convert to UTC time to VTT local time.
                utc_dt=dt.datetime.strptime(sample_DateUTC, "%Y-%m-%d %H:%M:%S")
                vtt_dt_aware = utc_dt.replace(tzinfo=from_zone).astimezone(to_zone)

                # convert to offset-naive from offset-aware datetimes
                vtt_dt=dt.datetime(*(vtt_dt_aware.timetuple()[:6]))

                ### WARNING: vtt_utc is not utc
                vtt_utc = dtime_to_unix([vtt_dt])

                # Check boundary condition
                if int((vtt_dt - t_slots[0]).total_seconds()) < 0 or \
                int((vtt_dt - t_slots[-1]).total_seconds()) >= TIMELET_INV.seconds:
                    #print 'skipping weather data out of analysis range...'
                    continue

                slot_idx = int((vtt_dt - t_slots[0]).total_seconds() / TIMELET_INV.seconds)
                cur_sec_val = (vtt_dt - t_slots[slot_idx]).total_seconds()

                if cur_sec_val>=TIMELET_INV.seconds:
                    print 'sec: ' ,cur_sec_val
                    raise NameError('Seconds from an hour idx cannot be greater than '+str(TIMELET_INV.seconds) +'secs')
                try:# time slot index a given weather sample time

                    for sample_idx,each_sample in enumerate(hour_samples):
                        try:# convert string type to float time if possible
                            each_sample=float(each_sample)
                        except ValueError:
                            each_sample=each_sample
                        sensor_name=sensor_name_list[sample_idx]
                        if sensor_name in data_dict:
                            if each_sample!='N/A' and each_sample!=[]:
                                #data_dict[sensor_name][vtt_dt_idx].append(each_sample)
                                data_dict[sensor_name][0][slot_idx].append(cur_sec_val)
                                data_dict[sensor_name][1][slot_idx].append(each_sample)
                                data_dict[sensor_name][2][0].append(vtt_utc)
                                data_dict[sensor_name][2][1].append(each_sample)

                        else:
                            raise NameError('Inconsistency in the list of weather data')
                except ValueError:
                    slot_idx=-1
            else: # hour_sample is list of weather filed name, discard
                hour_sample=[]
    return sensor_name_list

###############################################################################
# This is the list of non-digit symbolic weather data
###############################################################################
# The symbolic weather data is such as Conditions (e.g Cloudy or Clear)
# and Events (e.g. Rain or Fog ...)
# Those symblic data is replaced with integer state representation whose
# pairs are stored in a hash table using Dictionary. 
# If no data is given, key value is set to 0.
###############################################################################
def symbol_to_state(symbol_list):
    symbol_dict={};symbol_val=[];key_val=1
    for i,key_set in enumerate(symbol_list):
        symbol_val_let=[]
        for key in key_set:
            if key not in symbol_dict:
                if len(key)==0:
                    symbol_dict.update({key:0})
                    symbol_val_let.append(0)
                else:    
                    symbol_dict.update({key:key_val})
                    symbol_val_let.append(key_val)
                    key_val=key_val+1        
            else:
                symbol_val_let.append(symbol_dict[key])
        symbol_val.append(symbol_val_let)
    return symbol_val,symbol_dict


def construct_data_dict(sensor_data, ANS_START_T, ANS_END_T, TIMELET_INV, include_weather=1, binfilename='data_dict'):
    print '-' * 80
    print 'mapping sensor list into hasing table using dictionary'
    print 'Align sensor data into a single time_slots referece...'
    print '-' * 80

    # Variable Declare and initialization
    time_slots = list()
    start = ANS_START_T
    while start < ANS_END_T:
        time_slots.append(start)
        start = start + TIMELET_INV

    # Data dictionary
    # All sensor and weather data is processed and structred into
    # a consistent single data format -- Dictionary
    data_dict = dict()
    sensor_list = list()
    purge_list = list()

    # Data Access is following ....
    #data_dict[key][time_slot_idx][(min_idx=0 or values=1)]

    for sensor_uuid, sensor_reading in sensor_data.iteritems():

        print 'sensor uuid ', sensor_uuid
        len_time_slots = len(time_slots)

        # sensor value is read by time
        dict_sensor_val, dict_stime, utc_t, val =\
            get_val_timelet(sensor_reading, time_slots, ANS_START_T, ANS_END_T, TIMELET_INV)

        if dict_sensor_val == -1:
            print 'append purge list: dict_sensor_val=-1'
            purge_list.append(sensor_uuid)

        elif len(utc_t) < len_time_slots:
            print 'append purge list:len(utc_t)<len_time_slots'
            purge_list.append(sensor_uuid)

        elif len(val) < len_time_slots:
            print 'append purge list:len(val)<len_time_slots'
            purge_list.append(sensor_uuid)

        else:
            sensor_list.append(sensor_uuid)

            # Convert list to array type for bin file size and loading time,
            dict_sensor_val_temp = np.array([np.asarray(val_) for val_ in dict_sensor_val])
            dict_stime_temp = np.array([np.asarray(t_) for t_ in dict_stime])
            utc_t_val_temp = np.asarray([utc_t, val])
            data_dict.update({sensor_uuid:[dict_stime_temp, dict_sensor_val_temp, utc_t_val_temp]}) # [:-4] drops 'bin'
            print '-' * 20

    data_dict.update({'time_slots': time_slots})
    print '-' * 20

    # directly access internet
    if include_weather == 1:
        #weather_list -that is pretty much fixed from database
        #(*) is the data to be used for our analysis
        #0 TimeEEST
        #1 TemperatureC (*)
        #2 Dew PointC (*)
        #3 Humidity (*)
        #4 Sea Level PressurehPa
        #5 VisibilityKm
        #6 Wind Direction
        #7 Wind SpeedKm/h
        #8 Gust SpeedKm/h
        #9 Precipitationmm
        #10 Events (*)
        #11 Conditions (*)
        #12 WindDirDegrees
        #13 DateUTC
        #import pdb;pdb.set_trace()

        weather_list = get_weather_timelet(data_dict, time_slots, TIMELET_INV)
        # Convert symbols to Integer representaion

        data_dict['Conditions'][1], Conditions_dict = symbol_to_state(data_dict['Conditions'][1])
        data_dict['Events'][1], Events_dict = symbol_to_state(data_dict['Events'][1])
        data_dict.update({'sensor_list': sensor_list})
        data_dict.update({'weather_list' : weather_list})
        data_dict.update({'Conditions_dict': Conditions_dict})
        data_dict.update({'Events_dict' : Events_dict})

        # Change List to Array type
        for key_id in weather_list:
            temp_list=[]
            for k, list_val_ in enumerate(data_dict[key_id]):
                temp_list.append(np.asanyarray(list_val_))

            data_dict[key_id] = temp_list

    # use stored bin file
    elif include_weather == 2:
        print 'use weather_dict.bin'
        # This part to be filled with Khiem......

    else:
        print 'skip weather database...'

    return data_dict, purge_list


def get_val_timelet(reading, t_slots, ANS_START_T, ANS_END_T, TIMELET_INV):

    data = dict()
    data['value'] = np.array([r[1] for r in reading], dtype=float)

    ts_list = list()
    for r in reading:
        local_dt = dt.datetime.fromtimestamp(r[0])
        time_tup = local_dt.timetuple()
        ts_list.append([local_dt, time_tup[5], time_tup[4], time_tup[3], time_tup[6], time_tup[2], time_tup[1]])

    data['ts'] = np.array(ts_list)


    if not len(data):
        print 'Error in file reading: empty data  ', data, '... Skip and need to be purged from sensor list'
        sensor_read = -1; stime_read = -1; utc_t = -1; val = -1
        return sensor_read, stime_read, utc_t, val

    if (len(data["ts"]) < MIN_NUM_VAL_FOR_FLOAT) or (len(data["value"]) < MIN_NUM_VAL_FOR_FLOAT):
        print 'No data included ', data,' ... Skip and need to be purged from sensor list'
        sensor_read = -1; stime_read = -1; utc_t = -1; val = -1
        return sensor_read, stime_read, utc_t, val

    nan_idx_list = np.nonzero(np.isnan(data["value"]))[0]
    sensor_val = np.delete(data["value"], nan_idx_list, axis=0)
    time_val = np.delete(data["ts"], nan_idx_list, axis=0)

    # Create the list of lists for value
    sensor_read = [[] for i in range(len(t_slots))]

    # Create the list of lists for seconds index
    stime_read = [[] for i in range(len(t_slots))]

    utc_t = []
    val = []

    for t_sample, v_sample in zip(time_val, sensor_val):
        temp_dt = t_sample[DT_IDX]

        if temp_dt < ANS_START_T or temp_dt >= ANS_END_T:
            continue

        try:
            idx = int((temp_dt - ANS_START_T).total_seconds() / TIMELET_INV.seconds)
            sensor_read[idx].append(v_sample)
            #secs=t_sample[MIN_IDX]*MIN+t_sample[SEC_IDX]
            secs = (temp_dt - t_slots[idx]).total_seconds()
            if secs >= TIMELET_INV.seconds:
                print 'sec: ', secs
                raise NameError('Seconds from an hour idx cannot be greater than '+str(TIMELET_INV.seconds)+ 'secs')

            stime_read[idx].append(secs)

        except ValueError:
            idx = -1

        utc_temp = dtime_to_unix([t_sample[DT_IDX]])
        utc_t.append(utc_temp)
        val.append(v_sample)

    return sensor_read, stime_read, utc_t, val