# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:56:40 2014

@author: deokwoo, stkim1
"""
from __future__ import division # To forace float point division

import datetime as dt
import shlex
import subprocess
import time
import sys

import numpy as np
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql import *

import mytool as mt
import toolset
import retrieve_weather as rw
from shared_constants import *
from data_tools import *

PATH_PREFIX = "/user/stkim1/"
PATH_POSTFIX = ".seq"

def get_input_filenames(sc=None, sqlContext=None, bldg_key=None, sensor_keyword=None):

    if not (sc and sqlContext):
        print "SparkContext or SQLContext cannot be nil"
        sys.exit(1)

    # get input sequnece files
    seqFiles = sc.sequenceFile(PATH_PREFIX + "SDH_READING_META" + PATH_POSTFIX,
                               keyClass="org.apache.hadoop.io.Text",
                               valueClass="org.apache.hadoop.io.Text")

    sqlContext.applySchema(seqFiles, StructType([StructField('uuid', StringType(), True), StructField('filename', StringType(), True)]))\
              .registerTempTable("INPUT_SEQS")

    query = "SELECT * FROM INPUT_SEQS"

    if bldg_key:
        query += " WHERE filename LIKE '%{}%'".format(bldg_key.upper())

    if sensor_keyword:
        query += " AND filename LIKE '%{}%'".format(sensor_keyword.upper())

    results = sqlContext.sql(query)\
        .map(lambda p: (p.uuid, PATH_PREFIX + p.filename + PATH_POSTFIX))\
        .collect()
    #return list(set(results))
    return results


def get_file_readings(sc=None, sqlContext=None, input_files=[], ANS_START_T=dt.datetime(2000, 1, 1, 0, 0, 0), ANS_END_T=dt.datetime(2100, 1, 1, 0, 0, 0)):

    def datetime_tuple(timestamp):
        #apply offset to correct time measurement
        local_dt = dt.datetime.fromtimestamp(timestamp)
        time_tup = local_dt.timetuple()
        return [local_dt, time_tup[5], time_tup[4], time_tup[3], time_tup[6], time_tup[2], time_tup[1]]

    if not (sc and sqlContext):
        print "SparkContext or SQLContext cannot be nil"
        sys.exit(1)

    if not (input_files and len(input_files)):
        print "No input file"
        sys.exit(1)

    readings = {}
    for i, seqmeta in enumerate(input_files):
        (uuid, filename) = seqmeta

        print "\n\n ---- ", filename, "----\n"
        tablename = filename.split('/')[-1].replace(PATH_POSTFIX, "")
        lines = sc.sequenceFile(filename,keyClass="org.apache.hadoop.io.LongWritable", valueClass="org.apache.hadoop.io.DoubleWritable")
        sqlContext.applySchema(lines, StructType([StructField('ts', IntegerType(), True), StructField('value', DoubleType(), True)])).registerTempTable(tablename)
        query = "SELECT * FROM {} WHERE {} <= ts AND ts <= {}".format(tablename, toolset.date_to_timestamp(ANS_START_T), toolset.date_to_timestamp(ANS_END_T))
        results = sqlContext.sql(query).map(lambda p: [datetime_tuple(p.ts), p.value]).collect()

        if not len(results):
            print "No results for given period between ", ANS_START_T, " and ", ANS_END_T
            continue

        ts_list = [r[0] for r in results]
        value_list = [r[1] for r in results]
        if not (len(ts_list) and len(value_list)) or len(ts_list) != len(value_list):
            print filename, " contains invalid results for given period between ", ANS_START_T, " and ", ANS_END_T
            import pdb;pdb.set_trace()
            continue

        readings[seqmeta] = {"ts": np.vstack(ts_list), "value": np.hstack(value_list)}

    return readings


def construct_data_dict(file_reading, ANS_START_T, ANS_END_T, TIMELET_INV, include_weather=1, binfilename='data_dict'):

    print '-' * 80, "\n", 'mapping sensor list into hasing table using dictionary'
    # Data dictionary that map all types of sensor readings into a single hash table
    # Read out all sensor files in the file list
    print 'Read sensor bin files in a single time_slots referece...','\n','-' * 80

    # Variable Declare and initialization
    time_slots=[]
    start = ANS_START_T
    while start < ANS_END_T:
        time_slots.append(start)
        start = start + TIMELET_INV

    # Data dictionary
    # All sensor and weather data is processed and structred into
    # a consistent single data format -- Dictionary
    data_dict={}
    sensor_list = []
    purge_list = []

    # Data Access is following ....
    #data_dict[key][time_slot_idx][(min_idx=0 or values=1)]
    start__dictproc_t=time.time()

    for sensor_meta, sensor_reading in file_reading.iteritems():
        (sensor_uuid, sensor_path) = sensor_meta
        print 'sensor uuid ', sensor_uuid, ' : path ', sensor_path
        len_time_slots = len(time_slots)

        # sensor value is read by time
        dict_sensor_val, dict_stime, utc_t, val =\
            get_val_timelet(sensor_reading, time_slots, ANS_START_T, ANS_END_T, TIMELET_INV)

        if dict_sensor_val == -1:
            print 'append purge list: dict_sensor_val=-1'
            purge_list.append(sensor_meta)

        elif len(utc_t) < len_time_slots:
            print 'append purge list:len(utc_t)<len_time_slots'
            purge_list.append(sensor_meta)

        elif len(val) < len_time_slots:
            print 'append purge list:len(val)<len_time_slots'
            purge_list.append(sensor_meta)

        else:
            sensor_list.append(sensor_path)

            # Convert list to array type for bin file size and loading time,
            dict_sensor_val_temp = np.array([np.asarray(val_) for val_ in dict_sensor_val])
            dict_stime_temp = np.array([np.asarray(t_) for t_ in dict_stime])
            utc_t_val_temp = np.asarray([utc_t, val])
            data_dict.update({sensor_path:[dict_stime_temp, dict_sensor_val_temp, utc_t_val_temp]}) # [:-4] drops 'bin'
            print '--------------------------------------'

    data_dict.update({'time_slots': time_slots})
    print '--------------------------------------'

    end__dictproc_t=time.time()
    print 'the time of processing get_val_timelet is ', end__dictproc_t - start__dictproc_t, ' sec'

    # directly access internet
    if include_weather==1:
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

    if IS_SAVING_DATA_DICT:
        # Storing  data_dict
        print ' Storing data_dict into binary file, data_dict.bin....'
        mt.saveObjectBinaryFast(data_dict, binfilename + '.bin')
        sizeoutput = int(shlex.split(subprocess.check_output("stat -c %s "+ binfilename+ ".bin", shell=True))[0])
        print 'Saved bin file size is ', round(sizeoutput/10**6, 2), 'Mbyte'
    return data_dict,purge_list


def get_val_timelet(data, t_slots, ANS_START_T, ANS_END_T, TIMELET_INV):

    if not len(data):
        print 'Error in file reading: empty data  ', data, '... Skip and need to be purged from sensor list'
        sensor_read = -1; stime_read = -1; utc_t = -1; val = -1
        return sensor_read, stime_read, utc_t, val

    if (len(data["ts"]) < MIN_NUM_VAL_FOR_FLOAT) or (len(data["value"]) < MIN_NUM_VAL_FOR_FLOAT):
        print 'No data included ', data,' ... Skip and need to be purged from sensor list'
        sensor_read = -1; stime_read = -1; utc_t = -1; val = -1
        return sensor_read, stime_read, utc_t, val

    sensor_val = data["value"]
    time_val = data["ts"]

    nan_idx_list = np.nonzero(np.isnan(sensor_val))[0]
    sensor_val = np.delete(sensor_val, nan_idx_list, axis=0)
    time_val = np.delete(time_val, nan_idx_list, axis=0)

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


def get_weather_timelet(data_dict,t_slots,TIMELET_INV,USE_WEATHER_DATA_BIN=True):
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
