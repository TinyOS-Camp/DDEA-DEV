#!/adsc/DDEA_PROTO/bin/python

import simplejson
import urllib2
from metadata import getUUID
import datetime
from pathos.multiprocessing import ProcessingPool
from toolset import dill_save_obj, dill_load_obj

import os
import json
import datetime as dt
import pytz
from dateutil import tz
import time
import numpy as np

import traceback
import sys

import multiprocessing


#'KETI Motes'
#'http://new.openbms.org/backend/api/tags/Metadata__SourceName/KETI%20Motes'

#'SDH Dent Meters'
#'http://new.openbms.org/backend/api/tags/Metadata__SourceName/Soda%20Hall%20Dent%20Meters'

#'Sutardja Dai Hall BACnet'
#'http://new.openbms.org/backend/api/tags/Metadata__SourceName/Sutardja%20Dai%20Hall%20BACnet'

from const import *

def get_uuid_list(metadata):
    fs = [(lambda y: lambda uuid: getUUID(metadata[y]))(i)
          for i in xrange(len(metadata))]
    return [f(0) for f in fs]


def get_uuid_url(uuid, start_time, end_time):
    global UUID_PREFIX
    #url = UUID_PREFIX + uuid + "?starttime=" + str(start_time) + "&endtime=" + str(end_time) + "&"
    url = UUID_PREFIX + uuid + "?starttime=" + str(start_time) + "&endtime=" + str(end_time)
    return url


def get_start_end_time(start_time, end_time):
    ts_start = int(time.mktime(datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timetuple()) * 1000)
    ts_end = int(time.mktime(datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timetuple()) * 1000)
    return ts_start, ts_end


def construct_url_list(req_url, start_time, end_time, data_folder):

    ts_start, ts_end = get_start_end_time(start_time, end_time)
    rawdata = urllib2.urlopen(req_url).read()
    metadata = simplejson.loads(rawdata)
    #uuid_list = get_uuid_list(metadata)

    uuid_list = map(lambda x: str.strip(x), map(getUUID, metadata))
    strip_uuid_list = filter(lambda uuid: not (uuid == "" or uuid == "\n"), uuid_list)
    url_list = map(lambda uuid:get_uuid_url(uuid, ts_start, ts_end),strip_uuid_list)

    #url = 'l3805b128-c248-5c35-b901-0073e9af01b8?starttime=1385894160000&endtime=1386412560000&'
    queue = map(lambda url, uuid: (url, data_folder + uuid + DATA_EXT), url_list, strip_uuid_list)
    #return strip_uuid_list, url_list, queue
    return strip_uuid_list, queue


def save_url_to_file(url, filename):
    try:
        data = urllib2.urlopen(url).read()
        if data:
            dill_save_obj(data, filename)
    except:
        raise SystemExit(0)


def convert_to_bin(uuid, in_dir, out_dir):

    start = time.time()
    in_file = os.path.join(in_dir + uuid + DATA_EXT)
    out_file = os.path.join(out_dir + uuid + BIN_EXT)

    #faulty_list = []
    # check if file exists
    try:
        if os.path.isfile(out_file) and dill_load_obj(out_file):
            return
    except:
        # here, file is either not existing, or corrupted
        pass

    try:
        data = dill_load_obj(in_file)
        json_readings = json.loads(str(data))
        sensor_readings = json_readings[0]['Readings']

        if len(sensor_readings) == 0:
            return

        ts_list = []
        value_list = []

        for pair in sensor_readings:
            ts = pair[0] / 1000
            if pair[1] is None:
                continue

            value = float(pair[1])
            local_dt = dt.datetime.fromtimestamp(ts).replace(tzinfo=tz.gettz('UTC')).astimezone(pytz.timezone('US/Pacific'))
            time_tup = local_dt.timetuple()
            value_list.append(value)
            ts_list.append([local_dt, time_tup[5], time_tup[4], time_tup[3], time_tup[6], time_tup[2], time_tup[1]])

        # Save complete bin file
        data = {"ts":np.array(ts_list), "value":np.array(value_list)}
        dill_save_obj(data, out_file)

        # print some stats: time and filesize
        end = time.time()
        filesize = os.path.getsize(out_file)
        print "%s: %s (%.3f MB) in %.3f secs"%(dt.datetime.now(), out_file, filesize * 1.0 / 10**6, end - start)
    except:
        print '=== ERROR: ' + uuid + ' ==='
        print traceback.format_exc()
        #faulty_list.append(uuid)
        return uuid

    return ""
    #mt.saveObjectBinaryFast(np.array(faulty_list),'faulty_list.bin')

if __name__ == '__main__':

    print "-- START COLLECING SDH DATA --"
    uuid_list, queue = construct_url_list(
        "http://new.openbms.org/backend/api/tags/Metadata__SourceName/Soda%20Hall%20Dent%20Meters",
        "2014-08-01 9:00:00",
        "2014-08-02 9:00:00",
        SDH_DATA_FOLDER)

    #map(lambda meta_url: save_url_to_file(meta_url[0], meta_url[1]), queue)

    pool = ProcessingPool(nodes=multiprocessing.cpu_count())
    pool.map(lambda meta_url: save_url_to_file(meta_url[0], meta_url[1]), queue)

    print "-- START CONVERTING DATA --"
    faulty_list = pool.map(lambda uuid:convert_to_bin(uuid, SDH_DATA_FOLDER, SDH_BIN_FOLDER),uuid_list)
    faulty_list = filter(lambda fault_uuid: not fault_uuid == '', faulty_list)
    dill_save_obj(np.array(faulty_list), SDH_BIN_FOLDER + 'faulty_list.bin')

    print "-- END of COLLECING DATA --"
    exit(0)
