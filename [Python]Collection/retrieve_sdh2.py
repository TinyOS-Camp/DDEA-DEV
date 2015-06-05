#!/adsc/DDEA_PROTO/bin/python

import simplejson
import urllib2
from metadata import getUUID, getPath
from pathos.multiprocessing import ProcessingPool
import pathos.multiprocessing as pmp
from toolset import dill_save_obj, dill_load_obj

import os
import json

import datetime as dt
import pytz
from dateutil import tz
import time
import numpy as np

import traceback
import multiprocessing


#'KETI Motes'
#'http://new.openbms.org/backend/api/tags/Metadata__SourceName/KETI%20Motes'

#'SDH Dent Meters'
#'http://new.openbms.org/backend/api/tags/Metadata__SourceName/Soda%20Hall%20Dent%20Meters'

#'Sutardja Dai Hall BACnet'
#'http://new.openbms.org/backend/api/tags/Metadata__SourceName/Sutardja%20Dai%20Hall%20BACnet'

from const import *

PATH_REP = {"/": "_", "-": "_", " ": "_"}
PATH_PREFIX = "SDH"

def get_uuid_list(metadata):
    fs = [(lambda y: lambda uuid: getUUID(metadata[y]))(i)
          for i in xrange(len(metadata))]
    return [f(0) for f in fs]


def get_uuid_url(uuid, start_time, end_time):
    url = UUID_PREFIX + uuid + "?starttime=" + str(start_time) + "&endtime=" + str(end_time)
    return url


def get_start_end_time(start_time, end_time):
    ts_start = int(time.mktime(dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timetuple()) * 1000)
    ts_end = int(time.mktime(dt.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timetuple()) * 1000)
    return ts_start, ts_end

def get_start_end_date(start_time, end_time):
    ts_start = int(time.mktime(start_time.timetuple()) * 1000)
    ts_end = int(time.mktime(end_time.timetuple()) * 1000)
    return ts_start, ts_end

def get_path_sorted(path, path_prefix, dic):
    path = str.strip(path)
    for i, j in dic.iteritems():
        path = path.replace(i, j)
    return (path_prefix + path).upper()


def construct_url_list(req_url, start_time, end_time, path_prefix, path_reps):
    print "Retrieve UUID list from " + start_time.strftime("%Y-%m-%d %H:%M:%S")\
          + " to " + end_time.strftime("%Y-%m-%d %H:%M:%S") + "..."

    ts_start, ts_end = get_start_end_date(start_time, end_time)
    rawdata = urllib2.urlopen(req_url).read()
    metadata = simplejson.loads(rawdata)

    uuids = map(lambda data: (str.strip(getUUID(data)),get_path_sorted(getPath(data), path_prefix, path_reps)),metadata)
    stripped = filter(lambda u: not (u[0] == "" or u[0] == "\n" or u[0] == "-"), uuids)

    uuid_list = map(lambda u: u[0], stripped)
    file_name = map(lambda u: u[1], stripped)
    urls = map(lambda u: get_uuid_url(u, ts_start, ts_end), uuid_list)

    return uuid_list, urls, file_name

## concatenate two objects
def merge(filepath, addl):
    print "merging file " + filepath
    orig = dill_load_obj(filepath)
    os.remove(filepath)
    return {'ts': np.vstack((orig['ts'], addl['ts'])),
            'value': np.hstack((orig['value'], addl['value']))}


def convert_to_bin(uuid, url, filename, out_dir):
    start = time.time()
    out_file = os.path.join(out_dir + filename + BIN_EXT)

    #faulty_list = []
    # check if file exists
    try:
        data = urllib2.urlopen(url).read()

        if not data:
            return

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


        data = {"ts":np.array(ts_list), "value":np.array(value_list)}
        if os.path.isfile(out_file):
            data = merge(out_file, data)

        # Save complete bin file
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


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr, min(curr + delta, end)
        curr += delta


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


if __name__ == '__main__':

    start_time = dt.time(hour = 9, minute = 0, second = 0)
    start_date = dt.datetime.combine(dt.date(2014, 3, 24), start_time)
    #end_date = dt.datetime.combine(dt.date(2014, 10, 24), start_time)
    end_date = dt.datetime.combine(dt.date(2014, 10, 24), start_time)

    print "-- START COLLECING SDH DATA --"

    for s, e in perdelta(start_date, end_date, dt.timedelta(days=7)):

        uuid_list, url_list, file_name = construct_url_list(
            "http://new.openbms.org/backend/api/tags/Metadata__SourceName/Soda%20Hall%20Dent%20Meters",
            s,
            e,
            PATH_PREFIX, PATH_REP)

        print "-- START CONVERTING DATA from " + \
              s.strftime("%Y-%m-%d %H:%M:%S") + " " + \
              e.strftime("%Y-%m-%d %H:%M:%S") + "..."

        pool = ProcessingPool(nodes=multiprocessing.cpu_count())
        faulty_list = pool.map(lambda u, l, f: convert_to_bin(u, l, f, SDH_BIN_FOLDER), uuid_list, url_list, file_name)
        faulty_list = filter(lambda fault_uuid: not fault_uuid == '', faulty_list)
        #dill_save_obj(np.array(faulty_list), SDH_BIN_FOLDER + 'faulty_list.bin')

    print "-- END of COLLECING DATA --"
    exit(0)
