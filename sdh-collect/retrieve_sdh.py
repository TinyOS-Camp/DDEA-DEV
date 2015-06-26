#!/adsc/DDEA_PROTO/bin/python

from metadata import getUUID, getPath
from toolset import saveObjectBinaryFast
import datetime as dt
from dateutil import tz
import numpy as np
import os, json, pytz, time, traceback, urllib2, simplejson
import multiprocessing as mp

from const import *

PATH_REP = {"/": "_", "-": "_", " ": "_"}
PATH_PREFIX = "SDH"


def get_uuid_list(metadata):
    fs = [(lambda y: lambda uuid: getUUID(metadata[y]))(i)
          for i in xrange(len(metadata))]
    return [f(0) for f in fs]


def get_uuid_url(uuid, start_time, end_time):
    url = DATA_URL + uuid + "?starttime=" + str(start_time) + "&endtime=" + str(end_time)
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


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr, min(curr + delta, end)
        curr += delta


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


def construct_stub_url_list(req_url, path_prefix, path_reps, start_date, end_date, out_dir):
    rawdata = urllib2.urlopen(req_url).read()
    metadata = simplejson.loads(rawdata)
    uuids = map(lambda data: (str.strip(getUUID(data)),get_path_sorted(getPath(data), path_prefix, path_reps)),metadata)
    stripped = filter(lambda u: not (u[0] == "" or u[0] == "\n" or u[0] == "-"), uuids)
    uuid_list = map(lambda u: u[0], stripped)
    file_name = map(lambda u: u[1], stripped)

    start_data_list = [start_date] * len(uuid_list)
    end_data_list = [end_date] * len(uuid_list)
    out_dir_list = [out_dir] * len(uuid_list)
    return zip(uuid_list, file_name, start_data_list, end_data_list, out_dir_list)


def get_retrieve_url(uuid, start_time, end_time):
    ts_start, ts_end = get_start_end_date(start_time, end_time)
    return get_uuid_url(uuid, ts_start, ts_end)


def get_sensor_data(url):
    data = urllib2.urlopen(url).read()
    if not data: return None
    json_readings = json.loads(str(data))
    if not len(json_readings[0]['Readings']): return None
    return json_readings[0]['Readings']


def collect_and_save_bin(url_info):
    uuid, filename, start_date, end_date, out_dir = url_info
    try:
        start = time.time()
        out_file = os.path.join(out_dir + filename + BIN_EXT)
        sensor_readings = list()

        for s, e in perdelta(start_date, end_date, dt.timedelta(days=7)):
            url = get_retrieve_url(uuid, s, e)
            data = get_sensor_data(url)
            if data:
                sensor_readings.extend(data)

        if not len(sensor_readings):
            print "---- WE'VE GOT NOTHING FOR", filename, "-----"
            return

        ts_list = list()
        value_list = list()

        for pair in sensor_readings:
            ts = pair[0] / 1000
            if pair[1] is None:
                continue

            value = float(pair[1])
            local_dt = dt.datetime.fromtimestamp(ts) #.replace(tzinfo=tz.gettz('UTC')).astimezone(pytz.timezone('US/Pacific'))
            time_tup = local_dt.timetuple()
            value_list.append(value)
            ts_list.append([local_dt, time_tup[5], time_tup[4], time_tup[3], time_tup[6], time_tup[2], time_tup[1]])

        data = {"ts": np.array(ts_list), "value": np.array(value_list)}

        # Save complete bin file
        saveObjectBinaryFast(data, out_file)

        # print some stats: time and filesize
        end = time.time()
        filesize = os.path.getsize(out_file)
        print "----- COLLECTING & SAVING COMPLETE %s %s: %s (%.3f MB) in %.3f secs ------"%\
              (filename, dt.datetime.now(), out_file, filesize * 1.0 / 10**6, end - start)

    except:
        print '=== ERROR ==='
        print traceback.format_exc()

if __name__ == '__main__':

    start_time = dt.time(hour = 0, minute = 0, second = 0)
    start_date = dt.datetime.combine(dt.date(2011, 5, 22), start_time)
    end_date = dt.datetime.combine(dt.date(2012, 12, 31), start_time)

    stub_url = construct_stub_url_list(
        META_URL,
        PATH_PREFIX,
        PATH_REP,
        start_date,
        end_date,
        SDH_BIN_FOLDER)

    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(collect_and_save_bin, stub_url)
    pool.terminate()
    pool.join()
