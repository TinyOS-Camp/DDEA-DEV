#!/usr/bin/python

"""
Created on Wed Apr 16 10:56:40 2014

Author : Deokwoo Jung
E-mail : deokwoo.jung@gmail.com
"""

import simplejson, urllib2
from shared_constants import *
from data_tools import remove_dot
from mytool import fix_sensor_name

def read_quasar_url(uid, start_time, end_time):
    req_url = "http://localhost:9000/data/uuid/{}?starttime={}&endtime={}&unitoftime=ns".format(uid, start_time, end_time)
    rawdata = urllib2.urlopen(req_url).read()
    values = simplejson.loads(rawdata)[0]['Readings']
    return values

def read_sensor_data(sensor_hash, start_time, end_time):

    from log_util import log

    sensor_data = dict()
    for stitle, uid in sensor_hash.iteritems():
        tsvals = read_quasar_url(uid, start_time, end_time)

        if tsvals is None or len(tsvals) == 0:
            log.critical(stitle + " (" + uid + ") is unavailable from " + str(start_time) + " to " + str(end_time))
        else:
            log.debug(uid + " : " + stitle + " : TS-VAL size " + str(len(tsvals)))

            """
            log.info(uid + " : " + stitle + " : TS-VAL reading saved in JSON format...")
            with open(JSON_DIR + "reading-" + uid + ".json", 'w') as f:
                f.write(simplejson.dumps(tsvals))
            """

            sensor_data.update({stitle: tsvals})

    return sensor_data












