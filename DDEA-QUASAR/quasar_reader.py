# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:56:40 2014

@author: deokwoo
"""
from __future__ import division # To forace float point division
from qdf import quasar
from twisted.internet import defer, reactor
from twisted.internet.defer import inlineCallbacks, Deferred

def dlrv(result, files):
    s_data = dict()
    for r in result:
        ret, args = r
        stat, uid, rval = args
        ver, vals = rval
        print files[uid], ":", stat, " ver :", ver, " length (", len(vals), ")"

        if stat == "ok":
            tsval = [[v.time, v.value] for v in vals]
            s_data[files[uid]] = tsval

    return s_data


def onConnection(q, files, start_time, end_time):
    print "Connected to archiver", q, "..."

    dl = list()
    for uid, f in files.iteritems():
        d = q.queryStandardValues(uid, start_time, end_time)
        dl.append(d)
    dfl = defer.DeferredList(dl)
    dfl.addCallback(dlrv, files)
    return dfl


def getConnection(files, start_time, end_time):
    d = quasar.connectToArchiver("localhost")
    d.addCallback(onConnection, files, start_time, end_time)
    return d


@inlineCallbacks
def quasar_cb(input_files, start_time, end_time, timelet_inv, bldg_key, pname_key):
    import datetime as dt
    from ddea_proc import ddea_process

    print "Quasar Connection Initiated..."
    conn = yield getConnection(input_files, start_time, end_time)
    sensor_data = yield conn
    reactor.stop()

    ddea_process(sensor_data, start_time, end_time, timelet_inv, bldg_key, pname_key)


def read_sensor_data(input_files, start_time, end_time, timelet_inv, bldg_key, pname_key):
    print "Retrieve sensor data from quasar..."
    reactor.callWhenRunning(quasar_cb, input_files, start_time, end_time, timelet_inv, bldg_key, pname_key)
    reactor.run()

