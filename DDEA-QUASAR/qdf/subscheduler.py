#!/usr/bin/env python
__author__ = 'immesys'

import configobj
import sys
from pymongo import MongoClient
import os
import quasar
import qdf
import resource
from twisted.internet import defer, protocol, reactor

EXIT_BADCONF = 2
EXIT_SKIP = 3
EXIT_UNK  = 4
EXIT_NOCHANGE = 5
EXIT_CODE = None
statusmap = {0: "OK: Changes made", 1:"ERR: Internal", 2:"ERR: Bad config file", 3:"OK: Disabled in config", 4:"ERR: Unknown error", 5:"OK: No change in data"}
def setexit(code):
    global EXIT_CODE
    EXIT_CODE = code

_client = MongoClient(os.environ["QDF_MDB_HOST"])
db = _client.qdf
sys.path.append(os.environ["QDF_ALGBASE"])
sys.path.append(os.environ["QDF_BASE"])

def dload(name):
    mod = __import__(name[:name.rindex(".")])
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def load_config(c):
    rv = []
    if "global" not in c:
        print "Missing 'global' section"
        setexit(EXIT_BADCONF)
        reactor.stop()
    if "enabled" not in c["global"]:
        print "Missing global/enabled"
        setexit(EXIT_BADCONF)
        reactor.stop()
    if c["global"]["enabled"] not in ["true","True"]:
        setexit(EXIT_SKIP)
        reactor.stop()
    try:
        klass = dload(c["global"]["algorithm"])
        for k in c:
            if k == "global":
                continue
            print "Loading instance '%s'" % k
            i = klass()

            i.deps = dict(c[k]["deps"])
            i.params = dict(c[k]["params"])
            i._conf_outputs = {ki : c[k]["outputs"][ki] for ki in c[k]["outputs"]}
            i._paramver = int(c[k]["paramver"])
            i._mintime = qdf.QDF2Distillate.date(c[k]["mintime"])
            i._maxtime = qdf.QDF2Distillate.date(c[k]["maxtime"])
            if "runonce" in c[k]:
                i._runonce = bool(c[k]["runonce"])
            else:
                i._runonce = False
            print "deps are: ", repr(i.deps)
            rv.append(i)

    except KeyError as e:
        print "Bad config, missing key", e
        setexit(EXIT_BADCONF)
        reactor.stop()
        return None
    except ImportError as e:
        print "Could not locate driver: ",e
        setexit(EXIT_BADCONF)
        reactor.stop()
        return None
    except Exception as e:
        setexit(EXIT_BADCONF)
        reactor.stop()
        raise

    return rv

def onFail(param):
    print "Encountered error: ", param

@defer.inlineCallbacks
def process(qsr, algs):
    try:
        all_sigs = []
        for a in algs:
            print ("[QDF] initialising algorithm:",repr(a))
            a.bind_databases(db, qsr)
            a.initialize(**a.params)
            print ("[QDF] doing process")
            significant = yield a._process()
            all_sigs.append(significant)
        if any(all_sigs):
            setexit(0)
            reactor.stop()
        else:
            setexit(EXIT_NOCHANGE)
            reactor.stop()
    except Exception as e:
        setexit(EXIT_UNK)
        reactor.stop()
        return

def entrypoint():
    cfg = configobj.ConfigObj(sys.stdin)
    algs = load_config(cfg)
    if algs is None:
        return
    d = quasar.connectToArchiver(os.environ["QDF_QUASAR_HOST"], int(os.environ["QDF_QUASAR_PORT"]))
    d.addCallback(process, algs)
    d.addErrback(onFail)

if __name__ == "__main__":
    # Connor wants more cpu time
    #resource.setrlimit(resource.RLIMIT_CPU, (60*60, 60*60)) #1 hour of CPU time
    #resource.setrlimit(resource.RLIMIT_DATA, (32*1024*1024*1024, 32*1024*1024*1024)) #32 GB of ram
    reactor.callWhenRunning(entrypoint)
    reactor.run()


    if EXIT_CODE is None:
        EXIT_CODE = EXIT_UNK
    print "EXIT CODE:", EXIT_CODE
    if EXIT_CODE in statusmap:
        print "MEANING: ", statusmap[EXIT_CODE]
    sys.exit(EXIT_CODE)
