from quasar import *
import sys
import array
from twisted.internet import defer, protocol, reactor
import isodate
import datetime
import time
import uuid

def _fullname(o):
    return o.__module__ + "." + o.__class__.__name__

def _get_commit():
    return "commitstr"

def onFail(param):
    reactor.stop()
    raise Exception("Encountered error: ", param)

MIN_TIME    = -(16<<56)
MAX_TIME    = (48<<56)
NANOSECOND = 1
MICROSECOND = 1000*NANOSECOND
MILLISECOND = 1000*MICROSECOND
SECOND = 1000*MILLISECOND
MINUTE = 60*SECOND
HOUR = 60*MINUTE
DAY = 24*HOUR

class StreamData (object):
    __slots__ = ["times", "values", "uid", "bounds_start", "bounds_end"]
    def __init__(self, uid):
        self.uid = uid
        self.times = array.array("L")
        self.values = array.array("d")
        self.bounds_start = array.array("L")
        self.bounds_end = array.array("L")

    def addreading(self, time, value):
        self.values.append(value)
        self.times.append(time)

    def addbounds(self, start, end):
        self.bounds_start.append(start)
        self.bounds_end.append(end)

    def __repr__(self):
        rv = "StreamData::"
        rv += "\n      uuid ="+repr(self.uid)
        rv += "\n     times ="+repr(self.times)
        rv += "\n     values="+repr(self.values)
        rv += "\n    sbounds="+repr(self.bounds_start)
        rv += "\n    ebounds="+repr(self.bounds_end)+"\n"
        return rv

class RunReport (object):
    __slots__ = ["streams"]
    def __init__(self):
        self.streams = {}

    def addstream(self, uid, name):
        s = StreamData(uid)
        self.streams[name] = s
        return s

    def output(self, name):
        return self.streams[name]

    def __repr__(self):
        rv = "RunReport"
        for k in self.streams:
            rv += "\n> "+ k + "=" + repr(self.streams[k])
        return rv

class QDF2Distillate (object):

    #set by scheduler to {name:uuid}
    deps = []

    def __init__(self):
        self._outputs = {}
        self.inputs = []
        self._metadata = {}

    def initialize(self):
        pass

    def prereqs(self, changed_ranges):
        """

        :param changed_ranges: a list of (name, uuid, [[start_time, end_time], ...])
        :return: a list of (name, uuid, [[start_time, end_time], ...]) tuples.
        """
        return changed_ranges

    def bind_databases(self, mongo, quasar):
        self.mdb = mongo
        self._db = quasar

    @staticmethod
    def expand_prereqs_parallel(changed_ranges):
        ranges  = []
        for s in changed_ranges:
            ranges.append(s[2])

        combined_ranges = []
        num_streams = len(ranges)
        while True:
            progress = False
            combined = False
            #find minimum range
            minidx = 0
            for idx in xrange(num_streams):
                if len(ranges[idx]) == 0:
                    continue
                progress = True
                if ranges[idx][0][0] < ranges[minidx][0][0]:
                    minidx = idx
            #Now see if any of the other ranges starts lie before it's end
            for idx in xrange(num_streams):
                if len(ranges[idx]) == 0:
                    continue
                if idx == minidx:
                    continue
                if ranges[idx][0][0] <= ranges[minidx][0][1]:
                    if ranges[idx][0][1] > ranges[minidx][0][1]:
                        ranges[minidx][0][1] = ranges[idx][0][1]
                    ranges[idx] = ranges[idx][1:]
                    combined = True
            if not progress:
                break
            if not combined:
                combined_ranges.append(ranges[minidx][0])
                ranges[minidx] = ranges[minidx][1:]

        rv = []
        for s in changed_ranges:
            rv.append([s[0], s[1], combined_ranges[:]])
        return rv

    def _clamp_range(self, range):
        rv = [range[0], range[1]]
        if rv[0] < self._mintime:
            rv[0] = self._mintime
        if rv[1] > self._maxtime:
            rv[1] = self._maxtime
        if rv[0] >= rv[1]:
            return None
        return rv

    def _sync_metadata(self):
        for skey in self._outputs:
            path = "/%s/%s/%s" % (self._section, self._name, skey)
            #deps = ",".join("%s" % self.deps[ky] for ky in self.deps)
            doc = self.mdb.metadata.find_one({"Path":path})
            if doc is not None and doc["Metadata"]["Version"] != self._version:
                print "[QDF] rewriting metadata: version bump"
                self.mdb.metadata.remove({"Path":path})
                doc = None
            if doc is None:
                uid = self._outputs[skey][0]
                ndoc = {
                    "Path" : path,
                    "Metadata" :
                    {
                        "SourceName" : "Distillates",
                        "Algorithm": _fullname(self),
                        "Commit": _get_commit(),
                        "Version": self._version,
                        "Name": self._name,
                        "ParamVer": self._paramver,
                        "Parameters" : {},
                    },
                    "Dependencies" : {},
                    "uuid" : uid,
                    "Properties" :
                    {
                        "UnitofTime" : "ns",
                        "Timezone" : "UTC",
                        "UnitofMeasure" : self._outputs[skey][1],
                        "ReadingType" : "double"
                    }
                }
                for k in self._metadata:
                    ndoc["Metadata"][k] = self._metadata[k]
                for k in self.deps:
                    ndoc["Dependencies"][k] = self.deps[k]+"::0"
                for k in self.params:
                    ndoc["Metadata"]["Parameters"][k] = self.params[k]

                self.mdb.metadata.save(ndoc)
            else:
                if int(doc["Metadata"]["Version"]) < self._version:
                    print "[QDF] stream mismatch: ", int(doc["Metadata"]["Version"]), self._version
                    self._old_streams.append(skey)

                #self._streams[skey]["uuid"] = doc["uuid"]
                sdoc = {"Metadata.Version":self._version, "Metadata.Commit": _get_commit()}
                for k in self._metadata:
                    sdoc["Metadata."+k]= self._metadata[k]
                for k in self._dep_vers:
                    sdoc["Dependencies."+k] = self.deps[k]+"::"+str(self._dep_vers[k])
                sdoc["Metadata.Parameters"] = self.params #{k : self.params[k] for k in self.params}
                sdoc["Metadata.ParamVer"] = self._paramver
                self.mdb.metadata.update({"Path":path},{"$set":sdoc},upsert=True)
                print "[QDF] we inserted new metadata version"

    def get_last_version(self, sname):
        anystream = self._outputs.keys()[0]
        path = "/%s/%s/%s" % (self._section, self._name, anystream)
        r = self.mdb.metadata.find_one({"Path":path})
        if r is None:
            return 0
        return int(r["Dependencies"][sname].split("::")[1])

    def get_last_paramversion(self, sname):
        path = "/%s/%s/%s" % (self._section, self._name, sname)
        r = self.mdb.metadata.find_one({"Path":path})
        if r is None:
            return 0
        return int(r["Metadata"]["ParamVer"])

    def get_last_algversion(self, sname):
        path = "/%s/%s/%s" % (self._section, self._name, sname)
        r = self.mdb.metadata.find_one({"Path":path})
        if r is None:
            return 0
        return int(r["Metadata"]["Version"])

    def insertstreamdata(self, s):
        """
        :param s: a StreamData class
        :return: a deferred
        """
        BATCHSIZE=4000
        idx = 0
        total = len(s.times)
        dlist = []
        while idx < total:
            chunklen = BATCHSIZE if total - idx > BATCHSIZE else total-idx
            d = self._db.insertValuesEx(s.uid, s.times, idx, s.values, idx, chunklen)
            def rv((stat, arg)):
                pass
            d.addCallback(rv)
            d.addErrback(onFail)
            dlist.append(d)
            idx += chunklen
        return defer.DeferredList(dlist)

    @defer.inlineCallbacks
    def _nuke_stream(self, uid):
        print "[QDF] Algorithm or Param ver has changed. Nuking existing data (this can take a while)"
        rv = yield self._db.deleteRange(uid, MIN_TIME, MAX_TIME)

    def compute(self, changed_ranges, input_streams, params, report):
        """

        :param changed_ranges: a dictionary of {name: [start_time, end_time]}
        :param input_streams: a dictionary of {name: [(time, value)]}
        :return: a RunReport
        """
        pass

    def set_version(self, ver):
        self._version = ver

    def set_section(self, section):
        self._section = section

    def set_name(self, name):
        self._name = name

    def register_output(self, name, unit):
        try:
            uid = self._conf_outputs[name]
            self._outputs[name] = (uid, unit)
        except KeyError:
            print "MISSING CONFIGURATION OPTION FOR EXPECTED OUTPUT"
            raise

    def default_chunking_algorithm(self, chranges):
        ncr = []
        for cr in chranges:
            ranges = []
            for ran in cr[2]:
                n = self._clamp_range(ran)
                if n is not None:
                    ranges.append(n)
            ncr.append([cr[0],cr[1],ranges])


        if len(ncr) == 0:
            yield None
            return

        print "[QDF] DCA input: ", repr(chranges)
        print "[QDF] DCA clamp: ", repr(ncr)
        expanded = QDF2Distillate.expand_prereqs_parallel(ncr)
        print "[QDF] expanded: ", expanded
        # TODO I am taking the easier route here. Technically we should
        # not lie about changed ranges, but it does not hurt. I am going to
        # tell the algs that every stream has changed on the expanded ranges
        # even if they have not

        def emit(st, et):
            rv = [[s[0], s[1], [[st,et]] ] for s in ncr]
            return rv

        range_size = 1<<37 #kinda two minutes
        for cur in expanded[0][2]:
            #cur is [st, et]
            ptr = cur[0]
            while ptr < cur[1]:
                eslice = (ptr + range_size)
                eslice = eslice & ~(range_size-1)
                if eslice > cur[1]:
                    eslice = cur[1]
                yield emit(ptr, eslice)
                ptr = eslice


    def register_input(self, name):
        self.inputs.append([None, name])

    def set_metadata(self, key, value):
        self._metadata[key] = value


    @defer.inlineCallbacks
    def _process(self):

        stuff_happened = False

        # check for alg or paramver nukages
        override = False
        for sname in self._outputs:
            uid, _ = self._outputs[sname]
            cur_alg_ver = int(self._version)
            cur_param_ver = int(self._paramver)
            if (self.get_last_algversion(sname) != cur_alg_ver or
                self.get_last_paramversion(sname) != cur_param_ver):
                self._nuke_stream(uid)
                override = True

        if self._runonce == True and override == False:
            print "[QDF] Not running: runonce with no override"
            defer.returnValue(False)

        # get the dependency past versions
        lver = {}
        for k in self.deps:
            uid = self.deps[k]
            if override:
                lver[uid] = 0
            else:
                lver[uid] = self.get_last_version(k)

        # get the dependency current versions (freeze)
        cver_keys = [k for k in self.deps]
        cver_uids = [self.deps[k] for k in self.deps]
        uid_keymap = {cver_uids[i] : cver_keys[i] for i in xrange(len(cver_keys))}
        key_uidmap = {cver_keys[i] : cver_uids[i] for i in xrange(len(cver_keys))}
        status, v = yield self._db.queryVersion(cver_uids)
        try: #cab: added to give informative error on bad uuid inputs
          cver = {cver_uids[i] : v[0][uuid.UUID(cver_uids[i]).bytes] for i in xrange(len(cver_uids))}
        except KeyError:
          print "[QDF] key error on current version uids. Check input uids in config for errors"
          raise KeyError
        print "stream cver", cver
        print "meta lver", lver
        # get changed ranges
        chranges = []
        for k in lver:
            if lver[k] != cver[k]:
                    stat, rv = yield self._db.queryChangedRanges(k, lver[k], cver[k], 300000)
                    print "[QDF] CR RV: ", rv
                    if len(rv) > 0:
                        cr = [[v.startTime, v.endTime] for v in rv[0]]
                        cr = [x for x in cr]
                        chranges.append((k, uid_keymap[k], cr))
                    else:
                        chranges.append((k, uid_keymap[k], []))
            else:
                chranges.append((k, uid_keymap[k], []))

        # TODO chunk changed ranges
        # the correct final algorithm would be to take pw=37 slices out of the combined cr
        # and then feed only the changed ranges that are present in that slice to the
        # prereqs call.
        for cr_slice in self.default_chunking_algorithm(chranges):
            # get adjusted ranges
            prereqs = self.prereqs(cr_slice)
            print "[QDF] prereqs: ", repr(cr_slice)

            # query data
            data = {}
            data_defs = []
            for istream in self.deps:
                #locate the uuid and range in prereqs
                print "[QDF] istream: " + str(istream)
                idx = [i for i in xrange(len(prereqs)) if prereqs[i][1] == istream][0]
                st = prereqs[idx][2][0][0]
                et = prereqs[idx][2][0][1]
                uid = prereqs[idx][0]
                print "[QDF] uuid=",uid
                ver = cver[uid]
                d = self._db.queryStandardValues(uid, st, et, version=ver)

                def make_onret(a_uid):
                        def onret((statcode, (version, values))):
                            key = uid_keymap[a_uid]
                            data[key] = [[v.time, v.value] for v in values]
                        return onret
                d.addCallback(make_onret(uid))
                data_defs.append(d)
            print "[QDF] waiting for data precache"
            then = time.time()
            yield defer.DeferredList(data_defs)
            print "[QDF] precache finished in %.2f seconds" % (time.time() - then)
            print "[QDF] data:"
            for k in data:
                print "[QDF] > ", k, " : ", len(data[k])

            # prepare output streams
            runrep = RunReport()
            for ostream in self._outputs:
                #                uuid                       name
                runrep.addstream(self._outputs[ostream][0], ostream)

            # process
            if cr_slice is None:
                altered_changed_ranges = None
            else:
                altered_changed_ranges = {}
                for it in cr_slice:
                    # we are assuming the chunking algorithm won't give disjoint changed
                    # ranges to a single invocation
                    assert len(it[2]) == 1
                    altered_changed_ranges[it[1]] = it[2][0]
            self.compute(altered_changed_ranges, data, self.params, runrep)
            stuff_happened = True
            print "[QDF] compute completed successfully. erasing and inserting"
            # delete data in bounds ranges
            for strm in runrep.streams:
                for idx in xrange(len(runrep.streams[strm].bounds_start)):
                    sb = runrep.streams[strm].bounds_start[idx]
                    eb = runrep.streams[strm].bounds_end[idx]
                    uid = runrep.streams[strm].uid
                    print "[QDF] erasing range %d to %d (%d) in %s" % (sb, eb, eb-sb, uid)
                    yield self._db.deleteRange(uid, sb, eb)
                    print "[QDF] erase range complete"

            # insert data into DB
            insertDeferreds = []
            for s in runrep.streams:
                insertDeferreds.append(self.insertstreamdata(runrep.streams[s]))

            # wait for that to finish

            yield defer.DeferredList(insertDeferreds)
            print "[QDF] database insert complete"

        # end of chunking for loop

        # update the dep versions
        self._dep_vers = {}
        for n in cver_keys:
            self._dep_vers[n] = cver[key_uidmap[n]]

        # sync the metadata
        self._sync_metadata()

        defer.returnValue(stuff_happened)


    @staticmethod
    def date(dst):
        """
        This parses a modified isodate into nanoseconds since the epoch. The date format is:
        YYYY-MM-DDTHH:MM:SS.NNNNNNNNN
        Fields may be omitted right to left, but do not elide leading or trailing zeroes in any field
        """
        idate = dst[:26]
        secs = (isodate.parse_datetime(idate) - datetime.datetime(1970,1,1)).total_seconds()
        nsecs = int(secs*1000000) * 1000
        nanoS = dst[26:]
        if len(nanoS) != 3 and len(nanoS) != 0:
            raise Exception("Invalid date string!")
        if len(nanoS) == 3:
            nsecs += int(nanoS)
        return nsecs
