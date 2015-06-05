#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-
# vi:ts=4:et

#
# Usage: python retriever-multi.py <file with URLs to fetch> [<# of
#          concurrent connections>]
#

import sys
import pycurl
import datetime
import time


def construct_url_list(uuid_list_file, start_time, end_time):
    
    ### read uuid_list_file ###
    queue = []
    uuid_list = open(uuid_list_file).readlines()
    strip_uuid_list = []
    url_list = []
    for uuid in uuid_list:
                
        uuid = uuid.strip()
        if uuid == "" or uuid == "\n":
            continue
        strip_uuid_list.append(uuid)
        
        prefix = 'http://new.openbms.org/backend/api/data/uuid/'
        ts_start =  int(time.mktime(datetime.datetime.strptime(start_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000) 
        ts_end = int(time.mktime(datetime.datetime.strptime(end_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000)
        
        url = prefix + uuid + "?starttime=" + str(ts_start) + "&endtime=" + str(ts_end) + "&"
        url_list.append(url)
        
        filename = uuid + ".dat"
        queue.append((url, filename)) 
        #url = 'l3805b128-c248-5c35-b901-0073e9af01b8?starttime=1385894160000&endtime=1386412560000&'
    return strip_uuid_list, url_list, queue

#uuid_list, url_list = construct_url_list("uuid-list.dat", "2013-12-01 9:00:00", "2013-12-07 9:00:00")


# We should ignore SIGPIPE when using pycurl.NOSIGNAL - see
# the libcurl tutorial for more info.
try:
    import signal
    from signal import SIGPIPE, SIG_IGN
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except ImportError:
    pass


# Get args
num_conn = 10
try:
    if sys.argv[1] == "-":
        urls = sys.stdin.readlines()
    else:        
        ###urls = open(sys.argv[1]).readlines()
        uuid_file = sys.argv[1]
        start_time = sys.argv[2]
        end_time = sys.argv[3]
        #uuid_list, urls, queue = construct_url_list("uuid-list.dat", "2013-12-01 9:00:00", "2013-12-07 9:00:00")        
        uuid_list, urls, queue = construct_url_list(uuid_file, start_time, end_time)
    if len(sys.argv) >= 5:
        num_conn = int(sys.argv[4])
except:
    print("Usage: %s <file with URLs to fetch> [<# of concurrent connections>]" % sys.argv[0])
    raise SystemExit


# Make a queue with (url, filename) tuples
# queue = []
# for url in urls:
#     url = url.strip()
#     if not url or url[0] == "#":
#         continue
#     filename = "doc_%03d.dat" % (len(queue) + 1)
#     queue.append((url, filename))


# Check args
assert queue, "no URLs given"
num_urls = len(queue)
num_conn = min(num_conn, num_urls)
assert 1 <= num_conn <= 10000, "invalid number of concurrent connections"
print("PycURL %s (compiled against 0x%x)" % (pycurl.version, pycurl.COMPILE_LIBCURL_VERSION_NUM))
print("----- Getting", num_urls, "URLs using", num_conn, "connections -----")


# Pre-allocate a list of curl objects
m = pycurl.CurlMulti()
m.handles = []
for i in range(num_conn):
    c = pycurl.Curl()
    c.fp = None
    c.setopt(pycurl.FOLLOWLOCATION, 1)
    c.setopt(pycurl.MAXREDIRS, 5)
    c.setopt(pycurl.CONNECTTIMEOUT, 30)
    c.setopt(pycurl.TIMEOUT, 300)
    c.setopt(pycurl.NOSIGNAL, 1)
    m.handles.append(c)


# Main loop
freelist = m.handles[:]
num_processed = 0
while num_processed < num_urls:
    # If there is an url to process and a free curl object, add to multi stack
    while queue and freelist:
        url, filename = queue.pop(0)
        c = freelist.pop()
        c.fp = open(filename, "wb")
        c.setopt(pycurl.URL, url)
        c.setopt(pycurl.WRITEDATA, c.fp)
        m.add_handle(c)
        # store some info
        c.filename = filename
        c.url = url
    # Run the internal curl state machine for the multi stack
    while 1:
        ret, num_handles = m.perform()
        if ret != pycurl.E_CALL_MULTI_PERFORM:
            break
    # Check for curl objects which have terminated, and add them to the freelist
    while 1:
        num_q, ok_list, err_list = m.info_read()
        for c in ok_list:
            c.fp.close()
            c.fp = None
            m.remove_handle(c)
            print("Success:", c.filename, c.url, c.getinfo(pycurl.EFFECTIVE_URL))
            freelist.append(c)
        for c, errno, errmsg in err_list:
            c.fp.close()
            c.fp = None
            m.remove_handle(c)
            print("Failed: ", c.filename, c.url, errno, errmsg)
            freelist.append(c)
        num_processed = num_processed + len(ok_list) + len(err_list)
        if num_q == 0:
            break
    # Currently no more I/O is pending, could do something in the meantime
    # (display a progress bar, etc.).
    # We just call select() to sleep until some more data is available.
    m.select(1.0)


# Cleanup
for c in m.handles:
    if c.fp is not None:
        c.fp.close()
        c.fp = None
    c.close()
m.close()