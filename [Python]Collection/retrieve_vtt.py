#!/adsc/DDEA_PROTO/bin/python

import os
import numpy as np
import shlex
from datetime import datetime
import urllib2
from const import VTT_BIN_FOLDER

#from pathos.multiprocessing import ProcessingPool
import pathos.multiprocessing as pmp
from toolset import dill_save_obj, dill_load_obj, str2datetime

#finland = timezone('Europe/Helsinki')
URL_PREFIFX = "http://121.78.237.160:8080" #"keti3.oktree.com"

def read_vtt_data(sensor, start_time, end_time):
    global URL_PREFIFX

    filename = VTT_BIN_FOLDER + sensor + ".bin"
    req_url = URL_PREFIFX + "/dashboard/query/?start=" + \
              start_time.strftime('%Y/%m/%d-%H:%M:%S') + \
              "&end=" + end_time.strftime('%Y/%m/%d-%H:%M:%S') + \
              "&m=avg:"+sensor+"&ascii"
    print "Retreving " + sensor + " data..."

    lines = urllib2.urlopen(req_url).read().split("\n")
    times = []
    values = []

    def _parse_data_line(data_line):
        pt = shlex.split(data_line)
        if len(pt) > 1:
            ltime = datetime.fromtimestamp(int(pt[1]))

            if start_time <= ltime < end_time:
                stime = ltime.strftime('%Y-%m-%d %H:%M:%S')
                dtime = str2datetime(stime)
                dt = dtime.timetuple()

                times.append([dtime, dt[5], dt[4], dt[3], dt[6], dt[2], dt[1]])
                values.append(float(pt[2]))

    def _merge(filepath, addl):
        try:
            orig = dill_load_obj(filepath)
        except:
            #import traceback;print traceback.print_exc()
            return None
        finally:
            ## concatenate two objects
            return {'ts': np.vstack((orig['ts'], addl['ts'])),
                    'value': np.hstack((orig['value'], addl['value']))}

    try:
        map(lambda dl: _parse_data_line(dl), lines)
    finally:
        data = {"ts": np.array(times), "value": np.array(values)}

#        if os.path.isfile(filename):
#            data = _merge(filename, data)

        if data:
            dill_save_obj(data, filename)


def load_finland_ids():
    with open('finland_ids.csv', 'r') as f:
        lines = f.readlines()
    return map(lambda l: l.split(',')[0].strip(), lines)

if __name__ == '__main__':

    chunk_size = pmp.cpu_count() - 2
    id_list = load_finland_ids()
    start_time = datetime(2013,11,1)
    end_time = datetime(2013,11,02)

    id_list_group = (lambda l, s=chunk_size: [l[i:i+s] for i in range(0, len(l), s)])(id_list)

    print "chuck size : " + str(chunk_size) + " len(id_list_group) : " + str(len(id_list_group))

    try:
        for id_group in id_list_group:
            p = pmp.Pool(chunk_size)
            p.map(lambda fin_id: read_vtt_data(fin_id, start_time, end_time), id_group)
            p.close()
            p.join()
            p.terminate()

    except (KeyboardInterrupt, SystemExit):
        import traceback;print traceback.print_exc()
    finally:
        exit(0)
