import os
import sys
import mytool as mt
import traceback
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from multiprocessing import Pool
from dateutil import tz
import pytz
from datetime import datetime

FOLDER = 'tempbin/'
from_zone = 'UTC'
to_zone = 'Europe/Helsinki'
def fix_bin(filename,in_path='./',out_path=FOLDER):
    old_bin = mt.loadObjectBinaryFast(filename)
    uuid = filename.strip().split('/')[-1][:-4]
    print uuid
    new_filename = out_path+uuid+'.bin'
    if os.path.isfile(new_filename):
        return

    ts = []
    
    start = time.time()
    for t in old_bin['ts']:
        dt = t[0]
        
        utc_dt = time.mktime(dt.timetuple())
        dt2 = datetime.utcfromtimestamp(utc_dt).replace(tzinfo=tz.gettz(from_zone)).astimezone(pytz.timezone(to_zone)).replace(tzinfo=None)
        dt2_tuple = dt2.timetuple()
        ts.append([dt2, dt2_tuple[5], dt2_tuple[4], dt2_tuple[3], dt2_tuple[6], dt2_tuple[2], dt2_tuple[1]])
        #print dt,dt2
        
    old_bin.update({'ts':np.array(ts)})
    mt.saveObjectBinaryFast(old_bin,new_filename)
    
    end = time.time()
    filesize = os.path.getsize(new_filename)
    print "%s: %s (%.3f MB) in %.3f secs"%(datetime.now(), new_filename, filesize * 1.0 / 10**6, end - start)
        
def foo(filename):
    start = time.time()
    new_filename = "Binfiles_new2/%s.bin"%filename[len(FOLDER):-4]
    print 
    if os.path.isfile(new_filename):
        return

    openfile = open(filename, "r")
    ts = []
    value = []
    for line in openfile:
        try:
            tmp = line.rstrip().rsplit(",")
            dt = mt.str2datetime(tmp[0]).timetuple()
            #ts.append([dt[4], dt[3], dt[6], dt[2], dt[1]])
            ts.append([mt.str2datetime(tmp[0]), dt[5], dt[4], dt[3], dt[6], dt[2], dt[1]])
            value.append(float(tmp[1]))
        except:
            traceback.print_exc(file=sys.stdout)
            pass
    openfile.close()

    data = {"ts":np.array(ts), "value":np.array(value)}
    mt.saveObjectBinary(data, new_filename)
    end = time.time()
    filesize = os.path.getsize(filename)
    print "%s: %s (%.3f MB) in %.3f secs"%(datetime.now(), filename, filesize * 1.0 / 10**6, end - start)
#uuid =  'VAK1.CG_SYSTEM_REACTIVE_POWER_M.bin'
#fix_bin(uuid)
if __name__ == "__main__":
    filenames = glob.glob("Binfiles/*.bin")
    filenames.sort()
    p = Pool(12)
    p.map(fix_bin, filenames)
    p.close()
    p.join()

"""
filenames = glob.glob("Csvfiles_new2/*.csv")
filenames.sort()
p = Pool(12)
p.map(foo, filenames)
p.close()
p.join()
"""

# for filename in filenames:
#     foo(filename)
