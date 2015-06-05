import os
import mytool as mt
import traceback
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from multiprocessing import Pool
from datetime import datetime

def foo(filename):
    start = time.time()
    new_filename = "Binfiles/%s.bin"%filename[9:-4]
    if os.path.isfile(new_filename):
        return

    openfile = open(filename, "r")
    ts = []
    value = []
    for line in openfile:
        try:
            tmp = line.rstrip().rsplit(",")
            dt = mt.str2datetime(tmp[0]).timetuple()
            ts.append([dt[4], dt[3], dt[6], dt[2], dt[1]])
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


filenames = glob.glob("Csvfiles/*.csv")
filenames.sort()
p = Pool(20)
p.map(foo, filenames)

# for filename in filenames:
#     foo(filename)
