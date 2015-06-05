import os
import sys
import mytool as mt
import traceback
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from multiprocessing import Pool
from datetime import datetime

FOLDER = 'Binfiles_extra/'
SCRIPT_DIR = os.path.dirname(__file__)
def foo(filename,path=FOLDER):
    start = time.time()
    uuid = filename.split('/')[-1]
    new_filename = os.path.join(SCRIPT_DIR,path + uuid[:-4] + '.bin')
    #new_filename = "Binfiles_new2/%s.bin"%filename[len(FOLDER):-4]
    #print 
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
    print 'Save bin...'
    mt.saveObjectBinaryFast(data, new_filename)
    end = time.time()
    filesize = os.path.getsize(filename)
    print "%s: %s (%.3f MB) in %.3f secs"%(datetime.now(), filename, filesize * 1.0 / 10**6, end - start)


if __name__ == '__main__':
    """
    filenames = glob.glob("Csvfiles_extra/*.csv")
    #filenames = [filename.strip().split('/')[-1] for filename in filenames]
    filenames.sort()
    p = Pool(12)
    p.map(foo, filenames)
    p.close()
    p.join()
    # for filename in filenames:
    #     foo(filename)
    """
    faulty = mt.loadObjectBinaryFast('faulty_files.bin')
    print faulty
    filenames = [os.path.join(SCRIPT_DIR,'Csvfiles_new2/' + filename.split('/')[-1][:-4] + '.csv') for filename in faulty]

    print filenames
    for filename in filenames:
        print filename
        foo(filename,path='bin_temp/')
