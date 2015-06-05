import os
import sys
import mytool as mt
import traceback
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
import multiprocessing as mp
from multiprocessing import Pool
from datetime import datetime

FOLDER = './'

def foo(filename):
    start = time.time()
    new_filename = filename[:-4] + '.bin'
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

def check_file(filename):
    try:
        temp = mt.loadObjectBinaryFast(filename)
        print 'OK ' + filename
    except:
        print '====== NOT OK ' + filename
        return (filename,False)

    return (filename, True)
if __name__ == '__main__':
    """
    filenames = glob.glob("Csvfiles_new2/*.csv")
    filenames.sort()
    p = Pool(12)
    p.map(foo, filenames)
    p.close()
    p.join()
    """

    
    filenames = glob.glob("Binfiles_new3/*.bin")
    faulty_files = []
    faulty_dict = {}

    """
    for filename in filenames:
        try:
            temp = mt.loadObjectBinaryFast(filename)
            print 'OK ' + filename
        except:
            print '====== NOT OK ' + filename
            faulty_files.append(filename)

    mt.saveObjectBinaryFast(faulty_files,'faulty_files.bin')
    """
    p = Pool(12)    
    faulty_dict = dict(p.map(check_file,filenames))
    p.close()
    p.join()
    mt.saveObjectBinaryFast(faulty_dict,'faulty_dict.bin')

    """
    filenames = mt.loadObjectBinaryFast('faulty_files.bin')
    filenames = [fn.split('/')[-1][:-4] for fn in filenames]
    print filenames

    for filename in filenames:
        try:
            foo(filename + '.csv')
            print 'OK ' + filename
        except:
            print '====== NOT OK ' + filename
    """
