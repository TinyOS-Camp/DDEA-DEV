import types

import time
from datetime import datetime
import socket
import numpy as np
import pickle
import cPickle
import random
import resource
import mmap
import os
import dill

##################################################
# UDP TOOL
##################################################
def udpRx(ip, port):
    '''Simple UDP transmission template'''
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    while True:
        data, addr = sock.recvfrom(1000)
        print data

def udpTx(ip, port, message):
    '''Simple UDP receiving template'''
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message, (ip, port))

##################################################
# READ FILE
##################################################
def read(filename, index, delim="", ctype=""):
    openfile = open(filename, "r")
    lines = openfile.readlines()
    openfile.close()

    _ctype = type(ctype)
    value = []

    for line in lines:
        tmp = line.rstrip().rsplit(delim)
        for i in index:
            val = tmp[i]
            if _ctype == types.IntType:
                convertedval = int(val)
            elif _ctype == types.FloatType:
                convertedval = float(val)
            elif _ctype == types.StringType:
                convertedval = val
            elif _ctype == types.BooleanType:
                convertedval = bool(val)
            else:
                convertedval = val
            value.append(convertedval)

    return value


##################################################
# BINARY OBJECT
##################################################

def saveObjectBinary(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadObjectBinary(filename):
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return obj

### Faster version of loading and saving objects into binaries
### using cPickle library (modified by Khiem)
def saveObjectBinaryFast(obj, filename):
    with open(filename, "wb") as output:
        cPickle.dump(obj, output, cPickle.HIGHEST_PROTOCOL)

def loadObjectBinaryFast(filename):
    with open(filename, "rb") as input:
        obj = cPickle.load(input)
    return obj 


### MMAP version
def saveObjectBinMmap(obj, filename):
    with os.open(filename, os.O_RDWR) as f:
        try:

            buf = mmap.mmap(f.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_WRITE)
            #buf.seek(0)
            try:
                cPickle.dump(obj, buf, cPickle.HIGHEST_PROTOCOL)
            finally:
                buf.flush()
                buf.close()
        finally:
            f.close()


def loadObjectBinMmap(filename):
    with open(filename, os.O_RDONLY) as f:
        try:
            # goto the end of file
            f.seek(0, 2)
            size = f.tell()
            f.seek(0, 0)
            m = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)
            try:
                obj = pickle.loads(m)
            finally:
                m.close()
        finally:
            f.close()
    return obj

#multi-processing version
def dill_save_obj(obj, filename):
    with open(filename, "wb") as output:
        dill.dump(obj, output, dill.HIGHEST_PROTOCOL)
        #dill.dump(obj, output)

def dill_load_obj(filename):
    with open(filename, "rb") as input:
        obj = dill.load(input)
    return obj

#multi-processing version
def saveObject(obj, filename):
    with open(filename, "wb") as f:
        f.write(obj)

def loadObject(filename):
    with open(filename, os.O_RDONLY) as f:
        f.readall()


##################################################
# OTHERS
##################################################

def str2datetime(s):
    parts = s.split('.')
    dt = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
    
    if len(parts) == 1:
        return dt
    elif len(parts) == 2:
        return dt.replace(microsecond=int(parts[1]))

##################################################
# KALMAN FILTER
##################################################

def kalman(y, Q, R):
    # Q: process variance
    # R: estimate of measurement variance

    length = len(y)
    yhat = np.zeros(length)          # a posteri estimate of x
    P = np.zeros(length)             # a posteri error estimate
    yhatminus = np.zeros(length)     # a priori estimate of x
    Pminus = np.zeros(length)        # a priori error estimate
    K = np.zeros(length)             # gain or blending factor

    yhat[0] = y[0]
    P[0] = 1.0
    for k in range(1, length):
        # time update
        yhatminus[k] = yhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        yhat[k] = yhatminus[k]+K[k]*(y[k]-yhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    return yhat

##################################################
# INTERPOLATE
##################################################

def interpolate(data, ts, newts):
    i, j = 0, 0
    newdata = []

    for j in range(len(newts)):
        if newts[j] < ts[1]:
            a = 0
        elif newts[j] > ts[-2]:
            a = -2
        else:
            while ts[i+1] < newts[j]:
                i += 1
            a = i
        tmp = 1.0 * (data[a+1] - data[a]) * (newts[j] - ts[a+1]) / (ts[a+1] - ts[a]) + data[a+1]
        newdata.append(tmp)
    return newdata

##################################################
# MOVING AVERAGE
##################################################

def average(data, wsize):
    newdata = []
    for i in range(len(data)):
        end = max(i, wsize)
        s = sum(data[end - wsize:end])
        newdata.append(s * 1.0 / wsize)
    return newdata

##################################################
# DIFFERENTIATION
##################################################

def df(data):
    return [data[i+1] - data[i] for i in range(len(data)-1)]

##################################################
# MISC.
##################################################
def rand(fromval, toval):
    return fromval + random.random() * (toval - fromval)

def drange(start, stop, step):
    tmp = []
    currval = start
    while currval < stop:
        tmp.append(currval)
        currval += step

    return tmp

def diff(a, b):
    assert len(a) == len(b)
    return [a[i] - b[i] for i in range(len(a))]

def print_report(start_time):
    elap = time.time() - start_time
    print "Elapse time: %d min %.3f sec"%(int(elap / 60), elap % 60)
    print "Mem usage  : %.3f MB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)