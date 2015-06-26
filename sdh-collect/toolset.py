
import types, time, socket, pickle, cPickle, random, resource, mmap, os
from datetime import datetime
import numpy as np

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
def saveObject(obj, filename):
    with open(filename, "wb") as f:
        f.write(obj)

def loadObject(filename):
    with open(filename, os.O_RDONLY) as f:
        f.readall()