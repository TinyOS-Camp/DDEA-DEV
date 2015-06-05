# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:17:45 2014

@author: NGO Quang Minh Khiem
@e-mail: khiem.ngo@adsc.com.sg

"""
import os
import mytool as mt
import json
import datetime as dt
import pytz
from dateutil import tz
import time
import numpy as np
import multiprocessing as mp

#import tz

UUID_FILE = 'uuid-list.dat'
DATA_FOLDER = 'readings_1year/'
#BIN_FOLDER = 'sdh_binfiles/'
BIN_FOLDER = 'Binfiles_2013/'
DATA_EXT = '.dat'
BIN_EXT = '.bin'

SCRIPT_DIR = os.path.dirname(__file__)

def load_uuid_list():
    uuid_list = []
    uuid_filepath = os.path.join(SCRIPT_DIR, UUID_FILE)
    #temp_uuid_list = open(uuid_filepath).readlines()
    
    with open(uuid_filepath,'r') as f:
        temp_uuid_list = f.readlines()
        
    for uuid in temp_uuid_list:
        uuid = uuid.strip()
        if uuid == "":
            continue
        uuid_list.append(uuid)
        
    return uuid_list

def convert_to_bin(uuid,in_dir=DATA_FOLDER,out_dir=BIN_FOLDER):
    start = time.time()
    in_file = os.path.join(SCRIPT_DIR, DATA_FOLDER + uuid + DATA_EXT)
    out_file = os.path.join(SCRIPT_DIR, BIN_FOLDER + uuid + BIN_EXT)
    
    faulty_list = []
    # check if file exists
    try:
        if os.path.isfile(out_file) and mt.loadObjectBinaryFast(out_file):            
            print uuid
            return
    except:
        # here, file is either not existing, or corrupted
        pass
    
    try:
        with open(in_file,'r') as f:
            json_readings = json.load(f)
            sensor_readings = json_readings[0]['Readings']
            
        if len(sensor_readings) == 0:
            return
    
        ts_list = []
        value_list = []
    
        for pair in sensor_readings:
            ts = pair[0] / 1000
            if pair[1] is None:
                continue
            value = float(pair[1])
            #local_dt = dt.datetime.fromtimestamp(ts).replace(tzinfo=tz.gettz('UTC')).astimezone(pytz.timezone(zone))
            #utc_dt = dt.datetime.fromtimestamp(ts)
            local_dt = dt.datetime.fromtimestamp(ts).replace(tzinfo=tz.gettz('UTC')).astimezone(pytz.timezone('US/Pacific'))
            time_tup = local_dt.timetuple()
            #print time_tup[5], time_tup[4], time_tup[3], time_tup[6], time_tup[2], time_tup[1]
            
            value_list.append(value)
            ts_list.append([local_dt,time_tup[5], time_tup[4], time_tup[3], time_tup[6], time_tup[2], time_tup[1]])

        # Save complete bin file
        data = {"ts":np.array(ts_list), "value":np.array(value_list)}
        mt.saveObjectBinaryFast(data, out_file)
        
        # print some stats: time and filesize
        end = time.time()
        filesize = os.path.getsize(out_file)
        print "%s: %s (%.3f MB) in %.3f secs"%(dt.datetime.now(), out_file, filesize * 1.0 / 10**6, end - start)
        
    except:
        print '=== ERROR: ' + uuid + ' ==='
        faulty_list.append(uuid)
        #return
    
    mt.saveObjectBinaryFast(np.array(faulty_list),'faulty_list.bin')

if __name__ == '__main__':
    num_cores = mp.cpu_count()
    uuid_list = load_uuid_list()
    p = mp.Pool(num_cores)
    p.map(convert_to_bin,uuid_list)
    p.close()
    p.join()
    #uuid = '7aa3d7da-6d45-5205-92b2-4735cc25a6e6'
    
    #convert_to_bin(uuid)
