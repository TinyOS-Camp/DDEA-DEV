import os
import sys
import json
from datetime import datetime
import time
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylab as pl
import pickle


UUID_FILE = 'uuid-list.dat'
DATA_FOLDER = 'readings_1year/'
DATA_EXT = '.dat'

SCRIPT_DIR = os.path.dirname(__file__)

def saveObjectBinary(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadObjectBinary(filename):
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return obj

def upsampling(readings):

    delta_t = 1000
    start_time = "2013/11/01-00:00:00"
    end_time = "2013/11/07-23:59:59"
    ts_start =  int(time.mktime(datetime.strptime(start_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000)
    ts_end = int(time.mktime(datetime.strptime(end_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000)

    
    if len(readings) == 0:
        return
    
    #temp_readings = [[ts_start,0.0]] + readings + [[ts_end, 0.0]]    
    temp_readings = [[ts_start,0.0]]
    min_val = 0.0
    max_val = 0.0
    for pair in readings:
        if pair[1] is None:
            continue
        
        if int(pair[0]) > ts_end or int(pair[0]) < ts_start:
            continue
        if min_val > float(pair[1]):
            min_val = float(pair[1])
            
        if max_val < float(pair[1]):
            max_val = float(pair[1])
            
        if min_val == max_val:
            return []
         
        temp_readings.append(pair)
    temp_readings.append([ts_end, 0.0])
    if temp_readings[0][0] == temp_readings[1][0]:
        temp_readings.pop(0)     
        
    if temp_readings[-1][0] == temp_readings[-2][0]:
        temp_readings.pop()
        
    interpolated_readings = []
    #for pair in readings:
    for i in range(0, len(temp_readings) - 1):
        
        pair = [int(temp_readings[i][0]), float(temp_readings[i][1])]
        next_pair = [int(temp_readings[i + 1][0]), float(temp_readings[i + 1][1])]
        if len(pair) < 2:
            continue
         #print datetime.fromtimestamp(float(pair[0]) / 1000.0), pair[1]
        ### append the first point ###
        interpolated_readings.append(pair)
        temp_ts = pair[0] + delta_t
        while temp_ts < next_pair[0]:
            
            ### interpolate ###
            interpolated_reading = (next_pair[1] * (temp_ts - pair[0]) + pair[1] * (next_pair[0] - temp_ts)) / (next_pair[0] - pair[0])
            interpolated_readings.append([int(temp_ts), interpolated_reading])
            print pair, next_pair, int(temp_ts), interpolated_reading
            temp_ts = temp_ts + delta_t
            #print 'interpolated: ' + str(datetime.fromtimestamp(temp_ts / 1000.0)) + ' ' + str(interpolated_reading)
    #print len(interpolated_readings)
    return interpolated_readings
    
    #print len(readings)

def downsampling(readings):
    
    if len(readings) == 0:
        return []

    delta_t = 600000 # 600000 ms = 600s = 10 mins
    start_time = "2013/11/01-00:00:00"
    end_time = "2013/11/30-23:59:59"
    ts_start =  int(time.mktime(datetime.strptime(start_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000)
    ts_end = int(time.mktime(datetime.strptime(end_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000)

    down_readings = []
    delta_t = 600
    readings_len = len(readings)
    for i in range(0, readings_len, delta_t):
        j = min(i + delta_t, readings_len)
        sum = 0.0
        for k in range(i,j):
            sum = sum + readings[k][1]

        pair = [readings[i][0], sum / delta_t]
        
        down_readings.append(pair)
        
    return down_readings

def group_uuids(uuid_list):
    sensors_metadata = []
    for uuid in uuid_list:
        metadata_filepath = os.path.join(SCRIPT_DIR, 'metadata/meta_' + uuid + '.dat')
        
        ### open metadata file ###
        with open(str(metadata_filepath)) as f:
            #metadata = f.read().strip()
            #sensors_metadata.append(metadata)
            sensor_metadata = json.load(f)
            sensors_metadata.append((uuid, sensor_metadata[0]['Path']))
            
    sensors_metadata.sort(key=lambda tup: tup[1])
    #print sensors_metadata
    return sensors_metadata

### delta_t in ms ; max_sr in ms ###
### start_time = "2013/11/01-00:00:00"
### end_time = "2013/11/07-23:59:59"

def load_uuid_list():
    uuid_list = []
    uuid_filepath = os.path.join(SCRIPT_DIR, UUID_FILE)
    temp_uuid_list = open(uuid_filepath).readlines()
    all_sensors_readings = []
    for uuid in temp_uuid_list:
        uuid = uuid.strip()
        if uuid == "":
            continue
        uuid_list.append(uuid)
        
    return uuid_list
        
def print_readings(uuid):
    
    sensor_filepath = os.path.join(SCRIPT_DIR, DATA_FOLDER + uuid + '.dat')
    
    sensors_readings = []
    with open(str(sensor_filepath)) as f:
        # sensors_metadata.append(f.read())
        json_readings = json.load(f)
        sensors_readings = json_readings[0]['Readings']        
    
    if len(sensors_readings) == 0:
            return
    for pair in sensors_readings:
        if pair[1] is None:
            continue
        ts = pair[0]
        readable_ts = datetime.fromtimestamp(int(ts) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        reading = pair[1]
        print str(ts), str(readable_ts), reading
        
def compute_sampling_density(uuid, start_time, end_time, delta_t, max_sr):
        
    ### for testing ###
    #start_time = "2013/11/01-00:00:00"
    #end_time = "2013/11/07-23:59:59"
    
    start_ts =  int(time.mktime(datetime.strptime(start_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000)
    end_ts = int(time.mktime(datetime.strptime(end_time, "%Y/%m/%d-%H:%M:%S").timetuple()) * 1000)
    
    if (end_ts - start_ts) * 1.0 / delta_t == int ( math.floor((end_ts - start_ts) / delta_t)):        
        num_intervals =  int ( (end_ts - start_ts) / delta_t) + 1
    else:        
        num_intervals = int(math.ceil((end_ts - start_ts) * 1.0 / delta_t))
            
    sampling_density = [0] * num_intervals    
     
    ### open reading of uuid ###
    sensor_filepath = os.path.join(SCRIPT_DIR, DATA_FOLDER + uuid + DATA_EXT)
    with open(str(sensor_filepath)) as f:
        # sensors_metadata.append(f.read())
        json_readings = json.load(f)
        sensors_readings = json_readings[0]['Readings']
        if len(sensors_readings) == 0:
            return sampling_density
            ### return []
    
    for pair in sensors_readings:
        curr_ts = int(pair[0])
        #reading = float(pair[1])
        if curr_ts < start_ts:
            continue
        
        if curr_ts > end_ts:
            break
                
        if pair[1] is None:
            continue
        
        curr_reading_index = int( (curr_ts - start_ts) / delta_t)
        sampling_density[curr_reading_index] = sampling_density[curr_reading_index] + 1

    ### compute density
    max_num_samples = delta_t / max_sr
    
    for i in range(0, num_intervals):
        sampling_density[i] = sampling_density[i] * 1.0 / max_num_samples        
        
    return sampling_density

def compute_sampling_density_matrix(start_time, end_time, delta_t, max_sr):
    
    uuid_list = load_uuid_list()
    sampling_density_matrix = []
    for uuid in uuid_list:
        sampling_density = compute_sampling_density(uuid, start_time, end_time, delta_t, max_sr)
        if len(sampling_density) == 0:
            continue
        sampling_density_matrix.append(sampling_density)
        
    return sampling_density_matrix

def visualize_density_matrix(sampling_density_matrix):
    plt.imshow(sampling_density_matrix, interpolation="nearest", cmap=pl.cm.spectral)
    pl.savefig('density.png',dpi=400, bbox_inches='tight')

sys_argv = sys.argv
#uuid = "d8b6fed7-40ae-54da-a212-010c3e040321"
start_time = "2013/11/01-00:00:00"
end_time = "2013/11/30-23:59:59"
max_sr = 60000 ### 1000 ms = 1s
delta_t = 1800000 ### 600000 ms = 10 min
if len(sys_argv) == 5:        
    start_time = sys_argv[1]
    end_time = sys_argv[2]
    delta_t = int(sys_argv[3])
    max_sr = int(sys_argv[4])

### compute sampling density matrix and visualize    
sampling_density_matrix = np.array(compute_sampling_density_matrix(start_time, end_time, delta_t, max_sr))    
visualize_density_matrix(sampling_density_matrix)
    
