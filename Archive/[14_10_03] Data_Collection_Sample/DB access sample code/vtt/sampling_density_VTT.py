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


######
### Configurations
######
UUID_FILE = 'finland_ids.csv'
#DATA_FOLDER = 'VTT_week/'
DATA_FOLDER = 'data_year/'
DATA_EXT = '.csv'
SCRIPT_DIR = os.path.dirname(__file__)


def saveObjectBinary(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def loadObjectBinary(filename):
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return obj


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
    
    for line in temp_uuid_list:
        tokens = line.strip().split(',')
        if len(tokens) == 0:
            continue
        uuid_list.append(tokens[0].strip())
        
    return uuid_list


def print_readings(uuid):
    
    sensor_filepath = os.path.join(SCRIPT_DIR, 'readings/' + uuid + '.dat')
    
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
     
    ###### open reading of uuid - BERKELEY SDH BUILDING ######
#     sensor_filepath = os.path.join(SCRIPT_DIR, 'readings/' + uuid + '.dat')
#     with open(str(sensor_filepath)) as f:
#         # sensors_metadata.append(f.read())
#         json_readings = json.load(f)
#         sensors_readings = json_readings[0]['Readings']
#         if len(sensors_readings) == 0:
#             return sampling_density
    
    ###### open reading of uuid - VTT FINLAND ######
    sensor_filepath = os.path.join(SCRIPT_DIR, DATA_FOLDER + uuid + DATA_EXT)
    lines = open(str(sensor_filepath)).readlines()
    sensors_readings = []
    for line in lines:
        pair = []
        if line == "":
            continue
        
        tokens = line.strip().split(',')
        if len(tokens) < 2:
            continue
        #[curr_date, curr_time] = tokens[0].split(' ')
        #print curr_date.strip() + '-' + curr_time.strip()
        ts =  int(time.mktime(datetime.strptime(tokens[0].strip(), "%Y-%m-%d %H:%M:%S").timetuple()) * 1000)
        reading = float(tokens[1].strip())
        
        pair.append(ts)
        pair.append(reading)
        
        #print tokens[0].strip(), str(ts), str(reading)
        
        # sensors_metadata.append(f.read())
        
    
    ###for pair in sensors_readings:
        
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
    uuid_list = uuid_list[0:1000]
    sampling_density_matrix = []
    for uuid in uuid_list:
        sampling_density = compute_sampling_density(uuid, start_time, end_time, delta_t, max_sr)
        if len(sampling_density) == 0:
            continue
        sampling_density_matrix.append(sampling_density)
        
    return sampling_density_matrix


def visualize_density_matrix(sampling_density_matrix):
    plt.imshow(sampling_density_matrix, interpolation="nearest", cmap=pl.cm.spectral)
    pl.savefig('density.png', bbox_inches='tight')

######
### Example 
######

#uuid = "GW1.HA1_AS_TE_AH_FM"
start_time = "2013/11/01-00:00:00"
end_time = "2013/11/07-23:59:59"
max_sr = 300000     ### 1000 ms = 1s, 5mins
delta_t = 1200000   ### ms ; 20 mins
sys_argv = sys.argv

if len(sys_argv) == 5:        
    
    start_time = sys_argv[1]
    end_time = sys_argv[2]
    delta_t = int(sys_argv[3])
    max_sr = int(sys_argv[4])

### compute sampling density matrix and visualize     
sampling_density_matrix = np.asarray(compute_sampling_density_matrix(start_time, end_time, delta_t, max_sr))
visualize_density_matrix(sampling_density_matrix)
