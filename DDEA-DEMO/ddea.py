#!/usr/bin/python
# To force float point division
from __future__ import division

"""
Author : Deokwooo Jung
E-mail : deokwoo.jung@gmail.com

======================================================================
Learning and Visualizing the BMS sensor-time-weather data structure
======================================================================
This example employs several unsupervised learning techniques to extract
the energy data structure from variations in Building Automation System (BAS) 
and historial weather data.
The fundermental timelet for analysis are 15 min, referred to as Q. 
** currently use H (Hour) as a fundermental timelet, need to change later **

The following analysis steps are designed and to be executed. 

Data Pre-processing
--------------------------
- Data Retrieval and Standardization
- Outlier Detection
- Interpolation 

Data Summarization
--------------------------
- Data Transformation
- Sensor Clustering

Model Discovery Bayesian Network
--------------------------
- Automatic State Classification
- Structure Discovery and Analysis
"""

# General Modules

# Custom libraries
import datetime as dt
import mytool as mt
from shared_constants import *
import time
from quasar_url_reader import read_sensor_data
from ddea_proc import ddea_process
import json
##################################################################
# Processing Configuraiton Settings
##################################################################
#TODO : this whole section of selecting a building should be wrapped in param
# Building keyword.
bldg_key = 'GW2'
pname_key = '_POWER_'


# Setting Analysis period  where ANS_START_T and ANS_START_T are the starting and
# and the ending timestamp.
ANS_START_T = dt.datetime(2013, 6, 1, 0)
ANS_END_T = dt.datetime(2013, 7, 1, 0)

# Setting for analysis time interval where all BEMS and weather data is aligned
# for a slotted time line quantized by TIMELET_INV. 
TIMELET_INV = dt.timedelta(minutes=15)

print TIMELET_INV, 'time slot interval is set for this data set !!'
print '-------------------------------------------------------------------'

print "Clean up old output..."
mt.remove_all_files(FIG_DIR)
mt.remove_all_files(JSON_DIR)
mt.remove_all_files(PROC_OUT_DIR)

FILEHASH = \
    {'2e43475f-5048-4153-4531-5f4143544956': 'CG_PHASE1_ACTIVE_POWER_M'
    ,'2e43475f-5048-4153-4532-5f504f574552': 'CG_PHASE2_POWER_FACTOR_M'
    ,'2e43475f-5048-4153-4533-5f5245414354': 'CG_PHASE3_REACTIVE_POWER_M'
    ,'2e43475f-5048-4153-4531-5f504f574552': 'CG_PHASE1_POWER_FACTOR_M'
    ,'2e43475f-5048-4153-4532-5f5245414354': 'CG_PHASE2_REACTIVE_POWER_M'
    ,'2e43475f-5359-5354-454d-5f4143544956': 'CG_SYSTEM_ACTIVE_POWER_M'
    ,'2e43475f-5048-4153-4531-5f5245414354': 'CG_PHASE1_REACTIVE_POWER_M'
    ,'2e43475f-5048-4153-4533-5f4143544956': 'CG_PHASE3_ACTIVE_POWER_M'
    ,'2e43475f-5359-5354-454d-5f504f574552': 'CG_SYSTEM_POWER_FACTOR_M'
    ,'2e43475f-5048-4153-4532-5f4143544956': 'CG_PHASE2_ACTIVE_POWER_M'
    ,'2e43475f-5048-4153-4533-5f504f574552': 'CG_PHASE3_POWER_FACTOR_M'
    ,'2e43475f-5359-5354-454d-5f5245414354': 'CG_SYSTEM_REACTIVE_POWER_M'}

#----------------------------- DATA PRE-PROCESSING -----------------------------
#  Retrieving a set of sensors with specified key
print '#' * 80
print 'DATA PRE-PROCESSING FROM QUASAR WITH KEY ', bldg_key, '...'
print '#' * 80

start_time = int(time.mktime(ANS_START_T.timetuple()))
end_time = int(time.mktime(ANS_END_T.timetuple()))

sensor_hash = mt.sensor_name_uid_dict(bldg_key, FILEHASH)

mt.saveObjectBinaryFast(sensor_hash, PROC_OUT_DIR + 'gw2_sensor_hash.bin')

sensor_data = read_sensor_data(sensor_hash, start_time, end_time)

ddea_process(sensor_hash, sensor_data, start_time, end_time, TIMELET_INV, bldg_key, pname_key)


