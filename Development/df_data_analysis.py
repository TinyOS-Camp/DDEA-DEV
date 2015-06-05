"""
==============================================
Visualizing the enegy-sensor-weather structure
==============================================

This example employs several unsupervised learning techniques to extract
the energy data structure from variations in Building Automation System (BAS) 
and historial weather data.

The fundermental timelet for analysis are 15 min, referred to as Q. 
** currently use H (Hour) as a fundermental timelet, need to change later **


Learning a graph structure
--------------------------

We use sparse inverse covariance estimation to find which quotes are
correlated conditionally on the others. Specifically, sparse inverse
covariance gives us a graph, that is a list of connection. For each
symbol, the symbols that it is connected too are those useful to explain
its fluctuations.

Clustering
----------

We use clustering to group together quotes that behave similarly. Here,
amongst the :ref:`various clustering techniques <clustering>` available
in the scikit-learn, we use :ref:`affinity_propagation` as it does
not enforce equal-size clusters, and it can choose automatically the
number of clusters from the data.

Note that this gives us a different indication than the graph, as the
graph reflects conditional relations between variables, while the
clustering reflects marginal properties: variables clustered together can
be considered as having a similar impact at the level of the full stock
market.

Embedding in 2D space
---------------------

For visualization purposes, we need to lay out the different symbols on a
2D canvas. For this we use :ref:`manifold` techniques to retrieve 2D
embedding.


Visualization
-------------

The output of the 3 models are combined in a 2D graph where nodes
represents the stocks and edges the:

- cluster labels are used to define the color of the nodes
- the sparse covariance model is used to display the strength of the edges
- the 2D embedding is used to position the nodes in the plan

This example has a fair amount of visualization-related code, as
visualization is crucial here to display the graph. One of the challenge
is to position the labels minimizing overlap. For this we use an
heuristic based on the direction of the nearest neighbor along each
axis.
"""
#print(__doc__)
# Author: Deokwooo Jung deokwoo.jung@gmail.compile
from __future__ import division # To forace float point division
import os
import sys
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

import pylab as pl
from scipy import signal
from scipy import stats
from scipy import fftpack

from shared_constants import *

import matplotlib.pyplot as plt
from multiprocessing import Pool
#from datetime import datetime
import datetime as dt
from dateutil import tz
import shlex, subprocess
import mytool as mt
import time
import retrieve_weather as rw
import itertools
import mpl_toolkits.mplot3d.axes3d as p3
import calendar
import random
from sklearn.gaussian_process import GaussianProcess
from sklearn import cluster, covariance, manifold # Machine Learning Packeage
from sklearn import metrics
from matplotlib.collections import LineCollection
from classify_sensors import get_sim_mat

# Convert a unix time u to a datetime object d, and vice versa
def unix_to_dtime(u): return dt.datetime.utcfromtimestamp(u)
def dtime_to_unix(d): return calendar.timegm(d.timetuple())

# This option let you use  data_dict object saved in hard-disk from the recent execution
IS_USING_SAVED_DICT=0
# This option shoud be set one unless you are debugging this code
DO_DATA_PREPROCESSING=1
###############################################################################
# Constant global variables
###############################################################################
"""
# in seconds
MIN=60; HOUR=60*MIN; DAY=HOUR*24; MONTH=DAY*31
# Hour, Weekday, Day, Month
MIN_IDX=0;HR_IDX=1; WD_IDX=2; MD_IDX=3 ;MN_IDX=4
monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
weekDict={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thur', 4:'Fri', 5:'Sat', 6:'Sun'}
"""
# Define the period for analysis  - year, month, day,hour
# Note: The sample data in the currently downloaded files are from 1 Apr 2013 to
#       30 Nov 2013.
# This is the best data set
#ANS_START_T=dt.datetime(2013,7,8,0)
#ANS_END_T=dt.datetime(2013,7,15,0)

ANS_START_T=dt.datetime(2013,7,1,0)
ANS_END_T=dt.datetime(2013,7,3,0)

#ANS_END_T=dt.datetime(2013,8,30,0)
# Interval of timelet, currently set to 1 Hour
TIMELET_INV=dt.timedelta(hours=1)
# UTC time of weather data
from_zone = tz.gettz('UTC')
# VTT local time
to_zone = tz.gettz('Europe/Helsinki')
# Multi-dimensional lists of hash tables 
time_slots=[]
start=ANS_START_T
while start < ANS_END_T:
    #print start
    time_slots.append(start)
    start = start + TIMELET_INV
###############################################################################
# Data Dictionary Functions and Global Variables
###############################################################################


# Data dictionary
# All sensor and weather data is processed and structred into 
# a consistent single data format -- Dictionary
data_dict={}

def data_dict_purge(purge_list):
    for key in purge_list:
        print 'purge', key
        if key in data_dict.keys():
            data_dict.pop(key,None)
#data_dict_purge(weather_list)

def verify_data_format(key_list):
    # Verify there is no  [] or N/A in the list
    # Only FLoat or Int format is allowed
    print 'Checking any inconsisent data format.....'
    print '---------------------------------'
    list_of_wrong_data_format=[]
    for key in key_list:
        print 'checking ', key, '...'
        for i,samples in enumerate(data_dict[key][1]):
            for j,each_sample in enumerate(samples):
                if each_sample==[]:
                    list_of_wrong_data_format.append([key,i,j])
                    print each_sample, 'at', time_slots[j], 'in', key
                elif (isinstance(each_sample,int)==False and isinstance(each_sample,float)==False):
                    list_of_wrong_data_format.append([key,i,j])                    
                    print each_sample, 'at', time_slots[j], 'in', key
    print '---------------------------------'                    
    if len(list_of_wrong_data_format)==0:
        print 'No inconsistent data format'
    return  list_of_wrong_data_format
    
def minidx_to_secs(min_t):
    sec_t_ar=[]
    #sec_a=[]
    sec_t=np.array(min_t)*60
    sec_ar=np.zeros(len(sec_t))
    dup_min_cnt=0
    prv_min_idx=-1;cur_min_idx=-1
    pt_str=0;
    #pt_end=0
    for j,min_idx in enumerate(min_t):
        cur_min_idx=min_idx
        if cur_min_idx==prv_min_idx:
            dup_min_cnt=dup_min_cnt+1
            sec_ar[j]=dup_min_cnt
        else:
            if sec_ar[j-1]==0:
                sec_ar[j-1]=60/2
            else:
                sec_ar[pt_str:j]=sec_ar[pt_str:j]*(60/sec_ar[j-1])
                sec_ar[pt_str]=dup_min_cnt
                dup_min_cnt=0;pt_str=j
                
        prv_min_idx=cur_min_idx
        sec_t_ar=sec_t+sec_ar
    return sec_t_ar


###############################################################################
# Application Functions 
###############################################################################
def daterange(start, stop, step=dt.timedelta(days=1), inclusive=False):
  # inclusive=False to behave like range by default
  if step.days > 0:
    while start < stop:
      yield start
      start = start + step
      # not +=! don't modify object passed in if it's mutable
      # since this function is not restricted to
      # only types from datetime module
  elif step.days < 0:
    while start > stop:
      yield start
      start = start + step
  if inclusive and start == stop:
    yield start

###############################################################################
# Plotting tool
###############################################################################
def ref_time_matrix(t_slots):
    # Return reference time matrix for time_slots
    # Minute,Hour, Weekday, Day, Month - 5 column matrix
    time_mat=np.zeros([len(t_slots),5])
    for i, time_sample in enumerate(t_slots):
        time_mat[i,MIN_IDX]=time_sample.minute
        time_mat[i,HR_IDX]=time_sample.hour
        time_mat[i,WD_IDX]=time_sample.weekday()
        time_mat[i,MD_IDX]=time_sample.day
        time_mat[i,MN_IDX]=time_sample.month
    return time_mat
        
def plotting_data(plot_list):
    # Month indicator    
    num_col=int(np.ceil(np.sqrt(len(plot_list))))
    num_row=num_col
    
    time_mn_diff=np.diff(time_mat[:,MN_IDX])
    m_label_idx=time_mn_diff.nonzero()[0]; m_label_str=[]
    for m_num in time_mat[m_label_idx,MN_IDX]:
        m_label_str.append(monthDict[m_num])
    time_wk_diff=np.diff(time_mat[:,WD_IDX])
    w_label_idx=time_wk_diff.nonzero()[0]; w_label_str=[]
    for w_num in time_mat[w_label_idx,WD_IDX]:
        w_label_str.append(weekDict[int(w_num)])
    for k,sensor in enumerate(plot_list):
        num_samples=[];  mean_samples=[]
        for i,(t,samples) in enumerate(zip(time_slots,data_dict[sensor][1])):
            #import pdb;pdb.set_trace()
            num_samples.append(len(samples))
            mean_samples.append(np.mean(samples))
        plt.figure('Sampling Density')
        plt.subplot(num_col, num_row,k+1)
        plt.plot(time_slots,num_samples)
        plt.title(sensor,fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('# Samples/Hour',fontsize=8)
        if k<len(plot_list)-1:        
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
        plt.figure('Hourly Average')
        plt.subplot(num_col, num_row,k+1)
        plt.plot(time_slots,mean_samples)
        plt.title(sensor,fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('Avg Val/Hour',fontsize=8)
        if k<len(plot_list)-1:        
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
        #plt.xticks(w_label_idx.tolist(),w_label_str,fontsize=8)
        #plt.text(m_label_idx, np.max(num_samples)*0.8, m_label_str, fontsize=12)
        plt.figure('Sampling Intervals')
        t_secs_diff=np.diff(data_dict[sensor][2][0])
        plt.subplot(num_col, num_row,k+1)
        plt.plot(t_secs_diff)
        plt.title(sensor,fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('Samping Interval (secs)',fontsize=8)
        if k<len(plot_list)-1:        
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
    print ' End of Plotting'


###############################################################################
# Parsing sensor data 
###############################################################################       
def get_val(filename):
    data = mt.loadObjectBinary(filename)
    sensor_val = data["value"]
    time_val = data["ts"]
    #print 'list of input bin files: '
    return sensor_val,time_val

def get_val_timelet(filename,t_slots):
    print ' get_val_timelet'    
    data = mt.loadObjectBinary(filename)
    sensor_val = data["value"]
    time_val = data["ts"]
    sensor_read=[[] for i in range(len(t_slots))] # Creat the list of lists for value
    mtime_read=[[] for i in range(len(t_slots))] # Creat the list of lists for minute index
    for t_sample, v_sample in zip(time_val,sensor_val):
        
        ##################################################################################        
        #Replaced with a faster code
        # If data in 2013 is only available after Aprile, Otherwise it is 2014 data            
        if t_sample[MN_IDX]>3:
            temp_dt=dt.datetime(2013,t_sample[MN_IDX],t_sample[MD_IDX],t_sample[HR_IDX])
        else:
            temp_dt=dt.datetime(2014,t_sample[MN_IDX],t_sample[MD_IDX],t_sample[HR_IDX])
        try:
            idx=t_slots.index(temp_dt)
            sensor_read[idx].append(v_sample)
            mtime_read[idx].append(t_sample[MIN_IDX])
        except ValueError:
            idx=-1
        ##################################################################################
        
        """
        ##################################################################################
        # New Code        
        tmp_year = 2013 if t_sample[MN_IDX] > 3 else 2014
        temp_dt = dt.datetime(tmp_year, t_sample[MN_IDX], t_sample[MD_IDX], t_sample[HR_IDX])
        if temp_dt < ANS_START_T or temp_dt >= ANS_END_T:
            continue
        try:
            idx = int((temp_dt - ANS_START_T).total_seconds() / HOUR)
            sensor_read[idx].append(v_sample)
            mtime_read[idx].append(t_sample[MIN_IDX])
        except ValueError:
            idx = -1
        ##################################################################################            
        """
    return sensor_read, mtime_read

###############################################################################
# Retrive weather data from internet for the specified periods 
# prefix_order=TS (default) [Time][Sensor]
# prefix_order=ST            [Sensor][Time]
###############################################################################
###############################################################################
# Parsing sensor data
# Data samples are regularized for specified times with timelet
###############################################################################
# This is the list of non-digit symbolic weather data
# The symbolic weather data is such as Conditions (e.g Cloudy or Clear)
# and Events (e.g. Rain or Fog ...)
# Those symblic data is replaced with integer state representation whose
# pairs are stored in a hash table using Dictionary. 
# If no data is given, key value is set to 0.
def symbol_to_state(symbol_list):
    symbol_dict={};symbol_val=[];key_val=1
    print 'start'
    for i,key_set in enumerate(symbol_list):
        symbol_val_let=[]
        for key in key_set:
            if key not in symbol_dict:
                if len(key)==0:
                    symbol_dict.update({key:0})
                    symbol_val_let.append(0)
                else:    
                    symbol_dict.update({key:key_val})
                    symbol_val_let.append(key_val)
                    key_val=key_val+1        
            else:
                symbol_val_let.append(symbol_dict[key])
        symbol_val.append(symbol_val_let)
    return symbol_val,symbol_dict

def get_weather(t_start, t_end, data_format='DICT'):
    print 'getting weater data new '    
    print 'start time:', t_start, ' ~ end time:',t_end
    data_days=[] # Date iteration given start time and end-time
    for date in daterange(t_start, t_end, inclusive=True):
        print date.strftime("%Y-%m-%d")
        temp=date.strftime("%Y,%m,%d").rsplit(',')
        data_day=rw.retrieve_data('VTT', int(temp[0]), int(temp[1]), int(temp[2]), view='d')    
        data_day=data_day.split('\n')
        if data_format=='DICT': # Formatting the weather data for  Dictionary data type
            day_sample_parse=[]        
            for h_idx,hour_sample in enumerate(data_day):
                #print hour_sample
                if h_idx==0:
                    sensor_name_list=hour_sample.split(',')
                else:
                    hour_samples=hour_sample.split(',')
                    for sample_idx,each_sample in enumerate(hour_samples):
                        sensor_name=sensor_name_list[sample_idx]
                        if sensor_name in data_dict:
                            data_dict[sensor_name].append(each_sample)
                        else:
                            data_dict.update({sensor_name:[each_sample]})
        elif data_format=='CSV': # Formatting the weather data for CSV format
            day_sample_parse=[]        
            for hour_sample in data_day:
                day_sample_parse.append(hour_sample.split(','))
            data_days.append(day_sample_parse)
        else:
            raise NameError('data_format needs to either DICT or CSV')
    if data_format=='DICT':
        return sensor_name_list
    else:
        return data_days

def get_weather_timelet(t_slots):
    print 'getting weater data new ' 
    print '------------------------------------'
    t_start=t_slots[0] 
    t_end=t_slots[-1]
    print 'start time:', t_start, ' ~ end time:',t_end
    # Date iteration given start time and end-time
    # Iterate for each day for all weather data types
    for date_idx,date in enumerate(daterange(t_start, t_end, inclusive=True)):
        print date.strftime("%Y-%m-%d")
        temp=date.strftime("%Y,%m,%d").rsplit(',')
        data_day=rw.retrieve_data('VTT', int(temp[0]), int(temp[1]), int(temp[2]), view='d')    
        # split the data into t
        data_day=data_day.split('\n')
        # Iterate for each time index(h_idx) of a day  for all weather data types
        for h_idx,hour_sample in enumerate(data_day):
            hour_samples=hour_sample.split(',')
            # Initialize weather data lists of dictionary            
            # The first row is always the list of weather data types
            if (h_idx==0) and (date_idx==0):
                sensor_name_list=hour_sample.split(',')
                for sample_idx,each_sample in enumerate(hour_samples):
                    sensor_name=sensor_name_list[sample_idx]
                    sensor_read=[[] for i in range(len(t_slots))]    
                    mtime_read=[[] for i in range(len(t_slots))] # Creat the list of lists for minute index                    
                    #data_dict.update({sensor_name:sensor_read})
                    #data_dict.update({sensor_name:zip(mtime_read,sensor_read)})
                    data_dict.update({sensor_name:[mtime_read,sensor_read]})
            elif h_idx>0:
                # 'DateUTC' is the one  
                sample_DateUTC=hour_samples[sensor_name_list.index('DateUTC')]
                # convert to UTC time to VTT local time. 
                utc_dt=dt.datetime.strptime(sample_DateUTC, "%Y-%m-%d %H:%M:%S")
                vtt_dt_aware = utc_dt.replace(tzinfo=from_zone).astimezone(to_zone)
                #import pdb; pdb.set_trace()
                # convert to offset-naive from offset-aware datetimes              
                vtt_dt=dt.datetime(*(vtt_dt_aware.timetuple()[:4]))
                cur_minute_val=vtt_dt_aware.timetuple().tm_min
                try:# time slot index a given weather sample time
                    vtt_dt_idx=t_slots.index(vtt_dt)
                    for sample_idx,each_sample in enumerate(hour_samples):
                        try:# convert string type to float time if possible
                            each_sample=float(each_sample)
                        except ValueError:
                            each_sample=each_sample
                        sensor_name=sensor_name_list[sample_idx] 
                        if sensor_name in data_dict:
                            if each_sample!='N/A' and each_sample!=[]:
                                #data_dict[sensor_name][vtt_dt_idx].append(each_sample)
                                data_dict[sensor_name][0][vtt_dt_idx].append(cur_minute_val)
                                data_dict[sensor_name][1][vtt_dt_idx].append(each_sample)
                                                                
                        else:
                            raise NameError('Inconsistency in the list of weather data')
                except ValueError:
                    vtt_dt_idx=-1
            else: # hour_sample is list of weather filed name, discard
                hour_sample=[]
    return sensor_name_list

###############################################################################
# Gaussian Process Interpooation
###############################################################################
def make_colvec(x):
    x=np.atleast_2d(x)
    return x.reshape((max(x.shape),1))
    
def gp_interpol_test(x,y,num_data_loss,y_label=[]):
    #x -input variable, y- observed variable
    #x=np.atleast_2d(x);x= x.reshape((max(x.shape),1))
    #y=np.atleast_2d(y);y= y.reshape((max(y.shape),1))    
    x=make_colvec(x);y=make_colvec(y)
    y_org=y.copy()
    infty_indice=np.nonzero(y==np.infty)[0]
    noinfty_indice=np.nonzero(y!=np.infty)[0]
    num_infty_add=np.max([0, num_data_loss-len(infty_indice)])
    y[random.sample(noinfty_indice,num_infty_add)]=np.infty
        #for row_idx,row_val in enumerate(col_val):
        #    print 'X',[row_idx, col_idx], ': ', row_val,':', X[row_idx,col_idx]
    gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, \
                         random_start=100)
    input_var=[];obs_var=[]
    for (x_val,y_val) in zip(x,y):
        if y_val!=np.infty:
            input_var.append(x_val)
            obs_var.append(y_val)
    #input_var=np.atleast_2d(input_var);input_var=input_var.reshape((max(input_var.shape),1))
    #obs_var=np.atleast_2d(obs_var);obs_var=obs_var.reshape((max(obs_var.shape),1))
    input_var=make_colvec(input_var);obs_var=make_colvec(obs_var)
    # Instanciate a Gaussian Process model    
    #import pdb;pdb.set_trace()                            
    gp.fit(input_var,obs_var)
    #new_input_var=np.atleast_2d(np.r_[input_var[0]:input_var[-1]]).T
    new_input_var=x
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(new_input_var, eval_MSE=True)
    sigma = np.sqrt(MSE)
    if len(y_label)>0:
        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.plot(input_var, obs_var, 'r.', markersize=20, label=u'Observations')
        plt.plot(new_input_var,y_org,'s',label=u'Actual')
        plt.plot(new_input_var, y_pred, 'bx-', label=u'Prediction')
        plt.fill(np.concatenate([new_input_var, new_input_var[::-1]]), \
                np.concatenate([y_pred - 1.9600 * sigma,
                               (y_pred + 1.9600 * sigma)[::-1]]), \
                alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel(y_label)
        #plt.ylim(-10, 20)
        plt.legend(loc='upper right')
    return y_pred,sigma,infty_indice

###############################################################################
    
#############################################################################
#############################################################################
# Main file to be executed
#############################################################################
#############################################################################

##############################################################################
# Reading sensor data from  BIN files ( BIN is preprocesssed)-use linux commands
###############################################################################
def get_id_dict(grep_expr):
    # This get data point ids from id descriptoin file . 
    idinfo_file_name='KMEG_output.txt'
    temp = subprocess.check_output('cat  '+idinfo_file_name+ '|'+grep_expr, shell=True)
    temp2=temp.rsplit('\r\n')
    id_dict={}
    for each_temp2 in temp2:
        if len(each_temp2):
            temp3=each_temp2.rsplit(',')
            temp4=temp3[0].rsplit(' ')
            while temp4.count('') > 0:
                temp4.remove('')
            rest_desc=[]
            for desc in temp4[1:]:
                rest_desc.append(desc)
            for desc in temp3[1:]:
                rest_desc.append(desc)
            id_dict.update({temp4[0]:rest_desc})
    return id_dict

###############################################################################
input_files=[]
plt.ion() # Interactive mode for plotting

Using_LoopUp=0
###############################################################################
# This directly searches files from bin file name
if Using_LoopUp==0:
    temp = subprocess.check_output("ls ../data_year/*.bin |grep _ACTIVE_POWER_", shell=True)    
    #temp = subprocess.check_output("ls *.bin |grep GW2.CG_SYSTEM_ACTIVE_POWER_M.bin", shell=True)        
    #temp = subprocess.check_output("ls *.bin |grep '_POWER_\|TK.*VAK'", shell=True)    
    #temp = subprocess.check_output("ls *.bin |grep '_POWER_' ", shell=True)    
    input_files =shlex.split(temp)
###############################################################################
# This look-ups id description tables and find relavant bin files.  
elif Using_LoopUp==1:
    id_dict=get_id_dict('grep kW')
    for id_name in id_dict.keys():
        binfile_name=id_name+'.bin'
        input_files.append(binfile_name)
else:
    raise NameError('Lookup file or not using, no other options')

num_files=len(input_files)
num_col_subplot=np.ceil(np.sqrt(num_files))
###############################################################################
#  Analysis script starts here ....
# List of sensors  from BMS
print 'mapping sensor list into hasing table using dictionary'


if IS_USING_SAVED_DICT==0:
    ###############################################################################
    # Data dictionary that map all types of sensor readings into a single hash table
    # Read out all sensor files in the file list
    print 'Read sensor bin files in a single time_slots referece...'
    print '----------------------------------'
    sensor_list=[]
    start__dictproc_t=time.time()         
    # Data Access is following ....    
    #data_dict[key][time_slot_idx][(min_idx=0 or values=1)]
    for i,sensor_name in enumerate(input_files):
        print 'index ',i+1,': ', sensor_name
        # sensor value is read by time 
        dict_sensor_val, dict_mtime=get_val_timelet(sensor_name,time_slots)
        sensor_list.append(sensor_name[:-4])
        #data_dict.update({sensor_name[:-4]:dict_sensor_val}) # [:-4] drops 'bin'
        data_dict.update({sensor_name[:-4]:[dict_mtime,dict_sensor_val]}) # [:-4] drops 'bin'
        end__dictproc_t=time.time()     
        #print sensor_name,'- dict.proc time is ', end__dictproc_t-start__dictproc_t    
    print ' Total dict.proc time is ', end__dictproc_t-start__dictproc_t
    

    
    # Multiprocessing Implementation
    # p = Pool(20)
    # p.map(resample, filenames)
    """
    weather_list -that is pretty much fixed from database 
    (*) is the data to be used for our analysis
    0 TimeEEST
    1 TemperatureC (*)
    2 Dew PointC (*)
    3 Humidity (*)
    4 Sea Level PressurehPa
    5 VisibilityKm
    6 Wind Direction
    7 Wind SpeedKm/h
    8 Gust SpeedKm/h
    9 Precipitationmm
    10 Events (*)
    11 Conditions (*)
    12 WindDirDegrees
    13 DateUTC
    """
    
    weather_list=get_weather_timelet(time_slots)
    # Convert symbols to Integer representaion
    data_dict['Conditions'][1],Conditions_dict=symbol_to_state(data_dict['Conditions'][1])
    data_dict['Events'][1],Events_dict=symbol_to_state(data_dict['Events'][1])
    data_dict.update({'sensor_list':sensor_list})
    data_dict.update({'weather_list': weather_list})
    data_dict.update({'Conditions_dict':Conditions_dict})
    data_dict.update({'Events_dict':Events_dict})
    mt.saveObjectBinary(data_dict,'data_dict.bin')
else:
    data_dict = mt.loadObjectBinary('data_dict.bin')
    
if DO_DATA_PREPROCESSING==1:
    # finding reference time matrix
    time_mat=ref_time_matrix(time_slots)
    # Plotting for verificaiotn
    #plotting_data(sensor_list[0:2])
    # Weather data to be used
    weather_list_used = [data_dict['weather_list'][i] for i in [1,2,3,10,11]]
    sensor_list=data_dict['sensor_list']
    # All (sensor + weather) data to be used
    data_used=weather_list_used + sensor_list
    # Verify there is no  [] or N/A in the list
    list_of_wrong_data_format=verify_data_format(data_used)
    
    if len(list_of_wrong_data_format)!=0:
        raise NameError('Inconsistent data format in the list of data_used')
    
    
    ############################################################################
    # Time regularization in a single reference time    
    # Weighted averge to impute missing value
    # Imputing missing data -using weighted mean value
    ############################################################################s
    min_set=time_mat[:,MIN_IDX].astype(int)
    hr_set=time_mat[:,HR_IDX].astype(int)
    wd_set=time_mat[:,WD_IDX].astype(int)
    day_set=time_mat[:,MD_IDX].astype(int)
    mn_set=time_mat[:,MN_IDX].astype(int)
    cumnum_days_mn=np.r_[0,np.array([calendar.monthrange(2013, i)[1] for i in np.r_[1:12]]).cumsum()]
    daycount_set=[int(day+cumnum_days_mn[mn-1]) for i,(day,mn) in enumerate(zip(day_set,mn_set))]
    #np.atleast_2d(np.arange(len(hr_set))).T    
    hrcount_set=make_colvec(np.arange(len(hr_set)))
    print 'Adding second time stamps ....'
    print '--------------------------------------'
    for key_id in data_used:
        print key_id
        utc_t=[];val=[]
        for i,(min_t, sample) in enumerate(zip(data_dict[key_id][0],data_dict[key_id][1])):
            if len(sample)>0:
                num_samples_per_hr=len(min_t)
                sec_t_ar=minidx_to_secs(min_t)
                data_dict[key_id][0][i]=sec_t_ar
                tt = dtime_to_unix(dt.datetime(2013, mn_set[i], day_set[i], hr_set[i]))
                utc_temp=tt+sec_t_ar
                for a,b in zip(utc_temp,sample):
                    utc_t.append(a);val.append(b)
        data_dict[key_id].append([utc_t,val])
    print '--------------------------------------'
    
    #for (i,j) in zip(hr_set,hrcount_set):
    #    print i,j
    num_of_data=len(data_used)
    num_of_samples=len(time_slots)
    X=np.zeros([num_of_samples,num_of_data])
    INT_type_cols=[]
    FLOAT_type_cols=[]
    
    # Constrcut X matrix by summerizing hourly samples
    for j,key in enumerate(data_used):
        for i,sample in enumerate(data_dict[key][1]):
            if len(sample)==0: # Set infty if no sample is availble
                X[i,j]=np.infty
            elif isinstance(sample[0],int):
                X[i,j]=int(stats.mode(sample)[0])
                if i==0: INT_type_cols.append(j) 
            elif isinstance(sample[0],float):
                X[i,j]=np.mean(sample)
                if i==0: FLOAT_type_cols.append(j) 
            else:
                raise NameError('Sample type must either INT or FLOAT type')


###############################################################################
# Learn a graphical structure from the correlations

#X_INT=X[:,INT_type_cols]
#X_FLOAT=X[:,FLOAT_type_cols]
# Currently use only Float type data
X_INPUT=[]
num_of_missing=0
input_names=[]
zero_var_list=[] # whose variance is zero, hence carry no information, 
DO_INTERPOLATE=0
for i,test_idx in enumerate(FLOAT_type_cols):
    print '---------------------'    
    infty_indice=[]
    y=X[:,test_idx]
    #y_95=np.sort(y)[np.ceil(len(y)*0.05):np.floor(len(y)*0.95)]
    #y_low=np.mean(y_95)-2.575*np.std(y_95)/np.sqrt(len(y_95))
    #y_high=np.mean(y_95)+2.575*np.std(y_95)/np.sqrt(len(y_95))
    #y[(y<y_low)+(y>y_high)]=np.inf
    x=np.arange(X.shape[0])
    
    if len(np.nonzero(y==np.infty)[0]) >0 and DO_INTERPOLATE==1:
        y_pred,sigma,infty_indice=gp_interpol_test(x,y,num_of_missing)
    else:
        print 'no data loss'
        y_pred=y
    print data_used[test_idx]
    print infty_indice,'are interpoloated'
    #if np.var(y_pred)>0: # Variance of 95% of samples
    if np.var(np.sort(y_pred)[np.ceil(len(y_pred)*0.05):np.floor(len(y_pred)*0.95)])>0:
        idx_temp=np.nonzero(y_pred!=np.infty)
        #y_pred=fftpack.dct(y_pred)
        #X_INPUT.append((y_pred-np.mean(y_pred[idx_temp]))/np.max(np.abs(y_pred[idx_temp])))
        y_pred=y_pred-np.mean(y_pred[idx_temp])
        temp_val=y_pred/norm(y_pred[idx_temp])
        X_INPUT.append(temp_val)
        input_names.append(data_used[test_idx])
    else:
        zero_var_list.append(data_used[test_idx])
    print '---------------------'

X_INPUT=np.asanyarray(X_INPUT).T  

inf_idx_set=[]
for col_vec in X_INPUT.T:
    inf_idx=np.nonzero(col_vec==np.infty)[0]
    inf_idx_set=np.r_[inf_idx_set,inf_idx]
    #inf_idx_set.append(inf_idx)

X_INPUT_ORG=X_INPUT.copy() # Let preserve original copy of it for future use
inf_col_idx=list(set(list(inf_idx_set)))
X_INPUT=np.delete(X_INPUT,inf_col_idx, axis=0)





input_names=np.array(input_names)

"""
for i,idx in enumerate(FLOAT_type_cols):
    plt.figure(i)    
    plt.plot(X_FLOAT[:,i], 'bx-', label=u'Prediction')
    plt.plot(X[:,idx], 'r.', markersize=10, label=u'Observations')    
    plt.title(data_used[idx])
    plt.legend(loc='upper right')
"""    

#y_pred,sigma=gp_interpol_test(x,y,num_of_missing)
edge_model = covariance.GraphLassoCV()
#################################################
# By normalizing by std the estimted covariance matrix becomes correlation matrix
# Dont need to normaliz anymore as we use norm in the beginning 
#X_INPUT /= X_INPUT.std(axis=0)   

#################################################
# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery

################################################################################
# Covariance (Correlation Coefficien) Estimatoin
################################################################################
try:    
    edge_model.fit(X_INPUT)
    cov_mat=edge_model.covariance_
except:
    COV_MAT=np.zeros([X_INPUT.shape[1],X_INPUT.shape[1]])
    DIST_MAT=np.zeros([X_INPUT.shape[1],X_INPUT.shape[1]])
    for i in range(X_INPUT.shape[1]):
        for j in range(X_INPUT.shape[1]):
            sample1=X_INPUT[:,i]
            sample2=X_INPUT[:,j]
            COV_MAT[i,j]=np.sum((sample1-np.mean(sample1))*(sample2-np.mean(sample2)))/X_INPUT.shape[0]
            DIST_MAT[i,j]=sqrt(norm(sample1-sample2))
    cov_mat=COV_MAT
    
corr_mat=(np.diag(cov_mat)**(-0.5))*cov_mat*(np.diag(cov_mat)**(-0.5))


################################################################################
# Unsupervised clustering for sensors given the measurement correlation 
# Find only a few represetative sensors out of many sensors
################################################################################
# exemplars are a set of representative signals for each cluster
# Smaller dampding input will generate more clusers, default is 0.5
# 0.5 <= damping <=0.99
################################################################################
#exemplars, labels = cluster.affinity_propagation(cov_mat,damping=0.5)
exemplars, labels = cluster.affinity_propagation(cov_mat)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(input_names[labels == i])))


###############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=3)

embedding = node_position_model.fit_transform(X_INPUT.T).T

###############################################################################
# Visualization
plt.figure('Data strucutre map', facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')
# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.01)

"""
idx=3
ind = np.arange(len(partial_correlations))    # the x locations for the groups
plt.stem(ind,corr_mat[idx,:])
plt.xticks(ind,input_names,fontsize=8,rotation='vertical')
plt.title(input_names[idx])
"""
from sklearn import manifold
from sklearn.metrics import euclidean_distances
similarities = euclidean_distances(X_INPUT.T)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,  dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_


# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100*d**2,c=labels, cmap=pl.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=pl.cm.hot_r,
                    norm=pl.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(zip(input_names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=12,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            bbox=dict(facecolor='w',
                      edgecolor=plt.cm.spectral(label / float(n_labels)),
                      alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
        embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
        embedding[1].max() + .03 * embedding[1].ptp())



#################################################################################
# Graph strucutre analysis of sensor naming
#################################################################################
print '--------------------------------------------------'
print 'Graph strucutre analysis of sensor naming'
print '--------------------------------------------------'
print 'get simialirty matrix of sensor naming'
#sim_mat, uuid_list, phrases, key_description, phrase_count = get_sim_mat()

sim_mat = mt.loadObjectBinary('sim_mat.bin')
uuid_list = mt.loadObjectBinary('uuid_list.bin')
phrases = mt.loadObjectBinary('phrases.bin')
key_description = mt.loadObjectBinary('key_description.bin')
phrase_count = mt.loadObjectBinary('phrase_count.bin')


_, sf_labels = cluster.affinity_propagation(sim_mat)
n_sf_labels = sf_labels.max()

for i in range(n_sf_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(np.array(uuid_list)[sf_labels == i])))



#plotting_data(data_used)
print '**************************** End of Program ****************************'

"""
# If no data availalbe, then imputes the data by weighted mean
    print 'Before imputation'
    for i,key in enumerate(data_used):
        plt.figure(1)
        print key
        print [k for k in np.nonzero(X[:,i]==np.infty)[0]]
        plt.subplot(len(data_used),1,i+1)
        plt.plot(time_slots,X[:,i],'.')
        plt.title(key,fontsize=6)
        plt.xticks(fontsize=6);plt.yticks(fontsize=6)
    # If no data availalbe, then imputes the data by weighted mean
    print 'Impute misssing data'
    for i,key in enumerate(data_used):
        for inf_idx in np.nonzero(X[:,i]==np.infty)[0]:
            whgt_bottom_sum=0;whgt_top_sum=0
            for h_idx in np.nonzero(hr_set==hr_set[inf_idx])[0]:
                #import pdb; pdb.set_trace()            
                sample_temp=X[h_idx,i]
                if (sample_temp<np.infty and h_idx!=inf_idx):
                    wght=1/np.abs(daycount_set[h_idx]-daycount_set[inf_idx])
                    whgt_bottom_sum=whgt_bottom_sum+wght
                    whgt_top_sum=whgt_top_sum+wght*sample_temp
            new_sample=whgt_top_sum/whgt_bottom_sum
            X[inf_idx,i]=new_sample
        
    # If no data availalbe, then imputes the data by weighted mean
    print 'After imputation'
    for i,key in enumerate(data_used):
        plt.figure(1)
        print key
        print [k for k in np.nonzero(X[:,i]==np.infty)[0]]
        plt.subplot(len(data_used),1,i+1)
        plt.plot(time_slots,X[:,i])
        plt.title(key,fontsize=6)
        plt.xticks(fontsize=6);plt.yticks(fontsize=6)
 
 
 
 """
