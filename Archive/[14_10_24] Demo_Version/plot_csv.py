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
import pylab as pl
from scipy import stats
import matplotlib.pyplot as plt
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
from sklearn import cluster, covariance, manifold # Machine Learning Packeage
###############################################################################
# Constant global variables
###############################################################################
# in seconds
MIN=60; HOUR=60*MIN; DAY=HOUR*24; MONTH=DAY*31
# Hour, Weekday, Day, Month
MIN_IDX=0;HR_IDX=1; WD_IDX=2; MD_IDX=3 ;MN_IDX=4

# Define the period for analysis  - year, month, day,hour
# Note: The sample data in the currently downloaded files are from 1 Apr 2013 to
#       30 Nov 2013.
ANS_START_T=dt.datetime(2013,7,1,0)
ANS_END_T=dt.datetime(2013,7,5,0)
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

# Data dictionary
# All sensor and weather data is processed and structred into 
# a consistent single data format -- Dictionary
data_dict={}
# This is the list of non-digit symbolic weather data
# The symbolic weather data is such as Conditions (e.g Cloudy or Clear)
# and Events (e.g. Rain or Fog ...)
# Those symblic data is replaced with integer state representation whose
# pairs are stored in a hash table using Dictionary. 
# If no data is given, key value is set to 0.
Conditions_dict={};Conditions_val=[];key_val_c=0
Events_dict={};Events_val=[]; key_val_e=0
    
Is_CSV=bool(0)
Start_t=time.time()
argv_len=len(sys.argv)
print 'arg length:',argv_len

###############################################################################
# Function 
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
# Retrive weather data from internet for the specified periods 
# prefix_order=TS (default) [Time][Sensor]
# prefix_order=ST            [Sensor][Time]
###############################################################################
"""
def get_weather(t_start, t_end, perfix_order='TS'):
    print 'getting weater data '    
    print 'start time:', t_start, ' ~ end time:',t_end
    data_days=[]
    for date in daterange(t_start, t_end, inclusive=True):
        #print date.strftime("%Y-%m-%d")
        temp=date.strftime("%Y,%m,%d").rsplit(',')
        data_day=rw.retrieve_data('VTT', int(temp[0]), int(temp[1]), int(temp[2]), view='d')    
        data_day=data_day.split('\n')
        if perfix_order=='TS':
            # order by [Sensor][Time]
            # Paring the strings of daily weather data
            day_sample_parse=[]        
            for hour_sample in data_day:
                #print hour_sample
                day_sample_parse.append(hour_sample.split(','))
            
            data_days.append(day_sample_parse)
            
        else:
            # order by [Time][Sensor]
            # Paring the strings of daily weather data
            #f=open('weather_data.txt','w')
            day_sample_parse=[]        
            for h_idx,hour_sample in enumerate(data_day):
                #print hour_sample
                if h_idx==0:
                    sensor_name_list=hour_sample.split(',')
                   # f.write(str(sensor_name_list)+'\n')
                else:
                    hour_samples=hour_sample.split(',')
                    #print hour_samples
                    #f.write(str(hour_samples)+'\n')
                    for sample_idx,each_sample in enumerate(hour_samples):
                        sensor_name=sensor_name_list[sample_idx]
                        if sensor_name in data_dict:
                            data_dict[sensor_name].append(each_sample)
                        else:
                            data_dict.update({sensor_name:[each_sample]})


    if perfix_order=='TS':             
        return data_days
    else:
        return sensor_name_list
            
            #f.close()
 """                    




###############################################################################
# Plotting tool
###############################################################################
def plotting_data(plot_list,opt='val'):
    # times is seconds, but it might not correct for months with 30 days.    
    #times_in_secs=(time_val[:,[HR_IDX,MD_IDX,MN_IDX]]*[HOUR,DAY,MONTH]).sum(axis=1)    
    # Minute,Hour, Weekday, Day, Month - total 5 time fields
    time_mat=np.zeros([len(time_slots),5])
    for i, time_sample in enumerate(time_slots):
        time_mat[i,HR_IDX]=time_sample.hour
        time_mat[i,WD_IDX]=time_sample.weekday()
        time_mat[i,MD_IDX]=time_sample.day
        time_mat[i,MN_IDX]=time_sample.month
    
    monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    weekDict={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thur', 4:'Fri', 5:'Sat', 6:'Sun'}
    # Month indicator    
    time_mn_diff=np.diff(time_mat[:,MN_IDX])
    m_label_idx=time_mn_diff.nonzero()[0]
    m_label_str=[]
    for m_num in time_mat[m_label_idx,MN_IDX]:
        m_label_str.append(monthDict[m_num])
    
    time_wk_diff=np.diff(time_mat[:,WD_IDX])
    w_label_idx=time_wk_diff.nonzero()[0]
    w_label_str=[]
    for w_num in time_mat[w_label_idx,WD_IDX]:
        w_label_str.append(weekDict[int(w_num)])

    for k,sensor in enumerate(plot_list):
        #print k, sensor
        num_samples=[]
        mean_samples=[]
        for i,(t,samples) in enumerate(zip(time_slots,data_dict[sensor])):
            #print i,str(t),len(samples)
            num_samples.append(len(samples))
            # Mean value with masking
            mean_samples.append(np.mean(samples))
            #mean_samples.append(np.mean(np.ma.masked_invalid(samples))
        
        #sensor_samples.append(num_samples)
        plt.figure(1)
        plt.subplot(len(plot_list),1,k+1)
        plt.plot(time_slots,num_samples)
        plt.title(sensor,fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('# Samples/Hour',fontsize=8)
        if k<len(plot_list)-1:        
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
        
        #frame1.axes.get_yaxis().set_visible(False)
        
        plt.figure(2)
        plt.subplot(len(plot_list),1,k+1)
        plt.plot(time_slots,mean_samples)
        plt.title(sensor,fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('Avg Val/Hour',fontsize=8)
        if k<len(plot_list)-1:        
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            #frame1.axes.get_yaxis().set_visible(False)
        
        #plt.xticks(w_label_idx.tolist(),w_label_str,fontsize=8)
        #plt.text(m_label_idx, np.max(num_samples)*0.8, m_label_str, fontsize=12)
    print ' End of Plotting'
    return time_mat



      
###############################################################################
# Parsing sensor data 
###############################################################################       
def get_val(filename):
    if Is_CSV==True:
        openfile=open(filename,"r")
        sensor_val=[]
        time_val=[];
        for line in openfile:
            tmp=line.rstrip().rsplit(",")
            sensor_val.append(float(tmp[1]))
            temp=dt.datetime.strptime(tmp[0],"%Y-%m-%d %H:%M:%S")
            temp=temp.timetuple()
            # Hour, Weekday, Day, Month
            time_val.append([temp[3],temp[6],temp[2],temp[1]]) 
            
        openfile.close()
        #print 'list of input csv files: '
    else:
        data = mt.loadObjectBinary(filename)
        sensor_val = data["value"]
        time_val = data["ts"]
        #print 'list of input bin files: '
        
    return sensor_val,time_val
    
    
def get_val_timelet(filename,t_slots):
    print ' get_val_timelet'    
    if Is_CSV==True:
        openfile=open(filename,"r")
        sensor_val=[]
        time_val=[];
        for line in openfile:
            tmp=line.rstrip().rsplit(",")
            sensor_val.append(float(tmp[1]))
            temp=dt.datetime.strptime(tmp[0],"%Y-%m-%d %H:%M:%S")
            temp=temp.timetuple()
            # Hour, Weekday, Day, Month
            time_val.append([temp[3],temp[6],temp[2],temp[1]]) 
            
        openfile.close()
        #print 'list of input csv files: '
    else:
        data = mt.loadObjectBinary(filename)
        sensor_val = data["value"]
        time_val = data["ts"]
        # Creat the list of lists
        sensor_read=[[] for i in range(len(t_slots))]
        for t_sample, v_sample in zip(time_val,sensor_val):
            #import pdb; pdb.set_trace()
            # If data in 2013 is only available after Aprile, Otherwise it is 2014 data            
            if t_sample[MN_IDX]>3:
                temp_dt=dt.datetime(2013,t_sample[MN_IDX],t_sample[MD_IDX],t_sample[HR_IDX])
            else:
                temp_dt=dt.datetime(2014,t_sample[MN_IDX],t_sample[MD_IDX],t_sample[HR_IDX])
                
                
            #print temp_dt
            try:
                idx=t_slots.index(temp_dt)
                sensor_read[idx].append(v_sample)
            except ValueError:
                idx=-1
        
    return sensor_read, time_val

###############################################################################
# Parsing sensor data
# Data samples are regularized for specified times with timelet
###############################################################################
def symbol_to_state(symbol_list):
    #list(itertools.chain(*list_of_lists))
    
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
    

def get_weather(t_start, t_end, perfix_order='TS'):
    print 'getting weater data new '    
    print 'start time:', t_start, ' ~ end time:',t_end
    data_days=[]
    # Date iteration given start time and end-time
    for date in daterange(t_start, t_end, inclusive=True):
        print date.strftime("%Y-%m-%d")
        temp=date.strftime("%Y,%m,%d").rsplit(',')
        data_day=rw.retrieve_data('VTT', int(temp[0]), int(temp[1]), int(temp[2]), view='d')    
        data_day=data_day.split('\n')
        if perfix_order=='TS':
            # order by [Sensor][Time]
            # Paring the strings of daily weather data
            day_sample_parse=[]        
            for hour_sample in data_day:
                #print hour_sample
                day_sample_parse.append(hour_sample.split(','))
            
            data_days.append(day_sample_parse)
            
        else:
            # order by [Time][Sensor]
            # Paring the strings of daily weather data
            #f=open('weather_data.txt','w')
            day_sample_parse=[]        
            for h_idx,hour_sample in enumerate(data_day):
                #print hour_sample
                if h_idx==0:
                    sensor_name_list=hour_sample.split(',')
                   # f.write(str(sensor_name_list)+'\n')
                else:
                    hour_samples=hour_sample.split(',')
                    #print hour_samples
                    #f.write(str(hour_samples)+'\n')
                    for sample_idx,each_sample in enumerate(hour_samples):
                        sensor_name=sensor_name_list[sample_idx]
                        if sensor_name in data_dict:
                            data_dict[sensor_name].append(each_sample)
                        else:
                            data_dict.update({sensor_name:[each_sample]})
    if perfix_order=='TS':             
        return data_days
    else:
        return sensor_name_list
        

def get_weather_timelet(t_slots):
    print 'getting weater data new '   
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
                    data_dict.update({sensor_name:sensor_read})
            elif h_idx>0:
                # 'DateUTC' is the one  
                sample_DateUTC=hour_samples[sensor_name_list.index('DateUTC')]
                
                # convert to UTC time to VTT local time. 
                utc_dt=dt.datetime.strptime(sample_DateUTC, "%Y-%m-%d %H:%M:%S")
                vtt_dt_aware = utc_dt.replace(tzinfo=from_zone).astimezone(to_zone)
                # convert to offset-naive from offset-aware datetimes              
                vtt_dt=dt.datetime(*(vtt_dt_aware.timetuple()[:4]))
                # time slot index a given weather sample time
                try:
                    vtt_dt_idx=t_slots.index(vtt_dt)
                    for sample_idx,each_sample in enumerate(hour_samples):
                        # convert string type to float time if possible
                        try:
                            each_sample=float(each_sample)
                        except ValueError:
                            each_sample=each_sample
                            
                        sensor_name=sensor_name_list[sample_idx]
                        #import pdb; pdb.set_trace()                        
                        if sensor_name in data_dict:
                            if each_sample!='N/A' and each_sample!=[]:
                                data_dict[sensor_name][vtt_dt_idx].append(each_sample)
                        else:
                            raise NameError('Inconsistency in the list of weather data')
                except ValueError:
                    vtt_dt_idx=-1
            else:
                # hour_sample is list of weather filed name, discard
                hour_sample=[]
    
    return sensor_name_list
    

def data_dict_purge(purge_list):
    for key in purge_list:
        print 'purge', key
        if key in data_dict.keys():
            data_dict.pop(key,None)
    
#data_dict_purge(weather_list)

###############################################################################
# Reading sensor data from CSV or BIN files - use linux commands
###############################################################################
input_csvs=[]
num_csvs=[]
if argv_len==1:
    if Is_CSV==True:
        temp = subprocess.check_output("ls *.csv |grep _ACTIVE_POWER_", shell=True)
    else:
        temp = subprocess.check_output("ls *.bin |grep _ACTIVE_POWER_", shell=True)
    
    input_csvs =shlex.split(temp)
    plt.ion()
    print 'argv 1' 
elif argv_len>1:
    input_csvs=sys.argv[1:]
    print 'getting args'
else:
    input_csvs=[]
    print '...'

num_csvs=len(input_csvs)
num_col_subplot=np.ceil(np.sqrt(num_csvs))



###############################################################################
#  Analysis script starts here ....
# List of sensors  from BMS
print 'mapping sensor list into hasing table using dictionary'
sensor_list=input_csvs

# List of sensors from Weather data
# getting weather files
# Weather parameter list
#['TimeEEST', 'TemperatureC', 'Dew PointC', 'Humidity', 
# 'Sea Level PressurehPa', 'VisibilityKm', 'Wind Direction', 
# 'Wind SpeedKm/h', 'Gust SpeedKm/h', 'Precipitationmm', 
# 'Events', 'Conditions', 'WindDirDegrees', 'DateUTC']
# Note: We select 'TemperatureC', 'Dew PointC', 'Humidity',
# 'Events', 'Conditions' for the main weather parameter

#weather_list=get_weather(ANS_START_T, ANS_END_T,'ST')



# Checking length of weather sample data
print "lenth of dictionary"
for key in data_dict.keys():
        print 'len of ', key, len(data_dict[key])

# data dictionary that map all types of sensor readings into a single hash table

###############################################################################
# Read out all sensor files in the file list

time_set_temp=[]	
for i,argv in enumerate(sensor_list):
    print 'index ',i+1,': ', argv
    # sensor value is read by time 
    start__dictproc_t=time.time()         
    dict_sensor_val, dict_time_val=get_val_timelet(argv,time_slots)
    data_dict.update({argv:dict_sensor_val})
    end__dictproc_t=time.time()     
    
    print argv,'- dict.proc time is ', end__dictproc_t-start__dictproc_t    

print 'Check sample density over time slots'

time_mat=plotting_data(sensor_list[0:2])




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
data_dict['Conditions'],Conditions_dict=symbol_to_state(data_dict['Conditions'])
data_dict['Events'],Events_dict=symbol_to_state(data_dict['Events'])

# Weather data to be used
weather_list_used = [weather_list[i] for i in [1,2,3,10,11]]
# All (sensor + weather) data to be used
data_used=weather_list_used + sensor_list


def verify_data_format(key_list):
    # Verify there is no  [] or N/A in the list
    print 'Checking any inconsisent data format.....'
    print '---------------------------------'
    list_of_wrong_data_format=[]
    for key in key_list:
        print 'checking ', key, '...'
        for i,samples in enumerate(data_dict[key]):
            for j,each_sample in enumerate(samples):
                if each_sample==[]:
                    list_of_wrong_data_format.append([key,i,j])
                    print each_sample, 'at', time_slots[j], 'in', key
                elif (isinstance(each_sample,int)==False and isinstance(each_sample,float)==False):
                    list_of_wrong_data_format.append([key,i,j])                    
                    print each_sample, 'at', time_slots[j], 'in', key
    print '---------------------------------'                    
    if len(list_of_wrong_data_format)==0:
        print ' no inconsistent data format'
    return  list_of_wrong_data_format

    
# Verify there is no  [] or N/A in the list
list_of_wrong_data_format=verify_data_format(data_used)

if len(list_of_wrong_data_format)!=0:
    raise NameError('Inconsistent data format in the list of data_used')
    
# Weighted averge to impute missing value
# Imputing missing data -using weighted mean value
hr_set=time_mat[:,HR_IDX].astype(int)
wd_set=time_mat[:,WD_IDX].astype(int)
day_set=time_mat[:,MD_IDX].astype(int)
mn_set=time_mat[:,MN_IDX].astype(int)
cumnum_days_mn=np.r_[0,np.array([calendar.monthrange(2013, i)[1] for i in np.r_[1:12]]).cumsum()]
daycount_set=[ int(day+cumnum_days_mn[mn-1]) for i,(day,mn) in enumerate(zip(day_set,mn_set))]

#  X.shape (1258, 7)
# type(X) <type 'numpy.ndarray'>

# type(X) <type 'numpy.ndarray'>
num_of_data=len(data_used)
num_of_samples=len(time_slots)
X=np.zeros([num_of_samples,num_of_data])
INT_type_cols=[]
FLOAT_type_cols=[]
for j,key in enumerate(data_used):
    for i,sample in enumerate(data_dict[key]):
        if len(sample)==0:
            X[i,j]=np.infty
        elif isinstance(sample[0],int):
            X[i,j]=int(stats.mode(sample)[0])
            if i==0: INT_type_cols.append(j) 
        elif isinstance(sample[0],float):
            X[i,j]=np.mean(sample)
            if i==0: FLOAT_type_cols.append(j) 
        else:
            raise NameError('Sample type must either INT or FLOAT type')
                
        
# If no data availalbe, then imputes the data by weighted mean
print 'Before imputation'
for i,key in enumerate(data_used):
    print key
    print [k for k in np.nonzero(X[:,i]==np.infty)[0]]




# If no data availalbe, then imputes the data by weighted mean
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
    print key
    print [k for k in np.nonzero(X[:,i]==np.infty)[0]]
# If no data availalbe, then imputes the data by weighted mean


X_INT=X[:,INT_type_cols]
X_FLOAT=X[:,FLOAT_type_cols]
###############################################################################
# Learn a graphical structure from the correlations

edge_model = covariance.GraphLassoCV()
# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
edge_model.fit(X_FLOAT)

            
            
        



# Using mode if interger type, using mean if real type


"""
vak1_power_sys_sum=[]
vak1_power_p1_sum=[]
vak1_power_p2_sum=[]
vak1_power_p3_sum=[]
for i,(psys,p1,p2,p3) in enumerate(zip(vak1_power_sys,vak1_power_p1,vak1_power_p2,vak1_power_p3)):
    vak1_power_sys_sum.append(sum(psys))
    vak1_power_p1_sum.append(sum(p1))
    vak1_power_p2_sum.append(sum(p2))
    vak1_power_p3_sum.append(sum(p3))


plt.subplot(2,1,1)
plt.plot(vak1_power_sys_sum)
plt.plot(np.array(vak1_power_p1_sum)+np.array(vak1_power_p2_sum)+np.array(vak1_power_p3_sum),'-s')
plt.subplot(2,1,2)
plt.plot(vak1_power_p1_sum,'-*')
plt.plot(vak1_power_p2_sum,'-s')
plt.plot(vak1_power_p3_sum,'-o')
"""

# Using the following weather data for variables
# 

# Regularized the weather data into a single time referece
# For symbolic data, use mode, and for real number data, use average
# Gaussian Process (GP) model and interploation for power consumption data
#Conditions_dict,Events_dict


"""
3D plotting
fig=pl.figure()
ax = p3.Axes3D(fig)
ax.scatter(gw2_power_p1_sum.T, gw2_power_p2_sum, gw2_power_p3_sum, c=colors)
ax.set_xlabel('P1')
ax.set_ylabel('P2')
ax.set_zlabel('P3')
fig.add_axes(ax)
"""
        
if argv_len>1:
    print 'end of program'
    plt.show()



