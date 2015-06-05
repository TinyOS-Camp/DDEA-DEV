# coding: utf-8 

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
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
from stackedBarGraph import StackedBarGrapher

import pprint
import radar_chart

##################################################################
# ETE tree module
#from ete2 import Tree
##################################################################

##################################################################
# Machine Learing Modules
from sklearn.gaussian_process import GaussianProcess
from sklearn import cluster, covariance, manifold # Machine Learning Packeage
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import Ward
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn import mixture
##################################################################
# Custom library
##################################################################
from data_tools import *
from data_retrieval import *
from pack_cluster import *
from data_preprocess import *
from shared_constants import *
from pre_bn_state_processing import *
##################################################################
from matplotlib.collections import LineCollection
#from classify_sensors import get_sim_mat
plt.ion() 

def event_analysis_bldg(dict_dir,bldg_tag):
    print 'Loading data dictionary......'
    start__dictproc_t=time.time()         
    data_dict = mt.loadObjectBinaryFast(dict_dir+'data_dict.bin')
    end__dictproc_t=time.time()
    print 'the time of loading data dict.bin is ', end__dictproc_t-start__dictproc_t, ' sec'
    print '--------------------------------------'                
    # Copy related variables 
    time_slots=data_dict['time_slots'][:]
    Conditions_dict=data_dict['Conditions_dict'].copy()
    Events_dict=data_dict['Events_dict'].copy()
    sensor_list=data_dict['sensor_list'][:]
    weather_list=data_dict['weather_list'][:]
    weather_list_used = [data_dict['weather_list'][i] for i in [1,2,3,10,11]]
    
    # data_used is the list of refernece name for all measurements from now on. 
    data_used=sensor_list+weather_list_used
    # This is a global ID for data_used measurement
    data_used_idx=range(len(data_used))
    sensor_idx=range(len(sensor_list))
    weather_idx=range(len(sensor_list),len(data_used))
    
    diffdata_dict = mt.loadObjectBinary(dict_dir+'diffdata_dict.bin')
    avgdata_dict = mt.loadObjectBinary(dict_dir+'avgdata_dict.bin')
    
    
    # Irregualr Events 
    diffdata_state_mat=diffdata_dict['diffdata_state_mat']
    diffdata_weather_mat=diffdata_dict['diffdata_weather_mat']
    diffdata_time_mat=diffdata_dict['diffdata_time_mat']
    diff_time_slot=diffdata_dict['diff_time_slot']
    diffdata_exemplar=diffdata_dict['diffdata_exemplar']
    diffdata_zvar=diffdata_dict['diffdata_zvar']
    diffsensor_names=diffdata_dict['sensor_names']
    diffweather_names=diffdata_dict['weather_names']
    difftime_names=diffdata_dict['time_names']
    print 'size of diffdata_state_mat is ' , diffdata_state_mat.shape
    
    # Regualr Events         
    avgdata_state_mat=avgdata_dict['avgdata_state_mat']
    avgdata_weather_mat=avgdata_dict['avgdata_weather_mat']
    avgdata_time_mat=avgdata_dict['avgdata_time_mat']
    avg_time_slot=avgdata_dict['avg_time_slot']
    avgdata_exemplar=avgdata_dict['avgdata_exemplar']
    avgdata_zvar=avgdata_dict['avgdata_zvar']
    avgsensor_names=avgdata_dict['sensor_names']
    avgweather_names=avgdata_dict['weather_names']
    avgtime_names=avgdata_dict['time_names']
    print 'size of avgdata_state_mat is ' , avgdata_state_mat.shape
    
    print '*** avg feature****'
    Conditions_dict=data_dict['Conditions_dict'].copy()
    Events_dict=data_dict['Events_dict'].copy()
    data_state_mat=avgdata_state_mat
    data_time_mat=avgdata_time_mat
    data_weather_mat=avgdata_weather_mat
    sensor_names=avgsensor_names
    time_names=avgtime_names
    weather_names=avgweather_names
    trf_tag='avg_' # transformation tag
    avg_wtf_tuple,avg_weather_dict=wt_sensitivity_analysis(data_state_mat,data_time_mat,data_weather_mat,sensor_names,time_names,\
     Conditions_dict,Events_dict,bldg_tag,trf_tag,weather_names,dict_dir,dst_t='h')
    
    print '*** diff feature****'
    Conditions_dict=data_dict['Conditions_dict'].copy()
    Events_dict=data_dict['Events_dict'].copy()
    data_state_mat=diffdata_state_mat
    data_time_mat=diffdata_time_mat
    data_weather_mat=diffdata_weather_mat
    sensor_names=diffsensor_names
    time_names=difftime_names
    weather_names=diffweather_names
    trf_tag='diff_' # transformation tag
    diff_wtf_tuple,diff_weather_dict=wt_sensitivity_analysis(data_state_mat,data_time_mat,data_weather_mat,sensor_names,time_names,\
     Conditions_dict,Events_dict,bldg_tag,trf_tag,weather_names,dict_dir,dst_t='h')
     
    output_dict={}
    output_dict.update({'avg_wtf_tuple':avg_wtf_tuple})
    output_dict.update({'diff_wtf_tuple':diff_wtf_tuple})
    output_dict.update({'avg_weather_dict':avg_weather_dict})
    output_dict.update({'diff_weather_dict':diff_weather_dict})
    mt.saveObjectBinary(output_dict,dict_dir+'output_dict.bin')

"""
dict_dir='./'
bldg_tag='VAK_' # building tag
event_analysis_bldg(dict_dir,bldg_tag)
"""

print '========================='
print 'GW1 BLDG'
print '========================='
dict_dir='./GW1_results/'
bldg_tag='GW1_' # building tag
event_analysis_bldg(dict_dir,bldg_tag)

print '========================='
print 'GW2 BLDG'
print '========================='
dict_dir='./GW2_results/'
bldg_tag='GW2_' # building tag
event_analysis_bldg(dict_dir,bldg_tag)

print '========================='
print 'VAK1 BLDG'
print '========================='
dict_dir='./VAK1_results/'
bldg_tag='VAK1_' # building tag
event_analysis_bldg(dict_dir,bldg_tag)

print '========================='
print 'VAK2 BLDG'
print '========================='
dict_dir='./VAK2_results/'
bldg_tag='VAK2_' # building tag
event_analysis_bldg(dict_dir,bldg_tag)


