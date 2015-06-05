# coding: utf-8 

"""
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
#print(__doc__)
# Author: Deokwooo Jung deokwoo.jung@gmail.compile
##################################################################
# General Moduels
from __future__ import division # To forace float point division
import os
import sys
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import uuid

import pylab as pl
from scipy import signal
from scipy import stats
from scipy.interpolate import interp1d

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
import calendar
import random
from matplotlib.collections import LineCollection
from stackedBarGraph import StackedBarGrapher
import pprint
import radar_chart

##################################################################
# Machine Learing Modules
from sklearn import cluster, covariance, manifold # Machine Learning Packeage
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import Ward
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
##################################################################
# Custom library
from data_tools import *
from data_retrieval import *
from pack_cluster import *
from data_preprocess import *
from shared_constants import *
from pre_bn_state_processing import *
##################################################################

##################################################################
# Processing Configuraiton Settings
##################################################################
# This option let you use  data_dict object saved in hard-disk from the recent execution
IS_USING_SAVED_DICT=-1
# File selection method
Using_LoopUp=0
# Analysis period
ANS_START_T=dt.datetime(2013,7,1,0)
ANS_END_T=dt.datetime(2013,9,30,0)
# Interval of timelet, currently set to 1 Hour
#TIMELET_INV=dt.timedelta(hours=1)
TIMELET_INV=dt.timedelta(minutes=15)

# Interactive mode for plotting
plt.ion() 
##################################################################

input_files=[]
###############################################################################
# This directly searches files from bin file name
if (Using_LoopUp==0) and (IS_USING_SAVED_DICT==0):
    temp4 = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep _ACTIVE_POWER_|grep GW2", shell=True)
    #temp = subprocess.check_output("ls "+data_dir+"*.bin |grep '_POWER_\|TK.*VAK'", shell=True)  
    ha_ = subprocess.check_output("ls "+DATA_DIR+"*.bin |grep '\.HA.._'", shell=True)  
    ha1_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep '\.HA1_'", shell=True)
    ha2_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep '\.HA2_'", shell=True) 
    power_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep _POWER_", shell=True)  
    #ventilation 
    iv_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep IV_", shell=True)  
    # Solar 
    aurinko_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep AURINKO_", shell=True)  
    # weather
    saa_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep '\.SAA'", shell=True) 
    # cooling
    jaah_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep JAAH", shell=True) 
    # ground heat
    mlp_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep MLP", shell=True) 
    # GW1 GEO Thermal    
    gw1geo_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep GW1.GEO", shell=True) 
    # GW2 GEO Thermal    
    gw2geo_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep GW2.GEO", shell=True) 
    # VAK1 GEO Thermal    
    vak1geo_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep VAK1.GEO", shell=True) 
    # VAK2 GEO Thermal    
    vak2geo_ = subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep VAK2.GEO", shell=True) 
    
    temp=power_+iv_+aurinko_+mlp_+gw1geo_ +gw2geo_+vak1geo_+vak2geo_+ha1_+ha2_
    #temp=temp4
    input_files =shlex.split(temp)
    # Get rid of duplicated files
    input_files=list(set(input_files))
    print 'The total number of sensors selected for analysis is ', len(input_files),'......'
    
###############################################################################
# This look-ups id description tables and find relavant bin files.  
elif (Using_LoopUp==1) and (IS_USING_SAVED_DICT==0):
    id_dict=get_id_dict('grep kW')
    for id_name in id_dict.keys():
        binfile_name=id_name+'.bin'
        input_files.append(binfile_name)
else:
    print 'Search data_dict.bin....'

###############################################################################
#  Analysis script starts here ....
###############################################################################
if IS_USING_SAVED_DICT==0:
    start__dictproc_t=time.time()
    # IS_USING_PARALLEL_OPT        
    data_dict=construct_data_dict(input_files,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename='data_dict', IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
    
    end__dictproc_t=time.time()
    print 'the time of construct data dict.bin is ', end__dictproc_t-start__dictproc_t, ' sec'
    print '--------------------------------------'                

elif IS_USING_SAVED_DICT==1:
    print 'Loading data dictionary......'
    start__dictproc_t=time.time()         
    data_dict = mt.loadObjectBinaryFast('data_dict.bin')
    end__dictproc_t=time.time()
    print 'the time of loading data dict.bin is ', end__dictproc_t-start__dictproc_t, ' sec'
    print '--------------------------------------'
else:
    print 'Skip data dict' 

if IS_USING_SAVED_DICT>0:
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
    # [data_used[i] for i in sensor_idx]
    # [data_used[i] for i in weather_idx]
    # Verify there is no  [] or N/A in the list

CHECK_DATA_FORMAT=0
if CHECK_DATA_FORMAT==1:
    list_of_wrong_data_format=verify_data_format(data_used,data_dict,time_slots)
    if len(list_of_wrong_data_format)>0:
        print 'Measurement list below'    
        print '----------------------------------------'    
        print list_of_wrong_data_format
        raise NameError('Errors in data format')


EVENT_RETRIEVAL=0
if EVENT_RETRIEVAL==1:
    # sensor_list  --> float or int  --> clustering for float and int --> exemplar 
    # exemplar of floats --> states , int is states, 
    # weather_list --> float or int
    ####################################
    # Regular Event Extraction
    ####################################
    # Build feature matrix wiht data interpolation for both sensor and weather data
    X_Feature,X_Time,X_names\
    ,X_zero_var_list, X_zero_var_val\
    ,X_int_type_list,X_int_type_idx\
    ,X_float_type_list,X_float_type_idx\
    ,X_weather_type_idx,X_sensor_type_idx\
    =build_feature_matrix(data_dict,sensor_list,weather_list_used\
    ,time_slots,DO_INTERPOLATE=1\
    ,max_num_succ_idx_for_itpl=int(len(time_slots)*0.05))
    
    
    
    if len(X_names+X_zero_var_list)!=len(data_used):
        raise NameError('Missing name is found in X_names or X_zero_var_list')
    else:
        zero_var_idx=[data_used.index(name_str) for name_str in X_zero_var_list]
        nzero_var_idx=list(set(data_used_idx)-set(zero_var_idx))
        
    # From below all index are reference to X_Feature
    sf_idx=list(set(X_sensor_type_idx)&set(X_float_type_idx));
    # Equivalent to np.array(data_used)[np.array(nzero_var_idx)[sf_idx]]
    sf_name=list(np.array(X_names)[sf_idx])
    si_idx=list(set(X_sensor_type_idx)&set(X_int_type_idx));
    si_name=list(np.array(X_names)[si_idx])
    wf_idx=list(set(X_weather_type_idx)&set(X_float_type_idx));
    wf_name=list(np.array(X_names)[wf_idx])
    wi_idx=list(set(X_weather_type_idx)&set(X_int_type_idx));
    wi_name=list(np.array(X_names)[wi_idx])
     #Euclidian Distance Matrix of Floating type of data only   wf+o
    float_idx=list(set(sf_idx)| set(wf_idx))
    int_idx=list(set(si_idx)| set(wi_idx))
    # Float Type Measurement Clustering
    X_Feature_sfe,sf_exemplars_dict,exemplars_,labels_\
    =cluster_measurement_points(X_Feature[:,sf_idx],sf_name,corr_bnd=[0.1,0.9],alg='pack')
    sfe_idx=list(np.array(sf_idx)[exemplars_])
    # InT Type Measurement Clustering
    X_Feature_sie,si_exemplars_dict,exemplars_,labels_\
    =cluster_measurement_points(X_Feature[:,si_idx],si_name,corr_bnd=[0.0,0.9],alg='pack')
    sie_idx=list(np.array(si_idx)[exemplars_])
            
    sfe_state,sfe_corr_val=X_INPUT_to_states(X_Feature_sfe,CORR_VAL_OUT=1) # sensor -float type
    sie_state=X_Feature_sie # sensor -integer type
    wf_state,wf_corr_val=X_INPUT_to_states(X_Feature[:,wf_idx],CORR_VAL_OUT=1) # weather -float type
    wi_state=X_Feature[:,wi_idx] # weather -integer type
    
    empty_states=np.array([[] for i in range(len(X_Time))])
    if len(sfe_state)==0: sfe_state=empty_states
    
    if len(sie_state)==0: sie_state=empty_states
    
    if len(wf_state)==0: wf_state=empty_states
    
    if len(wi_state)==0: wi_state=empty_states

    # Exemplar sensor only    
    X_Sensor_STATE=np.append(sfe_state,sie_state, axis=1)
    X_Sensor_STATE=X_Sensor_STATE.astype(int)
    X_Sensor_NAMES=list(np.array(X_names)[sfe_idx])+list(np.array(X_names)[sie_idx])
    X_Weather_STATE=np.append(wf_state,wi_state, axis=1)
    X_Weather_STATE=X_Weather_STATE.astype(int)
    X_Weather_NAMES=list(np.array(X_names)[wf_idx])+list(np.array(X_names)[wi_idx])
    # months of a year,days of a week, and hours of a day
    # (Monday, Tuesday,Wendsday,Thursday,Saturday,Sunday) =(0,1,2,3,4,5,6)
    X_Time_STATE_temp=build_time_states(X_Time)
    X_Time_NAMES_temp=['MTH','WD','HR']
    X_Time_STATE=[]
    X_Time_NAMES=[]
    for xt_col,xt_name in zip(X_Time_STATE_temp.T,X_Time_NAMES_temp):
        if len(set(xt_col))>1:
            X_Time_STATE.append(xt_col)
            X_Time_NAMES.append(xt_name)
    
    X_Time_STATE=np.array(X_Time_STATE).T

    DO_PLOTTING=0
    if DO_PLOTTING==1:
        sensor_name_temp=['VAK1.HA1_SM_EP_KM','VAK1.HA1_SM_KAM','GW1.HA1_TE16_AH2_M']
        plot_compare_sensors(sensor_name_temp,X_Time,X_Feature,X_names)
        plot_compare_states(sensor_name_temp[0],data_dict,X_Time,X_Feature,X_names)
        

    #################################################
    # FORMATTED DATA  FOR REGUALR EVENT
    #################################################
    #DO_PROB_EST=1  ** Save this variables***
    #avgdata_mat = np.hstack([X_Sensor_STATE,X_Weather_STATE,X_Time_STATE])
    #avgdata_names = X_Sensor_NAMES+X_Weather_NAMES+X_Time_NAMES
    avgdata_exemplar=dict(sf_exemplars_dict.items()+si_exemplars_dict.items())
    avgdata_zvar=X_zero_var_list
    
    avgdata_dict={}
    #avgdata_dict.update({'avgdata_mat':avgdata_mat})
    avgdata_dict.update({'avgdata_state_mat':X_Sensor_STATE})
    avgdata_dict.update({'avgdata_weather_mat':X_Weather_STATE})
    avgdata_dict.update({'avgdata_time_mat':X_Time_STATE})
    avgdata_dict.update({'avg_time_slot':X_Time})
    #avgdata_dict.update({'avgdata_names':avgdata_names})
    avgdata_dict.update({'avgdata_exemplar':avgdata_exemplar})
    avgdata_dict.update({'avgdata_zvar':avgdata_zvar})
    avgdata_dict.update({'sensor_names':X_Sensor_NAMES})
    avgdata_dict.update({'weather_names':X_Weather_NAMES})
    avgdata_dict.update({'time_names':X_Time_NAMES})
    mt.saveObjectBinary(avgdata_dict,'avgdata_dict.bin')
    
    ####################################
    # Irregular Event Extraction
    ####################################
    # Interpolatoin with outlier removal, Here we exclude weather data from irregualr event analysis
    # since weather data noramlly show slow changes in time.so we dont expect in any meaningful diffs values
    measurement_point_set,num_type_set\
    =interpolation_measurement(data_dict,sensor_list,err_rate=1,sgm_bnd=20)
    # Irregualr matrix 
    Xdiff_Mat,Xdiff_Time,Xdiff_Names\
    ,Xdiff_zero_var_list, Xdiff_zero_var_val\
    ,Xdiff_int_type_list,Xdiff_int_type_idx\
    ,Xdiff_float_type_list,Xdiff_float_type_idx\
    =build_diff_matrix(measurement_point_set,time_slots,num_type_set,sensor_list,PARALLEL=IS_USING_PARALLEL_OPT)
    
    #==============================================================================
    # This code is to fix the dimension difference in diff sensor and weather
    # WARNING: this is just a quick fix. A more elegant solution should be implemented
    #==============================================================================
    time_slots_array = np.sort(np.array(list(set(Xdiff_Time) & set(X_Time))))
    # Extract subset of X_Weather_STATE
    removed_idx_list = []
    for ridx,slot in enumerate(X_Time):
        slot_idx = np.where(time_slots_array==slot)[0]
        if len(slot_idx) == 0: # slot not in common time slots
            removed_idx_list.append(ridx)
    XDIFF_Weather_STATE = np.delete(X_Weather_STATE, removed_idx_list,axis=0)
    
    # Extract subset of Xdiff_Mat
    removed_idx_list = []
    for ridx,slot in enumerate(Xdiff_Time):
        slot_idx = np.where(time_slots_array==slot)[0]
        if len(slot_idx) == 0: # slot not in common time slots
            removed_idx_list.append(ridx)
    Xdiff_Mat = np.delete(Xdiff_Mat,removed_idx_list,axis=0)
    
    # Update Xdiff_Time
    Xdiff_Time = time_slots_array
    XDIFF_Weather_STATE = np.array(XDIFF_Weather_STATE)    
    #==============================================================================
    # End of fix            
    #==============================================================================   

    # From below all index are reference to X_Feature
    xdiff_sf_idx=Xdiff_float_type_idx;
    xdiff_sf_name=Xdiff_float_type_list;
    xdiff_si_idx=Xdiff_int_type_idx;
    xdiff_si_name=Xdiff_int_type_list
            
    # Float Type Measurement Clustering
    X_Diff_sfe,sf_diff_exemplars_dict,exemplars_,labels_\
    =cluster_measurement_points(Xdiff_Mat[:,xdiff_sf_idx],xdiff_sf_name,corr_bnd=[0.1,0.9])
    xdiff_sfe_idx=list(np.array(xdiff_sf_idx)[exemplars_])
    # InT Type Measurement Clustering
    X_Diff_sie,si_diff_exemplars_dict,exemplars_,labels_\
    =cluster_measurement_points(Xdiff_Mat[:,xdiff_si_idx],xdiff_si_name,corr_bnd=[0.1,0.9])
    xdiff_sie_idx=list(np.array(xdiff_si_idx)[exemplars_])
     
    xdiff_sfe_state,xdiff_sfe_corr_val\
    =X_INPUT_to_states(X_Diff_sfe,CORR_VAL_OUT=1,PARALLEL =IS_USING_PARALLEL_OPT) # sensor -float type
    xdiff_sie_state=X_Diff_sie # sensor -integer type
    

    
    empty_states=np.array([[] for i in range(len(Xdiff_Time))])
    if len(xdiff_sfe_state)==0: xdiff_sfe_state=empty_states
    
    if len(xdiff_sie_state)==0: xdiff_sie_state=empty_states
    
    if len(wf_state)==0: wf_state=empty_states
    
    if len(wi_state)==0: wi_state=empty_states
    

    # Exemplar sensor only    
    XDIFF_Sensor_STATE=np.append(xdiff_sfe_state,xdiff_sie_state, axis=1)
    XDIFF_Sensor_STATE=XDIFF_Sensor_STATE.astype(int)
    XDIFF_Sensor_NAMES=list(np.array(Xdiff_Names)[xdiff_sfe_idx])+list(np.array(Xdiff_Names)[xdiff_sie_idx])
    
    # months of a year,days of a week, and hours of a day
    # (Monday, Tuesday,Wendsday,Thursday,Saturday,Sunday) =(0,1,2,3,4,5,6)
    XDIFF_Time_STATE_temp=build_time_states(Xdiff_Time)
    XDIFF_Time_NAMES_temp=['MTH','WD','HR']
    XDIFF_Time_STATE=[]
    XDIFF_Time_NAMES=[]
    for xt_col,xt_name in zip(XDIFF_Time_STATE_temp.T,XDIFF_Time_NAMES_temp):
        if len(set(xt_col))>1:
            XDIFF_Time_STATE.append(xt_col)
            XDIFF_Time_NAMES.append(xt_name)
    
    XDIFF_Time_STATE=np.array(XDIFF_Time_STATE).T
    #################################################
    # FORMATTED DATA  FOR IRREGUALR EVENT
    #################################################
    #** Save this variables***
    #diffdata_mat = np.hstack([XDIFF_Sensor_STATE,X_Weather_STATE,XDIFF_Time_STATE])
    #diffdata_names = XDIFF_Sensor_NAMES+X_Weather_NAMES+XDIFF_Time_NAMES
    diffdata_exemplar=dict(sf_diff_exemplars_dict.items()+si_diff_exemplars_dict.items())
    diffdata_zvar=Xdiff_zero_var_list
    
    diffdata_dict={}
    #diffdata_dict.update({'diffdata_mat':diffdata_mat})
    diffdata_dict.update({'diffdata_state_mat':XDIFF_Sensor_STATE})
    #diffdata_dict.update({'diffdata_weather_mat':X_Weather_STATE})
    diffdata_dict.update({'diffdata_weather_mat':XDIFF_Weather_STATE})
    diffdata_dict.update({'diffdata_time_mat':XDIFF_Time_STATE})
    diffdata_dict.update({'diff_time_slot':Xdiff_Time})
    #diffdata_dict.update({'diffdata_names':diffdata_names})
    diffdata_dict.update({'diffdata_exemplar':diffdata_exemplar})
    diffdata_dict.update({'diffdata_zvar':diffdata_zvar})
    diffdata_dict.update({'sensor_names':XDIFF_Sensor_NAMES})
    diffdata_dict.update({'weather_names':X_Weather_NAMES})
    diffdata_dict.update({'time_names':X_Time_NAMES})
    mt.saveObjectBinary(diffdata_dict,'diffdata_dict.bin')
    
    
EVENT_ANALYSIS=0
if EVENT_ANALYSIS==1:
    # 0-nb distance analysis
    ####################################################
    # Probabiity Computatoin
    #---------------------------------------------------
    #  - Ranking output. 
    #  - Effect Prob Analysis
    #  - Causal Prob Analysis
    ####################################################
    diffdata_dict = mt.loadObjectBinary('diffdata_dict.bin')
    avgdata_dict = mt.loadObjectBinary('avgdata_dict.bin')
    
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

    ###############################################################################################    
    # Regualr Event Analysis
    #avgdata_state_mat,avgdata_weather_mat, avgdata_time_mat, avg_time_slot
    #avgdata_exemplar, avgdata_zvar, avgsensor_names, avgweather_names, avgtime_names
    ###############################################################################################    
    
    
    #****** Complete Analysis Script***** #
    ######################################################################    
    #1. effect prob - time dependecy analysis
    ######################################################################    

    # Temporary for correcting month change    

    
    ######################################################################
    # Use this for special cases
    ######################################################################
    """
    monthly_variability,monthly_structure_score\
    =time_effect_analysis_all(data_mat,data_name,avgtime_names,avgsensor_names)
    start_t=time.time()
    s_name=avgsensor_names[0]
    state_list,s_prob_log,time_effect_mat_dist,score_in_structure,valid_mon_list,state_list=\
    time_effect_analysis(data_mat,data_name,avgtime_names,avgsensor_names[0],DO_PLOT=True)
    end_t=time.time()
    print 'Total-- ',end_t-start_t, 'secs'
    plot_time_effect(s_name,state_list,valid_mon_list,s_prob_log)
    
    wf_tuple=wf_tuple_t
    plot_weather_sensitivity(wf_tuple[0],wf_tuple[1],wf_tuple[2],wf_tuple[3],wf_tuple[4],\
    avgsensor_names,Conditions_dict,Events_dict,sort_opt='desc',num_of_picks=9)            
    """
    
    Conditions_dict=data_dict['Conditions_dict'].copy()
    Events_dict=data_dict['Events_dict'].copy()
    data_state_mat=avgdata_state_mat
    data_time_mat=avgdata_time_mat
    data_weather_mat=avgdata_weather_mat
    sensor_names=avgsensor_names
    time_names=avgtime_names
    weather_names=avgweather_names
    bldg_tag='VAK_' # building tag
    trf_tag='avg_' # transformation tag
    dst_t='h'
    vak_avg_wtf_tuple,vak_avg_weather_dict=wt_sensitivity_analysis(data_state_mat,data_time_mat,data_weather_mat,sensor_names,time_names,\
     Conditions_dict,Events_dict,bldg_tag,trf_tag,weather_names,dst_t='h')
    
    Conditions_dict=data_dict['Conditions_dict'].copy()
    Events_dict=data_dict['Events_dict'].copy()
    data_state_mat=diffdata_state_mat
    data_time_mat=diffdata_time_mat
    data_weather_mat=diffdata_weather_mat
    sensor_names=diffsensor_names
    time_names=difftime_names
    weather_names=diffweather_names
    bldg_tag='VAK_' # building tag
    trf_tag='diff_' # transformation tag
    dst_t='h'
    vak_diff_wtf_tuple,vak_diff_weather_dict=wt_sensitivity_analysis(data_state_mat,data_time_mat,data_weather_mat,sensor_names,time_names,\
     Conditions_dict,Events_dict,bldg_tag,trf_tag,weather_names,dst_t='h')

    
    ###############################################################################################    
    # Irregualr Event Analysis
    #avgdata_state_mat,avgdata_weather_mat, avgdata_time_mat, avg_time_slot
    #avgdata_exemplar, avgdata_zvar, avgsensor_names, avgweather_names, avgtime_names
    ###############################################################################################        
    #########################################################################
    # Computes the maximum screwness of distribution of sensors
    #  max_{i,j} abs(p_i-p_j)/p_i*p_j such that p_i, p_j ~=0 
    #########################################################################
    
    #plot(irr_state_mat[:,skewness_metric_sort_idx[12]],'-s')
    num_of_picks=10
    rare_event_sensors=list(np.array(diffsensor_names)[skewness_metric_sort_idx[0:num_of_picks]])
    rare_event_sensors_scores=list(skewness_metric_sort[0:num_of_picks])
    pprint.pprint(np.array([rare_event_sensors, rare_event_sensors_scores]).T)
    
    data_mat = np.hstack([diffdata_state_mat,diffdata_time_mat])
    # Temporary for correcting month change    
    #data_mat[:,-3]=data_mat[:,-3]-1
    data_name = diffsensor_names+difftime_names
    
    dst_t='h'
    mth_prob_map,mth_state_map, mth_sensitivity,mth_list\
    = param_sensitivity(data_mat,data_name,diffsensor_names,'MTH',dst_type=dst_t)   
    wday_prob_map,wday_state_map,wday_sensitivity,wday_list\
    = param_sensitivity(data_mat,data_name,diffsensor_names,'WD',dst_type=dst_t)   
    dhr_prob_map,dhr_state_map,dhr_sensitivity,dhr_list\
    = param_sensitivity(data_mat,data_name,diffsensor_names,'HR',dst_type=dst_t)   
    
    
    tf_tuple_mth=('MTH',mth_prob_map,mth_state_map,mth_sensitivity,mth_list)
    tf_tuple_wday=('WD',wday_prob_map,wday_state_map,wday_sensitivity,wday_list)
    tf_tuple_dhr=('HR',dhr_prob_map,dhr_state_map,dhr_sensitivity,dhr_list)

    #tf_tuple=tf_tuple_mth
    ##########################################################################################
    # Genarelize this plotting
    #plot_xxx_sensitivity(tf_tuple[0],tf_tuple[1],tf_tuple[2],tf_tuple[3],tf_tuple[4],\
    #                             avgsensor_names,Conditions_dict,Events_dict,sort_opt='desc',num_of_picks=9)
    ##########################################################################################
    tf_sstv_tuple=np.array([tf_tuple_mth[3],tf_tuple_wday[3],tf_tuple_dhr[3]])
    max_tf_sstv=tf_sstv_tuple[tf_sstv_tuple<np.inf].max()*2
    tf_sstv_tuple[tf_sstv_tuple==np.inf]=max_tf_sstv
    tf_sstv_total=np.sum(tf_sstv_tuple,0)
    arg_idx_s=argsort(tf_sstv_total)[::-1]
    arg_idx_is=argsort(tf_sstv_total)
    num_of_picks=9
    print 'Most time sensitive sensors'
    print '---------------------------------------------'
    Time_Sensitive_Sensors=list(np.array(diffsensor_names)[arg_idx_s[0:num_of_picks]])
    pprint.pprint(Time_Sensitive_Sensors)
    print 'Least time sensitive sensors'
    print '---------------------------------------------'
    Time_Insensitive_Sensors=list(np.array(diffsensor_names)[arg_idx_is[0:num_of_picks]])
    pprint.pprint(Time_Insensitive_Sensors)
    
    ####################################################################
    ## Rador Plotting for Weather_Sensitive_Sensors
    ####################################################################
    sensor_no = len(diffsensor_names)
    # convert 'inf' to 1
    sen_mth = [max_tf_sstv if val == float("inf") else val for val in tf_tuple_mth[3]]
    sen_wday = [max_tf_sstv if val == float("inf") else val for val in tf_tuple_wday[3]]
    sen_dhr = [max_tf_sstv if val == float("inf") else val for val in tf_tuple_dhr[3]]
    SEN = [[sen_mth[i], sen_wday[i], sen_dhr[i]] for i in range(sensor_no)]
    TOTAL_SEN = np.array([sum(SEN[i]) for i in range(sensor_no)])
    idx = np.argsort(TOTAL_SEN)[-6:] # Best 6 sensors
    
    spoke_labels = ["Month", "Day", "Hour"]
    data = [SEN[i] for i in idx]
    sensor_labels = [diffsensor_names[i] for i in idx]
    #import radar_chart
    radar_chart.plot(data, spoke_labels, sensor_labels, saveto="time_radar.png")    

    import pdb;pdb.set_trace()
    
    """
    diffdata_state_mat=diffdata_dict['diffdata_state_mat']
    diffdata_weather_mat=diffdata_dict['diffdata_weather_mat']
    diffdata_time_mat=diffdata_dict['diffdata_time_mat']
    diff_time_slot=diffdata_dict['diff_time_slot']
    diffdata_exemplar=diffdata_dict['diffdata_exemplar']
    diffdata_zvar=diffdata_dict['diffdata_zvar']
    diffsensor_names=diffdata_dict['sensor_names']
    diffweather_names=diffdata_dict['weather_names']
    difftime_names=diffdata_dict['time_names']    
    """

    do_sampling_interval_plot=1
    if do_sampling_interval_plot==1:
        num_of_picks=5
        fig=figure('sampling interval')
        for k in range(num_of_picks):
            ax=subplot(num_of_picks,1,k)
            m_idx=skewness_metric_sort_idx[k]
            sensor_name_=diffdata_names[m_idx]
            t_=unix_to_dtime(data_dict[sensor_name_][2][0])
            plot(t_[1:],abs(diff(data_dict[sensor_name_][2][0])))
            plt.title(sensor_name_,fontsize=14,y=0.8)
            ylabel('Sampling Intervals')
        fig.savefig(fig_dir+'sampling_intervals.png')
    
    do_rare_event_compare_plot=1
    if do_rare_event_compare_plot==1:
        num_of_picks=3
        for k in range(num_of_picks):
            fig=figure('irregualr event compare'+str(k))
            m_idx=skewness_metric_sort_idx[k]
            sensor_name_=diffdata_names[m_idx]
            irr_idx=irr_data_name.index(sensor_name_)
            
            t_=unix_to_dtime(data_dict[sensor_name_][2][0])
            val_=data_dict[sensor_name_][2][1]
            subplot(4,1,1)
            plt.title(sensor_name_+' samples',fontsize=14,y=0.8)
            plot(t_,val_)
            subplot(4,1,2)
            plt.title(sensor_name_+' differential',fontsize=14,y=0.8)
            plot(t_[1:],abs(diff(val_)))
            subplot(4,1,3)
            plot(measurement_point_set[irr_idx][0],measurement_point_set[irr_idx][1])
            subplot(4,1,4)
            plt.title(sensor_name_+' irregular states',fontsize=14,y=0.8)
            plot(diff_time_slot,irr_state_mat[:,m_idx])
            
            plt.get_current_fig_manager().window.showMaximized()
            fig.savefig(fig_dir+'irr_event_compare'+str(k)+'.png')
    

BLDG_ANALYSIS=1
if BLDG_ANALYSIS==1:
    #########################################################################    
    # Case by Case Analysis.
    #########################################################################    
    ##############################
    # VTT VTT_POWER data
    ##############################
    VTT_LOAD=0
    if VTT_LOAD==1:
        print 'VTT_POWER data loading ...'
        # VTT_POWER data loading ...
        avgdata_dict = mt.loadObjectBinaryFast('./VTT_POWER/avgdata_dict.bin')
        avgdata_dict=obj(avgdata_dict)
        gw2_power=mt.loadObjectBinaryFast('./VTT_POWER/GW2.CG_PHASE1_ACTIVE_POWER_M.bin')
        X_Feature=mt.loadObjectBinaryFast('./VTT_POWER/X_Feature.bin')
        X_names=mt.loadObjectBinaryFast('./VTT_POWER/X_names.bin')
        X_Time=mt.loadObjectBinaryFast('./VTT_POWER/X_Time.bin')
        Xdiff_Mat=mt.loadObjectBinaryFast('./VTT_POWER/Xdiff_Mat.bin')
        Xdiff_Names=mt.loadObjectBinaryFast('./VTT_POWER/Xdiff_Names.bin')
        Xdiff_Time=mt.loadObjectBinaryFast('./VTT_POWER/Xdiff_Time.bin')
    
    #########################################################################    
    ### Load all builings bin files. 
    #########################################################################
    bldg_dict={}
    RUN_VTT_BLDG=0
    if RUN_VTT_BLDG==1:
        sig_tag_set=['avg','diff']
        bldg_tag_set=['GW1_','GW2_','VAK1_','VAK2_']
        dict_dir_set=['./GW1_results/','./GW2_results/','./VAK1_results/','./VAK2_results/']
        pname_key='POWER'
        for dict_dir,bldg_tag in zip(dict_dir_set,bldg_tag_set):
            bldg_dict.update({bldg_tag:create_bldg_obj(dict_dir,bldg_tag,pname_key)})
        bldg_=obj(bldg_dict)
    else:
        bldg_dict={'GW1_':mt.loadObjectBinaryFast('GW1_.bin'),'GW2_':mt.loadObjectBinaryFast('GW2_.bin')\
        ,'VAK1_':mt.loadObjectBinaryFast('VAK1_.bin'),'VAK2_':mt.loadObjectBinaryFast('VAK2_.bin')}
        bldg_=obj(bldg_dict)
        
    
    RUN_GSBC_BLDG=1
    if RUN_GSBC_BLDG==1:
        #gsbc_dict_dir_set=['./GSBC/allsensors/','./GSBC/seleceted/']
        #gsbc_dict_dir_set=['./GSBC/allsensors/']
        gsbc_dict_dir_set=['./GSBC/selected/']
        bldg_tag_set=['GSBC_']
        print 'Building for ',bldg_tag_set, '....'
        gsbc_hcw_pname_key='3003....' # Hot and Cold water
        gsbc_main_1_pname_key='300401..' # Maing Buiding F1
        gsbc_main_2_pname_key='300402..' # Maing Buiding F2
        gsbc_hvac_pname_key='3006....' # HVAC
        for dict_dir,bldg_tag in zip(gsbc_dict_dir_set,bldg_tag_set):
            bldg_dict.update({bldg_tag:create_bldg_obj(dict_dir,bldg_tag,gsbc_hcw_pname_key)})
        bldg_=obj(bldg_dict)

    import pdb;pdb.set_trace()
    
    PLOTTING_LH=0
    if PLOTTING_LH==1:
        plotting_bldg_lh(bldg_,attr_class='sensor',num_picks=30)
        plotting_bldg_lh(bldgbldg_obj_,attr_class='time',num_picks=30)
        plotting_bldg_lh(bldg_,attr_class='weather',num_picks=30)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    import lib_bnlearn as rbn    
    ###########################
    # BLDG = GW2_ Analysis    
    ###########################
    
    #plotting_bldg_lh(bldg_,bldg_key='GW2_',attr_class='sensor',num_picks=30)
    #import pdb;pdb.set_trace()
    
    """
    ###############################################################
    # 1. Regualr Events
    ###############################################################
    def bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag='avg',num_picks_bn=15,learning_alg='hc'):
        cmd_str_='s_names=bldg_obj.'+sig_tag+'.sensor_names'
        exec(cmd_str_)
        p_idx=s_names.index(p_name)        
        cmd_str_='data_state_mat=bldg_obj.'+sig_tag+'.data_state_mat'
        exec(cmd_str_)
        if not (attr=='all') :        
            cmd_str_='optprob_set=bldg_obj.analysis.'+sig_tag+'.__dict__[p_name].'+attr+'.optprob_set'
            exec(cmd_str_)
            cmd_str_='optstate_set=bldg_obj.analysis.'+sig_tag+'.__dict__[p_name].'+attr+'.optstate_set'
            sort_idx=np.argsort(optprob_set)[::-1]
            
        if (attr=='sensor') :
            print 'power - sensors...'
            cmd_str_='s_names=bldg_obj.'+sig_tag+'.sensor_names'
            exec(cmd_str_)
            idx_select=[p_idx]+ list(sort_idx[:num_picks_bn])
            cmd_str_='bndata_mat=bldg_obj.'+sig_tag+'.data_state_mat[:,idx_select]'
            exec(cmd_str_)
            cols=[s_names[k] for k in idx_select]
        elif (attr=='weather'):
            print 'power - weather...'
            cmd_str_='w_names=bldg_obj.'+sig_tag+'.weather_names'
            exec(cmd_str_)
            cmd_str_='bndata_mat=np.vstack((bldg_obj.'+sig_tag+\
            '.data_state_mat[:,p_idx].T,bldg_obj.'+sig_tag+'.data_weather_mat_.T)).T'
            exec(cmd_str_)
            cols=[p_name]+[w_name for w_name in w_names] 
        elif (attr=='time'):
            print 'power - time...'
            cmd_str_='t_names=bldg_obj.'+sig_tag+'.time_names'
            exec(cmd_str_)
            cmd_str_='bndata_mat=np.vstack((bldg_obj.'+sig_tag+\
            '.data_state_mat[:,p_idx].T,bldg_obj.'+sig_tag+'.data_time_mat.T)).T'
            exec(cmd_str_)
            cols=[p_name]+[t_name for t_name in t_names] 
        elif (attr=='all'):
            print 'power - sensors + weather + time ...'
            s_cause_label,s_labels,s_hc=\
            bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag=sig_tag,num_picks_bn=num_picks_bn,learning_alg=learning_alg)
            t_cause_label,t_labels,t_hc=\
            bn_anaylsis(bldg_obj,p_name,attr='time',sig_tag=sig_tag,num_picks_bn=num_picks_bn,learning_alg=learning_alg)
            w_cause_label,w_labels,w_hc=\
            bn_anaylsis(bldg_obj,p_name,attr='weather',sig_tag=sig_tag,num_picks_bn=num_picks_bn,learning_alg=learning_alg)
            #s_cause_label=s_labels; w_cause_label=w_labels;t_cause_label=t_labels
            cmd_str_='s_cause_idx=[bldg_obj.'+sig_tag+'.sensor_names.index(name_) for name_ in s_cause_label]'
            exec(cmd_str_)
            cmd_str_='t_cause_idx=[bldg_obj.'+sig_tag+'.time_names.index(name_) for name_ in t_cause_label]'
            exec(cmd_str_)
            cmd_str_='w_cause_idx=[bldg_obj.'+sig_tag+'.weather_names.index(name_) for name_ in w_cause_label]'
            exec(cmd_str_)
            cmd_str_='bndata_mat=np.vstack((bldg_obj.'+sig_tag+'.data_state_mat[:,p_idx].T,\
            bldg_obj.'+sig_tag+'.data_state_mat[:,s_cause_idx].T, \
            bldg_obj.'+sig_tag+'.data_weather_mat_[:,w_cause_idx].T, \
            bldg_obj.'+sig_tag+'.data_time_mat[:,t_cause_idx].T)).T'
            exec(cmd_str_)
            cmd_str_='cols=[name_ for name_ in [p_name]+s_cause_label+w_cause_label+t_cause_label]'
            exec(cmd_str_)
        else:
            print 'error'
            return 0
        if (attr=='all'):
            b_arc_list = pair_in_idx([p_name],s_cause_label+ w_cause_label+t_cause_label)+\
            pair_in_idx(s_cause_label,w_cause_label+t_cause_label)+\
            pair_in_idx(w_cause_label,t_cause_label)+\
            pair_in_idx(t_cause_label,t_cause_label)
            #import pdb;pdb.set_trace()
        else:
            b_arc_list = pair_in_idx([cols[0]],cols[1:])
        black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
        factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
        data_frame = rbn.construct_data_frame(factor_data_mat,cols)
        if learning_alg=='tabu':
            hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
        elif learning_alg=='mmhc':
            hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
        else:
            hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
        amat = rbn.py_get_amat(hc_b)
        cause_label=list(np.array(cols)[np.nonzero(amat[:,0]==1)[0]])
        cause_idx=[cols.index(label_) for label_ in cause_label]
        return cause_label,cols,hc_b
"""

    
    
    bldg_obj=bldg_.GW2_
    p_name=bldg_obj.analysis.avg.__dict__.keys()[0]
    s_cause_label,s_labels,s_hc=bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag='avg',num_picks_bn=15)
    t_cause_label,t_labels,t_hc=bn_anaylsis(bldg_obj,p_name,attr='time',sig_tag='avg',num_picks_bn=15)
    w_cause_label,w_labels,w_hc=bn_anaylsis(bldg_obj,p_name,attr='weather',sig_tag='avg',num_picks_bn=15)
    all_cause_label,all_labels,all_hc=bn_anaylsis(bldg_obj,p_name,attr='all',sig_tag='avg',num_picks_bn=15)
    import pdb;pdb.set_trace()

    bldg_obj=bldg_.GSBC_
    p_name=bldg_obj.analysis.avg.__dict__.keys()[0]
    s_cause_label,s_labels,s_hc=bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag='avg',num_picks_bn=15)
    t_cause_label,t_labels,t_hc=bn_anaylsis(bldg_obj,p_name,attr='time',sig_tag='avg',num_picks_bn=15)
    w_cause_label,w_labels,w_hc=bn_anaylsis(bldg_obj,p_name,attr='weather',sig_tag='avg',num_picks_bn=15)
    all_cause_label,all_labels,all_hc=bn_anaylsis(bldg_obj,p_name,attr='all',sig_tag='avg',num_picks_bn=15)

    
    
   # Plotting....
    #plt.ioff()
    fig1=rbn.nx_plot(s_hc,s_labels)
    fig2=rbn.nx_plot(t_hc,t_labels)
    fig3=rbn.nx_plot(w_hc,w_labels)
    fig4=rbn.nx_plot(all_hc,all_labels)

    p_name=bldg_obj.analysis.avg.__dict__.keys()[0]
    all_cause_label,all_labels,all_hc=bn_anaylsis(bldg_obj,p_name,attr='all',sig_tag='avg',num_picks_bn=20)         
    fig=rbn.nx_plot(all_hc,all_labels)
    
    p_name=bldg_obj.analysis.diff.__dict__.keys()[0]
    all_cause_label,all_labels,all_hc=bn_anaylsis(bldg_obj,p_name,attr='all',sig_tag='diff',num_picks_bn=20)         
    fig=rbn.nx_plot(all_hc,all_labels)
    
    
    png_name=str(uuid.uuid4().get_hex().upper()[0:2])
    plt.savefig(fig_dir+p_name+'_'+sig_tag+'_bn_sensors_'+png_name+'.png', bbox_inches='tight')
    plt.close()
    plt.ion()
    
    import pdb;pdb.set_trace()

    #fig=figure(figsize=(10,10))
    plt.ioff()
    fig=figure()
    for k in range(len(cause_idx)):
        effect_idx=cols.index(p_name)
        peak_state_0, peak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[k]],[[effect_idx]],[[PEAK]])
        lowpeak_state_0, lowpeak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[k]],[[effect_idx]],[[LOW_PEAK]])
        sort_idx1=argsort(peak_state_0)
        sort_idx2=argsort(lowpeak_state_0)
        subplot(1,len(cause_idx),k+1)
        plot(sort(peak_state_0), np.array(peak_prob_0)[sort_idx1],'-^')
        plot(sort(lowpeak_state_0), np.array(lowpeak_prob_0)[sort_idx2],'-v')
        plt.legend(('measurements','classified states'))
        if k==0:
            plt.ylabel('Probability of Peak Power Demand')
        plt.grid()
        plt.legend(('High Peak', 'Low Peak'),loc='center right')
        plt.xlabel(s_cause_label[k])
        if len(peak_state_0)==len(stateDict.keys()):
            if sum(abs(sort(stateDict.keys())-sort(peak_state_0)))==0:
                plt.xticks(stateDict.keys(),stateDict.values(),rotation=0, fontsize=12)
                
    png_name=str(uuid.uuid4().get_hex().upper()[0:2])
    plt.savefig(fig_dir+p_name+'_'+sig_tag+'_bn_sensors_lh_out'+png_name+'.png', bbox_inches='tight')
    plt.close()
    plt.ion()

        
    #############################################3
    # plotting ant result
    #############################################3
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
      
    fcause_label=list(np.array(cols_fnames)[np.nonzero(amat[:,0]==1)[0]])
    start_t=datetime.datetime(2014, 1, 19, 0, 0, 0)
    end_t=datetime.datetime(2014, 1, 25, 0, 0, 0)
    data_2=get_data_set(fcause_label+effect_label,start_t,end_t)
    # data_x=get_data_set([cause_label[1]]+[cause_label[3]]+effect_label,start_t,end_t)
    png_namex=plot_data_x(data_2,stype='raw',smark='-')
    #png_namex=plot_data_x(data_x,stype='diff',smark='-^')    
        # Check the probability 
    plt.plot(peak_state_temp,peak_prob_temp,'-^')
    plt.plot(lowpeak_state_temp,lowpeak_prob_temp,'-v')
    plt.title(cause_label)
    plt.xlabel('Measurements')
    plt.ylabel('Probability of State of Power Demand')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.savefig(fig_dir+p_name_+'_'+sig_tag+'_cause_prob.png', bbox_inches='tight')    
    
    data_1=get_data_set(cause_label+effect_label)
    avg_png_name=plot_data_x(data_1)

    
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###############################################################
    # 1. Regualr Events for GW2.CG_SYSTEM_ACTIVE_POWER_M
    ###############################################################
    num_picks=30
    sig_tag='avg'
    optprob_set=GW2_.analysis.avg.GW2_CG_SYSTEM_ACTIVE_POWER_M.Sensor.optprob_set
    optstate_set=GW2_.analysis.avg.GW2_CG_SYSTEM_ACTIVE_POWER_M.Sensor.optstate_set
    s_names=GW2_.avgsensor_names
    p_name='GW2.CG_SYSTEM_ACTIVE_POWER_M'
    
    sort_idx=argsort(optprob_set)[::-1]
    sort_lh=optprob_set[sort_idx[:num_picks]].T
    sort_state=optstate_set[sort_idx[:num_picks]].T
    sort_label= list(np.array(s_names)[sort_idx[:num_picks]])
    
    data_state_mat=GW2_.avgdata_state_mat
    lh_threshold=0.9
    cause_idx=list(np.nonzero(optprob_set>lh_threshold)[0])
    cause_label=[GW2_.avgsensor_names[idx] for idx in cause_idx]
    effect_idx=GW2_.avgsensor_names.index(p_name)
    effect_label=[p_name]
    
    # For PEAK Demand
    obs_state=PEAK
    peak_state_temp, peak_prob_temp=compute_cause_likelihood(data_state_mat,cause_idx,[[effect_idx]],[[obs_state]])
    # For LOW PEAK Demand
    obs_state=LOW_PEAK
    lowpeak_state_temp,lowpeak_prob_temp=compute_cause_likelihood(data_state_mat,cause_idx,[[effect_idx]],[[obs_state]])
    # Check the probability 
    plt.plot(peak_state_temp,peak_prob_temp,'-^')
    plt.plot(lowpeak_state_temp,lowpeak_prob_temp,'-v')
    plt.title(cause_label)
    plt.xlabel('Measurements')
    plt.ylabel('Probability of State of Power Demand')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.savefig(fig_dir+p_name_+'_'+sig_tag+'_cause_prob.png', bbox_inches='tight')    
    
    data_1=get_data_set(cause_label+effect_label)
    avg_png_name=plot_data_x(data_1)
    
    import lib_bnlearn as rbn    
    num_picks=10
    p_idx=GW2_.avgsensor_names.index(p_name)
    idx_select=[p_idx]+ list(sort_idx[:num_picks])
    bndata_mat=GW2_.avgdata_state_mat[:,idx_select]
    # File name format - allowing dot
    cols_fnames=[GW2_.avgsensor_names[k] for k in idx_select]
    # Variable name format - replacing dot with underscore 
    cols=[remove_dot(GW2_.avgsensor_names[k]) for k in idx_select]
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    fig=rbn.nx_plot(hc_b,cols)
    amat = rbn.py_get_amat(hc_b)
    plt.savefig(fig_dir+p_name+'_'+sig_tag+'bn_sensors.png', bbox_inches='tight')
    
    
    s_cause_label=list(np.array(cols)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[cols.index(label_) for label_ in s_cause_label]
    #fig=figure(figsize=(10,10))
    fig=figure()
    for k in range(len(cause_idx)):
        effect_idx=cols_fnames.index(p_name)
        peak_state_0, peak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[k]],[[effect_idx]],[[PEAK]])
        lowpeak_state_0, lowpeak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[k]],[[effect_idx]],[[LOW_PEAK]])
        sort_idx1=argsort(peak_state_0)
        sort_idx2=argsort(lowpeak_state_0)
        subplot(1,len(cause_idx),k+1)
        plot(sort(peak_state_0), np.array(peak_prob_0)[sort_idx1],'-^')
        plot(sort(lowpeak_state_0), np.array(lowpeak_prob_0)[sort_idx2],'-v')
        plt.legend(('measurements','classified states'))
        if k==0:
            plt.ylabel('Probability of Peak Power Demand')
        plt.grid()
        plt.legend(('High Peak', 'Low Peak'),loc='center right')
        plt.xlabel(wcause_label[k])
        if len(peak_state_0)==len(stateDict.keys()):
            if sum(abs(sort(stateDict.keys())-sort(peak_state_0)))==0:
                plt.xticks(stateDict.keys(),stateDict.values(),rotation=0, fontsize=12)
        
        
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
      
    fcause_label=list(np.array(cols_fnames)[np.nonzero(amat[:,0]==1)[0]])
    start_t=datetime.datetime(2014, 1, 19, 0, 0, 0)
    end_t=datetime.datetime(2014, 1, 25, 0, 0, 0)
    data_2=get_data_set(fcause_label+effect_label,start_t,end_t)
    # data_x=get_data_set([cause_label[1]]+[cause_label[3]]+effect_label,start_t,end_t)
    png_namex=plot_data_x(data_2,stype='raw',smark='-')
    #png_namex=plot_data_x(data_x,stype='diff',smark='-^')
    
    
    
    
    ###############################################################
    # 2. Irregualr Events for GW2.CG_SYSTEM_ACTIVE_POWER_M
    ###############################################################
    bldg_tag='GW2_'
    sig_tag='diff'
    p_name='GW2.CG_PHASE2_ACTIVE_POWER_M'
    cmd_str_='optprob_set='+bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name)+'.Sensor.optprob_set'
    exec(cmd_str_)
    cmd_str_='optstate_set='+bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name)+'.Sensor.optstate_set'
    exec(cmd_str_)
    cmd_str_='s_names='+bldg_tag+sig_tag+'sensor_names'
    exec(cmd_str_)
    
    sort_idx=argsort(optprob_set)[::-1]
    sort_lh=optprob_set[sort_idx[:num_picks]].T
    sort_state=optstate_set[sort_idx[:num_picks]].T
    sort_label= list(np.array(s_names)[sort_idx[:num_picks]])
    # BN Network Learning
    import lib_bnlearn as rbn    
    num_picks=15
    p_idx=GW2_.diffsensor_names.index(p_name)
    idx_select=[p_idx]+ list(sort_idx[:num_picks])
    bndata_mat=GW2_.diffdata_state_mat[:,idx_select]
    # File name format - allowing dot
    cols_fnames=[GW2_.diffsensor_names[k] for k in idx_select]
    # Variable name format - replacing dot with underscore 
    cols=[remove_dot(GW2_.diffsensor_names[k]) for k in idx_select]
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    #fit = rbn.py_bn_fit(hc_b,data_frame)
    #index_temp=2
    #prob_dimnames,prob_factors,prob_mat = rbn.py_get_node_cond_mat(fit,index_temp)
    data_state_mat=GW2_.diffdata_state_mat
    cause_label=list(np.array(cols_fnames)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[GW2_.diffsensor_names.index(label_) for label_ in cause_label]
    effect_idx=GW2_.diffsensor_names.index(p_name)
    effect_label=[p_name]
    obs_state=PEAK
    peak_state_temp, peak_prob_temp=compute_cause_likelihood(data_state_mat,cause_idx,[[effect_idx]],[[obs_state]])
    
    obs_state=LOW_PEAK
    lowpeak_state_temp, lowpeak_prob_temp=compute_cause_likelihood(data_state_mat,cause_idx,[[effect_idx]],[[obs_state]])
    
    plt.plot(peak_state_temp,peak_prob_temp,'-^')
    plt.plot(lowpeak_state_temp,lowpeak_prob_temp,'-v')
    plt.title(cause_label,fontsize='large')
    plt.xlabel('Measurements',fontsize='large')
    plt.ylabel('Probability of State of Power Demand Variation',fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid()
    plt.legend(('High Variation', 'No Variation'),prop={'size':18})
    plt.savefig(fig_dir+p_name_+'_'+sig_tag+'_variaiton_cause_prob.png', bbox_inches='tight')    
    
    data_2=get_data_set(cause_label+effect_label)
    diff_png_name=plot_data_x(data_2,type='diff')
    #sensors_=list(np.array(cols_fnames)[np.nonzero(amat[:,0]==1)[0]])
    
    
    ###############################################################
    # 3. Time and Weahter Dependency Analysis
    # Weather data dependency
    # BN Network Learning
    ###############################################################
    fig1=figure()
    plot(GW2_.avg_time_slot,GW2_.avgdata_weather_mat[:,-1])
    plot(GW2_.avg_time_slot,GW2_.avgdata_weather_mat_[:,-1],'*r')
    ylabel(GW2_.avgweather_names[-1])
    plt.legend(('measurements','classified states'))
    mn_=min(GW2_.avgdata_weather_mat[:,-1])
    mx_=max(GW2_.avgdata_weather_mat[:,-1])
    ylim([mn_-0.1*abs(mn_),mx_+0.1*abs(mx_)])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig1.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig2=figure()
    plot(GW2_.avg_time_slot,GW2_.avgdata_weather_mat[:,-2])
    plot(GW2_.avg_time_slot,GW2_.avgdata_weather_mat_[:,-2],'*r')
    plt.legend(('measurements','classified states'))
    ylabel(GW2_.avgweather_names[-2])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig2.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig3=figure()
    plot(GW2_.avg_time_slot,GW2_.avgdata_weather_mat[:,-3])
    plot(GW2_.avg_time_slot,GW2_.avgdata_weather_mat_[:,-3],'*r')
    plt.legend(('measurements','classified states'))
    ylabel(GW2_.avgweather_names[-3])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig3.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    # Likelihood of weather factors
    optprob_set=GW2_.analysis.avg.GW2_CG_SYSTEM_ACTIVE_POWER_M.weather.optprob_set
    w_names=GW2_.avg.weather_names
    sort_idx=argsort(optprob_set)[::-1]
    sort_lh=optprob_set[sort_idx].T
    sort_state=optstate_set[sort_idx].T
    figw=figure(figsize=(15.0,10.0))
    #figw=figure()
    plt.subplot(2,1,1)
    plt.plot(sort_lh,'-s')
    x_label= list(np.array(w_names)[sort_idx])
    x_ticks=range(len(x_label))
    #plt.xticks(x_ticks,x_label, fontsize="small")
    plt.xticks(x_ticks,x_label,rotation=30, fontsize=12)
    plt.tick_params(labelsize='large')
    plt.ylabel('Likelihood (From 0 to 1)',fontsize=18)
    #plt.get_current_fig_manager().window.showMaximized()
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    figw.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    # regualr event
    sig_tag='avg'
    p_name='GW2_CG_SYSTEM_ACTIVE_POWER_M'
    p_idx=GW2_.avg.sensor_names.index(p_name)
    bndata_mat=np.vstack((GW2_.avg.data_state_mat[:,p_idx].T,GW2_.avg.data_weather_mat_.T)).T
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols=[p_name]+[w_name for w_name in GW2_.avg.weather_names]
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    wcause_label=list(np.array(cols)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[cols.index(label_) for label_ in wcause_label]
    effect_idx=cols_fnames.index(p_name)
    peak_state_0, peak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[0]],[[effect_idx]],[[PEAK]])
    lowpeak_state_0, lowpeak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[0]],[[effect_idx]],[[LOW_PEAK]])
    sort_idx1=argsort(peak_state_0)
    sort_idx2=argsort(lowpeak_state_0)
    fig0=figure()
    plot(sort(peak_state_0), np.array(peak_prob_0)[sort_idx1],'-^')
    plot(sort(lowpeak_state_0), np.array(lowpeak_prob_0)[sort_idx2],'-v')
    plt.legend(('measurements','classified states'))
    plt.ylabel('Probability of State of Power Demand')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.xlabel(wcause_label[0])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig0.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    # Irregualr event
    sig_tag='diff'
    p_name='GW2.CG_PHASE2_ACTIVE_POWER_M'
    p_idx=GW2_.diffsensor_names.index(p_name)
    bndata_mat=np.vstack((GW2_.diffdata_state_mat[:,p_idx].T,GW2_.diffdata_weather_mat_.T)).T
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols_fnames=[p_name]+[w_name for w_name in GW2_.diffweather_names]
    cols=[remove_dot(p_name)]+[remove_dot(w_name) for w_name in GW2_.diffweather_names]
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    wcause_label=list(np.array(cols)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[cols.index(label_) for label_ in wcause_label]
    effect_idx=cols_fnames.index(p_name)
    peak_state_0, peak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[0]],[[effect_idx]],[[PEAK]])
    lowpeak_state_0, lowpeak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[0]],[[effect_idx]],[[LOW_PEAK]])
    sort_idx1=argsort(peak_state_0)
    sort_idx2=argsort(lowpeak_state_0)
    fig0=figure()
    plot(sort(peak_state_0), np.array(peak_prob_0)[sort_idx1],'-^')
    plot(sort(lowpeak_state_0), np.array(lowpeak_prob_0)[sort_idx2],'-v')
    plt.legend(('measurements','classified states'))
    plt.ylabel('Probability of State of Power Demand')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.xlabel(wcause_label[0])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig0.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    """
    peak_state_1, peak_prob_1=compute_cause_likelihood(bndata_mat,[cause_idx[1]],[[effect_idx]],[[PEAK]])
    lowpeak_state_1, lowpeak_prob_1=compute_cause_likelihood(bndata_mat,[cause_idx[1]],[[effect_idx]],[[LOW_PEAK]])
    sort_idx1=argsort(peak_state_1)
    sort_idx2=argsort(lowpeak_state_1)
    fig1=figure()
    plot(sort(peak_state_1), np.array(peak_prob_1)[sort_idx1],'-^')
    plot(sort(lowpeak_state_1), np.array(lowpeak_prob_1)[sort_idx2],'-v')
    plt.legend(('measurements','classified states'))
    plt.ylabel('Probability of State of Power Demand')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.xlabel(wcause_label[1])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig1.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    """
    # Time data dependency - Likelihood of time factors
    # BN Network Learning 
    # Regualr event
    state_map=np.array(GW2_.analysis.avg.GW2_CG_SYSTEM_ACTIVE_POWER_M.Time.state_map)
    prob_map=np.array(GW2_.analysis.avg.GW2_CG_SYSTEM_ACTIVE_POWER_M.Time.prob_map)
    t_name_set=GW2_.avgtime_names
    # [MTH', 'WD', 'HR']
    sig_tag='avg'
    p_name='GW2.CG_SYSTEM_ACTIVE_POWER_M'
    p_idx=GW2_.avgsensor_names.index(p_name)
    bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_time_mat.T)).T
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols_fnames=[p_name]+[w_name for w_name in GW2_.avgtime_names]
    cols=[remove_dot(p_name)]+[remove_dot(w_name) for w_name in GW2_.avgtime_names]
    effect_idx=cols_fnames.index(p_name)
    time_high_peak_liklihood_set=[]
    time_low_peak_liklihood_set=[]
    for t_name in t_name_set:
        idx_t=cols.index(t_name)
        peak_state, peak_prob=compute_cause_likelihood(bndata_mat,[idx_t],[[effect_idx]],[[PEAK]])
        time_high_peak_liklihood_set.append(np.array([peak_state,peak_prob]))
        peak_state, peak_prob=compute_cause_likelihood(bndata_mat,[idx_t],[[effect_idx]],[[LOW_PEAK]])
        time_low_peak_liklihood_set.append(np.array([peak_state,peak_prob]))
    
    fig=figure()
    subplot(3,1,1)
    plot(time_high_peak_liklihood_set[0][0],time_high_peak_liklihood_set[0][1],'-^')
    plot(time_low_peak_liklihood_set[0][0],time_low_peak_liklihood_set[0][1],'-v')
    plt.xticks(monthDict.keys(),monthDict.values())
    plt.xlabel('Months of a year',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    subplot(3,1,2)
    plot(time_high_peak_liklihood_set[1][0],time_high_peak_liklihood_set[1][1],'-^')
    plot(time_low_peak_liklihood_set[1][0],time_low_peak_liklihood_set[1][1],'-v')
    plt.xticks(weekDict.keys(),weekDict.values())
    plt.xlabel('Days of a Week',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.tick_params(labelsize='large')
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    subplot(3,1,3)
    plot(time_high_peak_liklihood_set[2][0],time_high_peak_liklihood_set[2][1],'-^')
    plot(time_low_peak_liklihood_set[2][0],time_low_peak_liklihood_set[2][1],'-v')
    plt.xticks(hourDict.keys(),hourDict.values())
    plt.xlabel('Hours of a day',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'))
    plt.tick_params(labelsize='large')
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    # BN Network representaiton
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    #hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
        
    
    # Irregualr event
    state_map=np.array(GW2_.analysis.diff.GW2_CG_PHASE2_ACTIVE_POWER_M.Time.state_map)
    prob_map=np.array(GW2_.analysis.diff.GW2_CG_PHASE2_ACTIVE_POWER_M.Time.prob_map)
    t_name_set=GW2_.avgtime_names
    # [MTH', 'WD', 'HR']
    sig_tag='diff'
    p_name='GW2.CG_PHASE2_ACTIVE_POWER_M'
    p_idx=GW2_.diffsensor_names.index(p_name)
    bndata_mat=np.vstack((GW2_.diffdata_state_mat[:,p_idx].T,GW2_.diffdata_time_mat.T)).T
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols_fnames=[p_name]+[w_name for w_name in GW2_.difftime_names]
    cols=[remove_dot(p_name)]+[remove_dot(w_name) for w_name in GW2_.difftime_names]
    effect_idx=cols_fnames.index(p_name)
    time_high_peak_liklihood_set=[]
    time_low_peak_liklihood_set=[]
    for t_name in t_name_set:
        idx_t=cols.index(t_name)
        peak_state, peak_prob=compute_cause_likelihood(bndata_mat,[idx_t],[[effect_idx]],[[PEAK]])
        time_high_peak_liklihood_set.append(np.array([peak_state,peak_prob]))
        peak_state, peak_prob=compute_cause_likelihood(bndata_mat,[idx_t],[[effect_idx]],[[LOW_PEAK]])
        time_low_peak_liklihood_set.append(np.array([peak_state,peak_prob]))
    
    fig=figure()
    subplot(3,1,1)
    plot(time_high_peak_liklihood_set[0][0],time_high_peak_liklihood_set[0][1],'-^')
    plot(time_low_peak_liklihood_set[0][0],time_low_peak_liklihood_set[0][1],'-v')
    plt.xticks(monthDict.keys(),monthDict.values())
    plt.xlabel('Months of a year',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Variaiton', 'Low Variaiton'))
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    subplot(3,1,2)
    plot(time_high_peak_liklihood_set[1][0],time_high_peak_liklihood_set[1][1],'-^')
    plot(time_low_peak_liklihood_set[1][0],time_low_peak_liklihood_set[1][1],'-v')
    plt.xticks(weekDict.keys(),weekDict.values())
    plt.xlabel('Days of a Week',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Variaiton', 'Low Variaiton'))
    plt.tick_params(labelsize='large')
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    subplot(3,1,3)
    plot(time_high_peak_liklihood_set[2][0],time_high_peak_liklihood_set[2][1],'-^')
    plot(time_low_peak_liklihood_set[2][0],time_low_peak_liklihood_set[2][1],'-v')
    plt.xticks(hourDict.keys(),hourDict.values())
    plt.xlabel('Hours of a day',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Variaiton', 'Low Variaiton'))
    plt.tick_params(labelsize='large')
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    # BN Network representaiton
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    #hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    ###############################################################
    # 4. Sensor, Weather Time Dependency Analysis
    # BN Network Learning
    ###############################################################
    # For regualr event. 
    state_map=np.array(GW2_.analysis.avg.GW2_CG_SYSTEM_ACTIVE_POWER_M.Time.state_map)
    prob_map=np.array(GW2_.analysis.avg.GW2_CG_SYSTEM_ACTIVE_POWER_M.Time.prob_map)
    t_name_set=GW2_.avgtime_names
    # [MTH', 'WD', 'HR']
    sig_tag='avg'
    p_name=['GW2.CG_SYSTEM_ACTIVE_POWER_M']
    sensor_cause_label=['GW2.SAA_UV_INDEX_M','GW2.HA49_AS_TE_KH_FM']
    weather_cause_label=['Humidity']
    time_cause_label=['MTH', 'HR']
    p_idx=[GW2_.avgsensor_names.index(temp) for temp in p_name]
    s_idx=[GW2_.avgsensor_names.index(temp) for temp in sensor_cause_label]
    w_idx=[GW2_.avgweather_names.index(temp) for temp in weather_cause_label]
    t_idx=[GW2_.avgtime_names.index(temp) for temp in time_cause_label]
    
    bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,\
     GW2_.avgdata_state_mat[:,s_idx].T, \
     GW2_.avgdata_weather_mat_[:,w_idx].T, \
      GW2_.avgdata_time_mat[:,t_idx].T)).T
     
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols_fnames=[name_ for name_ in p_name+sensor_cause_label+weather_cause_label+time_cause_label]
    cols=[remove_dot(name_) for name_ in p_name+sensor_cause_label+weather_cause_label+time_cause_label]
    # BN Network representaiton
    b_arc_list = pair_in_idx([cols[0]],cols[1:])+pair_in_idx([cols[1]],cols[2:])+pair_in_idx([cols[2]],cols[3:])+pair_in_idx([cols[3]],cols[4:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    cause_label=list(np.array(cols_fnames)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[cols_fnames.index(label_) for label_ in cause_label]
    effect_idx=[cols_fnames.index(label_) for label_ in p_name]
    effect_label=p_name
    obs_state=PEAK
    peak_state_temp, peak_prob_temp=compute_cause_likelihood(bndata_mat,cause_idx,[effect_idx],[[obs_state]])
    obs_state=LOW_PEAK
    lowpeak_state_temp, lowpeak_prob_temp=compute_cause_likelihood(bndata_mat,cause_idx,[effect_idx],[[obs_state]])
    
    peak_state=np.array(peak_state_temp)
    peak_prob=np.array(peak_prob_temp)
    lowpeak_state=np.array(lowpeak_state_temp)
    lowpeak_prob=np.array(lowpeak_prob_temp)
    # Probability 
    fig=figure(figsize=(25.0,20.0))
    for i,mon in enumerate(yearMonths):
        subplot(3,4,mon+1)
        idx=np.nonzero(peak_state[:,1]==mon)[0]
        plot(peak_state[idx,0],peak_prob[idx],'-^')
        idx=np.nonzero(lowpeak_state[:,1]==mon)[0]
        plot(lowpeak_state[idx,0],lowpeak_prob[idx],'-v')
        plt.ylabel('Likelihood',fontsize='small')
        if i>7:
            plt.xlabel(cause_label[0]+' Measurements',fontsize='small')
        title(monthDict[mon]);plt.ylim([-0.05,1.05])
        plt.legend(('High Peak', 'Low Peak'),loc='center right')
        plt.tick_params(labelsize='small')
        plt.grid()
    
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    print '----------------------------------------'
    print 'Likelihoods '
    print '----------------------------------------'
    print cause_label+['Low Peak','High Peak']
    print '----------------------------------------'
    print np.vstack((np.int0(peak_state).T,np.int0(100*lowpeak_prob).T,np.int0(100*peak_prob).T)).T
    print '----------------------------------------'
    
    
    s_val_set=set(peak_state[:,0])
    m_val_set=set(peak_state[:,1])
    Z_peak=np.ones((len(s_val_set),len(m_val_set)))*np.inf
    for i,s_val in enumerate(s_val_set):
        for j,m_val in enumerate(m_val_set):
            idx=np.nonzero((peak_state[:,0]==s_val)&(peak_state[:,1]==m_val))[0][0]
            Z_peak[i,j]=peak_prob[idx]
            
    s_val_set=set(lowpeak_state[:,0])
    m_val_set=set(lowpeak_state[:,1])
    Z_lowpeak=np.ones((len(s_val_set),len(m_val_set)))*np.inf
    for i,s_val in enumerate(s_val_set):
        for j,m_val in enumerate(m_val_set):
            idx=np.nonzero((lowpeak_state[:,0]==s_val)&(lowpeak_state[:,1]==m_val))[0][0]
            Z_lowpeak[i,j]=lowpeak_prob[idx]
            
    Z_lowpeak=lowpeak_prob.reshape((len(s_val_set),len(m_val_set)))
    Z_peak=peak_prob.reshape((len(s_val_set),len(m_val_set)))
    fig1=figure()
    im = plt.imshow(Z_peak, cmap='hot',vmin=0, vmax=1,aspect='auto')
    plt.colorbar(im, orientation='horizontal')
    plt.xticks(monthDict.keys(),monthDict.values(),fontsize='large')
    plt.yticks(range(len(s_val_set)),list(s_val_set),fontsize='large')
    plt.xlabel(cause_label[1],fontsize='large')
    plt.ylabel(cause_label[0],fontsize='large')
    plt.title('Likelihood of High-Peak')
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig1.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig2=figure()
    im = plt.imshow(Z_lowpeak, cmap='hot',vmin=0, vmax=1,aspect='auto')
    plt.colorbar(im, orientation='horizontal')
    plt.xticks(monthDict.keys(),monthDict.values(),fontsize='large')
    plt.yticks(range(len(s_val_set)),list(s_val_set),fontsize='large')
    plt.xlabel(cause_label[1],fontsize='large')
    plt.ylabel(cause_label[0],fontsize='large')
    plt.title('Likelihood of Low-Peak')
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig2.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    
    ###############################################################
    # 3. Irregualr Events for VAK1_CG_SYSTEM_REACTIVE_POWER_M
    ###############################################################
    bldg_tag='VAK1_'
    sig_tag='diff'
    p_name='VAK1.CG_SYSTEM_REACTIVE_POWER_M'
    cmd_str_='optprob_set='+bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name)+'.Sensor.optprob_set'
    exec(cmd_str_)
    cmd_str_='optstate_set='+bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name)+'.Sensor.optstate_set'
    exec(cmd_str_)
    cmd_str_='s_names='+bldg_tag+sig_tag+'sensor_names'
    exec(cmd_str_)
    
    optprob_set=VAK1_.analysis.diff.VAK1_CG_SYSTEM_REACTIVE_POWER_M.Sensor.optprob_set
    optstate_set=VAK1_.analysis.diff.VAK1_CG_SYSTEM_REACTIVE_POWER_M.Sensor.optstate_set
    s_names=VAK1_.diffsensor_names
    
    sort_idx=argsort(optprob_set)[::-1]
    sort_lh=optprob_set[sort_idx[:num_picks]].T
    sort_state=optstate_set[sort_idx[:num_picks]].T
    sort_label= list(np.array(s_names)[sort_idx[:num_picks]])
    # BN Network Learning
    import lib_bnlearn as rbn    
    num_picks=30
    p_idx=VAK1_.diffsensor_names.index(p_name)
    idx_select=[p_idx]+ list(sort_idx[:num_picks])
    bndata_mat=VAK1_.diffdata_state_mat[:,idx_select]
    # File name format - allowing dot
    cols_fnames=[VAK1_.diffsensor_names[k] for k in idx_select]
    # Variable name format - replacing dot with underscore 
    cols=[remove_dot(VAK1_.diffsensor_names[k]) for k in idx_select]
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    plt.savefig(fig_dir+p_name+'_'+sig_tag+'bn_sensors.png', bbox_inches='tight')
    
    #fit = rbn.py_bn_fit(hc_b,data_frame)
    #index_temp=2
    #prob_dimnames,prob_factors,prob_mat = rbn.py_get_node_cond_mat(fit,index_temp)
    data_state_mat=VAK1_.diffdata_state_mat
    cause_label=list(np.array(cols_fnames)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[VAK1_.diffsensor_names.index(label_) for label_ in cause_label]
    effect_idx=VAK1_.diffsensor_names.index(p_name)
    effect_label=[p_name]
    
    obs_state=PEAK
    peak_state_13, peak_prob_13=compute_cause_likelihood(data_state_mat,[cause_idx[1],cause_idx[3]],[[effect_idx]],[[obs_state]])
    print_cond_table(peak_state_13, peak_prob_13,[cause_label[1],cause_label[3]])
    obs_state=LOW_PEAK
    lowpeak_state_13, lowpeak_prob_13=compute_cause_likelihood(data_state_mat,[cause_idx[1],cause_idx[3]],[[effect_idx]],[[obs_state]])
    print_cond_table(lowpeak_state_13, lowpeak_prob_13,[cause_label[1],cause_label[3]])
    
    
    plt.plot(range(len(peak_state_13)), peak_prob_13,'-^')
    plt.plot(range(len(lowpeak_state_13)), lowpeak_prob_13,'-v')
    plt.title(cause_label[1]+cause_label[3],fontsize='large')
    plt.xlabel('State',fontsize='large')
    plt.ylabel('Probability of State of Reactuve Power Variation',fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid()
    plt.legend(('High Variation', 'No Variation'),prop={'size':18})
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    start_t=datetime.datetime(2014, 2, 20, 15, 44, 52)
    end_t=datetime.datetime(2014, 2, 24, 16, 5, 12)
    data_x=get_data_set([cause_label[1]]+[cause_label[3]]+effect_label,start_t,end_t)
    png_namex=plot_data_x(data_x,stype='raw',smark='-^')
    png_namex=plot_data_x(data_x,stype='diff',smark='-^')
    
    
    
    ############################################################################
    ############################################################################
    #<--------------------------------------------------------------------
    #<--------------------------------------------------------------------
    #<--------------------------------------------------------------------
    ###############################################################
    # 3. Time and Weahter Dependency Analysis
    # Weather data dependency
    # BN Network Learning
    ###############################################################
    bldg_tag='VAK1_'
    sig_tag='diff'
    p_name='VAK1.CG_SYSTEM_REACTIVE_POWER_M'
    optprob_set=VAK1_.analysis.diff.VAK1_CG_SYSTEM_REACTIVE_POWER_M.Sensor.optprob_set
    optstate_set=VAK1_.analysis.diff.VAK1_CG_SYSTEM_REACTIVE_POWER_M.Sensor.optstate_set
    s_names=VAK1_.diffsensor_names
    
    # Likelihood of weather factors
    optprob_set=VAK1_.analysis.diff.VAK1_CG_SYSTEM_REACTIVE_POWER_M.Weather.optprob_set
    optstate_set=VAK1_.analysis.diff.VAK1_CG_SYSTEM_REACTIVE_POWER_M.Weather.optstate_set
    w_names=VAK1_.diffweather_names
    sort_idx=argsort(optprob_set)[::-1]
    sort_lh=optprob_set[sort_idx].T
    sort_state=optstate_set[sort_idx].T
    figw=figure(figsize=(15.0,10.0))
    #figw=figure()
    plt.subplot(2,1,1)
    plt.plot(sort_lh,'-s')
    x_label= list(np.array(w_names)[sort_idx])
    x_ticks=range(len(x_label))
    #plt.xticks(x_ticks,x_label, fontsize="small")
    plt.xticks(x_ticks,x_label,rotation=30, fontsize=12)
    plt.tick_params(labelsize='large')
    plt.ylabel('Likelihood (From 0 to 1)',fontsize=18)
    plt.title('Likelihood of peak differential measurement of '+p_name+' given weather factors')
    #plt.get_current_fig_manager().window.showMaximized()
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    figw.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    # regualr event
    import lib_bnlearn as rbn    
    p_idx=VAK1_.diffsensor_names.index(p_name)
    bndata_mat=np.vstack((VAK1_.diffdata_state_mat[:,p_idx].T,VAK1_.diffdata_weather_mat_.T)).T
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols_fnames=[p_name]+[w_name for w_name in VAK1_.diffweather_names]
    cols=[remove_dot(p_name)]+[remove_dot(w_name) for w_name in VAK1_.diffweather_names]
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    wcause_label=list(np.array(cols)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[cols.index(label_) for label_ in wcause_label]
    #fig=figure(figsize=(10,10))
    fig=figure()
    for k in range(len(cause_idx)):
        effect_idx=cols_fnames.index(p_name)
        peak_state_0, peak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[k]],[[effect_idx]],[[PEAK]])
        lowpeak_state_0, lowpeak_prob_0=compute_cause_likelihood(bndata_mat,[cause_idx[k]],[[effect_idx]],[[LOW_PEAK]])
        sort_idx1=argsort(peak_state_0)
        sort_idx2=argsort(lowpeak_state_0)
        subplot(1,len(cause_idx),k+1)
        plot(sort(peak_state_0), np.array(peak_prob_0)[sort_idx1],'-^')
        plot(sort(lowpeak_state_0), np.array(lowpeak_prob_0)[sort_idx2],'-v')
        plt.legend(('measurements','classified states'))
        if k==0:
            plt.ylabel('Probability of Peak Rective Power Variation')
        plt.grid()
        plt.legend(('High Peak', 'Low Peak'),loc='center right')
        plt.xlabel(wcause_label[k])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
        
    # Time data dependency - Likelihood of time factors
    # BN Network Learning 
    # Regualr event
    t_name_set=VAK1_.difftime_names
    # [MTH', 'WD', 'HR']
    p_idx=VAK1_.diffsensor_names.index(p_name)
    bndata_mat=np.vstack((VAK1_.diffdata_state_mat[:,p_idx].T,VAK1_.diffdata_time_mat.T)).T
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols_fnames=[p_name]+[w_name for w_name in VAK1_.difftime_names]
    cols=[remove_dot(p_name)]+[remove_dot(w_name) for w_name in VAK1_.difftime_names]
    effect_idx=cols_fnames.index(p_name)
    time_high_peak_liklihood_set=[]
    time_low_peak_liklihood_set=[]
    for t_name in t_name_set:
        idx_t=cols.index(t_name)
        peak_state, peak_prob=compute_cause_likelihood(bndata_mat,[idx_t],[[effect_idx]],[[PEAK]])
        time_high_peak_liklihood_set.append(np.array([peak_state,peak_prob]))
        peak_state, peak_prob=compute_cause_likelihood(bndata_mat,[idx_t],[[effect_idx]],[[LOW_PEAK]])
        time_low_peak_liklihood_set.append(np.array([peak_state,peak_prob]))
    
    fig=figure()
    subplot(3,1,1)
    plot(time_high_peak_liklihood_set[0][0],time_high_peak_liklihood_set[0][1],'-^')
    plot(time_low_peak_liklihood_set[0][0],time_low_peak_liklihood_set[0][1],'-v')
    plt.xticks(monthDict.keys(),monthDict.values())
    plt.xlabel('Months of a year',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'),loc='center right')
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    subplot(3,1,2)
    plot(time_high_peak_liklihood_set[1][0],time_high_peak_liklihood_set[1][1],'-^')
    plot(time_low_peak_liklihood_set[1][0],time_low_peak_liklihood_set[1][1],'-v')
    plt.xticks(weekDict.keys(),weekDict.values())
    plt.xlabel('Days of a Week',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'),loc='center right')
    plt.tick_params(labelsize='large')
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    subplot(3,1,3)
    plot(time_high_peak_liklihood_set[2][0],time_high_peak_liklihood_set[2][1],'-^')
    plot(time_low_peak_liklihood_set[2][0],time_low_peak_liklihood_set[2][1],'-v')
    plt.xticks(hourDict.keys(),hourDict.values())
    plt.xlabel('Hours of a day',fontsize='large')
    plt.ylabel('Likelihood',fontsize='large')
    plt.grid()
    plt.legend(('High Peak', 'Low Peak'),loc='center right')
    plt.tick_params(labelsize='large')
    plt.tick_params(labelsize='large')
    plt.ylim([-0.05,1.05])
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    # BN Network representaiton
    b_arc_list = pair_in_idx([cols[0]],cols[1:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().gVAK1_.diffet_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
        
    
    
    ###############################################################
    # 4. Sensor, Weather Time Dependency Analysis
    # BN Network Learning
    ###############################################################
    # For regualr event. 
    t_name_set=VAK1_.difftime_names
    # [MTH', 'WD', 'HR']
    sig_tag='diff'
    p_name=['VAK1.CG_SYSTEM_REACTIVE_POWER_M']
    sensor_cause_label=['VAK1.GEO_LM5_TE1_FM','VAK1.AK_TE50_4_M']
    weather_cause_label=['Dew PointC','Humidity']
    time_cause_label=['MTH', 'HR']
    p_idx=[VAK1_.diffsensor_names.index(temp) for temp in p_name]
    s_idx=[VAK1_.diffsensor_names.index(temp) for temp in sensor_cause_label]
    w_idx=[VAK1_.diffweather_names.index(temp) for temp in weather_cause_label]
    t_idx=[VAK1_.difftime_names.index(temp) for temp in time_cause_label]
    
    bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,\
     GW2_.avgdata_state_mat[:,s_idx].T, \
     GW2_.avgdata_weather_mat_[:,w_idx].T, \
      GW2_.avgdata_time_mat[:,t_idx].T)).T
     
    #bndata_mat=np.vstack((GW2_.avgdata_state_mat[:,p_idx].T,GW2_.avgdata_weather_mat.T)).T
    cols_fnames=[name_ for name_ in p_name+sensor_cause_label+weather_cause_label+time_cause_label]
    cols=[remove_dot(name_) for name_ in p_name+sensor_cause_label+weather_cause_label+time_cause_label]
    # BN Network representaiton
    b_arc_list = pair_in_idx([cols[0]],cols[1:])+pair_in_idx([cols[1]],cols[2:])+pair_in_idx([cols[2]],cols[3:])
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    #hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    fig=rbn.nx_plot(hc_b,cols)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    cause_label=list(np.array(cols_fnames)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[cols_fnames.index(label_) for label_ in cause_label]
    effect_idx=[cols_fnames.index(label_) for label_ in p_name]
    effect_label=p_name
    obs_state=PEAK
    peak_state_temp, peak_prob_temp=compute_cause_likelihood(bndata_mat,cause_idx,[effect_idx],[[obs_state]])
    obs_state=LOW_PEAK
    lowpeak_state_temp, lowpeak_prob_temp=compute_cause_likelihood(bndata_mat,cause_idx,[effect_idx],[[obs_state]])
    
    peak_state=np.array(peak_state_temp)
    peak_prob=np.array(peak_prob_temp)
    lowpeak_state=np.array(lowpeak_state_temp)
    lowpeak_prob=np.array(lowpeak_prob_temp)
    
    # Probability 
    fig=figure(figsize=(30.0,25.0))
    for i,mon in enumerate(yearMonths):
        subplot(3,4,mon+1)
        idx=np.nonzero(peak_state[:,2]==mon)[0]
        x_set=peak_state[idx,0:2]
        plot(range(len(x_set)),peak_prob[idx],'-^')
        idx=np.nonzero(lowpeak_state[:,2]==mon)[0]
        plot(range(len(x_set)),lowpeak_prob[idx],'-v')
        x_label=[(stateDict[peak_tpl[0]],stateDict[peak_tpl[1]]) for peak_tpl in x_set]
        x_ticks=range(len(x_set))
        plt.ylabel('Likelihood',fontsize='small')
        if i>7:
            #plt.xlabel(cause_label[0]+' Measurements',fontsize='small')
            plt.xticks(x_ticks,x_label,rotation=270, fontsize=10)
            plt.tick_params(labelsize='small')
        title(monthDict[mon]);plt.ylim([-0.05,1.05])
        plt.legend(('High Peak', 'Low Peak'),loc='center right')
        plt.tick_params(labelsize='small')
        plt.grid()
    
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    print '----------------------------------------'
    print 'Likelihoods '
    print '----------------------------------------'
    print cause_label+['Low Peak','High Peak']
    print '----------------------------------------'
    print np.vstack((np.int0(peak_state).T,np.int0(100*lowpeak_prob).T,np.int0(100*peak_prob).T)).T
    print '----------------------------------------'
    


#<----------------------------------------------------------------------
#import pdb;pdb.set_trace()
DO_BN_LEARN=0
# This is BN Learn example
if DO_BN_LEARN==1:
    import lib_bnlearn as rbn
    irr_state_mat,irr_state_prob,skewness_metric_sort,skewness_metric_sort_idx=irr_state_mapping(diffdata_state_mat,weight_coeff=10)
    bndata_dict = mt.loadObjectBinary('diffdata_dict.bin')
    
    bn_col=bndata_dict['diffdata_names']
    bn_sn=bndata_dict['sensor_names']
    bn_wn=bndata_dict['weather_names']
    bn_tn=bndata_dict['time_names']
    bndata_mat=bndata_dict['diffdata_mat']
    
    # If the variable is discrete, we should convert the data into R's factor data type
    #cols =  X_Sensor_NAMES+X_Time_NAMES
    
    for k,name_temp in enumerate(bn_wn):
        try:
            blank_idx=name_temp.index(' ')
            #print blank_idx,X_Weather_NAMES[k][blank_idx]
            bn_wn[k]=bn_wn[k].replace(' ','_')
        except:
            pass
    for k,name_temp in enumerate(bn_col):
        try:
            blank_idx=name_temp.index(' ')
            #print blank_idx,X_Weather_NAMES[k][blank_idx]
            bn_col[k]=bn_col[k].replace(' ','_')
        except:
            pass

    
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat[:,:len(bn_sn)])
    #cols = X_Sensor_NAMES+X_Weather_NAMES+X_Time_NAMES
    cols =bn_col[:len(bn_sn)]
    
    # Construct data frame, given data matrix (np.array) and column names
    # if column names are not given, we use column index [0,1,..] as the column names
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)

    #arc_list = pair_in_idx(X_Sensor_NAMES,X_Time_NAMES)
    # Black list        
    b_arc_list = pair_in_idx(bn_sn,bn_tn)\
    +pair_in_idx(bn_sn,bn_wn)\
    +pair_in_idx(bn_wn,bn_tn)\
    +pair_in_idx(bn_wn,bn_wn)\
    +pair_in_idx(bn_tn,bn_tn)
    
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    # White list    
    w_arc_list = pair_in_idx(bn_tn,bn_sn)\
    +pair_in_idx(bn_tn,bn_wn)
    white_arc_frame = rbn.construct_arcs_frame(w_arc_list)
       
    
    """
        Step2: Using bnlearn to learn graph structure from data frame
    """
    # Use hill-climbing learning algorithm
    # With blacklisting arcs
    hc = rbn.bnlearn.hc(data_frame,score='bic')
    hc_score=rbn.bnlearn.score(hc,data_frame,type="bic")
    hc_bw = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,whitelist=white_arc_frame,score='bic')
    hc_bw_score=rbn.bnlearn.score(hc_bw,data_frame,type="bic")
    hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    hc_b_score=rbn.bnlearn.score(hc_b,data_frame,type="bic")
    
    print 'hc_score: ',hc_score,'hc_b_score: ',hc_b_score,'hc_bw_score: ',hc_bw_score
    
    # Print some output from the learning process
    #print str(hc_b)
    # Get the adjacent matrix from the graph structure
    # the return is numpy array
    amat = rbn.py_get_amat(hc_b)
    """ 
        There are other learning algorithms available too 
        E.g.:
        gs = rbn.bnlearn.gs(data_frame)
    """    
    """
        Step 3: Plotting the graph, given the graph structure
        and the names of nodes
    """
    #hc = rbn.bnlearn.hc(data_frame,score='k2')
    figure(2)
    rbn.nx_plot(hc_b,cols)
    rbn.nx_plot(hc,cols)
    #rbn.nx_plot(hc,rbn.bnlearn.nodes(hc))
    """
        Step4: Fitting the data into graph structure
        to estimate the conditional probability
        
        NOTE: in order for fitting to happen, the graph must be completely directed
    """
    fit = rbn.py_bn_fit(hc_b,data_frame)
    #print str(fit)
    #index_temp=cols.index('GW1.HA1_SM_K')
    index_temp=1
    prob_dimnames,prob_factors,prob_mat = rbn.py_get_node_cond_mat(fit,index_temp)
    #rbn.write_to_file('fit.dat',str(fit))
    
    
###############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.


###############################################################################
DATA_EMBEDDING_ANALYSIS=0
if DATA_EMBEDDING_ANALYSIS==1:
    # Covariance Estimation    
    edge_model = covariance.GraphLassoCV()
    edge_model.fit(X_INPUT)
    cov_mat=edge_model.covariance_
    
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=X_INPUT.shape[1]-1)
    
    embedding = node_position_model.fit_transform(X_INPUT.T).T
    
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

DATA_NAME_ANALYSIS=0
if DATA_NAME_ANALYSIS==1:
    #################################################################################
    # Graph strucutre analysis of sensor naming
    #################################################################################
    print '--------------------------------------------------'
    print 'Graph strucutre analysis of sensor naming'
    print '--------------------------------------------------'
    print 'get simialirty matrix of sensor naming'
    #sim_mat, uuid_list, phrases, key_description, phrase_count = get_sim_mat()
    
    sim_mat = mt.loadObjectBinary('../data_year/sim_mat.bin')
    uuid_list = mt.loadObjectBinary('../data_year/uuid_list.bin')
    phrases = mt.loadObjectBinary('../data_year/phrases.bin')
    key_description = mt.loadObjectBinary('../data_year/key_description.bin')
    phrase_count = mt.loadObjectBinary('../data_year/phrase_count.bin')
        
    
    print 'build tree.....'
    for sensor_name in uuid_list:
        print len(sensor_name)
    

    
    
print '**************************** End of Program ****************************'




"""
# Obslete Lines
    ###########################################################################
    # Float Type Measurement Clustering
    ###########################################################################

    DIST_MAT_sf=find_norm_dist_matrix(X_Feature[:,sf_idx])
    # Find representative set of sensor measurements 
    min_dist_=np.sqrt(2*(1-(0.9)))
    max_dist_=np.sqrt(2*(1-(0.1)))
    distmat_input=DIST_MAT_sf
    DO_CLUSTERING_TEST=0
    if DO_CLUSTERING_TEST==1:
        CLUSTERING_TEST(distmat_input,min_corr=0.1,max_corr=0.9)

    pack_exemplars_float,pack_labels_float=max_pack_cluster(distmat_input,min_dist=min_dist_,max_dist=max_dist_)
    pack_num_clusters_float=int(pack_labels_float.max()+1)
    print '-------------------------------------------------------------------------'
    print pack_num_clusters_float, 'clusters out of ', len(pack_labels_float), ' float type measurements'
    print '-------------------------------------------------------------------------'
    validity,intra_dist,inter_dist=compute_cluster_err(distmat_input,pack_labels_float)
    print 'validity:',round(validity,2),', intra_dist: ',np.round(intra_dist,2),', inter_dist: ',np.round(inter_dist,2)
    print '-------------------------------------------------------------------------'
    sf_exemplars_dict={}
    sfe_name=list(np.array(sf_name)[pack_exemplars_float])
    sfe_idx=np.array(sf_idx)[pack_exemplars_float]
    for label_id,(m_idx,exemplar_label) in enumerate(zip(pack_exemplars_float,sfe_name)):
        print exemplar_label
        children_set=list(set(np.nonzero(pack_labels_float==label_id)[0])-set([m_idx]))
        print 'Label ', label_id, ': ',m_idx,'<--', children_set
        sf_exemplars_dict.update({exemplar_label:list(np.array(sf_name)[children_set])})
    # exemplar index    
    
    
    ###########################################################################
    # InT Type Measurement Clustering
    ###########################################################################
    
    DIST_MAT_si=find_norm_dist_matrix(X_Feature[:,si_idx])
    # Find representative set of sensor measurements 
    min_dist_=np.sqrt(2*(1-(0.9)))
    max_dist_=np.sqrt(2*(1-(0.1)))
    distmat_input=DIST_MAT_si
    DO_CLUSTERING_TEST=0
    if DO_CLUSTERING_TEST==1:
        CLUSTERING_TEST(distmat_input,min_corr=0.1,max_corr=0.9)

    pack_exemplars_int,pack_labels_int=max_pack_cluster(distmat_input,min_dist=min_dist_,max_dist=max_dist_)
    pack_num_clusters_int=int(pack_labels_int.max()+1)
    print '-------------------------------------------------------------------------'
    print pack_num_clusters_int, 'clusters out of ', len(pack_labels_int), ' int type measurements'
    print '-------------------------------------------------------------------------'
    validity,intra_dist,inter_dist=compute_cluster_err(distmat_input,pack_labels_int)
    print 'validity:',round(validity,2),', intra_dist: ',np.round(intra_dist,2),', inter_dist: ',np.round(inter_dist,2)
    print '-------------------------------------------------------------------------'
    si_exemplars_dict={}
    sie_name=list(np.array(si_name)[pack_exemplars_int])
    sie_idx=np.array(si_idx)[pack_exemplars_int]
    for label_id,(m_idx,exemplar_label_int) in enumerate(zip(pack_exemplars_int,sie_name)):
        print exemplar_label_int
        children_set=list(set(np.nonzero(pack_labels_int==label_id)[0])-set([m_idx]))
        print 'Label ', label_id, ': ',m_idx,'<--', children_set
        si_exemplars_dict.update({exemplar_label_int:list(np.array(si_name)[children_set])})

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
 
 
gmm_labels=gmm.predict(obs)
labels=gmm_labels
#kmean=KMeans(n_clusters=2).fit(obs[:,newaxis])
#labels=kmean.labels_  
subplot(3,1,1)
for i in range(num_cluster):
    plot(t_new[labels==i]-t_new[0],val_new[labels==i],'s')

title(input_names[k])
subplot(3,1,2)
plot(t_new[1:]-t_new[0],abs(diff(val_new))/max(abs(diff(val_new))))
subplot(3,1,3)
a=diff(val_new)
plot(t_new[1:]-t_new[0],a/max(abs(a)))
 
 #labels=kmean.labels_  len(sie_idx
subplot(2,1,1)
for i in range(opt_num_cluster):
    plot(t_new[label==i]-t_new[0],val_new[label==i],'*')

title(input_names[k])
subplot(2,1,2)
plot(t_new[1:]-t_new[0],abs(diff(val_new))/max(abs(diff(val_new))))
plot(t_new[0:50],label[0:50],'s')


#plt.ioff()
# Only do state classification for number of samples greater than 
k=0
dt=intpl_intv[k]
# Reference time unit is 5 min, 15 min, 30 min and 1 hour
num_samples_set=np.round(np.array([60*5,60*15,60*30, 60*60 ])*(1/dt))
min_num_samples_for_analysis=2**5
for i,nfft_temp in enumerate(num_samples_set):
    if nfft_temp>min_num_samples_for_analysis:
        NFFT=int(2**ceil(log2(nfft_temp)));break;

window_duration=NFFT*dt
Fs = (1.0/dt)  # the sampling frequency
# Pxx is the segments x freqs array of instantaneous power, freqs is
# the frequency vector, bins are the centers of the time bins in which
# the power is computed, and im is the matplotlib.image.AxesImage
# instance

 
 """
