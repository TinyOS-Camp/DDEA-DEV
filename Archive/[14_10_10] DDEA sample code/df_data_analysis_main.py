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

# Custom library
from data_tools import *
from data_retrieval import *
from pack_cluster import *
from data_preprocess import *
from shared_constants import *
from pre_bn_state_processing import *
##################################################################
# Interactive mode for plotting
plt.ion() 

##################################################################
# Processing Configuraiton Settings
##################################################################
# This option let you use  data_dict object saved in hard-disk from the recent execution
gsbc_name_dict= mt.loadObjectBinaryFast('gsbc_name_dict.bin')
#input_files_temp=gsbc_name_dict.items()[2][1]
#input_files_temp=list(set(gsbc_name_dict['main_bldg_facility']+gsbc_name_dict['main_bldg_power']+ gsbc_name_dict['machine_room_facility']))
input_files_temp=list(set(gsbc_name_dict['main_bldg_power']+ gsbc_name_dict['machine_room_facility']))
#input_files_temp=list(set(gsbc_name_dict['main_bldg_power_1']+ gsbc_name_dict['main_bldg_facility_1']+ gsbc_name_dict['machine_room_facility']))
input_files=[DATA_DIR+temp+'.bin' for temp in input_files_temp]

IS_USING_SAVED_DICT=-1
print 'Extract a common time range...'
# Analysis period
ANS_START_T=dt.datetime(2013,6,1,0)
ANS_END_T=dt.datetime(2013,12,30,0)
# Interval of timelet, currently set to 1 Hour
#TIMELET_INV=dt.timedelta(hours=1)
TIMELET_INV=dt.timedelta(minutes=30)

##################################################################
"""
['conference_bldg_power', 'office_bldg_facility',
 'conference_bldg_facility_3', 'lab_bldg_power',
 'lab_bldg_facility', 'main_bldg_power_3', 'main_bldg_power_2',
 'conference_bldg_facility', 'main_bldg_power_9',
 'main_bldg_power_8', 'main_bldg_power_5',
 'main_bldg_power_4', 'main_bldg_facility_6',
 'main_bldg_facility_7', 'main_bldg_power_1',
 'main_bldg_facility_1',
 'main_bldg_facility_2',
 'main_bldg_facility_3',
 'main_bldg_power',
 'machine_room_facility',
 'research_bldg_power',
 'main_bldg_facility_16',
 'main_bldg_facility_14',
 'main_bldg_facility_15',
 'main_bldg_facility_12',
 'main_bldg_facility_13',
 'main_bldg_facility_10',
 'main_bldg_facility_11',
 'main_bldg_power_15',
 'main_bldg_power_14',
 'main_bldg_power_16',
 'main_bldg_power_11',
 'main_bldg_power_10',
 'main_bldg_power_13',
 'main_bldg_power_12',
 'main_bldg_facility_8',
 'main_bldg_facility_9',
 'conference_bldg_facility_1',
 'main_bldg_power_6',
 'main_bldg_facility_4',
 'main_bldg_facility_5',
 'main_bldg_facility_top',
 'main_bldg_power_7',
 'main_bldg_facility']
 """

#import pdb;pdb.set_trace()
###############################################################################
# This directly searches files from bin file name
print '###############################################################################'
print '#  Data Pre-Processing'
print '###############################################################################'
# define input_files  to be read
if IS_USING_SAVED_DICT==0:
    ANS_START_T,ANS_END_T,input_file_to_be_included=time_range_check(input_files,ANS_START_T,ANS_END_T,TIMELET_INV)
    print 'time range readjusted  to  (' ,ANS_START_T, ', ', ANS_END_T,')'
    start__dictproc_t=time.time()
    if IS_SAVING_INDIVIDUAL==True:
        data_dict=construct_data_dict_2(input_files,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename='data_dict', IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
    else:
        data_dict,purge_list=construct_data_dict(input_file_to_be_included,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename='data_dict', IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
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

"""
no_data_list=[]
len_time_slots=len(data_dict['time_slots'])
for key_ in data_dict['sensor_list']:
    if len(data_dict[key_][2][0])<len_time_slots:
        print key_,': NO Data !!'
        no_data_list.append(key_)

if len(no_data_list)>0:
    raise Exception("Invaid data found !")
"""

"""
no_data_list=[]
#for key_ in data_dict['sensor_list'][100:120]:
for key_ in  X_names[0:5]:
    figure(key_)
    if len(data_dict[key_][2][0])==0:
        print key_,': NO Data !!'
        no_data_list.append(key_)
    else:
        t_=unix_to_dtime(data_dict[key_][2][0])
        val_=data_dict[key_][2][1]
        plot(t_,val_,'-s')
"""        
        
if IS_USING_SAVED_DICT>=0:
    # Copy related variables 
    time_slots=data_dict['time_slots'][:]
    Conditions_dict=data_dict['Conditions_dict'].copy()
    Events_dict=data_dict['Events_dict'].copy()
    sensor_list=data_dict['sensor_list'][:]
    weather_list=data_dict['weather_list'][:]
    #weather_list_used = [data_dict['weather_list'][i] for i in [1,2,3,10,11]]
    weather_list_used =['TemperatureC', 'Dew PointC', 'Humidity', 'Events', 'Conditions']
    
    # data_used is the list of refernece name for all measurements from now on. 
    data_used=sensor_list+weather_list_used
    # This is a global ID for data_used measurement
    data_used_idx=range(len(data_used))
    sensor_idx=range(len(sensor_list))
    weather_idx=range(len(sensor_list),len(data_used))

    
CHECK_DATA_FORMAT=0
if CHECK_DATA_FORMAT==1:
    if IS_SAVING_INDIVIDUAL==True:
        list_of_wrong_data_format=verify_data_format_2(data_used,data_dict,time_slots)
    else:
        list_of_wrong_data_format=verify_data_format(data_used,data_dict,time_slots)
    if len(list_of_wrong_data_format)>0:
        print 'Measurement list below'    
        print '----------------------------------------'    
        print list_of_wrong_data_format
        raise NameError('Errors in data format')


Data_Summarization=0
if Data_Summarization==1:
    print '###############################################################################'
    print '#  Data Summarization'
    print '###############################################################################'

    # sensor_list  --> float or int  --> clustering for float and int --> exemplar 
    # exemplar of floats --> states , int is states, 
    # weather_list --> float or int
    ####################################
    # Regular Event Extraction
    ####################################
    # Build feature matrix wiht data interpolation for both sensor and weather data
    if IS_SAVING_INDIVIDUAL==True:
        X_Feature,X_Time,X_names\
        ,X_zero_var_list, X_zero_var_val\
        ,X_int_type_list,X_int_type_idx\
        ,X_float_type_list,X_float_type_idx\
        ,X_weather_type_idx,X_sensor_type_idx\
        =build_feature_matrix_2(data_dict,sensor_list,weather_list_used\
        ,time_slots,DO_INTERPOLATE=1\
        ,max_num_succ_idx_for_itpl=int(len(time_slots)*0.05))
    else:
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
    
    if X_Feature.shape[0]>0:
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
        =cluster_measurement_points(X_Feature[:,sf_idx],sf_name,corr_bnd=[0.1,0.9],alg='aff')
        sfe_idx=list(np.array(sf_idx)[exemplars_])
        #plot_label(X_Feature,X_names,labels_,exemplars_,[4,5,6,7])
        # InT Type Measurement Clustering
        X_Feature_sie,si_exemplars_dict,exemplars_,labels_\
        =cluster_measurement_points(X_Feature[:,si_idx],si_name,corr_bnd=[0.0,0.9],alg='aff')
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
    if IS_SAVING_INDIVIDUAL==True:
        measurement_point_set,num_type_set\
        =interpolation_measurement_2(data_dict,sensor_list,err_rate=1,sgm_bnd=20)
    else:
        measurement_point_set,num_type_set\
        =interpolation_measurement(data_dict,sensor_list,err_rate=1,sgm_bnd=20)
    # Irregualr matrix 
    Xdiff_Mat,Xdiff_Time,Xdiff_Names\
    ,Xdiff_zero_var_list, Xdiff_zero_var_val\
    ,Xdiff_int_type_list,Xdiff_int_type_idx\
    ,Xdiff_float_type_list,Xdiff_float_type_idx\
    =build_diff_matrix(measurement_point_set,time_slots,num_type_set,sensor_list,PARALLEL=IS_USING_PARALLEL_OPT)
    
    if Xdiff_Mat.shape[0]>0:
        #==============================================================================
        # Restructure diff_marix's and weather matix  for the same common time slot
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
    

"""
start_t=datetime.datetime(2013, 1, 1, 15, 44, 52)
end_t=datetime.datetime(2014, 12, 24, 16, 5, 12)
sensor_list=data_dict['sensor_list']
data_x=get_data_set(X_names[0:5],start_t,end_t)
png_namex=plot_data_x(data_x,stype='raw',smark='-^')
png_namex=plot_data_x(data_x,stype='diff',smark='-^')
"""
    
print '###############################################################################'
print '#  Model_Discovery'
print '###############################################################################'
Model_Discovery=1
if Model_Discovery==1:
    bldg_dict={}
    RUN_VTT_BLDG=0
    if RUN_VTT_BLDG==1:
        # Load Building Object
        LOAD_BLDG_OBJ=1
        if LOAD_BLDG_OBJ==1:
            bldg_dict={'GW1_':mt.loadObjectBinaryFast('GW1_.bin'),'GW2_':mt.loadObjectBinaryFast('GW2_.bin'),\
            'VAK1_':mt.loadObjectBinaryFast('VAK1_.bin'),'VAK2_':mt.loadObjectBinaryFast('VAK2_.bin')}
            bldg_=obj(bldg_dict)
        else:
            sig_tag_set=['avg','diff']
            bldg_tag_set=['GW1_','GW2_','VAK1_','VAK2_']
            dict_dir_set=['./GW1_results/','./GW2_results/','./VAK1_results/','./VAK2_results/']
            pname_key='POWER'
            for dict_dir,bldg_tag in zip(dict_dir_set,bldg_tag_set):
                bldg_dict.update({bldg_tag:create_bldg_obj(dict_dir,bldg_tag,pname_key)})
            bldg_=obj(bldg_dict)
        # Plotting Likelihood plotting
        PLOTTING_LH=0
        if PLOTTING_LH==1:
            plotting_bldg_lh(bldg_,attr_class='sensor',num_picks=30)
            plotting_bldg_lh(bldg_,attr_class='time',num_picks=30)
            plotting_bldg_lh(bldg_,attr_class='weather',num_picks=30)
        
        import pdb;pdb.set_trace()
        # Exampl of BN Analysis 
        bldg_obj=bldg_.GW2_
        p_name=bldg_obj.analysis.avg.__dict__.keys()[0]
        s_cause_label,s_labels,s_hc=bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag='avg',num_picks_bn=15)
        t_cause_label,t_labels,t_hc=bn_anaylsis(bldg_obj,p_name,attr='time',sig_tag='avg',num_picks_bn=15)
        w_cause_label,w_labels,w_hc=bn_anaylsis(bldg_obj,p_name,attr='weather',sig_tag='avg',num_picks_bn=15)
        all_cause_label,all_labels,all_hc=bn_anaylsis(bldg_obj,p_name,attr='all',sig_tag='avg',num_picks_bn=15)
        figure('BN for Sensors')
        fig1=rbn.nx_plot(s_hc,s_labels)
        figure('BN for Time')
        fig2=rbn.nx_plot(t_hc,t_labels)
        figure('BN for Weather')
        fig3=rbn.nx_plot(w_hc,w_labels)
        figure('BN for Sensor-Time-Weather')
        fig4=rbn.nx_plot(all_hc,all_labels)
        
    
    RUN_GSBC_BLDG=1
    if RUN_GSBC_BLDG==1:
        LOAD_BLDG_OBJ=0
        if LOAD_BLDG_OBJ==1:
            print 'not yet ready'
        else:
            #gsbc_dict_dir_set=['./GSBC/allsensors/','./GSBC/seleceted/']
            gsbc_dict_dir_set=['./GSBC/main_bldg_1/','./GSBC/main_bldg_power_machine_room/']
            bldg_tag_set=['GSBC_main_bldg_1','GSBC_main_bldg_power_machine_room']
            print 'Building for ',bldg_tag_set, '....'
            gsbc_hcw_pname_key='3003....' # Hot and Cold water
            gsbc_main_1_pname_key='300401..' # Maing Buiding F1
            gsbc_main_2_pname_key='300402..' # Maing Buiding F2
            gsbc_pname='300.....'
            gsbc_hvac_pname_key='3006....' # HVAC
            #gsbc_name_dict['main_bldg_power']
            pname_key= gsbc_pname# 
            for dict_dir,bldg_tag in zip(gsbc_dict_dir_set,bldg_tag_set):
                bldg_dict.update({bldg_tag:create_bldg_obj(dict_dir,bldg_tag,pname_key)})
            bldg_=obj(bldg_dict)
        PLOTTING_LH=0
        if PLOTTING_LH==1:
            plotting_bldg_lh(bldg_,attr_class='sensor',num_picks=30)
            plotting_bldg_lh(bldg_,attr_class='time',num_picks=30)
            plotting_bldg_lh(bldg_,attr_class='weather',num_picks=30)
   
        
    #######################################################################################
    # Analysis For GSBC
    #######################################################################################
    gsbc_key_dict=mt.loadObjectBinaryFast('./gsbc_key_dict.bin')
   # Analysis of BN network result
    def convert_gsbc_name(id_labels):
        if isinstance(id_labels,list)==False:
            id_labels=[id_labels]
        out_name=[gsbc_key_dict[key_label_[2:]] if key_label_[2:] \
        in gsbc_key_dict else key_label_ for key_label_ in id_labels ]
        return out_name
    #bldg_obj=bldg_.GSBC_main_bldg_power_machine_room
    bldg_obj=bldg_.GSBC_main_bldg_power_machine_room
    bldg_.GSBC_main_bldg_power_machine_room.anal_out=bn_prob_analysis(bldg_obj,sig_tag_='avg')
    bldg_obj=bldg_.GSBC_main_bldg_1
    bldg_.GSBC_main_bldg_1.anal_out=bn_prob_analysis(bldg_obj,sig_tag_='avg')
    
    import pdb;pdb.set_trace()
    #--------------------------------------------------------------------------
    # Analysis Display
    #--------------------------------------------------------------------------
    # Data set 1 - GSBC_main_bldg_power_machine_room
    p_name_sets_1=bldg_.GSBC_main_bldg_power_machine_room.anal_out.__dict__.keys()
    bn_out_sets_1=bldg_.GSBC_main_bldg_power_machine_room.anal_out.__dict__
    # Data set 2 - GSBC_main_bldg_1
    p_name_sets_2=bldg_.GSBC_main_bldg_1.anal_out.__dict__.keys()
    bn_out_sets_2=bldg_.GSBC_main_bldg_1.anal_out.__dict__
    
    # Data set 2 Analysis
    print 'List power meters for analysis'
    print '------------------------------------'
    pprint.pprint(np.array([p_name_sets_1,convert_gsbc_name(p_name_sets_1)]).T)
    print '------------------------------------'
    p_name=p_name_sets_1[3]
    bn_out=bn_out_sets_1[p_name]
       
    fig_name='BN for Sensors '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    col_name=[str(np.array([[lab1],[remove_dot(lab2)]])) \
    for lab1,lab2 in zip(bn_out.s_labels, convert_gsbc_name(bn_out.s_labels))]
    rbn.nx_plot(bn_out.s_hc,col_name,graph_layout='spring',node_text_size=15)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig_name='BN for Time '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    rbn.nx_plot(bn_out.t_hc,convert_gsbc_name(bn_out.t_labels),graph_layout='spring',node_text_size=12)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    
    fig_name='BN for Weather '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    rbn.nx_plot(bn_out.w_hc,convert_gsbc_name(bn_out.w_labels),graph_layout='spring',node_text_size=12)
    png_name=fig_name+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig_name='BN for Sensor-Time-Weather '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    rbn.nx_plot(bn_out.all_hc,convert_gsbc_name(bn_out.all_labels),graph_layout='spring',node_text_size=20)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    
    fig_name='BN PEAK LH Analysis for Sensor-Time-Weather '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name, figsize=(30.0,30.0))
    subplot(2,1,1)
    plot(bn_out.all_cause_symbol_xtick,bn_out.high_peak_prob,'-^')
    plot(bn_out.all_cause_symbol_xtick,bn_out.low_peak_prob,'-v')
    plt.ylabel('Likelihood',fontsize='large')
    plt.xticks(bn_out.all_cause_symbol_xtick,bn_out.all_cause_symbol_xlabel,rotation=270, fontsize=10)
    plt.tick_params(labelsize='large')
    plt.legend(('High Peak', 'Low Peak'),loc='center right')
    plt.tick_params(labelsize='large')
    plt.grid();plt.ylim([-0.05,1.05])
    plt.title('Likelihood of '+ str(remove_dot(convert_gsbc_name(p_name)))+\
    ' given '+'\n'+str(remove_dot(convert_gsbc_name(bn_out.all_cause_label))))
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    # Compare with the raw data
    #-------------------------------------------    
    start_t=datetime.datetime(2013, 8, 9, 0, 0, 0)
    end_t=datetime.datetime(2013, 8, 13, 0, 0, 0)
    data_x=get_data_set([label_[2:] for label_ in bn_out.all_cause_label]+[p_name[2:]],start_t,end_t)
    png_namex=plot_data_x(data_x,stype='raw',smark='-^')
    png_namex=plot_data_x(data_x,stype='diff',smark='-^')

    name_list_out=[[p_name]+bn_out.all_cause_label,convert_gsbc_name([p_name]+bn_out.all_cause_label)]
    pprint.pprint(np.array(name_list_out).T)
    pprint.pprint(name_list_out)



    start_t=datetime.datetime(2013, 7, 1, 0, 0, 0)
    end_t=datetime.datetime(2013, 12, 31, 0, 0, 0)
    data_x=get_data_set([label_[2:] for label_ in bn_out.s_labels],start_t,end_t)
    png_namex=plot_data_x(data_x,stype='raw',smark='-^',fontsize='small',xpos=0.00)
    png_namex=plot_data_x(data_x,stype='diff',smark='-^')

    
    """
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
    
    """

print '**************************** End of Program ****************************'



