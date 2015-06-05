# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 18:18:41 2014

@author: Deokwoo Jung
"""

# To force float point division
from __future__ import division
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
import datetime as dt
from dateutil import tz
import shlex, subprocess
import time
import retrieve_weather as rw
import itertools
import calendar
import random
from matplotlib.collections import LineCollection
import pprint
from data_tools import *
from quasar_retrieval import *
from pack_cluster import *
from quasar_preprocess import *
from shared_constants import *
from pre_bn_processing import *

####################################
# Regular Event Extraction
####################################
# Build feature matrix wiht data interpolation for both sensor and weather data
#def data_summerization(bldg_key,data_dict,data_used,sensor_list,weather_list_used,time_slots,PROC_AVG=True,PROC_DIFF=True):
def data_summerization(bldg_key, data_dict, proc_avg=True, proc_diff=True):

    time_slots = data_dict['time_slots'][:]
    conditions_dict = data_dict['Conditions_dict'].copy()
    events_dict = data_dict['Events_dict'].copy()
    sensor_list = data_dict['sensor_list'][:]
    weather_list = data_dict['weather_list'][:]
    weather_list_used = ['TemperatureC', 'Dew PointC', 'Humidity', 'Events', 'Conditions']

    # data_used is the list of refernece name for all measurements from now on.
    data_used = sensor_list + weather_list_used
    # This is a global ID for data_used measurement
    data_used_idx = range(len(data_used))
    sensor_idx = range(len(sensor_list))
    weather_idx = range(len(sensor_list), len(data_used))
    dsout = {'data_dict': data_dict}

    if proc_avg:
        print '-' * 40
        print 'processing avg.feature..'
        print '-' * 40

        X_Feature, X_Time, X_names, X_zero_var_list, X_zero_var_val, X_int_type_list,\
        X_int_type_idx, X_float_type_list, X_float_type_idx, X_weather_type_idx, X_sensor_type_idx = \
            build_feature_matrix(data_dict, sensor_list, weather_list_used, time_slots, interpolation=1, max_num_succ_idx_for_itpl=int(len(time_slots)*0.05))

        build_feature_matrix_out = \
            {'X_Feature': X_Feature,
             'X_Time': X_Time,
             'X_names': X_names,
             'X_zero_var_list': X_zero_var_list,
             'X_zero_var_val': X_zero_var_val,
             'X_int_type_list': X_int_type_list,
             'X_int_type_idx': X_int_type_idx,
             'X_float_type_list': X_float_type_list,
             'X_float_type_idx': X_float_type_idx,
             'X_weather_type_idx': X_weather_type_idx,
             'X_sensor_type_idx': X_sensor_type_idx}
        build_feature_matrix_out = obj(build_feature_matrix_out)

        if len(X_names+X_zero_var_list) != len(data_used):
            raise NameError('Missing name is found in X_names or X_zero_var_list')

        else:
            zero_var_idx = [data_used.index(name_str) for name_str in X_zero_var_list]
            nzero_var_idx = list(set(data_used_idx)-set(zero_var_idx))
        
        if X_Feature.shape[0] > 0:
            # From below all index are reference to X_Feature
            sf_idx = list(set(X_sensor_type_idx)&set(X_float_type_idx))
            # Equivalent to np.array(data_used)[np.array(nzero_var_idx)[sf_idx]]
            sf_name = list(np.array(X_names)[sf_idx])
            si_idx = list(set(X_sensor_type_idx)&set(X_int_type_idx))
            si_name = list(np.array(X_names)[si_idx])
            wf_idx = list(set(X_weather_type_idx)&set(X_float_type_idx))
            wf_name = list(np.array(X_names)[wf_idx])
            wi_idx = list(set(X_weather_type_idx)&set(X_int_type_idx))
            wi_name = list(np.array(X_names)[wi_idx])

            #Euclidian Distance Matrix of Floating type of data only   wf+o
            float_idx = list(set(sf_idx)| set(wf_idx))
            int_idx = list(set(si_idx)| set(wi_idx))
            # Float Type Measurement Clustering
            X_Feature_sfe, sf_exemplars_dict, exemplars_, labels_ = \
                cluster_measurement_points(X_Feature[:, sf_idx], sf_name, corr_bnd=[0.1,0.9], alg='aff')

            sfe_idx = list(np.array(sf_idx)[exemplars_])
            #plot_label(X_Feature,X_names,labels_,exemplars_,[4,5,6,7])

            # InT Type Measurement Clustering
            X_Feature_sie, si_exemplars_dict, exemplars_, labels_ = \
                cluster_measurement_points(X_Feature[:,si_idx],si_name,corr_bnd=[0.0,0.9],alg='aff')
            sie_idx = list(np.array(si_idx)[exemplars_])
            
            sfe_state, sfe_corr_val = x_input_to_states(X_Feature_sfe,CORR_VAL_OUT=1) # sensor -float type

            # sensor -integer type
            sie_state = X_Feature_sie

            # weather -float type
            wf_state, wf_corr_val = x_input_to_states(X_Feature[:,wf_idx],CORR_VAL_OUT=1)
            wi_state = X_Feature[:, wi_idx] # weather -integer type
            empty_states = np.array([[] for i in range(len(X_Time))])
            if len(sfe_state) == 0:
                sfe_state = empty_states

            if len(sie_state) == 0:
                sie_state=empty_states

            if len(wf_state) ==0:
                wf_state = empty_states

            if len(wi_state) == 0:
                wi_state = empty_states

            # Exemplar sensor only    
            X_Sensor_STATE = np.append(sfe_state,sie_state, axis=1)
            X_Sensor_STATE = X_Sensor_STATE.astype(int)
            X_Sensor_NAMES = list(np.array(X_names)[sfe_idx])+list(np.array(X_names)[sie_idx])
            X_Weather_STATE = np.append(wf_state,wi_state, axis=1)
            X_Weather_STATE = X_Weather_STATE.astype(int)
            X_Weather_NAMES = list(np.array(X_names)[wf_idx])+list(np.array(X_names)[wi_idx])

            # months of a year,days of a week, and hours of a day
            # (Monday, Tuesday,Wendsday,Thursday,Saturday,Sunday) =(0,1,2,3,4,5,6)
            X_Time_STATE_temp = build_time_states(X_Time)
            X_Time_NAMES_temp = ['MTH', 'WD', 'HR']
            X_Time_STATE = []
            X_Time_NAMES = []

            for xt_col, xt_name in zip(X_Time_STATE_temp.T,X_Time_NAMES_temp):
                if len(set(xt_col)) > 1:
                    X_Time_STATE.append(xt_col)
                    X_Time_NAMES.append(xt_name)
            
            X_Time_STATE = np.array(X_Time_STATE).T

            #################################################
            # FORMATTED DATA  FOR REGUALR EVENT
            #################################################
            #DO_PROB_EST=1  ** Save this variables***
            #avgdata_mat = np.hstack([X_Sensor_STATE,X_Weather_STATE,X_Time_STATE])
            #avgdata_names = X_Sensor_NAMES+X_Weather_NAMES+X_Time_NAMES
            avgdata_exemplar = dict(sf_exemplars_dict.items()+si_exemplars_dict.items())
            avgdata_zvar = X_zero_var_list
            
            avgdata_dict = dict()
            avgdata_dict.update({'build_feature_matrix_out': build_feature_matrix_out})
            avgdata_dict.update({'avgdata_state_mat': X_Sensor_STATE})
            avgdata_dict.update({'avgdata_weather_mat': X_Weather_STATE})
            avgdata_dict.update({'avgdata_time_mat': X_Time_STATE})
            avgdata_dict.update({'avg_time_slot': X_Time})
            avgdata_dict.update({'avgdata_exemplar': avgdata_exemplar})
            avgdata_dict.update({'avgdata_zvar': avgdata_zvar})
            avgdata_dict.update({'sensor_names': X_Sensor_NAMES})
            avgdata_dict.update({'weather_names': X_Weather_NAMES})
            avgdata_dict.update({'time_names': X_Time_NAMES})

            dsout.update({'avgdata_dict': avgdata_dict})

    if proc_diff:
        print '-' * 40
        print 'processing diff.feature..'
        print '-' * 40
        ####################################
        # Irregular Event Extraction
        ####################################
        # Interpolatoin with outlier removal, Here we exclude weather data from irregualr event analysis
        # since weather data noramlly show slow changes in time.so we dont expect in any meaningful diffs values
        measurement_point_set,num_type_set = interpolation_measurement(data_dict, sensor_list, err_rate=1, sgm_bnd=20)

        # Irregualr matrix
        Xdiff_Mat,\
        Xdiff_Time,\
        Xdiff_Names,\
        Xdiff_zero_var_list,\
        Xdiff_zero_var_val,\
        Xdiff_int_type_list,\
        Xdiff_int_type_idx,\
        Xdiff_float_type_list,\
        Xdiff_float_type_idx =\
            build_diff_matrix(measurement_point_set, time_slots, num_type_set, sensor_list, PARALLEL=IS_USING_PARALLEL_OPT)

        build_diff_matrix_out=\
            {'Xdiff_Mat':Xdiff_Mat,
             'Xdiff_Time':Xdiff_Time,
             'Xdiff_Names':Xdiff_Names,
             'Xdiff_zero_var_list':Xdiff_zero_var_list,
             'Xdiff_zero_var_val':Xdiff_zero_var_val,
             'Xdiff_int_type_list':Xdiff_int_type_list,
             'Xdiff_int_type_idx':Xdiff_int_type_idx,
             'Xdiff_float_type_list':Xdiff_float_type_list,
             'Xdiff_float_type_idx':Xdiff_float_type_idx}
        build_diff_matrix_out = obj(build_diff_matrix_out)

        if Xdiff_Mat.shape[0] > 0:
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
            xdiff_sf_idx = Xdiff_float_type_idx;
            xdiff_sf_name = Xdiff_float_type_list;
            xdiff_si_idx = Xdiff_int_type_idx;
            xdiff_si_name = Xdiff_int_type_list
            # Float Type Measurement Clustering
            X_Diff_sfe,sf_diff_exemplars_dict,exemplars_,labels_ = \
                cluster_measurement_points(Xdiff_Mat[:,xdiff_sf_idx], xdiff_sf_name,corr_bnd=[0.1,0.9])
            xdiff_sfe_idx = list(np.array(xdiff_sf_idx)[exemplars_])
            # InT Type Measurement Clustering
            X_Diff_sie, si_diff_exemplars_dict, exemplars_, labels_ = \
                cluster_measurement_points(Xdiff_Mat[:, xdiff_si_idx], xdiff_si_name, corr_bnd=[0.1, 0.9])
            xdiff_sie_idx = list(np.array(xdiff_si_idx)[exemplars_])

            # sensor -float type
            xdiff_sfe_state, xdiff_sfe_corr_val =\
                x_input_to_states(X_Diff_sfe,CORR_VAL_OUT=1, PARALLEL=IS_USING_PARALLEL_OPT)
            xdiff_sie_state = X_Diff_sie # sensor -integer type
            empty_states = np.array([[] for i in range(len(Xdiff_Time))])

            if len(xdiff_sfe_state) == 0:
                xdiff_sfe_state = empty_states

            if len(xdiff_sie_state) == 0:
                xdiff_sie_state = empty_states

            if len(wf_state)==0:
                wf_state = empty_states

            if len(wi_state)==0:
                wi_state = empty_states

            # Exemplar sensor only    
            XDIFF_Sensor_STATE = np.append(xdiff_sfe_state,xdiff_sie_state, axis=1)
            XDIFF_Sensor_STATE = XDIFF_Sensor_STATE.astype(int)
            XDIFF_Sensor_NAMES = list(np.array(Xdiff_Names)[xdiff_sfe_idx])+list(np.array(Xdiff_Names)[xdiff_sie_idx])
            # months of a year,days of a week, and hours of a day
            # (Monday, Tuesday,Wendsday,Thursday,Saturday,Sunday) =(0,1,2,3,4,5,6)
            XDIFF_Time_STATE_temp = build_time_states(Xdiff_Time)
            XDIFF_Time_NAMES_temp = ['MTH','WD','HR']
            XDIFF_Time_STATE = []
            XDIFF_Time_NAMES = []
            for xt_col,xt_name in zip(XDIFF_Time_STATE_temp.T,XDIFF_Time_NAMES_temp):
                if len(set(xt_col))>1:
                    XDIFF_Time_STATE.append(xt_col)
                    XDIFF_Time_NAMES.append(xt_name)
            XDIFF_Time_STATE = np.array(XDIFF_Time_STATE).T
            #################################################
            # FORMATTED DATA  FOR IRREGUALR EVENT
            #################################################
            #** Save this variables***
            #diffdata_mat = np.hstack([XDIFF_Sensor_STATE,X_Weather_STATE,XDIFF_Time_STATE])
            #diffdata_names = XDIFF_Sensor_NAMES+X_Weather_NAMES+XDIFF_Time_NAMES
            diffdata_exemplar = dict(sf_diff_exemplars_dict.items()+si_diff_exemplars_dict.items())
            diffdata_zvar = Xdiff_zero_var_list
            
            diffdata_dict = dict()
            diffdata_dict.update({'build_diff_matrix_out': build_diff_matrix_out})
            diffdata_dict.update({'diffdata_state_mat': XDIFF_Sensor_STATE})
            diffdata_dict.update({'diffdata_weather_mat': XDIFF_Weather_STATE})
            diffdata_dict.update({'diffdata_time_mat': XDIFF_Time_STATE})
            diffdata_dict.update({'diff_time_slot': Xdiff_Time})
            diffdata_dict.update({'diffdata_exemplar': diffdata_exemplar})
            diffdata_dict.update({'diffdata_zvar': diffdata_zvar})
            diffdata_dict.update({'sensor_names': XDIFF_Sensor_NAMES})
            diffdata_dict.update({'weather_names': X_Weather_NAMES})
            diffdata_dict.update({'time_names': X_Time_NAMES})
            dsout.update({'diffdata_dict': diffdata_dict})

    dsout.update({'bldg_key': remove_dot(bldg_key)})
    return dsout


def interpolation_measurement(data_dict,input_names,err_rate=1,sgm_bnd=20):
    print '-' * 40
    print 'interploattion starts....'
    print '-' * 40
    measurement_point_set = []
    num_of_discrete_val = []
    sampling_interval_set = []
    num_type_set = []
    err_rate = 1
    sgm_bnd = 20

    for i, key_name in enumerate(input_names):
        print key_name, '...'
        t_ = np.array(data_dict[key_name][2][0])

        if len(t_) == 0:
            continue

        intpl_intv = np.ceil((t_[-1]-t_[0]) /len(t_))
        sampling_interval_set.append(intpl_intv)
        val_ = np.array(data_dict[key_name][2][1])
        num_of_discrete_val_temp = len(set(val_))
        num_of_discrete_val.append(num_of_discrete_val_temp)

        # filtering outlier
        # assuming 1% of errors and 30 x standard deviation rules
        outlier_idx = outlier_detect(val_, err_rate, sgm_bnd)

        if len(outlier_idx) > 0:
            print 'outlier samples are detected: ', 'outlier_idx:', outlier_idx
            t_ = np.delete(t_,outlier_idx)
            val_ = np.delete(val_,outlier_idx)
        t_new = np.r_[t_[0]:t_[-1]:intpl_intv]

        """
        if num_of_discrete_val_temp<MIN_NUM_VAL_FOR_FLOAT:
            num_type=INT_TYPE
            val_new=fast_nearest_interp(t_new, t_,val_)
        else:
            num_type=FLOAT_TYPE
            val_new = np.interp(t_new, t_,val_)
        """

        num_type = check_data_type(data_dict[key_name][2][1])
        if num_type == INT_TYPE:
            val_new = fast_nearest_interp(t_new, t_,val_)
        else:
            #num_type=FLOAT_TYPE
            val_new = np.interp(t_new, t_,val_)

        c = np.vstack([t_new,val_new])
        measurement_point_set.append(c)
        num_type_set.append(num_type)

    #return measurement_point_set,num_type_set,num_of_discrete_val,sampling_interval_set
    return measurement_point_set,np.array(num_type_set)


def x_input_to_states(xinput, CORR_VAL_OUT=0, PARALLEL = False):
    sinput = np.zeros(xinput.shape)
    num_samples = xinput.shape[0]
    num_sensors = xinput.shape[1]
    if num_samples < num_sensors:
        print '[WARN] number of samplesa are smaller than number of sensors'

    print 'Mapping', xinput.shape, ' marix to discrete states '

    for k, samples in enumerate(xinput.T):
        obs = samples[:,np.newaxis]
        label, opt_num_cluster, model, score, score_err_sum = state_retrieval(obs, max_num_cluster=6, est_method='kmean', PARALLEL=PARALLEL)
        high_peak_label_idx = np.argmax(model.cluster_centers_)
        low_peak_label_idx = np.argmin(model.cluster_centers_)
        high_peak_idx = np.nonzero(label == high_peak_label_idx)[0]
        sinput[high_peak_idx,k] = 1
        low_peak_idx = np.nonzero(label == low_peak_label_idx)[0]
        sinput[low_peak_idx, k] = -1

    corr_state_val = []
    if CORR_VAL_OUT == 1:
        print 'Compute Correlation Score....'
        for k,(row1, row2) in enumerate(zip(sinput.T, xinput.T)):
            corr_state_val.append(round(stats.pearsonr(row1, row2)[0],3))
    corr_state_val = np.array(corr_state_val)
    return sinput, corr_state_val