# coding: utf-8 

"""
==============================================
BN Learning script for BMS +Weather data
==============================================
"""
#print(__doc__)
# Author: Deokwooo Jung deokwoo.jung@gmail.compile
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
import lib_bnlearn as rbn    
##################################################################
from matplotlib.collections import LineCollection
#from classify_sensors import get_sim_mat
# Interactive mode for plotting
plt.ion() 
#########################################################################    
### Load all builings bin files. 
#########################################################################
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
        #gsbc_dict_dir_set=['./GSBC/allsensors/']
        gsbc_dict_dir_set=['./GSBC/selected/']
        bldg_tag_set=['GSBC_']
        print 'Building for ',bldg_tag_set, '....'
        gsbc_hcw_pname_key='3003....' # Hot and Cold water
        gsbc_main_1_pname_key='300401..' # Maing Buiding F1
        gsbc_main_2_pname_key='300402..' # Maing Buiding F2
        gsbc_hvac_pname_key='3006....' # HVAC
        pname_key=gsbc_hcw_pname_key # 
        for dict_dir,bldg_tag in zip(gsbc_dict_dir_set,bldg_tag_set):
            bldg_dict.update({bldg_tag:create_bldg_obj(dict_dir,bldg_tag,pname_key)})
        bldg_=obj(bldg_dict)
    PLOTTING_LH=0
    if PLOTTING_LH==1:
        plotting_bldg_lh(bldg_,attr_class='sensor',num_picks=30)
        plotting_bldg_lh(bldg_,attr_class='time',num_picks=30)
        plotting_bldg_lh(bldg_,attr_class='weather',num_picks=30)
        
    bldg_obj=bldg_.GSBC_
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

figure('a_20302005')
a_20302005=mt.loadObjectBinaryFast('20302005.bin')
a_20302005_t=[temp[0] for temp in a_20302005['ts']]
a_20302005_val= a_20302005['value']
s_idx=bldg_.GSBC_.avg.sensor_names.index('a_20302005')
plot(bldg_.GSBC_.avg.time_slot, bldg_.GSBC_.avg.data_state_mat[:,s_idx])
plot(a_20302005_t,a_20302005_val)

figure('a_20170204')
a_20170204=mt.loadObjectBinaryFast('20170204.bin')
a_20170204_t=[temp[0] for temp in a_20170204['ts']]
a_20170204_val= a_20170204['value']
s_idx=bldg_.GSBC_.avg.sensor_names.index('a_20170204')
plot(bldg_.GSBC_.avg.time_slot, bldg_.GSBC_.avg.data_state_mat[:,s_idx])
plot(a_20170204_t,a_20170204_val)


figure('a_20242206')
a_20242206=mt.loadObjectBinaryFast('20242206.bin')
a_20242206_t=[temp[0] for temp in a_20242206['ts']]
a_20242206_val= a_20242206['value']
s_idx=bldg_.GSBC_.avg.sensor_names.index('a_20242206')
plot(bldg_.GSBC_.avg.time_slot, bldg_.GSBC_.avg.data_state_mat[:,s_idx])
plot(a_20242206_t,a_20242206_val)





