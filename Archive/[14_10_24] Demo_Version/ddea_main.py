#!/adsc/DDEA_PROTO/bin/python
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
import pprint
import radar_chart

# Custom library
from data_tools import *
from data_retrieval import *
from pack_cluster import *
from data_preprocess import *
from shared_constants import *
from pre_bn_state_processing import *
from data_summerization import *
from mytool import remove_all_files
##################################################################
# Interactive mode for plotting
#plt.ion()

##################################################################
# Processing Configuraiton Settings
##################################################################
# all BEMS and weather data is built into a signel variable, 'data_dict'.
# 'data_dict' is a  'dictionary' data structure of python. 
# For debugging or experiment purpose, the program allows to store 'data_dict' variable 
# to data_dict.bin by setting a flag variabe, IS_USING_SAVED_DICT
# IS_USING_SAVED_DICT=0  (Default) : Build a new 'data_dict' variabe and store it to 'data_dict.bin'
# IS_USING_SAVED_DICT=1  :  Skip to build 'data_dict' and load 'data_dict.bin' instead 
# IS_USING_SAVED_DICT=-1  : Neither build nor load 'data_dict' 

# Default flag for processing
IS_USING_SAVED_DICT = 0
CHECK_DATA_FORMAT = 1
Data_Summarization = 1

# this is a vtt specific sensor name conversion
def convert_vtt_name(id_labels):
    if isinstance(id_labels,list)==False:
        id_labels=[id_labels]
    out_name=[key_label_ for key_label_ in id_labels ]
    return out_name


def ddea_analysis(subsystem, sensor, start_date, end_date, dr_point="_POWER_"):

    # Setting Analysis period  where ANS_START_T and ANS_START_T are the starting and
    # and the ending timestamp.
    #ANS_START_T=dt.datetime(2013,6,1,0)
    #ANS_END_T=dt.datetime(2013,12,1,0)

    print "\n\n"
    print '###############################################################################'
    print '###############################################################################'
    print 'Begining of Program'
    print '###############################################################################'
    print '###############################################################################'

    ANS_START_T=start_date
    ANS_END_T=end_date

    print "ANS_START_T : " + str(ANS_START_T)
    print "ANS_END_T   : " + str(ANS_END_T)

    # Setting for analysis time interval where all BEMS and weather data is aligned
    # for a slotted time line quantized by TIMELET_INV.
    TIMELET_INV=dt.timedelta(minutes=60)
    print TIMELET_INV, 'time slot interval is set for this data set !!'
    print '-------------------------------------------------------------------'

    # Compute Average Feature if  PROC_AVG ==True
    PROC_AVG=True
    # Compute Differential Feature if PROC_DIFF ==True
    PROC_DIFF=True

    ##################################################################
    # List buildings and substation names
    # Skip all data PRE_BN_STAGE
    #['GW1','GW2','VAK1','VAK2']
    #bldg_key_set=['GW2']
    bldg_key_set=subsystem.split(',')
    bldg_key_set_run=bldg_key_set

    print "Sub-System selected [" + subsystem + "]"
    if sensor:
        print "Sensor Keyword selected ["+sensor+ "]"

    print "Initial DR point [" + dr_point + "]"

    print "Clean up old output..."
    remove_all_files(BN_ANALYSIS_PNG)
    remove_all_files(LH_ANALYSIS_PNG)
    remove_all_files(SD_ANALYSIS_PNG)
    remove_all_files(BN_ANALYSIS_PNG_THUMB)
    remove_all_files(LH_ANALYSIS_PNG_THUMB)
    remove_all_files(SD_ANALYSIS_PNG_THUMB)


    #  Retrieving a set of sensors having a key value in bldg_key_set
    for bldg_key in bldg_key_set_run:
        print '###############################################################################'
        print '###############################################################################'
        print 'Processing '+ bldg_key+'.....'
        print '###############################################################################'
        print '###############################################################################'

        #temp=subprocess.check_output('ls '+DATA_DIR+'*'+bldg_key+'*.bin', shell=True)
        #temp=subprocess.check_output('ls '+DATA_DIR+'*'+bldg_key+'*.bin | grep POWER', shell=True)
        grep_cmd = 'ls '+DATA_DIR+'*'+bldg_key+'*.bin'
        if sensor:
            grep_cmd += ' | grep ' + sensor

        try:
            temp=subprocess.check_output(grep_cmd, shell=True)
        except:
            if sensor:
                print "! : " + sensor + " data for " + bldg_key + " is not found."
            else:
                print bldg_key + " data is not found."

            continue

        input_files_temp =shlex.split(temp)
        # Get rid of duplicated files
        input_files_temp=list(set(input_files_temp))
        input_files=input_files_temp

        ###############################################################################
        # This directly searches files from bin file name
        print '###############################################################################'
        print '#  Data Pre-Processing'
        print '###############################################################################'
        # define input_files  to be read
        if IS_USING_SAVED_DICT==0:
            print 'Extract a common time range...'
            ANS_START_T, ANS_END_T, input_file_to_be_included = time_range_check(input_files, ANS_START_T, ANS_END_T, TIMELET_INV)
            print 'time range readjusted  to  (' ,ANS_START_T, ', ', ANS_END_T,')'
            start__dictproc_t=time.time()
            data_dict, purge_list= construct_data_dict(input_file_to_be_included,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename=PROC_OUT_DIR + 'data_dict',IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
            end__dictproc_t=time.time()
            print 'the time of construct data dict.bin is ', end__dictproc_t-start__dictproc_t, ' sec'
            print '--------------------------------------'
        elif IS_USING_SAVED_DICT==1:
            print 'Loading data dictionary......'
            start__dictproc_t=time.time()
            data_dict = mt.loadObjectBinaryFast(PROC_OUT_DIR +'data_dict.bin')
            end__dictproc_t=time.time()
            print 'the time of loading data dict.bin is ', end__dictproc_t-start__dictproc_t, ' sec'
            print '--------------------------------------'
        else:
            print 'Skip data dict'

        if CHECK_DATA_FORMAT==1:
            # This is for data verification purpose
            # You cab skip it if you are sure that there would be no bug in the 'construct_data_dict' function.
            list_of_wrong_data_format=verify_data_format(data_dict)
            if len(list_of_wrong_data_format)>0:
                print 'Measurement list below'
                print '----------------------------------------'
                print list_of_wrong_data_format
                raise NameError('Errors in data format')

        # This perform data summerization process.
        if Data_Summarization==1:
            bldg_out=data_summerization(bldg_key,data_dict,PROC_AVG=True,PROC_DIFF=True)

    RECON_BLDG_BIN_OUT=0
    if RECON_BLDG_BIN_OUT==1:
        for bldg_key in ['GW1_','GW2_','VAK1_','VAK2_']:
            avgdata_dict=mt.loadObjectBinaryFast('./VTT/'+bldg_key+'avgdata_dict.bin')
            diffdata_dict=mt.loadObjectBinaryFast('./VTT/'+bldg_key+'diffdata_dict.bin')
            data_dict=mt.loadObjectBinaryFast('./VTT/'+bldg_key+'data_dict.bin')
            cmd_str=remove_dot(bldg_key)+'out={\'data_dict\':data_dict}'
            exec(cmd_str)
            cmd_str=remove_dot(bldg_key)+'out.update({\'avgdata_dict\':avgdata_dict})'
            exec(cmd_str)
            cmd_str=remove_dot(bldg_key)+'out.update({\'diffdata_dict\':diffdata_dict})'
            exec(cmd_str)
            cmd_str=remove_dot(bldg_key)+'out.update({\'bldg_key\':remove_dot(bldg_key)})'
            exec(cmd_str)
            cmd_str='mt.saveObjectBinaryFast('+remove_dot(bldg_key)+'out'+',\''+PROC_OUT_DIR+remove_dot(bldg_key)+'out.bin\')'
            exec(cmd_str)


    print '###############################################################################'
    print '#  Model_Discovery'
    print '###############################################################################'
    #bldg_key_set=['GW1','GW2','VAK1','VAK2']
    Model_Discovery=1
    #pwr_key='_POWER_';
    pwr_key=dr_point
    bldg_dict={}
    for bldg_load_key in  bldg_key_set:
        print 'Building for ',bldg_load_key, '....'
        try:
            bldg_tag='vtt_'+bldg_load_key
            bldg_load_out=mt.loadObjectBinaryFast(PROC_OUT_DIR+bldg_load_key+'_out.bin')
        except:
            print bldg_load_key+' bin file not found in PROC_OUT_DIR, skip....'
            pass
        mt.saveObjectBinaryFast(bldg_load_out['data_dict'],PROC_OUT_DIR+'data_dict.bin')
        if 'avgdata_dict' in bldg_load_out.keys():
            mt.saveObjectBinaryFast(bldg_load_out['avgdata_dict'],PROC_OUT_DIR+'avgdata_dict.bin')
        if 'diffdata_dict' in bldg_load_out.keys():
            mt.saveObjectBinaryFast(bldg_load_out['diffdata_dict'],PROC_OUT_DIR+'diffdata_dict.bin')
        pname_key= pwr_key
        bldg_dict.update({bldg_tag:create_bldg_obj(PROC_OUT_DIR,bldg_tag,pname_key)})
        bldg_=obj(bldg_dict)
        #cmd_str='bldg_.'+bldg_tag+'.data_out=obj(bldg_load_out)'
        #exec(cmd_str)
        cmd_str='bldg_obj=bldg_.'+bldg_tag
        exec(cmd_str)

        anal_out={}
        if 'avgdata_dict' in bldg_load_out.keys():
            anal_out.update({'avg':bn_prob_analysis(bldg_obj,sig_tag_='avg')})
        if 'diffdata_dict' in bldg_load_out.keys():
            anal_out.update({'diff':bn_prob_analysis(bldg_obj,sig_tag_='diff')})
        cmd_str='bldg_.'+bldg_tag+'.anal_out=obj(anal_out)'
        exec(cmd_str)

    # Save vtt building object file.
    mt.saveObjectBinaryFast(bldg_ ,PROC_OUT_DIR+'vtt_bldg_obj.bin')

    bldg_.convert_name=convert_vtt_name

    #######################################################################################
    # Analysis For VTT
    #######################################################################################
    # Analysis of BN network result - All result will be saved in fig_dir.
    BN_ANAL=1
    if BN_ANAL==1:
        # Plotting individual LHs
        PLOTTING_LH=0
        if PLOTTING_LH==1:
            plotting_bldg_lh(bldg_,attr_class='sensor',num_picks=30)
            plotting_bldg_lh(bldg_,attr_class='time',num_picks=30)
            plotting_bldg_lh(bldg_,attr_class='weather',num_picks=30)

        PLOTTING_BN=1
        if PLOTTING_BN==1:
            plotting_bldg_bn(bldg_)

    print '**************************** End of Program ****************************'



