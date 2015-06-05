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
#from stackedBarGraph import StackedBarGrapher
import pprint
#import radar_chart

# Custom library
from data_tools import *
from data_retrieval import *
from pack_cluster import *
from data_preprocess import *
from shared_constants import *
from pre_bn_state_processing import *
from data_summerization import *
##################################################################
PROC_OUT_DIR=Gvalley_out_dir
DATA_DIR=Gvalley_data_dir
WEATHER_DIR=Gvalley_weather_dir


# Interactive mode for plotting
plt.ion() 
# Gvalley dependect function and parameters
gvalley_bgid_dict={'10110102'	:'BDV1', '10110105'	:'UEZ1', '10110107'	:'HST',\
'10110110'	:'UEZ2','10110111'	:'MAT','10110113'	:'ACH1','10110118'	:'ACH2',
'10110119'	:'ACT8','10110182'	:'KLB1','10110187'	:'WKR','10110188'	:'ATW1',
'10110190'	:'ATW2','10110191'	:'EDT2','10110192'	:'DPT3','10110193'	:'GRH',
'10110194'	:'KLS1','10110214'	:'OMS','11810101'	:'MPC'}

gvalley_data_list=['MODEM_NO','HSHLD_INFO_SEQ','CKM_DATE','PF','MAX_DEMND_EPR',\
'CRMON_VLD_EENGY','CRMON_RACT_EENGY','LMON_VLD_EENGY','LMON_RACT_EENGY',\
'LP_FWD_VLD_EENGY','LP_GRD_RACT_EENGY','LP_TRTH_RACT_EENGY','LP_APET_RACT_EENGY',\
'LP_BWD_VLD_EENGY','EMS_GAS_USG','EMS_HTNG_USG','EMS_HOTWT_USG','EMS_TAPWT_USG','MAKEDAY',\
'CORRDAY','REC_CNT']
# 'BDV1' : 1500 sensors
# 'MAT': 3500 sensors
# ATW1 : 1500 sensors
# Define name conversion method
def convert_gvalley_name(id_labels):
    out_name=[]
    if isinstance(id_labels,list)==False:
        id_labels=[id_labels]
    for key_label_ in id_labels:
        if key_label_[2:10] in gvalley_bgid_dict:
            bldg_id_name=gvalley_bgid_dict[key_label_[2:10]]
            cdata_id=[data_id  for data_id in  gvalley_data_list if len(grep(data_id,[id_labels[0]]))>0]
            lastdigit_id=key_label_[len(key_label_)-len(cdata_id[0])-5:-1*len(cdata_id[0])-1]
            out_name.append(bldg_id_name+'_'+cdata_id[0]+'_'+lastdigit_id)
        else: 
            out_name.append(key_label_)
    return out_name


##################################################################
# Processing Configuraiton Settings
##################################################################
# Analysis BLDG ID 
BLDG_DATA_POINT_CNT=0
if BLDG_DATA_POINT_CNT==1:
    gvalley_data_id_cnt={}
    for bldg_id in gvalley_bgid_dict.keys():
        print 'process ', gvalley_bgid_dict[bldg_id], '...'
        data_id_cnt=[]
        for data_id in gvalley_data_list:
            temp=subprocess.check_output('ls '+DATA_DIR+'*'+bldg_id+'*.bin |grep '+data_id+' |wc -l', shell=True) 
            data_id_cnt.append(int(shlex.split(temp)[0]))
        gvalley_data_id_cnt.update({gvalley_bgid_dict[bldg_id] :data_id_cnt})
    
    max_count=max(max(gvalley_data_id_cnt.values()))
    fig_name='Counts of Data Points'
    fig=figure(fig_name,figsize=(30.0,30.0))
    for i,bldg_id in enumerate(gvalley_bgid_dict.keys()):
        bldg_name=gvalley_bgid_dict[bldg_id]
        plt.subplot(7,3,i+1)
        x_tick_val=range(len(gvalley_data_list))
        x_tick_label=gvalley_data_list
        plt.bar(x_tick_val,gvalley_data_id_cnt[bldg_name])
        plt.ylabel('# of data points',fontsize='small')
        plt.title(bldg_name,fontsize='large')
        if i>14:
            plt.xticks(x_tick_val,x_tick_label,rotation=270, fontsize=10)
            plt.tick_params(labelsize='small')
        else:
            plt.xticks(x_tick_val,['']*len(x_tick_val),rotation=270, fontsize=10)
            plt.tick_params(labelsize='small')
        plt.ylim([-0.05,max_count*1.2])
    png_name='Gvalley'+remove_dot(fig_name)+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')

PRE_BN_STAGE=0
if PRE_BN_STAGE==0:
    bldg_key_set=[]
    print 'skip PRE_BN_STAGE....'
else:
    bldg_key_set=gvalley_bgid_dict.values()

for bldg_key in bldg_key_set:
    print '###############################################################################'
    print '###############################################################################'
    print 'Processing '+ bldg_key+'.....'
    print '###############################################################################'
    print '###############################################################################'
    try:
        #bldg_key='BDV1'
        #bldg_key='GRH'
        bldg_id=[key_val[0] for key_val in gvalley_bgid_dict.items() if key_val[1]==bldg_key][0]
        data_id='CRMON_VLD_EENGY'
        #temp= subprocess.check_output("ls "+DATA_DIR+"*"+FL_EXT+" |grep "+bldg_id, shell=True) 
        temp=subprocess.check_output('ls '+DATA_DIR+'*'+bldg_id+'*.bin |grep '+data_id, shell=True) 
                    
        input_files_temp =shlex.split(temp)
        # Get rid of duplicated files
        input_files_temp=list(set(input_files_temp))
        input_files=input_files_temp
        #input_files=['../gvalley/Binfiles/'+temp for temp in input_files_temp]
        
        IS_USING_SAVED_DICT=0
        print 'Extract a common time range...'
        # Analysis period
        ANS_START_T=dt.datetime(2013,1,1,0)
        ANS_END_T=dt.datetime(2013,12,30,0)
        # Interval of timelet, currently set to 1 Hour
        TIMELET_INV=dt.timedelta(hours=1)
        #TIMELET_INV=dt.timedelta(minutes=60)
        print TIMELET_INV, 'time slot interval is set for this data set !!'
        print '-------------------------------------------------------------------'
        PROC_AVG=True
        PROC_DIFF=False
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
                data_dict,purge_list=construct_data_dict_2\
                (input_files,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename='data_dict', IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
            else:
                data_dict,purge_list=\
                construct_data_dict(input_file_to_be_included,ANS_START_T,ANS_END_T,TIMELET_INV,\
                binfilename='data_dict',IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
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
        Data_Summarization=1
        if Data_Summarization==1:
            bldg_out=data_summerization(bldg_key,data_dict,PROC_AVG=True,PROC_DIFF=False)
    except:
        print 'error occured, pass this '
    

print '###############################################################################'
print '#  Model_Discovery'
print '###############################################################################'
Model_Discovery=1
if Model_Discovery==1:
    gvalley_pwr_key='CRMON_VLD_EENGY'; gvalley_gas_key='EMS_GAS_USG'
    gvalley_heat_key='EMS_HTNG_USG'; gvalley_hwt_key='EMS_HOTWT_USG'
    gvalley_twt_key='EMS_TAPWT_USG'; dict_dir='./Gvalley/'
    LOAD_BLDG_OBJ=0
    if LOAD_BLDG_OBJ==1:
        print 'load Gvalley_bldg_obj.bin'
        #bldg_=mt.loadObjectBinaryFast(gvalley_dict_dir_set[0]+'Gvalley_bldg_obj.bin')
        bldg_=mt.loadObjectBinaryFast(PROC_OUT_DIR+'Gvalley_bldg_obj.bin')
    else:
        bldg_dict={}
        for bldg_load_key in gvalley_bgid_dict.values():
            print 'Building for ',bldg_load_key, '....'
            try:
                bldg_tag='Gvalley_'+bldg_load_key
                bldg_load_out=mt.loadObjectBinaryFast(dict_dir+bldg_load_key+'_out.bin')
            except:
                print 'not found, skip....'
                pass
            data_dict_keys=bldg_load_out['data_dict'].keys()
            del_idx=grep('CRMON_VLD_EENGY',data_dict_keys)
            for idx in del_idx:
                key=data_dict_keys[idx]
                if key in bldg_load_out['data_dict']:
                    del bldg_load_out['data_dict'][key]
                
            mt.saveObjectBinaryFast(bldg_load_out['data_dict'],dict_dir+'data_dict.bin')
            if 'avgdata_dict' in bldg_load_out.keys():
                mt.saveObjectBinaryFast(bldg_load_out['avgdata_dict'],dict_dir+'avgdata_dict.bin')
            if 'diffdata_dict' in bldg_load_out.keys():
                mt.saveObjectBinaryFast(bldg_load_out['avgdata_dict'],dict_dir+'diffdata_dict.bin')
            pname_key= gvalley_pwr_key
            bldg_dict.update({bldg_tag:create_bldg_obj(dict_dir,bldg_tag,pname_key)})
            bldg_=obj(bldg_dict)
            #cmd_str='bldg_.'+bldg_tag+'.data_out=obj(bldg_load_out)'
            #exec(cmd_str)
            cmd_str='bldg_.'+bldg_tag+'.gvalley_bgid_dict=gvalley_bgid_dict'
            exec(cmd_str)
            cmd_str='bldg_.'+bldg_tag+'.gvalley_data_list=gvalley_data_list'
            exec(cmd_str)
            cmd_str='bldg_obj=bldg_.'+bldg_tag
            exec(cmd_str)
            anal_out={}
            if 'avgdata_dict' in bldg_load_out.keys():
                anal_out.update({'avg':bn_prob_analysis(bldg_obj,sig_tag_='avg')})
            if 'diffdata_dict' in bldg_load_out.keys():
                anal_out.update({'diff':bn_prob_analysis(bldg_obj,sig_tag_='diff')})
            cmd_str='bldg_.'+bldg_tag+'.anal_out=obj(anal_out)'            
            exec(cmd_str)
        cmd_str='bldg_.'+'convert_name=convert_gvalley_name'
        exec(cmd_str)
        mt.saveObjectBinaryFast(bldg_ ,PROC_OUT_DIR+'Gvalley_bldg_obj.bin')
        mt.saveObjectBinaryFast('LOAD_BLDG_OBJ' ,PROC_OUT_DIR+'Gvalley_bldg_obj_is_done.txt')

    #######################################################################################
    # Analysis For GValley
    #######################################################################################
    # Analysis of BN network result
BN_ANAL=1
if BN_ANAL==1:
    # Plotting individual LHs
    PLOTTING_LH=1
    if PLOTTING_LH==1:
        plotting_bldg_lh(bldg_,attr_class='sensor',num_picks=30)
        plotting_bldg_lh(bldg_,attr_class='time',num_picks=30)
        plotting_bldg_lh(bldg_,attr_class='weather',num_picks=30)
    
    PLOTTING_BN=1
    if PLOTTING_BN==1:
        plotting_bldg_bn(bldg_)
    
More_BN_ANAL=0
if More_BN_ANAL==1:
    # Extra Steps for GValley network    
    bdv1_hc_b,cols,bdv1_amat=compute_bn_sensors(bldg_.Gvalley_BDV1)
    fig_name='BN for BDV1 power meters '+'CRMON_VLD_EENGY'
    plt.figure(fig_name,figsize=(25.0,25.0))
    rbn.nx_plot(bdv1_hc_b,convert_gvalley_name(cols),graph_layout='circular',node_text_size=30)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    plt.close()

    bdv1_in=np.array([ len(np.nonzero(col==1)[0]) for col in bdv1_amat])
    bdv1_out=np.array([ len(np.nonzero(col==1)[0]) for col in bdv1_amat.T])
    bdv1_in_out=bdv1_in+bdv1_out
    
    Mat_hc_b,cols,Mat_amat=compute_bn_sensors(bldg_.Gvalley_MAT)
    fig_name='BN for MAT power meters '+'CRMON_VLD_EENGY'
    plt.figure(fig_name,figsize=(25.0,25.0))
    rbn.nx_plot(bdv1_hc_b,convert_gvalley_name(cols),graph_layout='circular',node_text_size=30)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    plt.close()
    Mat_in=np.array([ len(np.nonzero(col==1)[0]) for col in Mat_amat])
    Mat_out=np.array([ len(np.nonzero(col==1)[0]) for col in Mat_amat.T])
    Mat_in_out=Mat_in+Mat_out
    
    atw1_hc_b,cols,atw1_amat=compute_bn_sensors(bldg_.Gvalley_ATW1)
    fig_name='BN for ATW1 power meters '+'CRMON_VLD_EENGY'
    plt.figure(fig_name,figsize=(25.0,25.0))
    rbn.nx_plot(bdv1_hc_b,convert_gvalley_name(cols),graph_layout='circular',node_text_size=30)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    plt.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    plt.close()
    atw1_in=np.array([ len(np.nonzero(col==1)[0]) for col in atw1_amat])
    atw1_out=np.array([ len(np.nonzero(col==1)[0]) for col in atw1_amat.T])
    atw1_in_out=atw1_in+atw1_out

BN_VERIFY=0
if BN_VERIFY==1:
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



