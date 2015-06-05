#!/usr/bin/python
"""

Author: Deokwooo Jung deokwoo.jung@gmail.com

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
# To force float point division
from __future__ import division

# Custom libraries
from data_summerization import *
from quasar_preprocess import *
from quasar_retrieval import *
import pre_bn_processing as pbp
import datetime as dt

def ddea_process(sensor_data, start_time, end_time, TIMELET_INV, bldg_key, pname_key):

    ANS_START_T = dt.datetime.fromtimestamp(start_time)
    ANS_END_T = dt.datetime.fromtimestamp(end_time)

    data_dict, purge_list = construct_data_dict(sensor_data, ANS_START_T, ANS_END_T, TIMELET_INV, binfilename=PROC_OUT_DIR + 'data_dict')

    # This perform data summerization process.
    print '-' * 40
    print 'VERIFY DATA FORMAT...'
    print '-' * 40

    # This is for data verification purpose
    # You cab skip it if you are sure that there would be no bug in the 'construct_data_dict' function.
    list_of_wrong_data_format = verify_data_format(data_dict, PARALLEL=IS_USING_PARALLEL_OPT)
    if len(list_of_wrong_data_format) > 0:
        print 'Measurement list below'
        print '-' * 40
        print list_of_wrong_data_format
        raise NameError('Errors in data format')

    bldg_list = list()
   #----------------------------- DATA SUMMERIZATION --------------------------
    # This perform data summerization process.
    print '#' * 80
    print 'DATA SUMMARIZATION...'
    print '#' * 80

    # Compute Average Feature if PROC_AVG == True
    # Compute Differential Feature if PROC_DIFF == True
    bldg_load_out = data_summerization(bldg_key, data_dict, proc_avg=True, proc_diff=True)

    #------------------------------- MODEL DISCOVERY ---------------------------
    print '#' * 80
    print 'MODEL DISCOVERY...'
    print '#' * 80

    print 'Building for ', bldg_key, '....'

    ## CREATE BUILDING OBJECT ##
    bldg = pbp.create_bldg_object(bldg_load_out['data_dict'],
                                  bldg_load_out['avgdata_dict'],
                                  bldg_load_out['diffdata_dict'],
                                  bldg_key,
                                  pname_key)

    ## BAYESIAN NETWORK PROBABILITY ANALYSIS OBJECT ##
    if 'avgdata_dict' in bldg_load_out.keys():
        bldg.anal_out.update({'avg': pbp.bn_probability_analysis(bldg, sig_tag='avg')})

    if 'diffdata_dict' in bldg_load_out.keys():
        bldg.anal_out.update({'diff': pbp.bn_probability_analysis(bldg, sig_tag='diff')})

    bldg_list.append(bldg)

    #------------------------------------ PLOTTING -----------------------------
    print '#' * 80
    print 'ANALYTICS PLOTTING...'
    print '#' * 80

    # Analysis of BN network result - All result will be saved in fig_dir.
    pbp.plotting_bldg_lh(bldg_list, attr='sensor', num_picks=30)
    pbp.plotting_bldg_lh(bldg_list, attr='time', num_picks=30)
    pbp.plotting_bldg_lh(bldg_list, attr='weather', num_picks=30)

    pbp.plotting_bldg_bn(bldg_list)

    print '**************************** End of Program ************************'
