#!/usr/bin/python
# To force float point division
from __future__ import division

"""
Created on Mon Mar 24 19:24:11 2014

Author : Deokwoo Jung
E-mail : deokwoo.jung@gmail.com

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

# Custom libraries
from data_summerization import *
from data_preprocess import *
from data_retrieval import *
import pre_bn_processing as pbp
import datetime as dt
from json_util import save_processed_json, save_avg_data_summary_json, save_diff_data_summary_json, save_bn_graph_json, conv_bn_graph_json


def ddea_setup():
    pass


def ddea_process(sensor_names_hash, sensor_data, start_time, end_time, timelet_inv, bldg_key, pname_key, plot_analysis=False):

    #----------------------------- DATA PRE-PROCESSING -------------------------
    from log_util import log

    log.info('#' * 80)
    log.info('#  Data Pre-Processing')
    log.info('#' * 80)

    ans_start_t = dt.datetime.fromtimestamp(start_time)
    ans_end_t = dt.datetime.fromtimestamp(end_time)

    data_dict, purge_list = \
            construct_data_dict(sensor_data, ans_start_t, ans_end_t, timelet_inv, PARALLEL=IS_USING_PARALLEL_OPT)

    # This perform data summerization process.
    log.info('-' * 40)
    log.info('VERIFY DATA FORMAT...')
    log.info('-' * 40)

    # This is for data verification purpose
    # You cab skip it if you are sure that there would be no bug in the 'construct_data_dict' function.
    list_of_wrong_data_format = \
        verify_data_format(data_dict, PARALLEL=IS_USING_PARALLEL_OPT)

    if len(list_of_wrong_data_format) > 0:
        log.critical('Measurement list below')
        log.critical('-' * 40)
        log.critical(str(list_of_wrong_data_format))
        raise NameError('Errors in data format')

    #----------------------------- DATA SUMMERIZATION --------------------------
    # This perform data summerization process.
    log.info('#' * 80)
    log.info('DATA SUMMARIZATION...')
    log.info('#' * 80)

    # Compute Average Feature if PROC_AVG == True
    # Compute Differential Feature if PROC_DIFF == True
    bldg_load_out = data_summerization(bldg_key, data_dict, proc_avg=True, proc_diff=True, PARALLEL=IS_USING_PARALLEL_OPT)

    # Save summarized Data in Bin Format
    log.info("Saving summarized building data in bin format...")
    mt.saveObjectBinaryFast(bldg_load_out, PROC_OUT_DIR + bldg_key.lower() + '_ds_out.bin')

    # Export Summarized Data to JSON
    log.info("Saving summarized building data in JSON format...")
    save_avg_data_summary_json(bldg_key, sensor_names_hash, bldg_load_out['avgdata_dict'])
    save_diff_data_summary_json(bldg_key, sensor_names_hash, bldg_load_out['diffdata_dict'])
    save_processed_json(sensor_names_hash, bldg_load_out)

    #------------------------------- MODEL DISCOVERY ---------------------------
    log.info('#' * 80)
    log.info('MODEL DISCOVERY...')
    log.info('#' * 80)

    log.info('Building for '+ bldg_key + '....')

    ## CREATE BUILDING OBJECT ##
    bldg = pbp.create_bldg_object(bldg_load_out['data_dict'],
                                  bldg_load_out['avgdata_dict'],
                                  bldg_load_out['diffdata_dict'],
                                  bldg_key,
                                  pname_key,
                                  PARALLEL=IS_USING_PARALLEL_OPT)

    ## BAYESIAN NETWORK PROBABILITY ANALYSIS OBJECT ##
    if 'avgdata_dict' in bldg_load_out.keys():
        avg = pbp.bn_probability_analysis(bldg, sig_tag='avg')
        bldg.anal_out.update({'avg': avg})

    if 'diffdata_dict' in bldg_load_out.keys():
        diff = pbp.bn_probability_analysis(bldg, sig_tag='diff')
        bldg.anal_out.update({'diff': diff})

    # Save a building data in Bin format
    log.info("Saving building graph in bin format...")
    mt.saveObjectBinaryFast(bldg, PROC_OUT_DIR + bldg_key.lower() + '_bldg_out.bin')

    # Export a building graph in json format
    log.info("Saving building graph in JSON format...")
    all_labels, all_edges = conv_bn_graph_json(bldg)
    save_bn_graph_json(bldg_key, sensor_names_hash, all_labels, all_edges)

    if plot_analysis:
        log.info('#' * 80)
        log.info('ANALYTICS PLOTTING...')
        log.info('#' * 80)

        # Analysis of BN network result - All result will be saved in fig_dir.
        pbp.plotting_bldg_lh(bldg, attr='sensor', num_picks=30)
        pbp.plotting_bldg_lh(bldg, attr='time', num_picks=30)
        pbp.plotting_bldg_lh(bldg, attr='weather', num_picks=30)

        pbp.plotting_bldg_bn(bldg)
