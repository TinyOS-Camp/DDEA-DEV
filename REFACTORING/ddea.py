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
import pre_bn_processing as pbp

##################################################################
# Processing Configuraiton Settings
##################################################################
# all sensor and weather data is built into a signel variable, 'data_dict'.
# 'data_dict' is a  'dictionary' data structure of python.

# For debugging or experiment purpose, the program allows to store 'data_dict'
# variable to data_dict.bin by setting a flag variabe, IS_USING_SAVED_DICT
# IS_USING_SAVED_DICT = 0  (Default) : Build a new 'data_dict' variabe and store it to 'data_dict.bin'
# IS_USING_SAVED_DICT = 1  :  Skip to build 'data_dict' and load 'data_dict.bin' instead
IS_USING_SAVED_DICT = 0

##################################################################
# Processing Configuraiton Settings
##################################################################

#TODO : this whole section of selecting a building should be wrapped in param
# Building keyword.
bldg_key = 'GW2'
pname_key = '_POWER_'

# Setting Analysis period  where ANS_START_T and ANS_START_T are the starting and
# and the ending timestamp.
ANS_START_T = dt.datetime(2013, 6, 1, 0)
ANS_END_T = dt.datetime(2013, 7, 1, 0)

# Setting for analysis time interval where all BEMS and weather data is aligned
# for a slotted time line quantized by TIMELET_INV. 
TIMELET_INV = dt.timedelta(minutes=15)

print TIMELET_INV, 'time slot interval is set for this data set !!'
print '-------------------------------------------------------------------'

print "Clean up old output..."
mt.remove_all_files(FIG_DIR)

bldg_list = list()
#----------------------------- DATA PRE-PROCESSING -----------------------------
#  Retrieving a set of sensors with specified key
print '#' * 80
print 'DATA PRE-PROCESSING ' + bldg_key + '...'
print '#' * 80


# This directly searches files from bin file name define input_files to be read
if IS_USING_SAVED_DICT == 0:

    try:
        temp = subprocess.check_output('ls '+DATA_DIR+'*'+bldg_key+'*.bin', shell=True)
    except Exception:
        print "in case there is no files to read, abort operation..."
        exit()

    # Get rid of duplicated files
    input_files_temp = shlex.split(temp)
    input_files_temp = list(set(input_files_temp))
    input_files = input_files_temp

    print 'Extract a common time range...'
    ANS_START_T, ANS_END_T, input_file_to_be_included = time_range_check(input_files, ANS_START_T, ANS_END_T, TIMELET_INV)
    print 'time range readjusted  to  (', ANS_START_T, ', ', ANS_END_T, ')'

    data_dict, purge_list = construct_data_dict(input_file_to_be_included, ANS_START_T, ANS_END_T, TIMELET_INV, binfilename=PROC_OUT_DIR + 'data_dict', IS_USING_PARALLEL=IS_USING_PARALLEL_OPT, IS_SAVING_DATA_DICT=True)
    print '-' * 40

else:
    print 'Loading data dictionary......'
    data_dict = mt.loadObjectBinaryFast(PROC_OUT_DIR + 'data_dict.bin')
    print '-' * 40

# This is for data verification purpose
# You cab skip it if you are sure that there would be no bug in the 'construct_data_dict' function.
list_of_wrong_data_format = verify_data_format(data_dict, PARALLEL=IS_USING_PARALLEL_OPT)
if len(list_of_wrong_data_format) > 0:
    print 'Measurement list below'
    print '-' * 40
    print list_of_wrong_data_format
    raise NameError('Errors in data format')

#----------------------------- DATA SUMMERIZATION ------------------------------
# This perform data summerization process.
print '#' * 80
print 'DATA SUMMARIZATION...'
print '#' * 80

# Compute Average Feature if PROC_AVG == True
# Compute Differential Feature if PROC_DIFF == True
bldg_load_out = data_summerization(bldg_key, data_dict, proc_avg=True, proc_diff=True)

#------------------------------- MODEL DISCOVERY -------------------------------
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

#------------------------------------ PLOTTING ---------------------------------
print '#' * 80
print 'ANALYTICS PLOTTING...'
print '#' * 80

# Analysis of BN network result - All result will be saved in fig_dir.
pbp.plotting_bldg_lh(bldg_list, attr='sensor', num_picks=30)
pbp.plotting_bldg_lh(bldg_list, attr='time', num_picks=30)
pbp.plotting_bldg_lh(bldg_list, attr='weather', num_picks=30)

pbp.plotting_bldg_bn(bldg_list)

print '**************************** End of Program ****************************'
