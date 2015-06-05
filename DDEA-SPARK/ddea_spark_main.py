# To force float point division
from __future__ import division

import os
import sys
import uuid
import datetime as dt
from dateutil import tz
import shlex

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

from scipy import signal
from scipy import stats
from scipy.interpolate import interp1d

import pylab as pl
import matplotlib.pyplot as plt
# Interactive mode for plotting
plt.ion()

import time
import itertools
import calendar
import random
from matplotlib.collections import LineCollection
import pprint
import pytz

############################# PREP SPARK CONTEXT ###############################

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.storagelevel import StorageLevel
from pyspark.sql import *
import atexit

print '-' * 36, " SETUP SPARK ", '-' * 36

conf = SparkConf()
sc = SparkContext('local', conf=conf)
atexit.register(lambda: sc.stop())
sqlContext = SQLContext(sc)

################################################################################
from spark_retrieval import *
import mytool as mt

# Compute Average Feature if  PROC_AVG ==True
PROC_AVG=True
# Compute Differential Feature if PROC_DIFF ==True
PROC_DIFF=True

def ddea_spark_terminate():
    sc.stop()

def ddea_spark_main(sub_system, keyword, sensor, time_inv, start_date, end_date):
    print "\n\n"
    print '###############################################################################'
    print '###############################################################################'
    print 'Begining of Program'
    print '###############################################################################'
    print '###############################################################################'

    ANS_START_T=start_date
    ANS_END_T=end_date

    print "ANS_START_T : " + str(ANS_START_T), type(ANS_START_T)
    print "ANS_END_T   : " + str(ANS_END_T), type(ANS_END_T)

    # Setting for analysis time interval where all BEMS and weather data is aligned
    # for a slotted time line quantized by TIMELET_INV.
    TIMELET_INV = dt.timedelta(minutes=time_inv)
    print TIMELET_INV, 'time slot interval is set for this data set !!'

    print "Clean up old output..."
    mt.remove_all_files(BN_ANALYSIS_PNG)
    mt.remove_all_files(LH_ANALYSIS_PNG)
    mt.remove_all_files(SD_ANALYSIS_PNG)
    mt.remove_all_files(BN_ANALYSIS_PNG_THUMB)
    mt.remove_all_files(LH_ANALYSIS_PNG_THUMB)
    mt.remove_all_files(SD_ANALYSIS_PNG_THUMB)

    ##################################################################
    # List buildings and substation names
    print "\n\n", '-' * 36, " PRE_BN_STAGE.... ", '-' * 36

    start = time.time()

    input_files = get_input_filenames(sc, sqlContext, keyword, sensor)
    print '#' * 80, '\n#  Data : Pre-Processing total ', len(input_files), " files\n"

    file_reading = get_file_readings(sc, sqlContext, input_files, ANS_START_T, ANS_END_T)
    print '#' * 80,"\n","Input list total ",len(input_files), "Output list total ",len(file_reading), "\n",'#' * 80

    data_dict, purge_list = construct_data_dict(file_reading, ANS_START_T, ANS_END_T, TIMELET_INV, include_weather=1, binfilename=PROC_OUT_DIR + 'data_dict')

    end = time.time()
    print " Task done in %.9f secs" %(end - start)