#!/adsc/DDEA_PROTO/bin/python

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

from data_retrieval import *

##################################################################
# Interactive mode for plotting
plt.ion()

import pytz
# Setting Analysis period  where ANS_START_T and ANS_START_T are the starting
# and the ending timestamp.
ANS_START_T = dt.datetime(2014,1,1,0).replace(tzinfo=tz.gettz('UTC')).astimezone(pytz.timezone('US/Pacific'))
ANS_END_T = dt.datetime(2014,2,1,0).replace(tzinfo=tz.gettz('UTC')).astimezone(pytz.timezone('US/Pacific'))

# Setting for analysis time interval where all BEMS and weather data is aligned
# for a slotted time line quantized by TIMELET_INV.
TIMELET_INV = dt.timedelta(minutes=60)
print TIMELET_INV, 'time slot interval is set for this data set !!'
print '-------------------------------------------------------------------'

# Compute Average Feature if  PROC_AVG ==True
PROC_AVG=True
# Compute Differential Feature if PROC_DIFF ==True
PROC_DIFF=True

##################################################################
# List buildings and substation names
print 'PRE_BN_STAGE....'
#bldg_key_set_run = ['POWER']
bldg_key_set_run = ['CURRENT']

start = time.time()
#  Retrieving a set of sensors having a key value in bldg_key_set
for bldg_key in bldg_key_set_run:
    print '#' * 80, '\n','Processing '+ bldg_key + '.....', '#' * 80, '\n'

    temp = subprocess.check_output('ls '+DATA_DIR+'*'+bldg_key+'*.bin', shell = True)
    #temp = subprocess.check_output('ls '+DATA_DIR+'*'+bldg_key+'*.bin | grep POWER', shell=True)
    input_files_temp = shlex.split(temp)

    # Get rid of duplicated files
    input_files_temp = list(set(input_files_temp))
    input_files = input_files_temp

# This step directly searches files from bin file name
print '#' * 80
print '#  Data : Pre-Processing total ', len(input_files), " files"
print '#' * 80

# define input_files to be read
print '-' * 80, '\n Extract a common time range...\n', '-' * 80, '\n'
ANS_START_T, ANS_END_T, input_file_to_be_included = \
    time_range_check(input_files, ANS_START_T, ANS_END_T, TIMELET_INV)

print '#' * 80, '\n#  Data : Pre-Processing total ', len(input_files), " files\n"
print '#' * 80,"\n","Input list total ", len(input_files), "Output list total ", "\n",'#' * 80

print 'time range readjusted  to  (' ,ANS_START_T, ', ', ANS_END_T,')'
data_dict,purge_list=construct_data_dict(input_file_to_be_included,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename=PROC_OUT_DIR + 'data_dict',IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)

end = time.time()
print 'the time of construct data dict.bin is ',
print '--------------------------------------'
print " Task done in %.9f secs" %(end - start)