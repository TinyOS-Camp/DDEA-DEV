# -*- coding: utf-8 -*-
#!/usr/bin/python
# To force float point division
from __future__ import division

"""
Created on Fri Mar 14 02:04:12 2014

Author : Deokwoo Jung
E-mail : deokwoo.jung@gmail.com

* Description 
- This file defines constant values shared among python modules.
- Should be included all python modules first. 
"""

###############################################################################
# Constant global variables
###############################################################################
"""
New format of timestamps stored in the bin files
[mt.str2datetime(tmp[0]), dt[5], dt[4], dt[3], dt[6], dt[2], dt[1]]

- str2datetime() converts the string representation of date "%Y-%m-%d %H:%M:%S" to Python datetime object (naive)
- dt[5], dt[4], dt[3], dt[6], dt[2], dt[1]: second, minute, hour, weekday, month day, month
	
E.g.: an item in the list of ts that represents "2013-4-1 22:43:16"
[datetime.datetime(2013, 4, 1, 22, 43, 16), 16, 43, 22, 0, 1, 4]
"""
from dateutil import tz
import multiprocessing

CPU_CORE_NUM = int(multiprocessing.cpu_count() - 3)

# Getting weather data from locally stored bin files if True
USE_WEATHER_DATA_BIN = True

PROC_OUT_DIR = '/data/DDEA/VTT/proc_out/'
DATA_DIR = '/data/DDEA/VTT/Binfiles/'
WEATHER_DIR = '/data/DDEA/VTT/weather/VTT_'
FIG_DIR = '/data/DDEA/VTT/png_files/'
FL_EXT = '.bin'
LEN_FL_EXT = len(FL_EXT)

IS_USING_PARALLEL_OPT = True

TS_INDEX = 0
VAL_INDEX = 1
JSON_DIR = 'resources/json/'
META_DIR = 'resources/meta/'


# in seconds
MIN = 60; HOUR = 60*MIN; DAY = HOUR*24; MONTH = DAY*31

# State definition...
PEAK=1;LOW_PEAK=-1;NO_PEAK=0;

# mininum number of discrete values to be float type
MIN_NUM_VAL_FOR_FLOAT = 10

# Data type
INT_TYPE = 0
FLOAT_TYPE = 1

# Hour, Weekday, Day, Month
SEC_IDX=1;MIN_IDX=2;HR_IDX=3; WD_IDX=4; MD_IDX=5 ;MN_IDX=6;DT_IDX=0;

hourDict={0:'0am',1:'1am',2:'2am',3:'3am',4:'4am',5:'5am',6:'6am',7:'7am', 8:'8am',9:'9am',10:'10am',11:'11am',12:'12pm',13:'1pm',14:'2pm', 15:'3pm',16:'4pm',17:'5pm',18:'6pm',19:'7pm',20:'8pm',21:'9pm',22:'10pm',23:'11pm'}

monthDict={0:'Jan', 1:'Feb', 2:'Mar', 3:'Apr', 4:'May', 5:'Jun', 6:'Jul', 7:'Aug', 8:'Sep', 9:'Oct', 10:'Nov', 11:'Dec'}

weekDict={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thur', 4:'Fri', 5:'Sat', 6:'Sun'}
stateDict={-1:'Low Peak',0:'No Peak',1:'High Peak'}

Weekday = range(0,5)
Weekend = range(5,7)
Week = range(0,7)

# Hours parameters
am0=0;am1=1;am2=2;am3=3;am4=4;am5=5;am6=6;am7=7;am8=8;am9=9;am10=10;am11=11;
pm12=12;pm1=13;pm2=14;pm3=15;pm4=16;pm5=17;pm6=18;pm7=19;pm8=20;pm9=21;pm10=22;pm11=23;

# Week parameters
Mon=0;Tue=1;Wed=2;Thur=3;Fri=4;Sat=5;Sun=6

# Month parameters
Jan=0;Feb=1;Mar=2;Apr=3;May=4;Jun=5;Jul=6;Aug=7;Sep=8;Oct=9;Nov=10;Dec=11;
DayHours=range(24)
yearMonths=range(12)

# Define the period for analysis  - year, month, day,hour
# Note: The sample data in the currently downloaded files are from 1 Apr 2013 to
#       30 Nov 2013.
# This is the best data set
#ANS_START_T=dt.datetime(2013,7,8,0)
#ANS_END_T=dt.datetime(2013,7,15,0)
# UTC time of weather data
from_zone = tz.gettz('UTC')
# VTT local time
#to_zone = tz.gettz('Europe/Helsinki')
to_zone = tz.gettz('Asia/Seoul')