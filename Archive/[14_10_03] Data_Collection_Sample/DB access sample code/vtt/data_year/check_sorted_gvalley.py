# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 19:28:19 2014

@author: nqmkhiem
"""

import os
##import sys
import pickle
import numpy as np
from multiprocessing import Pool
import mytool as mt

UUID_FILE = 'finland_ids.csv'
FOLDER1 = 'Binfiles_new2/'
FOLDER2 = 'Binfiles_extra/'
FOLDER_NEW = 'Binfiles_new3/'

def saveObjectBinary(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadObjectBinary(filename):
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return obj

def load_uuid_list():
    uuid_list = []
    uuid_filepath = os.path.join(UUID_FILE)
    temp_uuid_list = open(uuid_filepath).readlines()

    for uuid in temp_uuid_list:
        uuid = uuid.strip().split(',')[0].strip()
        if uuid == "":
            continue
        uuid_list.append(uuid)
        
    return uuid_list
    
def merge_gvalley(uuid):
    PATH1 = '/data/bigdata/gvalley/MDMS_TST_CKM_RTIME_INFO_1309/'
    PATH2 = '/data/bigdata/gvalley/MDMS_TST_CKM_RTIME_INFO_1310/'
    PATH3 = '/data/bigdata/gvalley/MDMS_TST_CKM_RTIME_INFO_1311/'
    PATH4 = '/data/bigdata/gvalley/MDMS_TST_CKM_RTIME_INFO_MAIN/'
    fn1 = PATH1 + uuid + '.bin'
    fn2 = PATH2 + uuid + '.bin'
    fn3 = PATH3 + uuid + '.bin'
    fn4 = PATH4 + uuid + '.bin'
    
    ts_list = []
    val_list = []
    if os.path.isfile(fn1):
        obj1 = mt.loadObjectBinaryFast(fn1)
        ts_list = list(obj1['ts'])
        val_list = list(obj1['value'])
        
    if os.path.isfile(fn2):
        obj2 = mt.loadObjectBinaryFast(fn2)
        ts_list = ts_list + list(obj2['ts'])
        val_list = val_list + list(obj2['value'])        
        
    if os.path.isfile(fn3):
        obj3 = mt.loadObjectBinaryFast(fn3)
        ts_list = ts_list + list(obj3['ts'])
        val_list = val_list + list(obj3['value'])
    
    data = {"ts":np.array(ts_list), "value":np.array(val_list)}
    mt.saveObjectBinaryFast(data,fn4)
    print 'Done for ' + uuid
        
def merge2(filename1,filename2,filename_new):
    print filename_new
    #filename_new = FOLDER_NEW + filename + '.bin'
    if os.path.isfile(filename_new):
        return

    try:
        #filename1 = FOLDER1 + filename + '.bin'
        obj1 = mt.loadObjectBinaryFast(filename1)
        
        #filename2 = FOLDER2 + filename + '.bin'
        obj2 = mt.loadObjectBinaryFast(filename2)
    except:
	print 'error file: ' + filename_new
        return
        
    ## concatenate two objects
    obj1['ts'] = np.vstack((obj1['ts'],obj2['ts']))
    obj1['value'] = np.hstack((obj1['value'],obj2['value']))
    
    
    mt.saveObjectBinaryFast(obj1, filename_new)
    
#    print np.shape(obj1['ts']), np.shape(obj2['ts'])
#    print np.shape(obj1['value']), np.shape(obj2['value'])
    return obj1
   
def merge(filename):
    print filename
    filename_new = FOLDER_NEW + filename + '.bin'
    if os.path.isfile(filename_new):
        return

    try:
        filename1 = FOLDER1 + filename + '.bin'
        obj1 = mt.loadObjectBinaryFast(filename1)
        
        filename2 = FOLDER2 + filename + '.bin'
        obj2 = mt.loadObjectBinaryFast(filename2)
    except:
	print 'error file: ' + filename
        return
        
    ## concatenate two objects
    obj1['ts'] = np.vstack((obj1['ts'],obj2['ts']))
    obj1['value'] = np.hstack((obj1['value'],obj2['value']))
    
    
    mt.saveObjectBinaryFast(obj1, filename_new)
    
#    print np.shape(obj1['ts']), np.shape(obj2['ts'])
#    print np.shape(obj1['value']), np.shape(obj2['value'])
    return obj1

def load_gvalley_list():
    with open('gvalley_uuids.bin','r') as f:
        lines = f.readlines()
        
    uuid_list = [i.strip() for i in lines]
    
    return uuid_list

def check_sorted(uuid):

    PATH = '/data/bigdata/gvalley/MDMS_TST_CKM_RTIME_INFO_MAIN/'
    obj = mt.loadObjectBinaryFast(PATH + uuid + '.bin')
    ts = obj['ts']
    is_sorted = all(ts[i][0] <= ts[i+1][0] for i in range(len(ts)-1))
    print uuid + ' ' + str(is_sorted)
    #return is_sorted
    
if __name__ == '__main__': 
    uuid_list = mt.loadObjectBinaryFast('gvalley_uuids.bin')
     #print uuid_list[0]
     #merge_gvalley(uuid_list[0])

    p = Pool(12)
    #p.map(merge_gvalley,uuid_list)
    p.map(check_sorted,uuid_list)
    p.close()
    p.join()
     
	#filename = 'VAK1.TK4_TF_SM_EP_KM'
	#obj1 = merge(filename)
