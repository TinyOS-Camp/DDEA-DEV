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


if __name__ == '__main__': 
	uuid_list = load_uuid_list()
	"""
	for i in range(0,len(uuid_list)):
	    uuid = uuid_list[i]
	    merge(uuid)
	    print str(i) + ": " + uuid
	""" 
	p = Pool(12)
	p.map(merge, uuid_list)
	p.close()
	p.join()
	#filename = 'VAK1.TK4_TF_SM_EP_KM'
	#obj1 = merge(filename)

