# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 19:28:19 2014

@author: nqmkhiem
"""

import os
import pickle
import numpy as np
from multiprocessing import Pool
import csv
from datetime import datetime

UUID_FILE = 'finland_ids.csv'
FOLDER1 = 'Csvfiles_new2/'
FOLDER2 = 'Csvfiles_extra/'
FOLDER_NEW = 'Csvfiles_new3/'


def writeCSV(fname, mat):
    with open(fname, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for ri in range(len(mat)):
            row = mat[ri]
            spamwriter.writerow(row)


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

def merge_csv(filename):
    #print filename
    filename_new = FOLDER_NEW + filename + '.csv'
    if os.path.isfile(filename_new):
        return
    else:
	print 'new: ' + filename
    try:
        filename1 = FOLDER1 + filename + '.csv'
        with open(filename1, 'rb') as f1:
            #f1.next()
            csv1 = f1.readlines()

        filename2 = FOLDER2 + filename + '.csv'
        with open(filename2, 'rb') as f2:
            #f2.next()
            csv2 = f2.readlines()


        ind = -1
        if len(csv1) > 0:
            last_date = datetime.strptime(csv1[-1].split(',')[0].strip(), '%Y-%m-%d %H:%M:%S')
            for i,line in enumerate(csv2):
                curr_date = datetime.strptime(line.split(',')[0].strip(), '%Y-%m-%d %H:%M:%S')
                if curr_date > last_date:
                    ind = i
                    break

        print last_date, ind, csv2[ind]

        if len(csv1) > 0:
            if ind < 0:
                csv_new = csv1
            else:
                csv_new = csv1 + csv2[ind:]

        else:
            csv_new = csv2

        filename_new = FOLDER_NEW + filename + '.csv'
        with open(filename_new, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for row in csv_new:
                spamwriter.writerow(row.strip().split(','))

    except:
	print 'error: ' + filename
        return



#    print np.shape(obj1['ts']), np.shape(obj2['ts'])
#    print np.shape(obj1['value']), np.shape(obj2['value'])
    return csv_new

#filenames = glob.glob("Csvfiles_extra/*.csv")
filenames = load_uuid_list()
#for i,filename in enumerate(filenames):
#    print i, filename
#    merge_csv(filename)
filenames.sort()
p = Pool(12)
p.map(merge_csv, filenames)
p.close()
p.join()



#merge_csv('GW1.24H_LASKIN_T')

#p = Pool(12)
#p.map(merge, uuid_list)
#p.close()
#p.join()
#filename = 'VAK1.TK4_TF_SM_EP_KM'
#obj1 = merge(filename)

