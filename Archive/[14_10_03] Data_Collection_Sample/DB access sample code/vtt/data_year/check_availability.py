# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:35:35 2014

@author: nqmkhiem
"""

import mytool as mt
import numpy as np
from math import sqrt
import csv

folder_list = ['aff_2013April/','aff_2013May/','aff_2013June/',\
    'aff_2013July/','aff_2013August/','aff_2013September/',\
    'aff_2013October/','aff_2013November/','aff_2013December/',\
    'aff_2014January/','aff_2014February/',]
    
def saveCSV(obj,filename):
    np.savetxt(filename, obj, delimiter=',')
    
def writeCSV(fname, mat):
    with open(fname, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for ri in range(len(mat)):
            row = mat[ri]
            spamwriter.writerow(row)    

def load_uuid_list():
    uuid_list = []
    index_dict = {}
    with open('finland_ids.csv', 'r') as f:
        lines = f.readlines()
    f.close()

    for i,line in enumerate(lines):
        uuid = line.split(',')[0].strip()
        uuid_list.append(uuid)
        index_dict[uuid] = i

    return uuid_list, index_dict
    
def get_start_end(fname):

    s = ''
    e = ''
    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
        f.close()
        
        s = lines[0].split(',')[0].split(' ')[0]
        e = lines[-1].split(',')[0].split(' ')[0]
        
        return s,e
    except:
        return s,e
        

def load_clusters(path='./'):
    input_names = mt.loadObjectBinaryFast(path + 'input_names.tmp')
    zero_var_list = mt.loadObjectBinaryFast(path + 'zero_var_list.tmp')
    exemplars = mt.loadObjectBinaryFast(path + 'aff_exemplars.tmp')
    labels = mt.loadObjectBinaryFast(path + 'aff_labels.tmp')
    dist_mat = mt.loadObjectBinaryFast(path + 'DIST_MAT.tmp')
    cluster_errors = mt.loadObjectBinaryFast(path + 'aff_cluster_err.tmp')

    return zero_var_list, input_names, exemplars, labels, dist_mat, cluster_errors

def construct_label_row(index_dict, zero_var_list, input_names, exemplars, labels):
    sparsity_row = [0] * len(index_dict)
    #label_row = [0] * len(index_dict)
    label_row = [None] * len(index_dict)
    ## Mark sensors with None (or ~0) data
    for uuid in zero_var_list:
        uuid_index = index_dict[uuid]
        label_row[uuid_index] = '-'

    ## Label sensors with data available
    for i in range(len(labels)):
        cluster_label = labels[i]
        uuid = input_names[i]
        uuid_index = index_dict[uuid]
        label_row[uuid_index] = cluster_label

    ## Mark sensors which are exemplars
    for i in exemplars:
        uuid = input_names[i]
        uuid_index = index_dict[uuid]
        label_row[uuid_index] = str(label_row[uuid_index]) + "*"
        sparsity_row[uuid_index] = 1

    return label_row, sparsity_row

def construct_full_dist_mat(index_dict, dist_mat, input_names):
    num_sensors = len(index_dict)
    num_available_sensors = len(input_names)
    # default entry is 2.0 (max distance)
    full_dist_mat = 2.0 + np.zeros((num_sensors, num_sensors))
    # entries on the diagonal
    for i in range(num_sensors):
        full_dist_mat[i,i] = 0.0

    # other entries in dist_mat
    for ri in range(num_available_sensors):
        for ci in range(ri):
            dist = dist_mat[ri,ci]
            uuid_ri = input_names[ri]
            uuid_ci = input_names[ci]
            global_uuid_ri_index = index_dict[uuid_ri]
            global_uuid_ci_index = index_dict[uuid_ci]

            full_dist_mat[global_uuid_ri_index,global_uuid_ci_index] = \
            full_dist_mat[global_uuid_ci_index,global_uuid_ri_index] = dist

    return full_dist_mat

def compute_norm_distance(mat1,mat2):
    a = np.mat(mat1 - mat2)

    frobenius_norm_dist = sqrt(np.matrix.trace(a.H * a))

    return frobenius_norm_dist

def compute_frobenius_mat(mat_list):
    n = len(mat_list)
    dist_mat = np.zeros((n,n))

    for ri in range(n):
        for ci in range(ri):
            print 'Compute frobenius distance [%d,%d]'% (ri,ci)
            dist_mat[ri,ci] = dist_mat[ci,ri] = \
            compute_norm_distance(mat_list[ri],mat_list[ci])

    return dist_mat

def load_mat_list():
    folder_list = ['aff_2013April/','aff_2013May/','aff_2013June/',\
    'aff_2013July/','aff_2013August/','aff_2013September/',\
    'aff_2013October/','aff_2013November/','aff_2013December/',\
    'aff_2014January/','aff_2014February/',]

    mat_list = []
    for path in folder_list:
        dist_mat_path = path + 'full_dist_mat.bin'
        print 'Loading ' + dist_mat_path + '...'
        mat_list.append(mt.loadObjectBinaryFast(dist_mat_path))

    return mat_list

def construct_label_mat():
    folder_list = ['aff_2013April/','aff_2013May/','aff_2013June/',\
    'aff_2013July/','aff_2013August/','aff_2013September/',\
    'aff_2013October/','aff_2013November/','aff_2013December/',\
    'aff_2014January/','aff_2014February/',]

    label_table = []
    sparsity_table = []
    for path in folder_list:
        label_path = path + 'label_row.bin'
        sparsity_path = path + 'sparsity_row.bin'
        print 'Loading ' + label_path + '...'

        label_row = mt.loadObjectBinaryFast(label_path)
        sparsity_row = mt.loadObjectBinaryFast(sparsity_path)

        label_table.append(label_row)
        sparsity_table.append(sparsity_row)
                
        
    label_table = np.array(label_table)
    sparsity_table = np.array(sparsity_table)
    
    
    #saveCSV(label_table,'label_table.csv')
    #saveCSV(sparsity_table,'sparsity_table.csv')
    #np.savetxt('label_table.csv',label_table)
    #np.savetxt('sparsity_table.csv',sparsity_table)
    mt.saveObjectBinary(label_table,'label_table.bin')
    mt.saveObjectBinary(sparsity_table,'sparsity_table.bin')

    return label_table, sparsity_table

def print_stats(index_dict, zero_var_list, input_names, exemplars, labels, cluster_errors):
    ## number of zero variance sensors
    ## number of clusters
    ## average number of sensors per cluster
    ## average weighted distance per cluster

    num_labeled_sensors = len(input_names)
    num_clusters = int(labels.max() + 1)
    avg_sensors_per_cluster = float(num_labeled_sensors) / num_clusters
    avg_error = cluster_errors[1]

    print 'Number of zero-variance sensors: ' + str(len(zero_var_list))
    print 'Number of labeled sensors: ' + str(num_labeled_sensors)
    print 'Number of clusters: ' + str(num_clusters)
    print 'Average number of sensors per cluster: ' + str(avg_sensors_per_cluster)
    print 'Average weighted distance per cluster: ' + str(avg_error)

if __name__ == '__main__':
        
    uuid_list, index_dict = load_uuid_list()
    """
    uuid = 'VAK2.TK3_SC08_A'
    s,e = get_start_end('VTT_year/data/' + uuid + '.csv')
    print s,e
    """
    l = []
    for uuid in uuid_list:
        s,e = get_start_end('Csvfiles_extra/' + uuid + '.csv')
        #print s,e
        l.append([uuid,s,e])
    
    writeCSV('availability2.csv',l)

    """
    uuid_list, index_dict = load_uuid_list()
    zero_var_list, input_names, exemplars, labels, dist_mat, cluster_errors = load_clusters()

    if False:
        mat_list = load_mat_list()
        frobenius_mat = compute_frobenius_mat(mat_list)
        mt.saveObjectBinaryFast(frobenius_mat, 'frobenius_mat.bin')
        
    if False:
        construct_label_mat()


    label_row, sparsity_row = construct_label_row(index_dict, zero_var_list, input_names, exemplars, labels)
    print_stats(index_dict, zero_var_list, input_names, exemplars, labels, cluster_errors)
    full_dist_mat = construct_full_dist_mat(index_dict, dist_mat, input_names)

    mt.saveObjectBinaryFast(index_dict, 'index_dict.bin')
    mt.saveObjectBinaryFast(uuid_list, 'uuid_list.bin')
    mt.saveObjectBinaryFast(label_row, 'label_row.bin')
    mt.saveObjectBinaryFast(sparsity_row, 'sparsity_row.bin')
    mt.saveObjectBinaryFast(full_dist_mat, 'full_dist_mat.bin')
    """
    #dist = compute_norm_distance(full_dist_mat, full_dist_mat)
