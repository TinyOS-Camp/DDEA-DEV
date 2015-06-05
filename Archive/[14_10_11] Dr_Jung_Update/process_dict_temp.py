# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:47:16 2014

@author: nqmkhiem
"""

import os
import mytool as mt
import numpy as np
import pandas as pd

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)
               
def load_sensor_list(filename):
    with open(filename,'rb') as f:
        lines = f.readlines()
        
    ret = [line.strip()[:-4] for line in lines]
    
    idx_dict = {}
    for (idx,uuid) in enumerate(ret):
        idx_dict.update({uuid:idx})
    return ret,idx_dict

def construct_idx_dict(ret):
    
    idx_dict = {}
    for (idx,uuid) in enumerate(ret):
        idx_dict.update({uuid:idx})
    return idx_dict
    
def load_dict(prefix):
    
    adict = mt.loadObjectBinaryFast(prefix+'_results/avgdata_dict.bin')
    #ddict = mt.loadObjectBinaryFast(prefix+'_results/diffdata_dict.bin')
    
    #adict = mt.loadObjectBinaryFast(prefix+'_results/pack/avgdata_dict.bin')
    #ddict = mt.loadObjectBinaryFast(prefix+'_results/pack/diffdata_dict.bin')
    
    print prefix,':'
    print 'avg: len of exemplars: ',str(len(adict['avgdata_exemplar']))    
    print 'avg: len of zero var: ',str(len(adict['avgdata_zvar']))
    
    #print 'diff: len of exemplars: ',str(len(ddict['diffdata_exemplar']))    
    #print 'diff: len of zero var: ',str(len(ddict['diffdata_zvar']))
    print '------------------------------'
    ddict = dict()
    return adict,ddict

def generate_csv(dict_name,csv_name = None,path='./'):
    d = mt.loadObjectBinaryFast(path+dict_name)
    ad = d['avgdata_dict']
    data_dict = d['data_dict']
    sensor_names = data_dict['sensor_list']
    idx_dict = construct_idx_dict(sensor_names)
    
    data_mat = generate_table(ad,sensor_names,idx_dict)
    name = csv_name
    if name is None:
        #name = dict_name.split('_')[0]
        name = dict_name[:-4]
    #np.savetxt('%s.csv'%name,data_mat,delimiter=',',fmt='%s')
    df = pd.DataFrame(data_mat,columns=['SENSOR ID','ZERO-VARIANCE','EXEMPLAR','CLUSTER LABEL'])
    df.to_csv('%s.csv'%name,index=False)
    
    print dict_name,':'
    print 'num of sensors: ', str(len(sensor_names))
    print 'avg: len of zero var: ',str(len(ad['avgdata_zvar']))
    print 'avg: len of exemplars: ',str(len(ad['avgdata_exemplar']))    
    
    
    
def generate_table(d,sensor_names,idx_dict):
    num_sensors = len(sensor_names)
    zvar = [False] * num_sensors
    exemplars = [None] * num_sensors
    sensor_labels = [None] * num_sensors
    
    cluster_exemplars = {}
    for (idx,k) in enumerate(d['avgdata_exemplar'].keys()):
        cluster_exemplars.update({k:idx})
        exemplar_uuid = k
        exemplar_idx = idx_dict[exemplar_uuid]
        exemplars[exemplar_idx] = idx
        sensor_labels[exemplar_idx] = idx
        for uuid in d['avgdata_exemplar'][k]:
            uuid_idx = idx_dict[uuid]
            sensor_labels[uuid_idx] = idx
            
    for uuid in d['avgdata_zvar']:
        uuid_idx = idx_dict[uuid]
        zvar[uuid_idx] = True
    data_mat = np.vstack([sensor_names,zvar,exemplars,sensor_labels])
    data_mat = data_mat.T
    return data_mat
    
def generate_table_irr(d,sensor_names,idx_dict):
    num_sensors = len(sensor_names)
    zvar = [False] * num_sensors
    exemplars = [None] * num_sensors
    sensor_labels = [None] * num_sensors
    
    cluster_exemplars = {}
    for (idx,k) in enumerate(d['diffdata_exemplar'].keys()):
        cluster_exemplars.update({k:idx})
        exemplar_uuid = k
        exemplar_idx = idx_dict[exemplar_uuid]
        exemplars[exemplar_idx] = idx
        sensor_labels[exemplar_idx] = idx
        for uuid in d['diffdata_exemplar'][k]:
            uuid_idx = idx_dict[uuid]
            sensor_labels[uuid_idx] = idx
            
    for uuid in d['diffdata_zvar']:
        uuid_idx = idx_dict[uuid]
        zvar[uuid_idx] = True
    data_mat = np.vstack([sensor_names,zvar,exemplars,sensor_labels])
    data_mat = data_mat.T
    return data_mat
    
if __name__ == "__main__":
    """
    sensor_names,idx_dict = load_sensor_list('vak1_list.dat')
    adict,ddict = load_dict('VAK1')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('vak1_regular_pack.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('vak1_irregular_pack.csv',data_mat_irr,delimiter=',',fmt='%s')
    
    sensor_names,idx_dict = load_sensor_list('vak2_list.dat')
    adict,ddict = load_dict('VAK2')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('vak2_regular_pack.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('vak2_irregular_pack.csv',data_mat_irr,delimiter=',',fmt='%s')

    
    sensor_names,idx_dict = load_sensor_list('gw1_list.dat')
    adict,ddict = load_dict('GW1')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('gw1_regular_pack.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('gw1_irregular_pack.csv',data_mat_irr,delimiter=',',fmt='%s')
    """
    """
    sensor_names,idx_dict = load_sensor_list('gw2_list.dat')
    adict,ddict = load_dict('GW2')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('gw2_regular_pack.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('gw2_irregular_pack.csv',data_mat_irr,delimiter=',',fmt='%s')
    """
    
    """
    sensor_names,idx_dict = load_sensor_list('gsbc_results/gsbc_list.dat')
    adict,ddict = load_dict('gsbc')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    #data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('gsbc_regular_pack.csv',data_mat,delimiter=',',fmt='%s')
    #np.savetxt('gw2_irregular_pack.csv',data_mat_irr,delimiter=',',fmt='%s')
    """
    """
    l = ['ACH1_out.bin',  'ACT8_out.bin',  'ATW1_out.bin',  'ATW2_out.bin',  'BDV1_out.bin',  'DPT3_out.bin',  'EDT2_out.bin',\
    'GRH_out.bin',  'KLB1_out.bin',  'KLS1_out.bin',  'MAT_out.bin',  'MPC_out.bin',  'UEZ1_out.bin',  'WKR_out.bin']
    
    l = ['EDT2_out.bin','GRH_out.bin','KLB1_out.bin','KLS1_out.bin', 'MAT_out.bin', 'MPC_out.bin','UEZ1_out.bin', 'WKR_out.bin']
    """
    l = ['conference_bldg_out.bin', 'main_bldg_10_out.bin', 'main_bldg_12_out.bin', 'main_bldg_14_out.bin',\
    'main_bldg_16_out.bin', 'main_bldg_2_out.bin', 'main_bldg_4_out.bin','main_bldg_6_out.bin',\
    'main_bldg_8_out.bin', 'elec_machine_room_bldg_out.bin','main_bldg_11_out.bin','main_bldg_13_out.bin',\
    'main_bldg_15_out.bin','main_bldg_1_out.bin','main_bldg_3_out.bin', 'main_bldg_5_out.bin',\
    'main_bldg_7_out.bin', 'main_bldg_9_out.bin']
    for i in l:
	#print i
        generate_csv(i,path='./GSBC/')
    """
    sensor_names,idx_dict = load_sensor_list('vak1_list.dat')
    adict,ddict = load_dict('VAK1')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('vak1_regular.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('vak1_irregular.csv',data_mat_irr,delimiter=',',fmt='%s')
    
    
    sensor_names,idx_dict = load_sensor_list('vak2_list.dat')
    adict,ddict = load_dict('VAK2')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('vak2_regular.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('vak2_irregular.csv',data_mat_irr,delimiter=',',fmt='%s')
    
    
    sensor_names,idx_dict = load_sensor_list('gw1_list.dat')
    adict,ddict = load_dict('GW1')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('gw1_regular.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('gw1_irregular.csv',data_mat_irr,delimiter=',',fmt='%s')
    
    sensor_names,idx_dict = load_sensor_list('gw2_list.dat')
    adict,ddict = load_dict('GW2')
    data_mat = generate_table(adict,sensor_names,idx_dict)
    data_mat_irr = generate_table_irr(ddict,sensor_names,idx_dict)
    np.savetxt('gw2_regular.csv',data_mat,delimiter=',',fmt='%s')
    np.savetxt('gw2_irregular.csv',data_mat_irr,delimiter=',',fmt='%s')
    """
    
    
    """
    load_dict('VAK2')
    load_dict('GW1')
    load_dict('GW2')
    """
