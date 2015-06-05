#!/usr/bin/python
# To force float point division
from __future__ import division

"""
Created on Mon Mar 24 19:24:11 2014

@author: NGO Quang Minh Khiem
@e-mail: khiem.ngo@adsc.com.sg

"""

import mytool as mt
from shared_constants import *
import json, pickle, time, ujson
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

r = robjects.r
utils = importr("utils")
bnlearn = importr("bnlearn")
rgraphviz = importr("Rgraphviz")

# this is important to seamlessly convert from pandas to R data frame
pandas2ri.activate()


BN_SENSOR = 'BN-SENSOR'
BN_TIME = 'BN-TIME'
BN_WEATHER = 'BN-WEATHER'
BN_ALL = 'BN-ALL' #'BN for Sensor-Time-Weather'


class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return json.JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}


def itrator(obj, lvl):


    if isinstance(obj, dict):

        dobj = obj
        for k, v in obj.iteritems():
            print '\t' * lvl, k, "(", type(obj), ")"

            """
            if isinstance(v, np.ndarray):
                l = v.tolist()
                dobj.update({k: l})
                itrator(l, lvl + 1)
                continue
            """

            itrator(v, lvl + 1)

    elif isinstance(obj, list):
        print '\t' * lvl, len(obj), "(", type(obj), ")"

        return
        dlist = obj
        for i, v in enumerate(obj):
            if isinstance(v, np.ndarray):
                dlist[i] = v.tolist()


    elif isinstance(obj, np.ndarray):
        print '\t' * lvl, obj.shape, "***************** (", type(obj), ") *****************"
        obj = obj.tolist()
        #print '\t' * lvl, obj.tolist()

    else:
        print '\t' * lvl, obj, "(", type(obj), ")"



def epoch_list(ts_list):
    ts = list()
    for i, d in enumerate(ts_list):
        ts.append(int(time.mktime(d.timetuple())))

    return ts


def ts_val(ts_list, val_list):

    ts = epoch_list(ts_list)

    val = val_list.tolist()
    """
    for i in enumerate(val_list):
        vval = list()
        for j, vv in enumerate(v):
            float()
        val.append(float(v))
    """
    return zip(ts, val)


def save_feature_matrix(bldg_key, ds_out):

    avg = ts_val(ds_out['avgdata_dict']['build_feature_matrix_out']['X_Time'],
                 ds_out['avgdata_dict']['build_feature_matrix_out']['X_Feature'])

    avg_name = ds_out['avgdata_dict']['build_feature_matrix_out']['X_names']

    diff = ts_val(ds_out['diffdata_dict']['build_diff_matrix_out']['Xdiff_Time'],
                  ds_out['diffdata_dict']['build_diff_matrix_out']['Xdiff_Mat'])

    diff_name = ds_out['diffdata_dict']['build_diff_matrix_out']['Xdiff_Names']

    feature = {'avg':avg, 'avg_name':avg_name, 'diff':diff, 'diff_name':diff_name}

    #ds_out_json = json.dumps(ds_out, cls=PythonObjectEncoder)
    ds_out_json = json.dumps(feature)
    with open(JSON_DIR + bldg_key.lower() + "_feature.json", mode="w") as f:
        f.write(ds_out_json)

def ts_npval(ts_list, val_list):
    ts = np.array(epoch_list(ts_list))
    val = np.array(val_list)
    return ts, val


def save_processed_json(sensor_names_hash, ds_out):
    diff_name_list = ds_out['diffdata_dict']['build_diff_matrix_out'].Xdiff_Names

    avg_ts, avg_val = ts_npval(ds_out['avgdata_dict']['build_feature_matrix_out'].X_Time,
                                ds_out['avgdata_dict']['build_feature_matrix_out'].X_Feature)

    diff_ts, diff_val = ts_npval(ds_out['diffdata_dict']['build_diff_matrix_out'].Xdiff_Time,
                                ds_out['diffdata_dict']['build_diff_matrix_out'].Xdiff_Mat)

    ts_indices = list()
    ts_list = None
    if avg_ts.shape[0] < diff_ts.shape[0]:
        ts_indices = [np.nonzero(diff_ts == ts)[0] for ts in avg_ts]
        ts_list = avg_ts.tolist()
    else:
        ts_indices = [np.nonzero(avg_ts == ts)[0] for ts in diff_ts]
        ts_list = diff_ts.tolist()

    for i, name in enumerate(diff_name_list):

        avg = avg_val[:, i].tolist()
        if len(ts_indices) < avg_val[:, i].shape[0]:
            avg = avg_val[:, i][ts_indices][:, 0].tolist()

        diff = diff_val[:, i].tolist()
        if len(ts_indices) < diff_val[:, i].shape[0]:
            diff = diff_val[:, i][ts_indices][:, 0].tolist()

        json_out = list()
        for t_idx in xrange(0, len(ts_indices)):
            json_out.append([ts_list[t_idx], avg[t_idx], diff[t_idx]])

        uid = sensor_names_hash[name]
        with open(JSON_DIR + "preproc-" + uid + ".json", 'w') as f:
            f.write(json.dumps(json_out))


def save_avg_data_summary_json(bldg_key, sensor_names_hash, avg_out):

    X_Sensor_STATE = avg_out['avgdata_state_mat']
    #X_Weather_STATE = avg_out['avgdata_weather_mat']

    avgdata_exemplar = dict()
    for ae, cl in avg_out['avgdata_exemplar'].iteritems():
        kuid = sensor_names_hash[ae]
        clist = list()
        for cn in cl:
            vuid = sensor_names_hash[cn]
            clist.append(vuid)
        avgdata_exemplar.update({kuid: clist})

    #X_Sensor_NAMES = avg_out['sensor_names']
    #X_Weather_NAMES = avg_out['weather_names']

    X_Sensor_NAMES = list()
    for an in avg_out['sensor_names']:
        uid = sensor_names_hash[an]
        X_Sensor_NAMES.append(uid)

    num_cols = X_Sensor_STATE.shape[1]

    x_states = list()
    for c in xrange(0, num_cols):
        x_states.append(X_Sensor_STATE[:, c].tolist())

    x_sensor_avg = {"sensor-names": X_Sensor_NAMES
                    ,"sensor-exemplar": avgdata_exemplar
                    ,"sensor-state": x_states}

    with open(JSON_DIR + bldg_key.lower() + "_sensor_feature_avg.json", 'w') as f:
        f.write(json.dumps(x_sensor_avg))


def save_diff_data_summary_json(bldg_key, sensor_names_hash, diff_out):

    XDIFF_Sensor_STATE = diff_out['diffdata_state_mat']
    #XDIFF_Weather_STATE = diff_out['diffdata_weather_mat']
    #XDIFF_Time_STATE = diff_out['diffdata_time_mat']

    #Xdiff_Time = diff_out['diff_time_slot']
    #diffdata_zvar = diff_out['diffdata_zvar']

    diffdata_exemplar = dict()
    for de, cl in diff_out['diffdata_exemplar'].iteritems():
        kuid = sensor_names_hash[de]
        clist = list()
        for cn in cl:
            vuid = sensor_names_hash[cn]
            clist.append(vuid)
        diffdata_exemplar.update({kuid: clist})

    XDIFF_Sensor_NAMES = list()
    for dn in diff_out['sensor_names']:
        uid = sensor_names_hash[dn]
        XDIFF_Sensor_NAMES.append(uid)

    #X_Weather_NAMES = diff_out['weather_names']
    #X_Time_NAMES = diff_out['time_names']

    num_cols = XDIFF_Sensor_STATE.shape[1]

    x_states = list()
    for c in xrange(0, num_cols):
        x_states.append(XDIFF_Sensor_STATE[:, c].tolist())

    x_sensor_diff = {"sensor-names": XDIFF_Sensor_NAMES
                    ,"sensor-exemplar": diffdata_exemplar
                    ,"sensor-state": x_states}

    with open(JSON_DIR + bldg_key.lower() + "_sensor_feature_diff.json", 'w') as f:
        f.write(json.dumps(x_sensor_diff))



def bldg_itrator(obj, lvl):

    if isinstance(obj, dict):
        dobj = obj
        for k, v in obj.iteritems():
            print '\t' * lvl, k, "(", type(obj), ")"
            bldg_itrator(v, lvl + 1)

    elif isinstance(obj, list):
        print '\t' * lvl, len(obj), "(", type(obj), ")"

    elif isinstance(obj, np.ndarray):
        print '\t' * lvl, obj.shape, "***************** (", type(obj), ") *****************"
        obj = obj.tolist()
        #print '\t' * lvl, obj.tolist()
    else:
        print '\t' * lvl, obj, "(", type(obj), ")"


def extract_edges(r_graph, cols_names):
    edges = list()
    np_amat = np.asarray(bnlearn.amat(r_graph))
    for ri in range(np_amat.shape[0]):
        for ci in range(np_amat.shape[1]):
            if np_amat[ri, ci] == 1:
                edges.append((cols_names[ri], cols_names[ci]))
    return edges

def save_bn_graph_json(bldg_key, sensor_names_hash, all_labels, all_edges):

    nodes_uids = [{"uid": sensor_names_hash[l], "group": 0} for l in all_labels if l in sensor_names_hash]

    nodes_index = dict()
    for i, n in enumerate(nodes_uids):
        nodes_index.update({n["uid"]:i})

    node_links = {'avg': [], 'diff': []}

    for sig_tag, sensor_edge in all_edges.iteritems():
        for sensor, edge_group in sensor_edge.iteritems():

            s_uid = sensor_names_hash[sensor]

            for e_type, edges in edge_group.iteritems():

                # we only take BN for Sensors, for now
                if e_type == BN_SENSOR:

                    links = list()
                    for e in edges:

                        v0_uid = sensor_names_hash[e[0]]
                        v0_idx = nodes_index[v0_uid]

                        v1_uid = sensor_names_hash[e[1]]
                        v1_idx = nodes_index[v1_uid]

                        #print (PREFIX_LINKS + sig_tag), s_uid, e_type, v0_idx, v1_idx,

                        l = {"source": v0_idx, "target": v1_idx, "value": 1}
                        links.append(l)

                    node_links[sig_tag].append({"sensor": s_uid,
                                                "link_group":{"graph_type": e_type,
                                                                "links": links}})


    with open(JSON_DIR + bldg_key.lower() + "_bn_graph.json", 'w') as f:
        f.write(json.dumps({"nodes":nodes_uids, "links": node_links}))


def conv_bn_graph_json(bldg_out):

    all_labels = list()
    all_edges = {'avg': {}, 'diff': {}}

    for sig_tag, anal_out in bldg_out.anal_out.iteritems():

        for bn_prob in anal_out:

            p_name = bn_prob['p_name']

            s_labels = bn_prob['s_labels']
            s_edges = extract_edges(bn_prob['s_hc'], s_labels)

            t_labels = bn_prob['t_labels']
            t_edges = extract_edges(bn_prob['t_hc'], t_labels)

            w_labels = bn_prob['w_labels']
            w_edges = extract_edges(bn_prob['w_hc'], w_labels)

            a_labels = bn_prob['all_labels']
            a_edges = extract_edges(bn_prob['all_hc'], a_labels)

            all_labels += s_labels + t_labels + w_labels + a_labels

            edges = {BN_SENSOR: s_edges,
                     BN_TIME: t_edges,
                     BN_WEATHER: w_edges,
                     BN_ALL: a_edges}

            all_edges[sig_tag].update({p_name: edges})

    all_labels = list(set(all_labels))

    return all_labels, all_edges


if __name__ == "__main__":

    import pre_bn_processing as pbp
    #print "-" * 80
    #ds_out = mt.loadObjectBinaryFast(PROC_OUT_DIR + 'ds_out.bin')
    #itrator(ds_out, 0)
    #save_feature_matrix(ds_out)

    print "~" * 80

    sensor_names_hash = mt.loadObjectBinaryFast(PROC_OUT_DIR + 'gw2_sensor_hash.bin')

    bldg_out = mt.loadObjectBinaryFast(PROC_OUT_DIR + 'gw2_bldg_out.bin')
    #import pdb;pdb.set_trace()

    #pbp.plotting_bldg_bn(bldg_out)
    all_labels, all_edges = conv_bn_graph_json(bldg_out)

    save_bn_graph_json('gw2', sensor_names_hash, all_labels, all_edges)

    #bldg_itrator(bldg_out.__dict__, 0)
    print "-" * 80
