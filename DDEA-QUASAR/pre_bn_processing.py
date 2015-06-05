# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:28:13 2014

@author: deokwooj
"""
# To force float point division
from __future__ import division
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from sklearn import mixture
from sklearn.cluster import KMeans
import time
from data_tools import *
from shared_constants import *
import pprint
import lib_bnlearn as rbn

from multiprocessing import Pool

from data_repr import *

def compute_joint_prob(data_mat, state_idx_set, state_val_set):
    num_samples = data_mat.shape[0]
    num_states = data_mat.shape[1]

    if len(state_idx_set) != len(state_val_set):
        raise NameError('the length of state_set and state_val must be same')

    joint_idx = set(range(num_samples))
    for k, state_idx in enumerate(state_idx_set):
        samples = data_mat[:, state_idx]
        sub_joint_idx = set([])
        for state_val in state_val_set[k]:
            sub_joint_idx = sub_joint_idx | set(np.nonzero(samples == state_val)[0])
        joint_idx = joint_idx & sub_joint_idx
    joint_prob = len(joint_idx)/num_samples

    if num_samples == 0:
        return 0
    else:
        return joint_prob

def compute_cond_prob(data_mat, state_idx_set, state_val_set, cond_idx_set, cond_val_set):
    joint_state_idx_set = state_idx_set + cond_idx_set
    joint_state_val_set = state_val_set + cond_val_set
    all_prob = compute_joint_prob(data_mat, joint_state_idx_set, joint_state_val_set)
    partial_prob = compute_joint_prob(data_mat, cond_idx_set, cond_val_set)
    if partial_prob == 0:
        return 0
    else:
        return all_prob/partial_prob

def compute_cause_likelihood(data_mat, cause_idx_set, effect_idx_set, effect_val_set):
    """
    find f_A*(B)=P(A=A*|B)
    generate a set of all possible states
    :param data_mat:
    :param cause_idx_set:
    :param effect_idx_set:
    :param effect_val_set:
    :return: state_set, likelihood_set
    """
    state_set = []

    for k, idx in enumerate(cause_idx_set):
        #print idx, ':', list(set(data_mat[:,idx]))
        #set(list(data_mat[idx,:]))
        if k == 0:
            state_set = list(set(data_mat[:, idx]))
        else:
            state_set = pair_in_idx(state_set, list(set(data_mat[:, idx])))

    likelihood_set = []
    for state_val in state_set:
        if isinstance(state_val, list):
            input_val_set = [[val] for val in state_val]
        else:
            input_val_set = [[state_val]]
        prob_temp = compute_cond_prob(data_mat, effect_idx_set, effect_val_set, cause_idx_set, input_val_set)
        likelihood_set.append(prob_temp)

    return state_set, likelihood_set


def find_cond_lh_set(data_state_mat, cause_idx_set, effect_idx, obs_state):
    """
    find conditional Likelihood set

    :param data_state_mat:
    :param cause_idx_set:
    :param effect_idx:
    :param obs_state:
    :return: optstate_set, optprob_set
    """
    optprob_set = np.zeros(len(cause_idx_set))
    optstate_set = np.zeros(len(cause_idx_set))

    for i, cause_idx in enumerate(cause_idx_set):
        # Compute liklihoood of GW2 avg data state map among sensors
        avg_state_temp, avg_prob_temp = compute_cause_likelihood(data_state_mat, [cause_idx], [[effect_idx]], [[obs_state]])

        # masking its own effect
        if cause_idx == effect_idx:
            # and its state
            max_opt_state = np.nan
            # and its probability
            max_opt_prob = -np.inf
        else:
            # find sensor index giving the maximum likelihood
            max_idx = np.argmax(avg_prob_temp)
            # and its state
            max_opt_state = avg_state_temp[max_idx]
            # and its probability
            max_opt_prob = avg_prob_temp[max_idx]
        optprob_set[i] = max_opt_prob
        optstate_set[i] = max_opt_state
    return optstate_set, optprob_set


def compute_log_ll(label_in,obs_in):
    log_ll_sum = 0
    for i in range(label_in.max()+1):
        idx = np.nonzero(label_in == i)[0]
        val_set = obs_in[idx]
        log_val = stats.norm.logpdf(val_set, loc=np.mean(val_set), scale=np.std(val_set))
        log_ll_sum = log_ll_sum + sum(log_val[log_val != -np.inf])
    return log_ll_sum


def cluster_state_retrieval(tup):
    obs = tup[0]
    num_clusters = tup[1]
    est_method = tup[2]
    if est_method == 'kmean':
        kmean = KMeans(n_clusters=num_clusters).fit(obs)
        model = kmean
        score= compute_log_ll(kmean.labels_,obs)
        #score=-1*np.log(-1*np.sum(kmean.score(obs)))
    elif est_method == 'gmm':
        gmm = mixture.GMM(n_components=num_clusters).fit(obs)
        model = gmm
        score = np.sum(gmm.score(obs))
    return (num_clusters - 1, [model , score])


def state_retrieval(obs, max_num_cluster=6, off_set=0, est_method='kmean', PARALLEL = False):
    print '-' * 40
    print 'Retrieving discrete states from data using', est_method, 'model...'
    print '-' * 40
    print 'try ', max_num_cluster, ' clusters..... '
    score = np.zeros(max_num_cluster)
    model_set = []

    if not PARALLEL:
        for num_cluster in range(max_num_cluster):
            #print 'Try ',num_cluster+1, ' clusters '
            #print '-----------------------------------'
            if est_method == 'kmean':
                kmean = KMeans(n_clusters=num_cluster+1).fit(obs)
                model_set.append(kmean)
                #score[num_cluster]=-1*np.log(-1*np.sum(kmean.score(obs)))
                #score[num_cluster]=kmean.score(obs)
                #score[num_cluster]=kmean.score(obs)-.5*(num_cluster+1)*1*log10(len(obs))
                #log_ll_val=compute_log_ll(kmean.labels_,obs)
                score[num_cluster] = compute_log_ll(kmean.labels_, obs)

            elif est_method == 'gmm':
                gmm = mixture.GMM(n_components=num_cluster+1).fit(obs)
                model_set.append(gmm)
                score[num_cluster] = np.sum(gmm.score(obs))

            else:
                raise NameError('not supported est_method')
    else:
        print 'Parallel enabled...'
        model_set = [0] * max_num_cluster
        score = [0] * max_num_cluster
        p = Pool(max_num_cluster)
        params = [(obs,i+1,est_method) for i in range(max_num_cluster)]
        model_dict = dict(p.map(cluster_state_retrieval,params))
        for k, v in model_dict.iteritems():
            model_set[k] = v[0]
            score[k] = v[1]
        p.close()
        p.join()

    score_err_sum = np.zeros(max_num_cluster)
    print 'Finding knee points of log likelihood...'

    for i in range(max_num_cluster):
        a_0 = score[:(i)]
        if len(a_0) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(a_0)),a_0)
            sqr_sum_err0 = sum(((slope*np.arange(len(a_0)) + intercept)-a_0)**2)
        else:
            sqr_sum_err0=0
        a_1 = score[(i):]
        if len(a_1) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(a_1)),a_1)
            sqr_sum_err1 = sum(((slope*np.arange(len(a_1)) + intercept)-a_1)**2)
        else:
            sqr_sum_err1 = 0
        score_err_sum[i] = sqr_sum_err0 + sqr_sum_err1
    # Optimum number of clusters.
    min_idx = np.argmin(score_err_sum)
    opt_num_cluster = min_idx+1
    print 'opt_num_cluster: ', opt_num_cluster

    if est_method == 'kmean':
        label = model_set[min_idx].labels_
    elif est_method == 'gmm':
        label = model_set[min_idx].predict(obs)
    else:
        raise NameError('not supported est_method')
    return label, opt_num_cluster, model_set[min_idx], score, score_err_sum


def weather_convert(wdata_mat, wdata_name, Conditions_dict,Events_dict):
    """
    New dictionary by state classification of weather data
    """
    weather_dict = dict()
    # coolect index of weather data point in previous data

    try:
        temp_idx = wdata_name.index('TemperatureC')
    except:
        temp_idx = list()

    try:
        dewp_idx = wdata_name.index('Dew_PointC')
    except:
        dewp_idx = list()

    try:
        humd_idx = wdata_name.index('Humidity')
    except:
        humd_idx = list()

    try:
        evnt_idx = wdata_name.index('Events')
    except:
        evnt_idx = list()

    try:
        cond_idx = wdata_name.index('Conditions')
    except:
        cond_idx = list()

    ############################################################################
    # Weather state classification
    ############################################################################
    for class_idx in [temp_idx, dewp_idx, humd_idx]:
        obs = wdata_mat[:, class_idx][:, np.newaxis]
        label, opt_num_cluster, model, score, score_err_sum=\
            state_retrieval(obs, max_num_cluster=30, off_set=1, est_method='kmean', PARALLEL=IS_USING_PARALLEL_OPT)
        #import pdb;pdb.set_trace()
        if class_idx == temp_idx:
            print 'Temp state classification...'
            weather_dict.update({'Temp':model.cluster_centers_})
        elif class_idx == dewp_idx:
            print 'Dewp state classification...'
            weather_dict.update({'Dewp':model.cluster_centers_})
        elif class_idx == humd_idx:
            print 'Humd state classification...'
            weather_dict.update({'Humd':model.cluster_centers_})
        else:
            print 'not found'

        for label_id in range(label.max()+1):
            label_idx = np.nonzero(label==label_id)[0]
            wdata_mat[label_idx, class_idx] = np.round(model.cluster_centers_[label_id][0])

    ##################################################
    # Reclassify the Condition states into clarity of the sky
    ##################################################
    cond_state=[[]]*9
    # Clear
    cond_state[8] = ['Clear']
    # 'Partly Cloudy'
    cond_state[7] = ['Partly Cloudy','Scattered Clouds']
    # 'Overcast'
    cond_state[6] = ['Mostly Cloudy','Overcast']
    # Light Rain
    cond_state[5] = ['Fog','Mist', 'Shallow Fog','Patches of Fog','Light Freezing Fog']
    cond_state[4] = ['Drizzle', 'Heavy Drizzle','Light Drizzle','Light Freezing Drizzle']
    # Heavy Rain
    cond_state[3] = ['Rain','Rain Showers','Thunderstorms and Rain','Heavy Rain','Heavy Rain Showers', 'Freezing Rain','Light Freezing Rain','Light Rain Showers','Light Rain','Light Thunderstorms and Rain']
    cond_state[2] = ['Ice Pellets', 'Ice Crystals','Light Ice Crystals','Light Ice Pellets']
    # 'Snow'
    cond_state[1] = ['Snow','Snow Showers','Light Snow','Light Snow Grains','Light Snow Showers']
    cond_state[0] = ['Unknown']
    cond_data_array = wdata_mat[:,cond_idx].copy()

    print 'Condition state classification...'
    for k in range(len(cond_state)):
        for cond_str in cond_state[k]:
            if cond_str in Conditions_dict.keys():
                cond_val_old = Conditions_dict[cond_str]
                idx_temp = np.nonzero(cond_data_array==cond_val_old)[0]
                if len(idx_temp)>0:
                    wdata_mat[idx_temp,cond_idx]=k

    Conditions_dict_temp={}
    Conditions_dict_temp.update({'Clear':8})
    Conditions_dict_temp.update({'Cloudy':7})
    Conditions_dict_temp.update({'Overcast':6})
    Conditions_dict_temp.update({'Fog':5})
    Conditions_dict_temp.update({'Drizzle':4})
    Conditions_dict_temp.update({'Rain':3})
    Conditions_dict_temp.update({'Ice':2})
    Conditions_dict_temp.update({'Snow':1})
    Conditions_dict_temp.update({'Unknown':0})
    # Abbr' of weather factor type is
    weather_dict.update({'Cond':Conditions_dict_temp})
    ####################################################################
    # Reclassify the Event states into rain/snow/fog weather conditons
    ####################################################################
    event_state=[[]]*4
    # No event
    event_state[0]=['']
    # Snow
    event_state[1]=['Rain-Snow','Snow','Fog-Snow']
    # Rain
    event_state[2]=['Rain','Thunderstorm','Rain-Thunderstorm']
    # Fog
    event_state[3]=['Fog','Fog-Rain']
    print 'Event state classification...'
    event_data_array=wdata_mat[:,evnt_idx].copy()
    for k in range(len(event_state)):
        for event_str in event_state[k]:
            if event_str in Events_dict.keys():
                event_val_old=Events_dict[event_str]
                idx_temp=np.nonzero(event_data_array==event_val_old)[0]
                if len(idx_temp)>0:
                    wdata_mat[idx_temp,evnt_idx]=k

    Events_dict_temp={}
    Events_dict_temp.update({'NoEvent':0})
    Events_dict_temp.update({'Snow':1})
    Events_dict_temp.update({'Rain':2})
    Events_dict_temp.update({'Fog':3})
    weather_dict.update({'Event':Events_dict_temp})
    return wdata_mat, weather_dict


def bldg_obj_weather_convert(bldg_obj, sig_tag='avg'):

    bldg_sigtag = bldg_obj.sigtags[sig_tag]

    if bldg_sigtag.data_weather_mat is not None and isinstance(bldg_sigtag.data_weather_mat, np.ndarray):
        wdata_mat = bldg_sigtag.data_weather_mat.copy()
        wdata_name = bldg_sigtag.names['weather']
        conditions_dict = bldg_obj.Conditions_dict.copy()
        events_dict = bldg_obj.Events_dict.copy()
        wdata_mat, weather_dict = weather_convert(wdata_mat, wdata_name, conditions_dict, events_dict)
        bldg_sigtag.weather_dict = weather_dict
        bldg_sigtag.data_weather_mat_ = wdata_mat


#=======================================================================================================================
def _sigtag_property(data_dict, pname_key, sig_tag):
    data_state_mat = data_dict[sig_tag + 'data_state_mat']
    data_weather_mat = data_dict[sig_tag + 'data_weather_mat']
    data_time_mat = data_dict[sig_tag + 'data_time_mat']
    time_slot = data_dict[sig_tag + '_time_slot']
    data_exemplar = data_dict[sig_tag + 'data_exemplar']
    data_zvar = remove_dot(data_dict[sig_tag + 'data_zvar'])
    sensor_names = remove_dot(data_dict['sensor_names'])
    weather_names = remove_dot(data_dict['weather_names'])
    time_names = remove_dot(data_dict['time_names'])

    #TODO: Name correction for exemplar
    if isinstance(pname_key, list):
        p_idx = [sig_tag + sensor_names.index(p_name) for p_name in pname_key]
        p_names = pname_key
    else:
        p_idx = grep(pname_key, sensor_names)
        p_names = list(np.array(sensor_names)[p_idx])
    p_names = remove_dot(p_names)

    print '-' * 40
    print 'Power sensor selected -' + sig_tag
    print '-' * 40
    pprint.pprint(p_idx)
    pprint.pprint(sensor_names)
    pprint.pprint(p_names)

    return BuildingSigtagProperty(sig_tag,  data_state_mat, data_weather_mat, data_time_mat, time_slot, data_exemplar,
                                  data_zvar, sensor_names, weather_names, time_names, p_idx, p_names)


def _compute_lh_value(blgd_property, bldg_analysis, sig_tag):
    print '-' * 40
    print 'Compute LH values for ' + sig_tag
    print '-' * 40

    all_data_state_mat = np.vstack((blgd_property.data_state_mat.T, blgd_property.data_time_mat.T,
                                    blgd_property.data_weather_mat_.T)).T
    p_idx = blgd_property.p_idx
    p_names = blgd_property.p_names
    len_sensor = blgd_property.data_state_mat.shape[1]
    len_time = blgd_property.data_time_mat.shape[1]
    len_weather = blgd_property.data_weather_mat.shape[1]
    sensor_cause_idx_set = range(len_sensor)
    time_cause_idx_set = range(len_sensor, len_sensor + len_time)
    weather_cause_idx_set = range(len_sensor + len_time, len_sensor + len_time + len_weather)

    for k, effect_idx in enumerate(p_idx):
        p_name = remove_dot(p_names[k])
        print 'compute cond. prob of ' + p_name

        for i in xrange(len(bldg_analysis)):
            bldg_anal_obj = bldg_analysis[i]

            if bldg_anal_obj.sensor_tag == p_name:
                # check weather it is in the set
                effect_state_set = np.array(list(set(all_data_state_mat[:, effect_idx])))
                eff_state = effect_state_set.max()
                bldg_anal_obj.peak_eff_state = eff_state

                s_optstate_set_temp, s_optprob_set_temp = \
                    find_cond_lh_set(all_data_state_mat, sensor_cause_idx_set, effect_idx, eff_state)
                bldg_anal_obj.attrs['sensor'].optprob_set = s_optprob_set_temp
                bldg_anal_obj.attrs['sensor'].optstate_set = s_optstate_set_temp

                w_optstate_set_temp, w_optprob_set_temp = \
                    find_cond_lh_set(all_data_state_mat, weather_cause_idx_set, effect_idx, eff_state)
                bldg_anal_obj.attrs['weather'].optprob_set = w_optprob_set_temp
                bldg_anal_obj.attrs['weather'].optstate_set = w_optstate_set_temp

                t_optstate_set_temp, t_optprob_set_temp = \
                    find_cond_lh_set(all_data_state_mat, time_cause_idx_set, effect_idx, eff_state)
                bldg_anal_obj.attrs['time'].optprob_set = t_optprob_set_temp
                bldg_anal_obj.attrs['time'].optstate_set = t_optstate_set_temp


def create_bldg_object(data_dict, avgdata_dict,  diffdata_dict, bldg_tag, pname_key):
    print '-' * 40
    print 'create object for', bldg_tag
    print '-' * 40

    bldg_object = BuildingObject(bldg_tag)

    # average data
    bldg_object.sigtags['avg'] = _sigtag_property(avgdata_dict, pname_key, 'avg')

    # variance data
    bldg_object.sigtags['diff'] = _sigtag_property(diffdata_dict, pname_key, 'diff')

    #TODO: Name correction for exemplar
    bldg_object.Conditions_dict = data_dict['Conditions_dict']
    bldg_object.Events_dict = data_dict['Events_dict']

    bldg_obj_weather_convert(bldg_object, 'avg')
    bldg_obj_weather_convert(bldg_object, 'diff')

    # Create classs strucutre for data analysis
    avg_p_name = [BuildingAnalysis(remove_dot(p_name)) for p_name in bldg_object.sigtags['avg'].p_names]
    diff_p_name = [BuildingAnalysis(remove_dot(p_name)) for p_name in bldg_object.sigtags['diff'].p_names]
    bldg_object.analysis = {'avg': avg_p_name, 'diff': diff_p_name}

    _compute_lh_value(bldg_object.sigtags['avg'], bldg_object.analysis['avg'], 'avg')
    _compute_lh_value(bldg_object.sigtags['diff'], bldg_object.analysis['diff'], 'diff')

    return bldg_object



def _bn_anaylsis(bldg_obj, p_name, attr='sensor', sig_tag='avg', num_picks_bn=15, learning_alg='hc'):
    s_names = bldg_obj.sigtags[sig_tag].names['sensor']
    p_idx = s_names.index(p_name)
    data_state_mat = bldg_obj.sigtags[sig_tag].data_state_mat

    anlist = bldg_obj.analysis[sig_tag]

    optprob_set = None
    optstate_set = None
    for anal in anlist:
        if anal.sensor_tag == p_name:
            optprob_set = anal.attrs[attr].optprob_set
            optstate_set = anal.attrs[attr].optstate_set
            break

    if optprob_set is None or optstate_set is None:
        raise Exception("Invalid p_name", p_name)

    sort_idx = np.argsort(optprob_set)[::-1]

    if attr == 'sensor':
        print 'power - sensors...'
        idx_select = [p_idx] + list(sort_idx[:num_picks_bn])
        cols = [s_names[k] for k in idx_select]
        bndata_mat = bldg_obj.sigtags[sig_tag].data_state_mat[:, idx_select]
        b_arc_list = pair_in_idx([cols[0]], cols[1:])

    elif attr == 'weather':
        print 'power - weather...'
        w_names = bldg_obj.sigtags[sig_tag].names['weather']
        cols = [p_name] + [w_name for w_name in w_names]
        bndata_mat = np.vstack((bldg_obj.sigtags[sig_tag].data_state_mat[:, p_idx].T,
                                bldg_obj.sigtags[sig_tag].data_weather_mat.T)).T
        b_arc_list = pair_in_idx([cols[0]], cols[1:])

    elif attr == 'time':
        print 'power - time...'
        t_names = bldg_obj.sigtags[sig_tag].names['time']
        cols = [p_name] + [t_name for t_name in t_names]
        bndata_mat = np.vstack((bldg_obj.sigtags[sig_tag].data_state_mat[:, p_idx].T,
                                bldg_obj.sigtags[sig_tag].data_time_mat.T)).T
        b_arc_list = pair_in_idx([cols[0]], cols[1:]) + pair_in_idx(cols[1:], cols[1:])

    else:
        print 'error'
        return 0

    # this is the heart and soul of ddea
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat, cols)

    if learning_alg == 'tabu':
        hc_b = rbn.bnlearn.tabu(data_frame, blacklist=black_arc_frame, score='bic')
    elif learning_alg == 'mmhc':
        hc_b = rbn.bnlearn.mmhc(data_frame, blacklist=black_arc_frame, score='bic')
    else:
        hc_b = rbn.bnlearn.hc(data_frame, blacklist=black_arc_frame, score='bic')

    amat = rbn.py_get_amat(hc_b)
    cause_label = list(np.array(cols)[np.nonzero(amat[:, 0] == 1)[0]])
    cause_idx = [cols.index(label_) for label_ in cause_label]
    return cause_label, cols, hc_b, amat, bndata_mat

def _bn_anaylsis_all(bldg_obj, p_name, sig_tag='avg', num_picks_bn=15, learning_alg='hc'):
    s_names = bldg_obj.sigtags[sig_tag].names['sensor']
    p_idx = s_names.index(p_name)
    data_state_mat = bldg_obj.sigtags[sig_tag].data_state_mat

    print 'power - sensors + weather + time ...'
    s_cause_label, s_labels, s_hc, s_cp_mat, s_bndata_mat = \
        _bn_anaylsis(bldg_obj, p_name, attr='sensor', sig_tag=sig_tag, num_picks_bn=num_picks_bn, learning_alg=learning_alg)

    t_cause_label, t_labels, t_hc, t_cp_mat, t_bndata_mat = \
        _bn_anaylsis(bldg_obj, p_name, attr='time', sig_tag=sig_tag, num_picks_bn=num_picks_bn, learning_alg=learning_alg)

    w_cause_label, w_labels, w_hc, w_cp_mat, w_bndata_mat = \
        _bn_anaylsis(bldg_obj, p_name, attr='weather', sig_tag=sig_tag, num_picks_bn=num_picks_bn, learning_alg=learning_alg)
    #s_cause_label=s_labels; w_cause_label=w_labels;t_cause_label=t_labels

    s_cause_idx = [bldg_obj.sigtags[sig_tag].names['sensor'].index(name) for name in s_cause_label]
    t_cause_idx = [bldg_obj.sigtags[sig_tag].names['time'].index(name) for name in t_cause_label]
    w_cause_idx = [bldg_obj.sigtags[sig_tag].names['weather'].index(name) for name in w_cause_label]

    bndata_mat = np.vstack( (bldg_obj.sigtags[sig_tag].data_state_mat[:, p_idx].T,\
        bldg_obj.sigtags[sig_tag].data_state_mat[:, s_cause_idx].T, \
        bldg_obj.sigtags[sig_tag].data_weather_mat_[:, w_cause_idx].T, \
        bldg_obj.sigtags[sig_tag].data_time_mat[:, t_cause_idx].T)).T

    cols = [name for name in [p_name] + s_cause_label + w_cause_label + t_cause_label]

    b_arc_list = \
        pair_in_idx([p_name], s_cause_label + w_cause_label + t_cause_label) + \
        pair_in_idx(s_cause_label, w_cause_label+t_cause_label) + \
        pair_in_idx(w_cause_label, t_cause_label) + \
        pair_in_idx(t_cause_label, t_cause_label)

    # this is the heart and soul of ddea
    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat, cols)
    if learning_alg == 'tabu':
        hc_b = rbn.bnlearn.tabu(data_frame, blacklist=black_arc_frame, score='bic')
    elif learning_alg == 'mmhc':
        hc_b = rbn.bnlearn.mmhc(data_frame, blacklist=black_arc_frame, score='bic')
    else:
        hc_b = rbn.bnlearn.hc(data_frame, blacklist=black_arc_frame, score='bic')
    amat = rbn.py_get_amat(hc_b)
    cause_label = list(np.array(cols)[np.nonzero(amat[:, 0] == 1)[0]])
    cause_idx = [cols.index(label_) for label_ in cause_label]
    return cause_label, cols, hc_b, amat, bndata_mat


def _peak_analysis(cause_label, effect_label, col_labels, bndata_mat):

    if isinstance(cause_label, list):
        cause_idx = [col_labels.index(label_) for label_ in cause_label]
    else:
        cause_idx = [col_labels.index(label_) for label_ in [cause_label]]

    if isinstance(effect_label, list):
        effect_idx = [col_labels.index(label_) for label_ in effect_label]
    else:
        effect_idx=[col_labels.index(label_) for label_ in [effect_label]]

    effect_state_set = list(set(bndata_mat[:, effect_idx].T[0]))
    low_peak_state_effect = np.min(effect_state_set)
    high_peak_state_effect = np.max(effect_state_set)
    high_peak_state_temp, high_peak_prob_temp = \
        compute_cause_likelihood(bndata_mat, cause_idx, [effect_idx], [[high_peak_state_effect]])
    low_peak_state_temp, low_peak_prob_temp = \
        compute_cause_likelihood(bndata_mat, cause_idx,[effect_idx], [[low_peak_state_effect]])
    high_peak_state = np.array(high_peak_state_temp)
    high_peak_prob = np.array(high_peak_prob_temp)
    low_peak_state = np.array(low_peak_state_temp)
    low_peak_prob = np.array(low_peak_prob_temp)
    return low_peak_state, low_peak_prob, high_peak_state, high_peak_prob


def _get_tick_symbol(tick_state_val, cause_labels, event, cond):
    if len(cause_labels) == 1:
        iter_zip = zip(cause_labels, tick_state_val.T[np.newaxis,:])
    else:
        iter_zip = zip(cause_labels, tick_state_val.T)

    symbol_tuple = []
    for cause_label, state_val in iter_zip:
        symbol_out = []
        if (isinstance(state_val, np.ndarray) == False) and (isinstance(state_val, list) == False):
            state_val = [state_val]
        temp = list(set(state_val))
        if list(np.sort(temp)) == [-1, 0, 1]:
            cause_label = 'PEAK'

        for s_val in state_val:
            if cause_label == 'MTH':
                symbol_out.append(monthDict[s_val])
            elif cause_label == 'WD':
                symbol_out.append(weekDict[s_val])
            elif cause_label == 'HR':
                symbol_out.append(hourDict[s_val])
            elif cause_label == 'Dew_PointC':
                symbol_out.append(str(s_val)+'C')
            elif cause_label == 'Humidity':
                symbol_out.append(str(s_val)+'%')
            elif cause_label == 'Events':
                symbol_out.append([key for key, val in event.items() if val == s_val])
            elif cause_label == 'Conditions':
                symbol_out.append([key for key, val in cond.items() if val == s_val])
            elif cause_label == 'TemperatureC':
                symbol_out.append(str(s_val) + 'C')
            elif cause_label == 'PEAK':
                symbol_out.append(stateDict[s_val])
            else:
                symbol_out.append(str(s_val))
        symbol_tuple.append(symbol_out)
    temp_ = np.array(symbol_tuple)
    temp2 = temp_.reshape(len(cause_labels), int(np.prod(temp_.shape) / len(cause_labels))).T
    return [tuple(symbol) for symbol in temp2]


def bn_probability_analysis(bldg_obj, sig_tag='avg'):

    p_name_set = [analysis.sensor_tag for analysis in bldg_obj.analysis[sig_tag]]

    if sig_tag == "avg":
        event = bldg_obj.sigtags['avg'].weather_dict['Event']
        cond = bldg_obj.sigtags['avg'].weather_dict['Cond']

    elif sig_tag == "diff":
        event = bldg_obj.sigtags['diff'].weather_dict['Event']
        cond = bldg_obj.sigtags['diff'].weather_dict['Cond']

    bn_out_set = list()

    for p_name in p_name_set:
        try:
            # bn analysis - Power-Sensor
            s_cause_label, s_labels, s_hc, s_cp_mat, s_bndata_mat = \
                _bn_anaylsis(bldg_obj, p_name, attr='sensor', sig_tag=sig_tag, num_picks_bn=5)

            # bn analysis -Power-Time
            t_cause_label, t_labels, t_hc, t_cp_mat, t_bndata_mat = \
                _bn_anaylsis(bldg_obj, p_name, attr='time', sig_tag=sig_tag, num_picks_bn=10)

            # bn analysis -Power-Weather
            w_cause_label, w_labels, w_hc, w_cp_mat, w_bndata_mat = \
                _bn_anaylsis(bldg_obj, p_name, attr='weather', sig_tag=sig_tag,num_picks_bn=10)

            # bn analysis -Power-Sensor+Time+Weather
            all_cause_label, all_labels, all_hc, all_cp_mat, all_bndata_mat=\
                _bn_anaylsis_all(bldg_obj, p_name, sig_tag=sig_tag, num_picks_bn=20)

            # prob analysis -Power-Sensor+Time+Weather
            cause_label = all_cause_label
            col_labels = all_labels
            effect_label = p_name
            bndata_mat = all_bndata_mat
            low_peak_state, low_peak_prob, high_peak_state, high_peak_prob = \
                _peak_analysis(cause_label, effect_label, col_labels, bndata_mat)

            x_set = low_peak_state
            all_cause_symbol_xlabel = _get_tick_symbol(x_set, all_cause_label, event, cond)
            all_cause_symbol_xtick = range(len(low_peak_state))

            # BN-PROB STORE
            bn_out = \
                {'p_name': p_name,
                 's_cause_label': s_cause_label,
                 's_labels': s_labels,
                 's_hc': s_hc,
                 's_cp_mat': s_cp_mat,
                 's_bndata_mat': s_bndata_mat,
                 't_cause_label': t_cause_label,
                 't_labels': t_labels,
                 't_hc': t_hc,
                 't_cp_mat': t_cp_mat,
                 't_bndata_mat': t_bndata_mat,
                 'w_cause_label': w_cause_label,
                 'w_labels': w_labels,
                 'w_hc': w_hc,
                 'w_cp_mat': w_cp_mat,
                 'w_bndata_mat': w_bndata_mat,
                 'all_cause_label': all_cause_label,
                 'all_labels': all_labels,
                 'all_hc': all_hc,
                 'all_cp_mat': all_cp_mat,
                 'all_bndata_mat': all_bndata_mat,
                 'low_peak_state': low_peak_state,
                 'low_peak_prob': low_peak_prob,
                 'high_peak_state': high_peak_state,
                 'high_peak_prob': high_peak_prob,
                 'all_cause_symbol_xlabel': all_cause_symbol_xlabel,
                 'all_cause_symbol_xtick': all_cause_symbol_xtick}
            bn_out_set.append(bn_out)

        except Exception as e:
            print '*** Error in processing bn_prob for ', p_name, '! ****\n', e
            pass
    return bn_out_set


def plotting_bldg_lh(bldg_list, bldg_key=[], attr='sensor', num_picks=30):
    print '-' * 40
    print 'plotting lh for ' + attr
    print '-' * 40
    sig_tag_set = ['avg', 'diff']
    plt.ioff()

    if not len(bldg_key):
        bldg_tag_set = [bldg.bldg_tag for bldg in bldg_list]
    else:
        bldg_tag_set = [bldg_key]

    for bldg_tag in bldg_tag_set:
        for blgd in bldg_list:
            if bldg_tag == bldg.bldg_tag:
                print '-' * 40
                print bldg_tag, "is to be plotted..."
                print '-' * 40

                for sig_tag in sig_tag_set:
                    try:
                        p_names = bldg.sigtags[sig_tag].p_names

                        for pname in p_names:
                            try:
                                blank_idx = pname.index('.')
                                pname = pname.replace('.', '_')
                            except:
                                pass

                            optprob_set = None
                            optstate_set = None
                            for anal in bldg.analysis[sig_tag]:
                                if anal.sensor_tag == pname:
                                    optprob_set = anal.attrs[attr].optprob_set
                                    optstate_set = anal.attrs[attr].optstate_set
                                    break

                            s_names = bldg.sigtags[sig_tag].names[attr]

                            num_picks = 30
                            sort_idx = np.argsort(optprob_set)[::-1]
                            sort_lh = optprob_set[sort_idx[:num_picks]].T
                            sort_state = optstate_set[sort_idx[:num_picks]].T
                            x_label = list(np.array(s_names)[sort_idx[:num_picks]])
                            x_ticks = range(len(x_label))

                            plt.figure(figsize=(20.0, 15.0))
                            plt.subplot(2, 1, 1)
                            plt.plot(sort_lh, '-*')
                            plt.xticks(x_ticks, x_label, rotation=270, fontsize="small")
                            if sig_tag == 'avg':
                                plt.title('Most relavant ' + attr + ' attributes to the peak (demand) of '+ pname, fontsize=20)
                            else:
                                plt.title('Most relavant ' + attr + ' attributes to the peak variations of '+ pname, fontsize=20)
                            plt.tick_params(labelsize='large')
                            plt.ylim([-0.05, 1.05])
                            plt.ylabel('Likelihood (From 0 to 1)', fontsize=18)
                            plt.savefig(FIG_DIR + bldg_tag + '_' + pname + '_' + attr + '_' + sig_tag + '_lh_sensors.png', bbox_inches='tight')
                            plt.close()

                    except Exception as e:
                        print "Plot error ", e
                        pass
    plt.close()
    plt.ion()


def plotting_bldg_bn(bldg_list):
    plt.ioff()

    for bldg in bldg_list:
        print 'Getting anal_out from ', bldg.bldg_tag

        try:
            for sig_tag, anal_out in bldg.anal_out.iteritems():

                for bn_prob in anal_out:

                    p_name = bn_prob['p_name']

                    try:
                        fig_name = 'BN for Sensors ' + p_name
                        plt.figure(fig_name, figsize=(30.0, 30.0))
                        col_name = bn_prob['s_labels']
                        rbn.nx_plot(bn_prob['s_hc'], col_name, graph_layout='spring', node_text_size=30)
                        plt.savefig(FIG_DIR + bldg.bldg_tag + '_' + p_name + '_' + sig_tag + '_bn_sensors' + get_pngid() + '.png', bbox_inches='tight')
                        plt.close()

                    except Exception as e:
                        print 'error in '+fig_name, e
                        pass

                    try:
                        fig_name = 'BN for Time ' + p_name
                        plt.figure(fig_name, figsize=(30.0,30.0))
                        rbn.nx_plot(bn_prob['t_hc'], bn_prob['t_labels'], graph_layout='spring', node_text_size=30)
                        plt.savefig(FIG_DIR + bldg.bldg_tag + '_' + p_name + '_' + sig_tag + '_bn_time' + get_pngid() + '.png', bbox_inches='tight')
                        plt.close()

                    except Exception as e:
                        print 'error in '+fig_name, e
                        pass

                    try:
                        fig_name = 'BN for Weather ' + p_name
                        plt.figure(fig_name, figsize=(30.0,30.0))
                        rbn.nx_plot(bn_prob['w_hc'], bn_prob['w_labels'], graph_layout='spring', node_text_size=30)
                        plt.savefig(FIG_DIR + bldg.bldg_tag + '_' + p_name + '_' + sig_tag + '_bn_weather' + get_pngid() +'.png', bbox_inches='tight')
                        plt.close()

                    except Exception as e:
                        print 'error in '+fig_name, e
                        pass

                    try:
                        fig_name = 'BN for Sensor-Time-Weather ' + p_name
                        plt.figure(fig_name, figsize=(30.0,30.0))
                        rbn.nx_plot(bn_prob['all_hc'], bn_prob['all_labels'], graph_layout='spring', node_text_size=30)
                        plt.savefig(FIG_DIR + bldg.bldg_tag + '_' + p_name + '_' + sig_tag + '_bn_sensor_time_weather' + get_pngid() + '.png', bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print 'error in '+fig_name, e
                        pass

                    try:
                        fig_name = 'BN PEAK LH Analysis for Sensor-Time-Weather ' + p_name
                        plt.figure(fig_name, figsize=(30.0, 30.0))
                        plt.subplot(2, 1, 1)
                        plt.plot(bn_prob['all_cause_symbol_xtick'], bn_prob['high_peak_prob'], '-^')
                        plt.plot(bn_prob['all_cause_symbol_xtick'], bn_prob['low_peak_prob'], '-.v')
                        plt.ylabel('Likelihood', fontsize=20)

                        plt.xticks(bn_prob['all_cause_symbol_xtick'], bn_prob['all_cause_symbol_xlabel'], rotation=270, fontsize=20)
                        plt.tick_params(labelsize=20)
                        plt.legend(('High Peak', 'Low Peak'), loc='center right', prop={'size':25})
                        plt.tick_params(labelsize=20)

                        plt.grid()
                        plt.ylim([-0.05,1.05])
                        plt.title('Likelihood of '+ str(remove_dot(p_name))+' given '+'\n'+str(remove_dot(bn_prob['all_cause_label'])), fontsize=20)
                        plt.savefig(FIG_DIR + bldg.bldg_tag + '_' + p_name + '_' + sig_tag + '_LH_sensor_time_weather' + get_pngid() + '.png', bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print 'error in '+fig_name, e
                        pass

        except Exception as e:
            print e
            pass

    plt.ion()