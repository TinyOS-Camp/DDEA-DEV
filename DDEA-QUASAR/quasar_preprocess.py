# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 01:34:41 2014

@author: deokwoo
"""
from __future__ import division # To forace float point division
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
from shared_constants import *
from data_tools import *
from scipy.stats import stats 
import time
import multiprocessing as mp

def verify_sensor_data_format(tup):
    key = tup[0]
    data_list = tup[1]
    time_slots = tup[2]
    q = tup[3]
    
    print 'checking ', key, '...'
    for i, samples in enumerate(data_list):
        for j, each_sample in enumerate(samples):
            if each_sample == []:
                q.put([key,i,j])
                print each_sample, 'at', time_slots[i], 'in', key

            elif ( isinstance(each_sample,int) == False and isinstance(each_sample, float) == False):
                q.put([key, i, j])
                print each_sample, 'at', time_slots[i], 'in', key


def verify_data_format(data_dict, PARALLEL=False):
    # Verify there is no  [] or N/A in the list
    # Only FLoat or Int format is allowed
    print 'Checking any inconsisent data format...'
    print '-' * 40

    list_of_wrong_data_format = []
    time_slots = data_dict['time_slots']
    weather_list_used = [data_dict['weather_list'][i] for i in [1, 2, 3, 10, 11]]
    key_list = weather_list_used+ data_dict['sensor_list']
    if not PARALLEL:
        for key in key_list:
            print 'checking ', key, '...'
            for i, samples in enumerate(data_dict[key][1]):
                for j, each_sample in enumerate(samples):

                    if each_sample == []:
                        list_of_wrong_data_format.append([key,i,j])
                        print each_sample, 'at', time_slots[i], 'in', key

                    elif isinstance(each_sample, int) == False and isinstance(each_sample, float) == False:
                        list_of_wrong_data_format.append([key, i, j])
                        print each_sample, 'at', time_slots[i], 'in', key
        print '-' * 40
    
    # PARALLEL
    else:
        manager = mp.Manager()
        q = manager.Queue()
        
        p = mp.Pool(CPU_CORE_NUM)
        param_list = [(key,data_dict[key][1],time_slots,q) for key in key_list]
        p.map(verify_sensor_data_format,param_list)
        
        p.close()
        p.join()

        while not q.empty():
            item = q.get()
            print 'queue item: ' + str(item)
            list_of_wrong_data_format.append(item)
    
    if len(list_of_wrong_data_format)>0:
        raise NameError('Inconsistent data format in the list of data_used')
    return list_of_wrong_data_format



def verify_data_mat(X):
    num_err_temp=np.array([[len(np.nonzero(np.isnan(sample))[0]),len(np.nonzero(sample==np.inf)[0]),len(np.nonzero(np.var(sample)==0)[0])] for sample in X])
    num_err=np.sum(num_err_temp,axis=0)
    for err_idx in np.argwhere(num_err>0):
        if err_idx==0:
            NameError('nan entry found')
        if err_idx==1:
            NameError('inf entry found')
        if err_idx==2:
            NameError('zero var found') 
    print 'all entry values of data matrix are verifed ok'


def normalize_data(data_input):
    y_pred = data_input.copy()
    y_temp = np.delete(y_pred, np.nonzero(y_pred==np.infty), axis=0)
    y_temp_sort = np.sort(y_temp)[int(np.ceil(len(y_temp)*0.05)):int(np.floor(len(y_temp)*0.95))]
    var_temp = np.var(y_temp_sort)

    if var_temp > 0: # At least 2 non-infty elements in y_pred
        no_inf_idx = np.nonzero(y_pred!=np.infty)
        y_pred[no_inf_idx] = y_pred[no_inf_idx] - np.mean(y_pred[no_inf_idx])
        temp_val = y_pred/norm(y_pred[no_inf_idx])
        temp_status = 0
    else:
        temp_val = list(set(y_temp_sort))
        temp_status = -1
    return temp_val, temp_status


def interploate_data(x_temp, num_type, max_num_succ_idx_for_itpl):
    num_of_samples = x_temp.shape[0]
    inf_idx = np.nonzero(x_temp == np.inf)[0]
    noinf_idx = np.nonzero(x_temp != np.inf)[0]

    # Dont interploate the values on bondary.
    inter_idx = np.delete(inf_idx, np.nonzero(inf_idx == 0))
    inter_idx = np.delete(inter_idx, np.nonzero(inter_idx == num_of_samples-1))

    #############################################################################################
    # Dont interploate the values unknown  successively more than num_succ_idx_no_interploate
    # Then deletea any index that meet the condition above, 
    # inter_idx=np.delete(inter_idx,those index)
    # Need to be completed  .....
    #############################################################################################

    # Find successive inf indices
    succ_inf_idx = []
    for i in range(0, len(noinf_idx) - 1):
        # number of successive inf between two non-inf indices
        num_succ_inf = noinf_idx[i+1] - noinf_idx[i] - 1

        if num_succ_inf > max_num_succ_idx_for_itpl:
            succ_inf_idx = succ_inf_idx + range(noinf_idx[i]+1, noinf_idx[i+1])

    # Remove successive inf indices
    inter_idx = list(set(inter_idx) - set(succ_inf_idx))
    if num_type == FLOAT_TYPE:
        #f = interp1d(noinf_idx,x_temp[noinf_idx,0],'linear')
        val_new = np.interp(inter_idx,noinf_idx, x_temp[noinf_idx,0])
        #val_new = np.interp(t_new, t_,val_)
    elif num_type == INT_TYPE:
        #f = interp1d(noinf_idx,x_temp[noinf_idx,0],'nearest')
        val_new=fast_nearest_interp(inter_idx,noinf_idx, x_temp[noinf_idx, 0])
    else:
        raise NameError('Sample type must either INT or FLOAT type')

    #x_temp[inter_idx,0]=f(inter_idx)
    x_temp[inter_idx,0] = val_new
    print 'No sample in time slot', inf_idx
    print len(inter_idx), '/', len(inf_idx), ' time slots are interplated'
    return x_temp


def get_feature(data_dict_samples,num_type):
    x_temp = []
    for i, sample in enumerate(data_dict_samples):

        # If sample=[], np.std returns 0. Avoid zero std, add a infitestimal number

        # Set infty if no sample is availble
        if len(sample) == 0:
            x_temp.append(np.inf)                

        else:
            if num_type == INT_TYPE:
                x_temp.append(int(stats.mode(sample)[0]))                
            elif num_type == FLOAT_TYPE:
                x_temp.append(np.mean(sample))                
            else:
                raise NameError('Sample type must either INT or FLOAT type')

    x_temp = np.array(x_temp)[:,np.newaxis]
    return x_temp

# Mean value measure
def build_feature_matrix(data_dict, sensor_list, weather_list, time_slots, interpolation=1, max_num_succ_idx_for_itpl=4):

    data_used = sensor_list + weather_list
    print 'Build data feature matrix now.....'

    if interpolation == 1:
        print 'Missing samples will be interpolated upto', max_num_succ_idx_for_itpl, 'successive time slots'
    else:
        print 'All time slots with any missing sample will be removed without interpolatoin '

    num_of_data = len(data_used)
    num_of_samples = len(time_slots)

    # Declare as 2-d list for exception.
    X = []
    INT_type_list = []
    FLOAT_type_list = []
    input_names = []
    weather_type_idx = []
    sensor_type_idx = []
    INT_type_idx = []
    FLOAT_type_idx = []
    zero_var_list = []
    zero_var_val = []

    #import pdb; pdb.set_trace()


    # whose variance is zero, hence carry no information,
    # Constrcut X matrix by summerizing hourly samples
    for j, key in enumerate(data_used):
        print '-' * 40
        print 'building for ', key
        
        try:
            num_type = check_data_type(data_dict[key][2][1])

            # Avg. value feature
            x_temp = get_feature(data_dict[key][1], num_type)

            non_inf_idx = np.nonzero(x_temp < np.inf)[0]
            #if non_inf_idx <len(time_slots):measurement_point_set

            # Outlier removal, different parameters for sensors and weather data
            if len(sensor_list) <= j:
                # weather data
                is_weather_data = True
                outlier_idx = outlier_detect(x_temp[non_inf_idx], 5, 10)
            else:
                is_weather_data = False
                outlier_idx = outlier_detect(x_temp[non_inf_idx], 1, 20)

            if len(outlier_idx) > 0:
                print 'outlier samples are detected: ', 'outlier_idx:', outlier_idx
                x_temp[non_inf_idx[outlier_idx]] = np.inf
            
            # interplolation data, use nearest for int type, use linear for float type
            if interpolation == 1:
                x_temp = interploate_data(x_temp, num_type, max_num_succ_idx_for_itpl)

            norm_data_vec, output_status = normalize_data(x_temp[:, 0])
            if len(np.nonzero(norm_data_vec == np.inf)[0]) > num_of_samples/5:
                raise

        except Exception as e:
            print ' Error in processing data feature, excluded from analysis'
            output_status = -1
            norm_data_vec = None

        if output_status == -1:
            zero_var_list.append(key)
            zero_var_val.append(norm_data_vec)
            print 'too small variance for float type, added to zero var list'

        else:
            input_names.append(key)
            print j, 'th sensor update'

            if (num_type == FLOAT_TYPE) and (is_weather_data == False):
                X.append(norm_data_vec)
                FLOAT_type_idx.append(len(X)-1)
                FLOAT_type_list.append(key)

            elif (num_type == INT_TYPE) or (is_weather_data == True):
                X.append(x_temp[:, 0])
                INT_type_idx.append(len(X)-1)
                INT_type_list.append(key)

            else:
                raise NameError('Sample type must either INT or FLOAT type')

            if key in weather_list:
                weather_type_idx.append(len(X)-1)

            elif key in sensor_list:
                sensor_type_idx.append(len(X)-1)
            else:
                raise NameError('Sample type must either Weather or Sensor type')

    # Linear Interpolate
    X = np.array(X).T
    if X.shape[0] != num_of_samples:
        raise NameError('The numeber of rows in feature matrix and the number of the time slots are  different ')

    if X.shape[1]+len(zero_var_list) != num_of_data:
        raise NameError('The sume of the numeber of column in feature matrix  and the number of zero var column are  different from the number of input measurements ')

    deleted_timeslot_idx=[]
    print '-' * 20
    print 'removing time slots having no sample...'
    inf_idx_set = []
    for col_vec in X.T:
        inf_idx = np.nonzero(col_vec ==np.infty)[0]
        inf_idx_set = np.r_[inf_idx_set, inf_idx]
    inf_col_idx = list(set(list(inf_idx_set)))
    deleted_timeslot_idx = np.array([int(x) for x in inf_col_idx])

    print 'time slots', deleted_timeslot_idx, ' removed...'
    print '-' * 20
    X = np.delete(X, deleted_timeslot_idx, axis=0)
    new_time_slot = np.delete(time_slots, deleted_timeslot_idx)

    # Checking whether it has any ill entry value
    verify_data_mat(X)

    return X, new_time_slot, input_names, zero_var_list, zero_var_val, INT_type_list, INT_type_idx, FLOAT_type_list, FLOAT_type_idx, weather_type_idx, sensor_type_idx

# Abs Diff value measure
def build_diff(tup):

    k = tup[0]
    time_slots = tup[1]
    conf_lev = tup[2]
    set_val = tup[3]
    set_name = tup[4]
    num_type = tup[5]

    print set_name
    try:
        diff_mean=get_diff(set_val,time_slots,num_type,conf_lev)
        if num_type==FLOAT_TYPE:
            #norm_diff_mean,output_status=normalize_data(diff_mean[:,0])
            norm_diff_mean,output_status=normalize_data(diff_mean)
        elif num_type==INT_TYPE:
            #num_discrete_vals=len(set(list(diff_mean[:,0])))
            num_discrete_vals=len(set(list(diff_mean)))
            print 'num_discrete_vals :', num_discrete_vals
            if num_discrete_vals>1:
                output_status=0
                norm_diff_mean=diff_mean
            else:
                output_status=-1
                norm_diff_mean=list(set(diff_mean))
                #norm_diff_mean=list(set(diff_mean[:,0]))
        else:
            pass
        
    except Exception:
        print ' Error in processing data feature, excluded from analysis'
        output_status=-1
        norm_diff_mean=None
        return (k,[output_status,norm_diff_mean])

    return (k,[output_status,norm_diff_mean])


def get_diff(set_val,time_slots,num_type,conf_lev):
    time_slots_utc=dtime_to_unix(time_slots)
    TIMELET_INV_seconds=(time_slots[1]-time_slots[0]).seconds
    diff_mean=[]
    for r,utc_t in enumerate(time_slots_utc):
        utc_t_s=utc_t
        utc_t_e=utc_t+TIMELET_INV_seconds
        idx=np.nonzero((set_val[0]>=utc_t_s) & (set_val[0]<utc_t_e))[0]
        if len(idx)<2:
            diff_val=np.inf
        else:
            temp_val=abs(np.diff(set_val[1][idx]))
            upper_val=np.sort(temp_val)[int(np.floor(len(temp_val)*conf_lev)):]
            if len(upper_val)==0:
                 diff_val=np.inf
            else:
                if num_type==FLOAT_TYPE:
                    diff_val=np.mean(upper_val)
                    #print 'float type'
                elif num_type==INT_TYPE:
                    diff_val=int(stats.mode(upper_val)[0])
                    #print 'int type'
                else:
                    raise NameError('Sample type must either INT or FLOAT type')
            #diff_val=max(abs(diff(set_val[1][idx])))
            #sort(abs(diff(set_val[1][idx])))[::-1]
        diff_mean.append(diff_val)
    #diff_mean=np.array(diff_mean)[:,np.newaxis]
    diff_mean=np.array(diff_mean)
    return diff_mean
    
# Abs Diff value measure
def build_diff_matrix(measurement_point_set,time_slots,num_type_set,irr_data_name,conf_lev=0.5,PARALLEL=False):

    #time_slots_utc = dtime_to_unix(time_slots)
    Xdiff = []
    input_names = []
    INT_type_list = []
    FLOAT_type_list = []
    INT_type_idx = []
    FLOAT_type_idx = []
    zero_var_list = []
    # whose variance is zero, hence carry no information,
    zero_var_val = []
    num_of_samples = len(time_slots)
    #TIMELET_INV_seconds = (time_slots[1]-time_slots[0]).seconds
    print '=' * 40
    if not PARALLEL:
        for k, (set_val, set_name) in enumerate(zip(measurement_point_set, irr_data_name)):
            print irr_data_name[k]
            try:
                num_type = num_type_set[k]
                diff_mean = get_diff(set_val, time_slots, num_type, conf_lev)
                if num_type == FLOAT_TYPE:
                    #norm_diff_mean,output_status=normalize_data(diff_mean[:,0])
                    norm_diff_mean,output_status=normalize_data(diff_mean)
                elif num_type == INT_TYPE:
                    #num_discrete_vals=len(set(list(diff_mean[:,0])))
                    num_discrete_vals=len(set(list(diff_mean)))
                    print 'num_discrete_vals :', num_discrete_vals
                    if num_discrete_vals>1:
                        output_status=0
                        norm_diff_mean=diff_mean
                    else:
                        output_status=-1
                        #norm_diff_mean=list(set(diff_mean[:,0]))
                        norm_diff_mean=list(set(diff_mean))
                else:
                    pass
                if len(np.nonzero(norm_diff_mean == np.inf)[0])>num_of_samples/5:
                    raise 
            except Exception as e:
                print ' Error in processing data feature, excluded from analysis'
                output_status=-1
                norm_diff_mean=None

            if output_status == -1:
                zero_var_list.append(set_name);#zero_var_flag=1
                zero_var_val.append(norm_diff_mean)
                print 'too small variance for float type or a single value for int type, added to zero var list'
            else:
                input_names.append(set_name)
                Xdiff.append(norm_diff_mean)
                if num_type == FLOAT_TYPE:
                    FLOAT_type_list.append(set_name)
                    FLOAT_type_idx.append(len(Xdiff)-1)
                elif num_type == INT_TYPE:
                    INT_type_list.append(set_name)
                    INT_type_idx.append(len(Xdiff)-1)
            print '-' * 20
        print '=' * 40

    # PARALLEL ENABLED
    else:
        print 'Build diff matrix: Parallel enabled...'
        # Construct param list for workers
        param_list = []
        for k,(set_val,set_name) in enumerate(zip(measurement_point_set,irr_data_name)):
            param_list.append((k,time_slots,conf_lev,set_val,set_name,num_type_set[k]))

        p = mp.Pool(CPU_CORE_NUM)
        ret_dict = dict(p.map(build_diff,param_list))
        p.close()
        p.join()

        for k in sorted(ret_dict.keys()):
            v = ret_dict[k]
            output_status = v[0]
            norm_diff_mean = v[1]

            set_name = irr_data_name[k]
            num_type = num_type_set[k]

            if output_status==-1:
                zero_var_list.append(set_name)
                #zero_var_flag=1
                zero_var_val.append(norm_diff_mean)
                print 'too small variance for float type or a single value for int type, added to zero var list'
            else:
                input_names.append(set_name)
                try:
                    Xdiff.append(norm_diff_mean)
                except:
                    import pdb;pdb.set_trace()

                if num_type == FLOAT_TYPE:
                    FLOAT_type_list.append(set_name)
                    FLOAT_type_idx.append(len(Xdiff)-1)

                elif num_type == INT_TYPE:
                    INT_type_list.append(set_name)
                    INT_type_idx.append(len(Xdiff)-1)
            print '-' * 20

    Xdiff = np.array(Xdiff).T
    deleted_timeslot_idx = []
    print '-' * 20
    print 'removing time slots having no sample...'
    inf_idx_set=[]
    for col_vec in Xdiff.T:
        inf_idx = np.nonzero(col_vec == np.infty)[0]
        inf_idx_set=np.r_[inf_idx_set, inf_idx]
    inf_col_idx = list(set(list(inf_idx_set)))
    deleted_timeslot_idx = np.array([int(x) for x in inf_col_idx]).astype(int)
    print 'time slots', deleted_timeslot_idx, ' removed...'
    print '-' * 20

    Xdiff = np.delete(Xdiff, deleted_timeslot_idx, axis=0)
    new_time_slot = np.delete(time_slots, deleted_timeslot_idx)

    # Checking whether it has any ill entry value
    verify_data_mat(Xdiff)

    print "*-" * 20
    print "* deleted_timeslot_idx :", deleted_timeslot_idx
    print "*-" * 20

    return Xdiff,\
           new_time_slot,\
           input_names,\
           zero_var_list,\
           zero_var_val, \
           INT_type_list,\
           INT_type_idx,\
           FLOAT_type_list,\
           FLOAT_type_idx