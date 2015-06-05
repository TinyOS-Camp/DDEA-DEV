# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:11:01 2014

@author: deokwoo
"""
from __future__ import division # To forace float point division
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime
import datetime as dt
import calendar
from numpy.linalg import norm
from shared_constants import *
import re
import mytool as mt
import pandas
import uuid

###############################################################################
# Application Functions 
###############################################################################

###############################################################################
# Plotting tool
###############################################################################
# example
#plotting_data(['GW2.CG_PHASE2_ACTIVE_POWER_M'],data_dict,time_slots[50:100])       
def plotting_data(plot_list,data_dict,time_slots,opt='avg'):
    # Month indicator   
    time_mat=build_time_states(time_slots)
    time_mat_NAMES=['MTH','WD','HR']
    month_idx=0; weekday_idx=1; hour_idx=2
    num_col=int(np.ceil(np.sqrt(len(plot_list))))
    num_row=num_col
    time_mn_diff=np.diff(time_mat[:,month_idx])
    m_label_idx=time_mn_diff.nonzero()[0]; m_label_str=[]
    for m_num in time_mat[m_label_idx,month_idx]:
        m_label_str.append(monthDict[m_num])
    time_wk_diff=np.diff(time_mat[:,weekday_idx])
    w_label_idx=time_wk_diff.nonzero()[0]; w_label_str=[]
    for w_num in time_mat[w_label_idx,weekday_idx]:
        w_label_str.append(weekDict[int(w_num)])
    
    sample_slot_idx=[data_dict['time_slots'].index(dt_val) for dt_val in time_slots]
    for k,sensor in enumerate(plot_list):
        num_samples=[];  mean_samples=[]
        for i,(t,samples) in enumerate(zip(time_slots,data_dict[sensor][1][sample_slot_idx])):
            #import pdb;pdb.set_trace()
            num_samples.append(len(samples))
            mean_samples.append(np.mean(samples))
        if opt=='sd' or opt=='all':
            plt.figure('Sampling Density')
            plt.subplot(num_col, num_row,k+1)
            plt.plot(time_slots,num_samples)
            plt.title(sensor,fontsize=8)
#==============================================================================
#             plt.xticks(fontsize=8)
#==============================================================================
            plt.yticks(fontsize=8)
            plt.ylabel('# Samples/Hour',fontsize=8)
            if k<len(plot_list)-1:        
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
        if opt=='avg' or opt=='all':    
            plt.figure('Hourly Average')
            plt.subplot(num_col, num_row,k+1)
            plt.plot(time_slots,mean_samples)
            plt.title(sensor,fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.ylabel('Avg Val/Hour',fontsize=8)
            if k<len(plot_list)-1:        
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
            #plt.xticks(w_label_idx.tolist(),w_label_str,fontsize=8)
            #plt.text(m_label_idx, np.max(num_samples)*0.8, m_label_str, fontsize=12)
        if opt=='si' or opt=='all':
            plt.figure('Sampling Intervals')
            t_secs_diff=np.diff(data_dict[sensor][2][0])
            plt.subplot(num_col, num_row,k+1)
            plt.plot(t_secs_diff)
            plt.title(sensor,fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.ylabel('Samping Interval (secs)',fontsize=8)
            if k<len(plot_list)-1:        
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
    print ' End of Plotting'


def daterange(start, stop, step=dt.timedelta(days=1), inclusive=False):
  # inclusive=False to behave like range by default
  if step.days > 0:
    while start < stop:
      yield start
      start = start + step
      # not +=! don't modify object passed in if it's mutable
      # since this function is not restricted to
      # only types from datetime module
  elif step.days < 0:
    while start > stop:
      yield start
      start = start + step
  if inclusive and start == stop:
    yield start
    
    
# Convert a unix time u to a datetime object d, and vice versa
#def unix_to_dtime(u): return dt.datetime.utcfromtimestamp(u)
def unix_to_dtime(u):
    if len(u)==1:
        return dt.datetime.utcfromtimestamp(u[0])
    elif len(u)>1:
        return [dt.datetime.utcfromtimestamp(x) for x in u]        
    else:
        raise NameError('length of vector must be greater 1')

def dtime_to_unix(d):
    if len(d)==1:
        return calendar.timegm(d[0].timetuple())
    elif len(d)>1:
        return [calendar.timegm(ds.timetuple()) for ds in d]        
    else:
        raise NameError('length of vector must be greater 1')

    
    #return calendar.timegm(d.timetuple())

def find_norm_dist_matrix(X_INPUT):
    #SIMM_MAT=np.zeros([X_INPUT.shape[1],X_INPUT.shape[1]])
    #if X_INPUT.shape[1] > X_INPUT.shape[0]:
     #   print 'WARNING: num of samples are smaller than num of sensors'
    DIST_MAT=np.zeros([X_INPUT.shape[1],X_INPUT.shape[1]])
    for i in range(X_INPUT.shape[1]):
        for j in range(X_INPUT.shape[1]):
            sample1=X_INPUT[:,i].copy()
            sample2=X_INPUT[:,j].copy()
            temp_dist=norm(sample1-sample2)
            DIST_MAT[i,j]=temp_dist
            #SIMM_MAT[i,j] = 2-temp_dist
    return DIST_MAT
    
def outlier_detect(val,err_rate=10,sgm_bnd=3):
    min_num_samples = 10
    val_len = len(val)
    val_sort = sorted(val)
    if err_rate < 0 or err_rate > 100:
        raise NameError('error rate must be between 0 and 100')

    start_idx = int( val_len * ( err_rate / 2) / 100 )
    end_idx = int( val_len - val_len * ( err_rate / 2 ) / 100 )
    #import pdb;pdb.set_trace()
    #print 'start_idx: ',start_idx,'end_idx: ',end_idx

    if end_idx - start_idx > min_num_samples:
        mean_val = np.mean(val_sort[start_idx:end_idx])
        std_val = np.std(val_sort[start_idx:end_idx])
        min_val = np.min(val_sort[start_idx:end_idx])
        max_val = np.max(val_sort[start_idx:end_idx])
    else:
        return []
    #print 'min_val: ',min_val,'max_val: ',max_val,'sgm_bnd: ',sgm_bnd
    #val_bnd_high=mean_val+sgm_bnd*std_val
    val_bnd_high= max_val + sgm_bnd * std_val
    #val_bnd_low = mean_val-sgm_bnd*std_val
    val_bnd_low = min_val - sgm_bnd * std_val
    #print 'val_bnd_low: ',val_bnd_low,'val_bnd_high: ',val_bnd_high
    return np.nonzero((val_bnd_high<val ) | (val_bnd_low>val))[0]
    

# Average Sampling Interval
# Interploated data
def fast_nearest_interp(xi, x, y):
    # Assumes that x is monotonically increasing!!.
    # Shift x points to centers
    spacing = np.diff(x) / 2
    x = x + np.hstack([spacing, spacing[-1]])
    # Append the last point in y twice for ease of use
    y = np.hstack([y, y[-1]])
    return y[np.searchsorted(x, xi)]


###
# Convert time (datetime object or utc time) to state [month,weekday,hour]
# if time is datetime ojbect, we assume it is local (Helsinki) time
# otherwise, it is UTC time, and we give the corresponding [month,wweekday,hour]
# in the local time
###
def convert_time_to_state(ts,readable_format=False,from_zone='Europe/Helsinki',to_zone='Europe/Helsinki'):
    time_state = []

    # If the timestamp is datetime object,
    # we assume it is already in local time (Helsinki)
    # If not: we assume it is utc time, and we need to convert it to
    # local time zone (Helsinki)
    if type(ts) == dt.datetime:
        local_dt = ts
    else:
        #local_dt = dt.datetime.fromtimestamp(ts).replace(tzinfo=tz.gettz('UTC')).astimezone(pytz.timezone(zone))
        local_dt = dt.datetime.utcfromtimestamp(ts).replace(tzinfo=tz.gettz(from_zone)).astimezone(pytz.timezone(to_zone))

    if not readable_format:
        time_state = [local_dt.month-1, local_dt.weekday(), local_dt.hour]
    else:
        time_state = [monthDict[local_dt.month-1],weekDict[local_dt.weekday()], str(local_dt.hour) + 'h']

    return time_state
###
# Construct time state matrix Nx3 [Month,Weekday,Hour] from the list of time
# ts_list, where N = len(ts_list)
# If the item in the list is datetime object, we assume it is already in local time (Helsinki)
# Otherwise, we assume the list item is in UTC time
# Setting readable_format to True will result in the time in understandable format
###
def build_time_states(ts_list,readable_format=False,from_zone='Europe/Helsinki',to_zone='Europe/Helsinki'):

    time_mat = []    
#    if not readable_format:
    for ts in ts_list:
        time_state = convert_time_to_state(ts,readable_format=readable_format,from_zone=from_zone,to_zone=to_zone)        
        time_mat.append(time_state)

    return np.array(time_mat)
    
    

def pair_in_idx(a,b=[],FLATTEN=True):
    pair_set=[]
    if len(b)==0:
        for idx1 in range(len(a)):
            for idx2 in range(idx1+1,len(a)):
                if FLATTEN==True:
                    if (isinstance(a[idx1],list)==True) and (isinstance(a[idx2],list)==True):
                        pair_set.append([a[idx1]+a[idx2]][0])
                    elif isinstance(a[idx1],list)==True and isinstance(a[idx2],list)==False:
                        pair_set.append([list([a[idx2]])+a[idx1]][0])
                    elif isinstance(a[idx1],list)==False and isinstance(a[idx2],list)==True:
                        pair_set.append([a[idx2]+list([a[idx1]])][0])
                    else:
                        pair_set.append([a[idx1],a[idx2]])
                else:
                    pair_set.append([a[idx1],a[idx2]])
    else:
        for idx1 in a:
            for idx2 in b:
                if FLATTEN==True:
                    if (isinstance(idx1,list)==True) and (isinstance(idx2,list)==True):
                        pair_set.append([idx1+idx2][0])
                    elif isinstance(idx1,list)==True and isinstance(idx2,list)==False:
                        pair_set.append([idx1+list([idx2])][0])
                    elif isinstance(idx1,list)==False and isinstance(idx2,list)==True:
                        pair_set.append([list([idx1])+idx2][0])
                    else:
                        pair_set.append([idx1,idx2])
                else:
                    pair_set.append([idx1,idx2])
    return pair_set


def plot_compare_sensors(sensor_names,X_Time,X_Feature,X_names):
    num_sensors=len(sensor_names)
    #sensor_name=data_used[k]
    fig = plt.figure('Compare')
    fig.suptitle('Compare')
    for k,sensor_name in enumerate(sensor_names):
        plt.subplot(num_sensors,1,k+1);
        plt.plot(X_Time,X_Feature[:,X_names.index(sensor_name)])
        plt.title(sensor_name)
    plt.get_current_fig_manager().window.showMaximized()

def plot_compare_states(x_idx,data_dict,X_Time,X_Feature,X_STATE,X_names):
    if X_STATE.shape!=X_Feature.shape:
        raise NameError('the size of state and feature matrix must be same')
    if (X_STATE.shape[0]!=X_Time.shape[0]):
        raise NameError('the row length of state /feature matrix and time array must be same')
    if (X_STATE.shape[1]!=len(X_names)):
        raise NameError('the column length of state and name array must be same')
    sensor_name=X_names[x_idx]
    fig = plt.figure('Regualar Event Classification')
    fig.suptitle('Regualar Event Classification');
    plt.subplot(3,1,1);
    plt.plot(unix_to_dtime(data_dict[sensor_name][2][0]),data_dict[sensor_name][2][1])
    plt.ylabel('Power, KWatt')
    plt.title(sensor_name+' - Measurements');
    plt.subplot(3,1,2);
    plt.plot(X_Time,X_Feature[:,x_idx]);
    plt.title(X_names[x_idx]+' - Hourly Average');
    plt.ylabel('Normalized Measurement')
    plt.subplot(3,1,3);
    low_peak_idx=np.nonzero(X_STATE[:,x_idx]==-1)[0]
    no_peak_idx=np.nonzero(X_STATE[:,x_idx]==0)[0]
    high_peak_idx=np.nonzero(X_STATE[:,x_idx]==1)[0]
    plt.plot(X_Time[low_peak_idx],X_STATE[low_peak_idx,x_idx],'rv');
    plt.plot(X_Time[high_peak_idx],X_STATE[high_peak_idx,x_idx],'b^');
    plt.plot(X_Time[no_peak_idx],X_STATE[no_peak_idx,x_idx],'g.');
    plt.plot(X_Time,X_STATE[:,x_idx]);
    plt.title(sensor_name+' - Classified States ');
    plt.ylabel('States'); plt.xlabel('Dates'); plt.ylim([-1.2,1.2])
    plt.yticks([-1, 0, 1], ['Low Peak', 'No Peak', 'High Peak'])
    plt.get_current_fig_manager().window.showMaximized()
    #time.sleep(3)
    #fig.savefig(fig_dir+'Reg_Event_Classification_'+input_names[k]+'.png')


def check_data_type(values):
    num_samples=len(values)
    num_values=len(set(values))
    if num_values==0:
        return None
        
    if num_samples>1000:
        if num_values<MIN_NUM_VAL_FOR_FLOAT:
            data_type=INT_TYPE
        else:
            data_type=FLOAT_TYPE
    else:
        comp_ratio=num_samples/num_values
        if comp_ratio>100:
            data_type=INT_TYPE
        else:
            data_type=FLOAT_TYPE
    return data_type
    
    
def grep(pattern,l):
    
    expr = re.compile(pattern)
    idx_list = [idx for idx in range(len(l)) if expr.search(l[idx])]
    #ret = [(idx,elem) for elem in l if expr.search(elem)]
    return list(set(idx_list))
    
    
    
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

def remove_dot(pname_):

    if isinstance(pname_, list) == True:
        pname_list = []
        for name_ in pname_:
            try:
                blank_idx = name_.index('.')
                name_ = name_.replace('.', '_')
            except Exception as e:
                pass

            try:
                blank_idx= name_.index(' ')
                name_= name_.replace(' ','_')
            except Exception as e:
                pass

            if name_[0].isdigit():
                name_='a_'+name_
            pname_list.append(name_)
        return pname_list

    else:
        try:
            blank_idx = pname_.index('.')
            pname_ = pname_.replace('.', '_')
        except Exception as e:
            pass

        try:
            blank_idx = pname_.index(' ')
            pname_ = pname_.replace(' ', '_')
        except Exception as e:
            pass

        if pname_[0].isdigit():
            pname_ = 'a_'+pname_
        return pname_

# cause sensor names
def get_data_set(f_names,start_t=[],end_t=[]):
    s_names={}
    for s_name in f_names:
        filename=DATA_DIR+s_name+FL_EXT
        data = mt.loadObjectBinary(filename)
        sensor_val = data["value"]
        time_val = data["ts"]
        if (start_t!=[]) and (end_t!=[]):
            temp_t=np.array([ t_[0] for t_ in time_val])
            s_t_idx=np.nonzero((temp_t>start_t) & (temp_t<end_t))[0]
            time_val=np.array(time_val)[s_t_idx]
            sensor_val=np.array(sensor_val)[s_t_idx]

        sensor_dtime=[time_val_[0] for time_val_ in time_val]
        temp_obj=obj({'time':sensor_dtime,'val':sensor_val})
        s_names.update({remove_dot(s_name):temp_obj})
    
    return obj(s_names)
    
    
def plot_data_x(data_x,stype='raw',smark='-',fontsize='small',xpos=0.5):
    plt.ioff()
    fig=plt.figure(figsize=(20.0,10.0))    
    sensor_names_x=data_x.__dict__.keys()
    num_plots=len(sensor_names_x)
    for k ,key in enumerate(sensor_names_x):
        plt.subplot(num_plots,1,k+1)
        if stype=='diff':
            t_=data_x.__dict__[key].time[1:]
            val_=abs(np.diff(data_x.__dict__[key].val))
            plt.title(key+'- Differential',fontsize=fontsize,x=xpos)
        else:
            t_=data_x.__dict__[key].time 
            val_=data_x.__dict__[key].val
            plt.title(key,fontsize=fontsize,x=xpos)
        
        plt.plot(t_,val_,smark)
        mn_=min(val_);mx_=max(val_)
        plt.ylim([mn_-0.1*abs(mn_),mx_+0.1*abs(mx_)])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    plt.close()
    plt.ion()
    return png_name
    
    
    
def print_cond_table(peak_state_temp, peak_prob_temp,cause_label):
    print '---------------------------------------------'
    print 'Conditoinal probability of PEAK given ', cause_label
    print '---------------------------------------------'

    peak_state_set=[[stateDict[s_]  for s_ in ss] for ss in peak_state_temp]
    print '----------------------------------------'
    print 'Conditional Probability'
    print '---------------     ---------------'
    print 'Sensor State        Cond.Prob of PEAK'
    print '---------------     ---------------'
    print pandas.DataFrame(np.array(peak_prob_temp),np.array(peak_state_set))
    print '----------------------------------------'
    
    
    
def plotting_feature_mat(all_psensors,X_names,X_Feature,start_t,end_t):
    for name_ in all_psensors:
        idx=grep(name_,X_names)
        dt_=X_Time
        val_=X_Feature[:,idx]
        s_t_idx=np.nonzero((np.array(dt_)>start_t) & (np.array(dt_)<end_t) )[0]    
        dt_=np.array(dt_)[s_t_idx]
        val_=np.array(val_)[s_t_idx]
        fig=figure(figsize=(20.0,10.0))
        if len(idx)>0:
            plt.plot(dt_,val_)
            plt.ylabel('Power',fontsize=18)
            plt.tick_params(labelsize='large')
            mn_=min(val_);mx_=max(val_)
            ylim([mn_-0.1*abs(mn_),mx_+0.1*abs(mx_)])
        title(name_+'- Feature value',fontsize=18)
        fig.savefig(fig_dir+name_+'_'+start_t.strftime("%B %d, %Y") +' - '+end_t.strftime("%B %d, %Y")+'.png')
        

def is_empty_idx(a):
    import pdb;pdb.set_trace()
    try:
        [ len(a_1) for a_1 in a].index(0)
        return True
    except:
        return False        
        
        
def get_pngid():
    png_id='_'+str(uuid.uuid4().get_hex().upper()[0:2])+'_'
    return png_id 