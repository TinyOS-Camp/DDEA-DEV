# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:28:13 2014

@author: deokwooj
"""
from __future__ import division # To forace float point division
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from sklearn import mixture
#from sklearn.cluster import Ward
from sklearn.cluster import KMeans
import time
##################################################################
# Custom library
##################################################################
from data_tools import *
from shared_constants import *
import pprint
import lib_bnlearn as rbn    

def X_INPUT_to_states(xinput,CORR_VAL_OUT=0,PARALLEL = False):
    #import pdb;pdb.set_trace()
    sinput=np.zeros(xinput.shape)
    num_samples=xinput.shape[0]
    num_sensors=xinput.shape[1]
    if num_samples <num_sensors:
        print 'Warning number of samplesa are smaller than number of sensors'

    print 'Mapping',xinput.shape, ' marix to discrete states '
    for k,samples in enumerate(xinput.T):
        obs=samples[:,np.newaxis]
        label,opt_num_cluster,model,score,score_err_sum= state_retrieval(obs,max_num_cluster=6,est_method='kmean',PARALLEL=PARALLEL,PRINTSHOW=False)
        high_peak_label_idx=np.argmax(model.cluster_centers_)
        low_peak_label_idx=np.argmin(model.cluster_centers_)
        high_peak_idx=np.nonzero(label==high_peak_label_idx)[0]
        sinput[high_peak_idx,k]=1
        low_peak_idx=np.nonzero(label==low_peak_label_idx)[0]
        sinput[low_peak_idx,k]=-1
    
    corr_state_val=[]    
    if CORR_VAL_OUT==1:
        print 'Compute Correlation Score....'
        for k,(row1, row2) in enumerate(zip(sinput.T, xinput.T)):
            corr_state_val.append(round(stats.pearsonr(row1,row2)[0],3))
    corr_state_val=np.array(corr_state_val)
    return sinput,corr_state_val
    
def interpolation_measurement(data_dict,input_names,err_rate=1,sgm_bnd=20):
    print 'interploattion starts....'
    measurement_point_set=[]
    num_of_discrete_val=[]
    sampling_interval_set=[]
    num_type_set=[]
    err_rate=1;sgm_bnd=20
    """
    try:
        import pdb;pdb.set_trace()
    except ValueError:
        import pdb;pdb.set_trace()
    """
    for i,key_name in enumerate(input_names):
        print key_name,'.....'
        t_=np.array(data_dict[key_name][2][0])
        
        if len(t_) == 0:
            continue
        
        intpl_intv=np.ceil((t_[-1]-t_[0]) /len(t_))
        sampling_interval_set.append(intpl_intv)
        val_=np.array(data_dict[key_name][2][1])
        num_of_discrete_val_temp=len(set(val_))
        num_of_discrete_val.append(num_of_discrete_val_temp)
        # filtering outlier
        # assuming 1% of errors and 30 x standard deviation rules
        outlier_idx=outlier_detect(val_,err_rate,sgm_bnd)
        if len(outlier_idx)>0:
            print 'outlier samples are detected: ', 'outlier_idx:', outlier_idx
            t_=np.delete(t_,outlier_idx)
            val_=np.delete(val_,outlier_idx)
        t_new=np.r_[t_[0]:t_[-1]:intpl_intv]
        """        
        if num_of_discrete_val_temp<MIN_NUM_VAL_FOR_FLOAT:
            num_type=INT_TYPE
            val_new=fast_nearest_interp(t_new, t_,val_)
        else:
            num_type=FLOAT_TYPE
            val_new = np.interp(t_new, t_,val_)
        """
        num_type=check_data_type(data_dict[key_name][2][1])
        if num_type==INT_TYPE:
            val_new=fast_nearest_interp(t_new, t_,val_)
        else:
            #num_type=FLOAT_TYPE
            val_new = np.interp(t_new, t_,val_)
        
        c=np.vstack([t_new,val_new])
        measurement_point_set.append(c)
        num_type_set.append(num_type)
        print '-----------------------------------------------------------------'
    #return measurement_point_set,num_type_set,num_of_discrete_val,sampling_interval_set
    return measurement_point_set,np.array(num_type_set)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window_label_mode(label,r_window):
    if (r_window/2)==int(r_window/2):
        r_window=int(r_window+1)
        #raise NameError('length of window size must be odd')
    offset=int(r_window/2)
    rw_label_temp=stats.mode(rolling_window(label, r_window),1)[0]
    head= rw_label_temp[0]*np.ones([offset,1])
    body=rw_label_temp
    tail= rw_label_temp[-1]*np.ones([offset,1])
    rw_label=np.r_[head,body,tail]
    return rw_label

def rolling_window_label_binary(label,r_window):
    if (r_window/2)==int(r_window/2):
        r_window=int(r_window+1)
        #raise NameError('length of window size must be odd')
    offset=int(r_window/2)
    rw_label_temp=np.array([ np.sum(temp)/r_window for temp in rolling_window(label, r_window)])
    #import pdb;pdb.set_trace()
    # rw_label_temp=stats.mode(rolling_window(label, r_window),1)[0]
    head= rw_label_temp[0]*np.ones([offset,1])
    body=rw_label_temp
    tail= rw_label_temp[-1]*np.ones([offset,1])
    rw_label=np.r_[head,body[:,np.newaxis],tail]
    return rw_label

"""


def state_retrieval(obs,max_num_cluster=6,est_method='kmean'):
    #print '========================================================================='    
    #print 'Retrieving discrete states from data using ',est_method, ' model...'
    #print '========================================================================='
    score=np.zeros(max_num_cluster)
    model_set=[]
    #print 'try ',max_num_cluster, ' clusters..... '
    for num_cluster in range(max_num_cluster):
        #print 'Try ',num_cluster+1, ' clusters '
        #print '-----------------------------------'
        if est_method=='kmean':
            kmean=KMeans(n_clusters=num_cluster+1).fit(obs)
            model_set.append(kmean)
            #import pdb;pdb.set_trace()
            score[num_cluster]=np.sum(kmean.score(obs))
        elif est_method=='gmm':
            gmm = mixture.GMM(n_components=num_cluster+1).fit(obs) 
            model_set.append(gmm)
            score[num_cluster]=np.sum(gmm.score(obs))
        else:
            raise NameError('not supported est_method')
    score_err_sum=np.zeros(max_num_cluster)
    #print 'Finding knee points of log likelihood...'
    for i in range(max_num_cluster):
        a_0=score[:(i)]
        if len(a_0)>1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(a_0)),a_0)
            sqr_sum_err0=sum(((slope*np.arange(len(a_0))+ intercept)-a_0)**2)
        else:
            sqr_sum_err0=0
        a_1=score[(i):]
        if len(a_1)>1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(a_1)),a_1)
            sqr_sum_err1=sum(((slope*np.arange(len(a_1))+ intercept)-a_1)**2)
        else:
            sqr_sum_err1=0
        score_err_sum[i]=sqr_sum_err0+sqr_sum_err1
    # Optimum number of clusters. 
    min_idx=np.argmin(score_err_sum)
    opt_num_cluster=min_idx+1
    #print 'opt_num_cluster: ' , opt_num_cluster
    if est_method=='kmean':
        label=model_set[min_idx].labels_
    elif est_method=='gmm':
        label=model_set[min_idx].predict(obs)
    else:
        raise NameError('not supported est_method')
    return label,opt_num_cluster, model_set[min_idx],score,score_err_sum

"""


def cluster_state_retrieval(tup):
    obs = tup[0]
    num_clusters = tup[1]
    est_method = tup[2]
    #print 'num clusters = ' + str(num_clusters)
    if est_method=='kmean':
        kmean=KMeans(n_clusters=num_clusters).fit(obs)
        model = kmean
        score=compute_log_ll(kmean.labels_,obs)
        #score=-1*np.log(-1*np.sum(kmean.score(obs)))
    elif est_method=='gmm':
        gmm = mixture.GMM(n_components=num_clusters).fit(obs)
        model = gmm
        score=np.sum(gmm.score(obs))
    #print 'Done ' + str(num_clusters)        
    return (num_clusters-1, [model,score])
from multiprocessing import Pool    


def compute_log_ll(label_in,obs_in):
    log_ll_sum=0
    for i in range(label_in.max()+1):
        idx=np.nonzero(label_in==i)[0]
        val_set=obs_in[idx]
        log_val=stats.norm.logpdf(val_set,loc=np.mean(val_set),scale=np.std(val_set))
        log_ll_sum=log_ll_sum+sum(log_val[log_val!=-np.inf])
    return log_ll_sum


def state_retrieval(obs,max_num_cluster=6,off_set=0,est_method='kmean',PARALLEL = False,PRINTSHOW=False):
    if PRINTSHOW==True:    
        print '========================================================================='
        print 'Retrieving discrete states from data using ',est_method, ' model...'
        print '========================================================================='
        print 'try ',max_num_cluster, ' clusters..... '
    score=np.zeros(max_num_cluster)
    model_set=[]
    if not PARALLEL:
        for num_cluster in range(max_num_cluster):
            #print 'Try ',num_cluster+1, ' clusters '
            #print '-----------------------------------'
            if est_method=='kmean':
                kmean=KMeans(n_clusters=num_cluster+1).fit(obs)
                model_set.append(kmean)
                #import pdb;pdb.set_trace()
                #score[num_cluster]=-1*np.log(-1*np.sum(kmean.score(obs)))
                #score[num_cluster]=kmean.score(obs)
                #score[num_cluster]=kmean.score(obs)-.5*(num_cluster+1)*1*log10(len(obs)) 
                #log_ll_val=compute_log_ll(kmean.labels_,obs)
                score[num_cluster]=compute_log_ll(kmean.labels_,obs)
            elif est_method=='gmm':
                gmm = mixture.GMM(n_components=num_cluster+1).fit(obs)
                model_set.append(gmm)
                score[num_cluster]=np.sum(gmm.score(obs))
            else:
                raise NameError('not supported est_method')
    else:
        if PRINTSHOW==True:    
            print 'Parallel enabled...'
        model_set = [0] * max_num_cluster
        score = [0] * max_num_cluster
        p = Pool(max_num_cluster)
        params = [(obs,i+1,est_method) for i in range(max_num_cluster)]
        model_dict = dict(p.map(cluster_state_retrieval,params))
        for k,v in model_dict.iteritems():
            model_set[k] = v[0]
            score[k] = v[1]
        p.close()
        p.join()
    
    score_err_sum=np.zeros(max_num_cluster)
    if PRINTSHOW==True:    
        print 'Finding knee points of log likelihood...'
    for i in range(max_num_cluster):
        a_0=score[:(i)]
        if len(a_0)>1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(a_0)),a_0)
            sqr_sum_err0=sum(((slope*np.arange(len(a_0))+ intercept)-a_0)**2)
        else:
            sqr_sum_err0=0
        a_1=score[(i):]
        if len(a_1)>1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(a_1)),a_1)
            sqr_sum_err1=sum(((slope*np.arange(len(a_1))+ intercept)-a_1)**2)
        else:
            sqr_sum_err1=0
        score_err_sum[i]=sqr_sum_err0+sqr_sum_err1
    # Optimum number of clusters.
    min_idx=np.argmin(score_err_sum)
    opt_num_cluster=min_idx+1
    if PRINTSHOW==True:    
        print 'opt_num_cluster: ' , opt_num_cluster
    if est_method=='kmean':
        label=model_set[min_idx].labels_
    elif est_method=='gmm':
        label=model_set[min_idx].predict(obs)
    else:
        raise NameError('not supported est_method')
    return label,opt_num_cluster, model_set[min_idx],score,score_err_sum


########################################################################
# Function Irregualr event table retrieval
########################################################################
def mesurement_to_states(measurement_point_set,alpha=0.5,max_num_cluster=8,est_method='kmean',PARALLEL=False):
    print '==============================================================================='      
    print 'Mapping measurement to states by ', est_method, ', Parallel Enabled: ',str(PARALLEL)
    print '==============================================================================='      
    model_set=[]
    label_set=[]
    irr_event_set=[]
    start_t=time.time()
    for k,data_set in enumerate(measurement_point_set):
        print 'working on ',k,'th measurement point... '
        val_new=data_set[1]
        val_set=list(set(val_new))
        num_of_discrete_val=len(val_set)
        t_new=data_set[0]
        # average sampling interval
        sr=(t_new[-1]-t_new[0]) /len(t_new)
        # transformed observatoin data for state retrieval
        if num_of_discrete_val<10:
            print 'the number of discrete values are less than 10'
            print 'no states retrieval needed '
            cnt_num_occurances=[len(np.nonzero(val_new==state)[0]) for state in val_set]
            #import pdb;pdb.set_trace()
            label=val_new
            label_set.append(np.vstack([t_new, label]))
            min_label_idx=val_set[np.argmin(cnt_num_occurances)]
            irregualr_event=np.zeros(label.shape)
            irregualr_event[label==min_label_idx]=1
            
        elif num_of_discrete_val<100:
            print 'the number of discrete values are less than 100'
            print 'use K-MEAN clustering by default '
            obs=abs(np.diff(val_new))[:,np.newaxis]
            label,opt_num_cluster,model,score,score_err_sum=state_retrieval(obs,max_num_cluster,est_method='kmean',PARALLEL=PARALLEL,PRINTSHOW=False)
            max_label_idx=np.argmax(model.cluster_centers_)
            max_label=np.zeros(label.shape)
            max_label[label==max_label_idx]=1
            irregualr_event=np.r_[max_label[0],max_label]

        else:
            obs=abs(np.diff(val_new))[:,np.newaxis]
            label,opt_num_cluster,model,score,score_err_sum=state_retrieval(obs,max_num_cluster,est_method=est_method,PARALLEL=PARALLEL,PRINTSHOW=False)
            #import pdb;pdb.set_trace()
            if est_method=='kmean':
                #label,opt_num_cluster,model,score,score_err_sum=state_retrieval_kmean(obs,max_num_cluster)
                max_label_idx=np.argmax(model.cluster_centers_)
            elif est_method=='gmm':
                #label,opt_num_cluster,model,score,score_err_sum=state_retrieval(obs,max_num_cluster)
                max_label_idx=np.argmax(model.means_)
            else:
                raise NameError('not supported est_method')
            model_set.append(model)    
            label_set.append(np.vstack([t_new[1:], label]))
            # Irregualr state mapping
            max_label=np.zeros(label.shape)
            max_label[label==max_label_idx]=1
            irregualr_event=np.r_[max_label[0],max_label]
        
        irregualr_event_inter_arr_times=np.diff(t_new[irregualr_event==1])
        if (len(irregualr_event_inter_arr_times)>10) and (num_of_discrete_val>10):
            loc_x, scale_x =stats.expon.fit(irregualr_event_inter_arr_times)
            inter_arr_times_alpha=stats.expon.ppf(alpha,loc=loc_x,scale=scale_x)
            window_size=int(inter_arr_times_alpha/sr)
            rw_irregualr_event=rolling_window_label_binary(irregualr_event,window_size)[:,0]
            irr_event_set.append(np.vstack([t_new, rw_irregualr_event]))
        else:
            irr_event_set.append(np.vstack([t_new, irregualr_event]))
    
    end_proc_t=time.time()
    print 'the time of processing mesurement_to_states ', end_proc_t-start_t, ' sec'
    return irr_event_set
#########################################################################

#########################################################################
# Binary Table Extraction
#########################################################################
def get_common_time_reference(ts_list):
    list_len = len(ts_list)
    start_ts_list = np.array([ts_list[i][0] for i in range(list_len)])
    end_ts_list = np.array([ts_list[i][-1] for i in range(list_len)])

    common_start_ts = np.max(start_ts_list)
    common_end_ts = np.min(end_ts_list)

    common_ts = []
    for i in range(list_len):
        #common_ts = common_ts + ts_list[i]
        common_ts = np.hstack([common_ts,ts_list[i]])
    # remove duplicated ts
    common_ts = np.asarray(sorted(list(set(common_ts))))

    common_ts = np.delete(common_ts,np.nonzero(common_ts < common_start_ts)[0])
    common_ts = np.delete(common_ts,np.nonzero(common_ts > common_end_ts)[0])

    return common_ts

def interpolate_state_nearest(available_ts,available_values, intplt_ts):

    f =  interp1d(available_ts,available_values,kind='nearest')
    ## Interpolation
    print 'len of intplt points: ' + str(len(intplt_ts))
    intplt_values = f(intplt_ts)

    return intplt_values

def find_consecutive_dup_rows(mat):
    nrows = len(mat)
    dup_idx_list = []
    for r_idx in range(nrows-1,0,-1):
        if all(mat[r_idx] == mat[r_idx-1]):
            dup_idx_list.append(r_idx)
    return dup_idx_list

def binary_table_extract(irr_event_set, binary_state_cut_off=-1, rm_dup=False):
    #print 'this extract binary state based on state transition of composit binary states (default) or reference time using interploation '
    #print 'return N-by-P matrix where N is the number of transitions and P is the number of sensors'
    num_of_sensors=len(irr_event_set)
    #num_of_transition=1 # to be updated
    state_table = []       
    """
        Steps to find state transition
        1. Find the common time frame of all sensors
            start = max{ts[0]} for all ts: list of time of each sensor
            end = min{ts[-1]} for all ts: list of time of each sensor
        2. Find all ts that at least one sensor data available [start,end]
            TS = Union of all ts, within [start_end]
            
        3. Interpolate sensor state for each sensor during [start,end]
            Before interpolation, convert states into binary (optional)
        4. Remove duplicated state transitions (optional)

    """
    ### Step 1+2: Get common time reference
    ts_list = []
    for i in range(num_of_sensors):
        ts_list.append(irr_event_set[i][0])
    print ts_list
    common_ts = get_common_time_reference(ts_list)


    ### interpolate state for each sensor, during common_ts
    for i in range(num_of_sensors):

        # convert state probability to binary state
        if (binary_state_cut_off >= 0):
            positive_prob_idx=np.nonzero(irr_event_set[i][1] > binary_state_cut_off)[0]
            irr_event_set[i][1][:]=0
            irr_event_set[i][1][positive_prob_idx]=1

        intplt_states = interpolate_state_nearest(irr_event_set[i][0],irr_event_set[i][1],common_ts)
        state_table.append(intplt_states)

    state_table = np.asarray(state_table).T
    # column: sensor, row: state sample
    ### Remove duplicated state transitions
    if rm_dup==True:
        dup_idx_list = find_consecutive_dup_rows(state_table)
        state_table = np.delete(state_table,dup_idx_list,axis=0)        
        common_ts = np.delete(common_ts,dup_idx_list)
        
    return common_ts,state_table
    


###################################################################################
# Probability Computation Functions
###################################################################################
# Example Codes
###################################################################################
# data_mat_set=np.array([[1,1,0],[1,1,0],[0,1,0],[1,1,1],[0,1,0],[1,0,0],[0,0,0],[0,1,0]])
# data_mat_set2=np.array([[1,11,100],[1,11,100],[0,11,100],[1,11,101],[0,11,100],[1,10,100],[0,10,100],[0,11,100]])
#data_mat_set=np.array([[1,1,0],[1,1,0],[0,1,0],[1,1,1],[0,1,0],[1,0,0],[0,0,0],[0,1,0]])
#compute_joint_prob(data_mat_set,[0,1],[[0,1],[0]])
#compute_cond_prob(data_mat_set,[0],[[1]],[1],[[1]])
#state_tmp,prob_tmp=compute_effect_prob(data_mat_set,[0],[1],[[1]])
#state_tmp,likelihood_tmp=compute_cause_likelihood(data_mat_set,[0],[1],[[1]])


def compute_joint_prob(data_mat,state_idx_set,state_val_set):
    num_samples=data_mat.shape[0]
    num_states=data_mat.shape[1]
    if len(state_idx_set)!=len(state_val_set):
        raise NameError('the length of state_set and state_val must be same')
    joint_idx=set(range(num_samples))
    for k,state_idx in enumerate(state_idx_set):
        samples=data_mat[:,state_idx]
        sub_joint_idx=set([])
        for state_val in state_val_set[k]:
            sub_joint_idx=sub_joint_idx| set(np.nonzero(samples==state_val)[0])
        joint_idx=joint_idx & sub_joint_idx
    joint_prob=len(joint_idx)/num_samples
    if num_samples==0:
        return 0
    else:
        return joint_prob
    
        
    
#def compute_cond_prob(data_mat,state_idx_set,state_val_set,cond_idx_set):
def compute_cond_prob(data_mat,state_idx_set,state_val_set,cond_idx_set,cond_val_set):
    joint_state_idx_set=state_idx_set+cond_idx_set
    joint_state_val_set=state_val_set+cond_val_set
    all_prob=compute_joint_prob(data_mat,joint_state_idx_set,joint_state_val_set)
    partial_prob=compute_joint_prob(data_mat,cond_idx_set,cond_val_set)
    if partial_prob==0:
        return 0
    else:
        return all_prob/partial_prob
    

def compute_effect_prob(data_mat,effect_idx_set,cause_idx_set,cause_val_set):
    # find f_B*(A)=P(A|B=B*)
    # generate a set of all possible states
    state_set=[]
    for k,idx in enumerate(effect_idx_set):
        #print idx, ':', list(set(data_mat[:,idx]))
        #set(list(data_mat[idx,:]))
        if k==0:            
            state_set=list(set(data_mat[:,idx]))
        else:
            state_set=pair_in_idx(state_set,list(set(data_mat[:,idx])))
    prob_set=[]                
    for state_val in state_set:
        #import pdb;pdb.set_trace()
        if isinstance(state_val,list):
            input_val_set=[[val] for val in state_val]
        else:
            input_val_set=[[state_val]]
        
        prob_temp=compute_cond_prob(data_mat,effect_idx_set,input_val_set,cause_idx_set,cause_val_set)
        prob_set.append(prob_temp)
    
    return state_set,prob_set


def compute_cause_likelihood(data_mat,cause_idx_set,effect_idx_set,effect_val_set):
    # find f_A*(B)=P(A=A*|B)
    # generate a set of all possible states
    state_set=[]
    for k,idx in enumerate(cause_idx_set):
        #print idx, ':', list(set(data_mat[:,idx]))
        #set(list(data_mat[idx,:]))
        #import pdb;pdb.set_trace()
        if k==0:            
            state_set=list(set(data_mat[:,idx]))
        else:
            state_set=pair_in_idx(state_set,list(set(data_mat[:,idx])))
            
    likelihood_set=[]                
    for state_val in state_set:
        #import pdb;pdb.set_trace()
        if isinstance(state_val,list):
            input_val_set=[[val] for val in state_val]
        else:
            input_val_set=[[state_val]]
        prob_temp=compute_cond_prob(data_mat,effect_idx_set,effect_val_set,cause_idx_set,input_val_set)
        likelihood_set.append(prob_temp)
    
    return state_set,likelihood_set
    
def irr_state_mapping(state_mat,weight_coeff=10):
    peak_prob=np.array([compute_joint_prob(state_mat,[k],[[PEAK]]) for k in range(state_mat.shape[1])])
    low_prob=np.array([compute_joint_prob(state_mat,[k],[[LOW_PEAK]]) for k in range(state_mat.shape[1])])
    no_prob=np.array([compute_joint_prob(state_mat,[k],[[NO_PEAK]]) for k in range(state_mat.shape[1])])
    irr_state_prob=np.zeros(state_mat.shape[1])
    irr_state_mat=np.zeros(state_mat.shape)
    skewness_metric_sort=np.zeros(peak_prob.shape)
    
    idx_state_map=[PEAK,NO_PEAK,LOW_PEAK]
    for k,prob_set in enumerate(np.vstack([peak_prob,no_prob,low_prob]).T):
        # Processing probaiblity data for each sensor 
        prob_sort_idx=np.argsort(prob_set) 
        prob_sort=prob_set[prob_sort_idx]
        #import pdb;pdb.set_trace()    
#        if k==16:
 #           import pdb;pdb.set_trace()
        if weight_coeff*(prob_sort[0]+prob_sort[1]) <prob_sort[2]:
            irr_prob=prob_sort[0]+prob_sort[1]
            reg_prob=prob_sort[2]
            irr_state_mat[(state_mat[:,k]==idx_state_map[prob_sort_idx[0]]) | (state_mat[:,k]==idx_state_map[prob_sort_idx[1]]),k]=1
        else:
            irr_prob=prob_sort[0]
            reg_prob=prob_sort[1]+prob_sort[2]
            irr_state_mat[state_mat[:,k]==idx_state_map[prob_sort_idx[0]],k]=1
        
        temp=abs(irr_prob-reg_prob)/np.sqrt(reg_prob*irr_prob)
        if temp<np.inf:
            skewness_metric_sort[k]=temp
            irr_state_prob[k]=irr_prob    
    desc_sort_idx=np.argsort(-1*skewness_metric_sort)
    return irr_state_mat
    #return irr_state_mat,irr_state_prob,skewness_metric_sort[desc_sort_idx],desc_sort_idx
    
###################################################################################
# Probability Analysis Functions
###################################################################################

def time_effect_analysis(data_mat,data_name,avgtime_names,s_name,DO_PLOT=False):
    s_idx=data_name.index(s_name)
    t_idx=[[data_name.index(ntemp)] for ntemp in avgtime_names] #['MTH', 'WD', 'HR']
    m_list=list(set(data_mat[:,data_name.index('MTH')]))
    state_list=list(set(data_mat[:,s_idx]))
    s_prob_log=[[]]*len(yearMonths)
    print 'Monthy analysis...'
    for m_idx in yearMonths:
        print monthDict[m_idx]
        if m_idx not in m_list:
            print 'no data for this month'
            print '-----------------------------'
            continue
        prob_map=np.zeros([len(state_list),len(Week),len(DayHours)])
        #for h_idx in DayHours:
        start_t=time.time()
        for dh_pair in pair_in_idx(Week,DayHours):
            #state_tmp,prob_tmp=compute_effect_prob(data_mat,[s_idx],t_idx,[[m_idx],Weekday,[h_idx]])
            state_tmp,prob_tmp=compute_effect_prob(data_mat,[s_idx],t_idx,[[m_idx],[dh_pair[0]],[dh_pair[1]]])
            for state in state_list:
                prob_map[state_list.index(state) ,dh_pair[0],dh_pair[1]]=prob_tmp[state_tmp.index(state)]
        end_t=time.time()
        print 'spend ' ,end_t-start_t,'secs'
        s_prob_log[m_idx]=prob_map
        #m_prob_log
        print '-----------------------------'
        #s_m_data_valid=[ False if sum(prob)==0 else True for prob in s_prob_log]
    valid_mon_list=[month_val for month_val in yearMonths if len(s_prob_log[month_val])>0]
    if DO_PLOT==True:
        plot_time_effect(s_name,state_list,valid_mon_list,s_prob_log)    
    
    valid_mon_pair=pair_in_idx(valid_mon_list)
    time_effect_mat_dist=np.zeros([len(state_list),len(valid_mon_pair)])
    for i,state_idx in enumerate(range(len(state_list))):    
        for j,mon_idx_pair in enumerate(valid_mon_pair):
            val_temp=norm(np.array(s_prob_log[mon_idx_pair[0]][state_idx])-np.array(s_prob_log[mon_idx_pair[1]][state_idx]))
            time_effect_mat_dist[i,j]=val_temp
    score_in_structure=[]
    for k,mon_idx in enumerate(valid_mon_list):
        score_val=[]
        for state_idx,state_val in enumerate(state_list):
            mat_input=np.array(s_prob_log[mon_idx][state_idx])
            dst_col=find_norm_dist_matrix(mat_input)
            dst_row=find_norm_dist_matrix(mat_input.T)
            score_val.append(dst_col.mean()+dst_row.mean())
        score_in_structure.append(np.sum(score_val))
    return state_list,s_prob_log,time_effect_mat_dist,score_in_structure,valid_mon_list,state_list
    

def plot_time_effect(s_name,state_list,valid_mon_list,s_prob_log):
    plt.figure(s_name)
    for i,state_val in enumerate(state_list):    
        for j, mon_val in enumerate(valid_mon_list):
            plt_idx=len(valid_mon_list)*i+j+1
            plt.subplot(len(state_list),len(valid_mon_list),plt_idx)
            im = plt.imshow(s_prob_log[mon_val][state_list.index(state_val)],interpolation='none',vmin=0, vmax=1,aspect='auto')
            if set(stateDict.keys())==set(state_list):
                plt.title(monthDict[mon_val]+' , state: '+ stateDict[state_val])
            else:
                plt.title(monthDict[mon_val]+' , state: '+ str(state_val))
            plt.yticks(weekDict.keys(),weekDict.values())
            plt.colorbar()
            #plt.xlabel('Hours of day')
            if i == len(state_list) - 1:
                plt.xlabel('Hours of day')
    
    #plt.subplots_adjust(right=0.95)
    #cbar_ax = plt.add_axes([0.95, 0.15, 0.05, 0.7])
    #cax,kw = mpl.colorbar.make_axes([ax for ax in pl   t.axes().flat])
    #plt.colorbar(im, cax=cax, **kw)
    #plt.colorbar(im,cbar_ax)
                    
                
def time_effect_analysis_all(data_mat,data_name,avgtime_names,avgsensor_names):
    monthly_structure_score=[]
    monthly_variability=[]
    for s_name in avgsensor_names:
        print s_name
        print '==============================='
        state_list,s_prob_log,time_effect_mat_dist,score_in_structure,valid_mon_list,state_list\
        =time_effect_analysis(data_mat,data_name,avgtime_names,s_name,DO_PLOT=False)
        monthly_variability.append(time_effect_mat_dist.mean())
        monthly_structure_score.append(score_in_structure)
    return np.array(monthly_variability),np.array(monthly_structure_score)


###############################################################################################    
# Analysis- Sensitivity of state distribution for parameters\
# Use Bhattacharyya distance  to compute the distance between two probabilities
#D_b(p,q)= - ln (BC(p,q)) where BC(p,q)=\sum_x \sqrt{p(x)q(x)}
# Due to triagnular propety we use Hellinger distance , D_h(p,q)=\sqrt{1-BC(p,q)}
###############################################################################################    
def param_sensitivity(data_mat, data_name,avgsensor_names,wfactor,dst_type):
    wfactor_prob_map=[]
    wfactor_state_map=[]
    wfactor_sensitivity=[]
    wfactor_idx=data_name.index(wfactor)
    wfactor_list=list(set(data_mat[:,wfactor_idx]))
    for k,s_name in enumerate(avgsensor_names):
        s_idx=data_name.index(s_name)
        state_list=list(set(data_mat[:,s_idx]))
        prob_map=np.zeros([len(state_list),len(wfactor_list)])
        state_map=np.zeros([len(state_list),len(wfactor_list)])
        for i,wfactor_state in enumerate(wfactor_list):
            state_tmp,prob_tmp=compute_effect_prob(data_mat,[s_idx],[[wfactor_idx]],[[wfactor_state]])
            state_map[:,i]=state_tmp
            prob_map[:,i]=prob_tmp
        wfactor_prob_map.append(np.round(prob_map,2))
        wfactor_state_map.append(state_map)
        D_=[]
        for probset in  pair_in_idx(prob_map.T,prob_map.T):
            BC=min(1,sum(np.sqrt(probset[0]*probset[1])))
            if dst_type=='b':
                D_.append(-1*np.log(BC))
            elif dst_type=='h':
                D_.append(np.sqrt(1-BC))
            elif dst_type=='v':
                D_.append(0.5*min(1,sum(abs(probset[0]-probset[1]))))
            else:
                print 'error'; return
        #import pdb;pdb.set_trace()
        #BC=np.min(1,sum(np.sqrt(probset[0]*probset[1])))
        #if dst_type=='b':
        #    D_=[-1*np.log(np.min(1,sum(np.sqrt(probset[0]*probset[1])))) for probset in  pair_in_idx(prob_map.T,prob_map.T)]
        #elif dst_type=='h':
        #    D_=[np.sqrt(1-np.min(1,sum(np.sqrt(probset[0]*probset[1])))) for probset in  pair_in_idx(prob_map.T,prob_map.T)]
        #else:
        #    print 'error'; return
        wfactor_sensitivity.append(np.mean(D_))
    return wfactor_prob_map,wfactor_state_map, wfactor_sensitivity,wfactor_list
###############################################################################################    

def plot_weather_sensitivity(wf_type,wf_prob_map,wf_state_map,wf_sensitivity,wf_list,\
                             avgsensor_names,Conditions_dict,Events_dict,sort_opt='desc',num_of_picks=9):
    # Plotting bar graph
    if sort_opt=='desc':
        argsort_idx=np.argsort(wf_sensitivity)[::-1]
    elif sort_opt=='asc':
        argsort_idx=np.argsort(wf_sensitivity)
    else:
        print 'error in type'
        return 
    wf_sort_idx=np.argsort(wf_list)
    width = 0.5       # the width of the bars
    color_list=['b','g','r','c','m','y','k','w']
    num_col=floor(np.sqrt(num_of_picks))
    num_row=ceil(num_of_picks/num_col)
    for i in range(num_of_picks):
        subplot(num_col,num_row,i+1)
        bar_idx=argsort_idx[i]
        prob_bar_val=wf_prob_map[bar_idx]
        prob_bar_name=avgsensor_names[bar_idx]
        prob_bar_wf_state=[str(wf) for wf in np.array(wf_list)[wf_sort_idx]]
        prob_bar_sensor_state=wf_state_map[bar_idx]
        N =prob_bar_sensor_state.shape[0]
        M =prob_bar_sensor_state.shape[1]
        ind = np.arange(N)  # the x locations for the groups
        state_ticks=[]
        state_label=[]
        for k,(val,state) in enumerate(zip(prob_bar_val[:,wf_sort_idx].T,prob_bar_sensor_state[:,wf_sort_idx].T)):
            x=ind+k*5
            x_sort_idx=np.argsort(state)
            bar(x, val[x_sort_idx], width, color=color_list[k%len(color_list)])
            state_ticks=state_ticks+list(x)
            state_label=state_label+list(state[x_sort_idx].astype(int))
            #category_ticks=category_ticks+[int(mean(x))]
            if wf_type=='T':
                start_str='TP';end_str='C'
                statek=prob_bar_wf_state[k];fontsize_val=10
                init_str=start_str+'=  '+statek+ end_str
            elif wf_type=='D':
                start_str='DP';end_str='C'
                statek=prob_bar_wf_state[k];fontsize_val=10
                init_str=start_str+'=  '+statek+ end_str
            elif wf_type=='H':
                start_str='HD';end_str='%'
                statek=prob_bar_wf_state[k];fontsize_val=10
                init_str=start_str+'=  '+statek+ end_str
            elif wf_type=='E':
                start_str='EV';end_str=''
                statek=\
                Events_dict.keys()[Events_dict.values().index(int(prob_bar_wf_state[k]))];fontsize_val=6
                if statek=='': statek='none'
                #statek=prob_bar_wf_state[k];fontsize_val=10
                init_str=start_str+'=  '+statek+ end_str
            elif wf_type=='C':
                start_str='CD';end_str=''
                statek=prob_bar_wf_state[k];fontsize_val=10                    
                #statek=\
                #Conditions_dict.keys()[Conditions_dict.values().index(int(prob_bar_wf_state[k]))];fontsize_val=6
                if statek=='': statek='none'
                init_str=''
            else:
                print 'no such type'
                return                    
            if k==0:
                category_str= init_str
            else:
                category_str=statek+ end_str
            plt.text(int(mean(x)),1.1,category_str,fontsize=fontsize_val)
        plt.xticks(state_ticks,state_label )
        plt.xlabel('State',fontsize=10)
        plt.ylabel('Probability',fontsize=10)
        ylim([0,1.3]); title(prob_bar_name,fontsize=10)
        
        
        
        
def wt_sensitivity_analysis(data_state_mat,data_time_mat,data_weather_mat,sensor_names,time_names,\
Conditions_dict,Events_dict,bldg_tag,trf_tag,weather_names,dict_dir,dst_t='h'):
    import pprint
    import radar_chart
    data_mat = np.hstack([data_state_mat,data_time_mat])
    data_name = sensor_names+time_names
    
    print 'Parameter sensitivty for Months....'
    mth_prob_map,mth_state_map, mth_sensitivity,mth_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'MTH',dst_type=dst_t)   
    print 'Parameter sensitivty for Days....'
    wday_prob_map,wday_state_map,wday_sensitivity,wday_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'WD',dst_type=dst_t)   
    print 'Parameter sensitivty for Hours....'
    dhr_prob_map,dhr_state_map,dhr_sensitivity,dhr_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'HR',dst_type=dst_t)   

    #Month Sensitivty bar Plot. 
    tf_tuple_mth=('MTH',mth_prob_map,mth_state_map,mth_sensitivity,mth_list)
    tf_tuple_wday=('WD',wday_prob_map,wday_state_map,wday_sensitivity,wday_list)
    tf_tuple_dhr=('HR',dhr_prob_map,dhr_state_map,dhr_sensitivity,dhr_list)


    tf_sstv_tuple=np.array([tf_tuple_mth[3],tf_tuple_wday[3],tf_tuple_dhr[3]])
    max_tf_sstv=tf_sstv_tuple[tf_sstv_tuple<np.inf].max()*2
    tf_sstv_tuple[tf_sstv_tuple==np.inf]=max_tf_sstv
    tf_sstv_total=np.sum(tf_sstv_tuple,0)
    arg_idx_s=np.argsort(tf_sstv_total)[::-1]
    arg_idx_is=np.argsort(tf_sstv_total)
    num_of_picks=9
    print 'Most time sensitive sensors'
    print '---------------------------------------------'
    Time_Sensitive_Sensors=list(np.array(sensor_names)[arg_idx_s[0:num_of_picks]])
    pprint.pprint(Time_Sensitive_Sensors)
    
    ####################################################################
    ## Rador Plotting for Hour_Sensitive_Sensors
    ####################################################################
    sensor_no = len(sensor_names)
    # convert 'inf' to 1
    sen_mth = [max_tf_sstv if val == float("inf") else val for val in tf_tuple_mth[3]]
    sen_wday = [max_tf_sstv if val == float("inf") else val for val in tf_tuple_wday[3]]
    sen_dhr = [max_tf_sstv if val == float("inf") else val for val in tf_tuple_dhr[3]]
    SEN = [[sen_mth[i], sen_wday[i], sen_dhr[i]] for i in range(sensor_no)]
    TOTAL_SEN = np.array([sum(SEN[i]) for i in range(sensor_no)])
    idx = np.argsort(TOTAL_SEN)[-num_of_picks:] # Best 9 sensors
    
    spoke_labels = ["Month", "Day", "Hour"]
    data = [SEN[i] for i in idx]
    sensor_labels = [sensor_names[i] for i in idx]
    radar_chart.subplot(data, spoke_labels, sensor_labels, saveto=dict_dir+bldg_tag+trf_tag+'time_radar.png')
    ######################################################################    
    #1. effect prob - weather dependecy analysis
    ######################################################################    
    data_mat = np.hstack([data_state_mat,data_weather_mat])
    # Temporary for correcting month change    
    #data_mat[:,-3]=data_mat[:,-3]-1
    data_name = sensor_names+weather_names
    # State classification of weather data
    temp_idx=data_name.index('TemperatureC')
    dewp_idx=data_name.index('Dew PointC')
    humd_idx=data_name.index('Humidity')
    evnt_idx=data_name.index('Events')
    cond_idx=data_name.index('Conditions')
    ######################################################################    
    # Weather state classification
    ######################################################################    
    weather_dict={}
    for class_idx in [temp_idx,dewp_idx,humd_idx]:
        obs=data_mat[:,class_idx][:,np.newaxis]    
        label,opt_num_cluster,model,score,score_err_sum=\
        state_retrieval(obs,max_num_cluster=10,est_method='kmean',PARALLEL=IS_USING_PARALLEL_OPT,PRINTSHOW=True)
        if class_idx==temp_idx:
            weather_dict.update({'Temp':model.cluster_centers_})
        elif class_idx==dewp_idx:
            weather_dict.update({'Dewp':model.cluster_centers_})
        elif class_idx==humd_idx:
            weather_dict.update({'Humd':model.cluster_centers_})
        else:
            print 'not found'
            
        for label_id in range(label.max()+1):
            label_idx=np.nonzero(label==label_id)[0]
            data_mat[label_idx,class_idx]=np.round(model.cluster_centers_[label_id][0])
    ##################################################
    # Reclassify the Condition states into clarity of the sky 
    ##################################################
    #Conditions_dict=data_dict['Conditions_dict'].copy()
    #data_mat = np.hstack([avgdata_state_mat,avgdata_weather_mat])
    cond_state=[[]]*6    
    cond_state[5]=['Clear'] # Clear
    cond_state[4]=['Partly Cloudy','Scattered Clouds'] # 'Partly Cloudy'
    cond_state[3]=['Mostly Cloudy','Overcast'] # 'Overcast'
    cond_state[2]=['Light Drizzle','Mist', 'Shallow Fog',  'Patches of Fog',\
    'Light Snow', 'Light Freezing Rain', 'Light Rain Showers','Light Freezing Fog','Light Snow Showers', 'Light Rain'] # Light Rain
    cond_state[1]=['Rain','Rain Showers','Thunderstorms and Rain'\
    ,'Heavy Rain','Heavy Rain Showers','Drizzle', 'Heavy Drizzle', 'Fog'] # Heavy Rain
    cond_state[0]=['Unknown']
    cond_data_array=data_mat[:,cond_idx].copy()
    for k in range(len(cond_state)):
        for cond_str in cond_state[k]:
            cond_val_old=Conditions_dict[cond_str]
            idx_temp=np.nonzero(cond_data_array==cond_val_old)[0]
            if len(idx_temp)>0:
                data_mat[idx_temp,cond_idx]=k
           
    #plt.plot(data_mat[:,cond_idx],'.')
    Conditions_dict_temp={}
    Conditions_dict_temp.update({'Clear':5})
    Conditions_dict_temp.update({'Partly Cloudy':4})
    Conditions_dict_temp.update({'Overcast':3})
    Conditions_dict_temp.update({'Light Rain':2})
    Conditions_dict_temp.update({'Heavy Rain':1})
    Conditions_dict_temp.update({'Unknown':0})
    # Abbr' of weather factor type is 
    weather_dict.update({'Cond':Conditions_dict_temp})
    
    ####################################################################
    # Reclassify the Event states into rain/snow/fog weather conditons
    ####################################################################
    event_state=[[]]*4    
    event_state[0]=[''] # No event
    event_state[1]=['Rain-Snow','Snow'] # Snow
    event_state[2]=['Rain','Thunderstorm','Rain-Thunderstorm']  # Rain
    event_state[3]=['Fog','Fog-Rain']  # Fog
    
    event_data_array=data_mat[:,evnt_idx].copy()
    for k in range(len(event_state)):
        for event_str in event_state[k]:
            event_val_old=Events_dict[event_str]
            idx_temp=np.nonzero(event_data_array==event_val_old)[0]
            if len(idx_temp)>0:
                data_mat[idx_temp,evnt_idx]=k
                
    Events_dict_temp={}
    Events_dict_temp.update({'NoEvent':0})
    Events_dict_temp.update({'Snow':1})
    Events_dict_temp.update({'Rain':2})
    Events_dict_temp.update({'Fog':3})
    weather_dict.update({'Event':Events_dict_temp})

    # T,D,H,E,C
    print 'Parameter sensitivty for TemperatureC....'
    tempr_prob_map,tempr_state_map, tempr_sensitivity,tempr_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'TemperatureC',dst_type=dst_t)
    print 'Parameter sensitivty for Dew PointC....'
    dewp_prob_map,dewp_state_map, dewp_sensitivity, dewp_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'Dew PointC',dst_type=dst_t)
    print 'Parameter sensitivty for Humidity....'
    humd_prob_map,humd_state_map, humd_sensitivity,humd_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'Humidity',dst_type=dst_t)
    print 'Parameter sensitivty for Events....'
    event_prob_map,event_state_map,event_sensitivity, event_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'Events',dst_type=dst_t)
    print 'Parameter sensitivty for Conditions....'
    cond_prob_map,cond_state_map,cond_sensitivity,cond_list\
    = param_sensitivity(data_mat,data_name,sensor_names,'Conditions',dst_type=dst_t)

    wf_tuple_t=('T',tempr_prob_map,tempr_state_map,tempr_sensitivity,tempr_list)
    wf_tuple_d=('D',dewp_prob_map,dewp_state_map,dewp_sensitivity,dewp_list)
    wf_tuple_h=('H',humd_prob_map,humd_state_map,humd_sensitivity,humd_list)
    wf_tuple_e=('E',event_prob_map,event_state_map,event_sensitivity,event_list)
    wf_tuple_c=('C',cond_prob_map,cond_state_map,cond_sensitivity,cond_list)

    wf_sstv_tuple=np.array([wf_tuple_t[3],wf_tuple_d[3],wf_tuple_h[3],wf_tuple_e[3],wf_tuple_c[3]])
    max_wf_sstv=wf_sstv_tuple[wf_sstv_tuple<np.inf].max()*2
    wf_sstv_tuple[wf_sstv_tuple==np.inf]=max_wf_sstv
    wf_sstv_total=np.sum(wf_sstv_tuple,0)
    arg_idx_s=np.argsort(wf_sstv_total)[::-1]
    print 'Most weather sensitive sensors'
    print '---------------------------------------------'
    Weather_Sensitive_Sensors=list(np.array(sensor_names)[arg_idx_s[0:num_of_picks]])
    pprint.pprint(Weather_Sensitive_Sensors)

    ####################################################################
    ## Radar Plotting for Weather_Sensitive_Sensors
    ####################################################################
    sensor_no = len(sensor_names)
    # convert 'inf' to 1
    sen_t = [max_wf_sstv if val == float("inf") else val for val in wf_tuple_t[3]]
    sen_d = [max_wf_sstv if val == float("inf") else val for val in wf_tuple_d[3]]
    sen_h = [max_wf_sstv if val == float("inf") else val for val in wf_tuple_h[3]]
    sen_e = [max_wf_sstv if val == float("inf") else val for val in wf_tuple_e[3]]
    sen_c = [max_wf_sstv if val == float("inf") else val for val in wf_tuple_c[3]]
    
    SEN = [[sen_t[i], sen_d[i], sen_h[i], sen_e[i], sen_c[i]] for i in range(sensor_no)]
    TOTAL_SEN = np.array([sum(SEN[i]) for i in range(sensor_no)])
    idx = np.argsort(TOTAL_SEN)[-num_of_picks:] # Best 6 sensors
    
    spoke_labels = ["Temperature", "Dew Point", "Humidity", "Events", "Conditions"]
    data = [SEN[i] for i in idx]
    sensor_labels = [sensor_names[i] for i in idx]
    import radar_chart
    radar_chart.subplot(data, spoke_labels, sensor_labels, saveto=dict_dir+bldg_tag+trf_tag+'weather_radar.png')        
    #radar_chart.plot(data, spoke_labels, sensor_labels, saveto="weather_radar.png")    

    ####################################################################
    ## Bar Plotting for Weather and time sensitive_Sensors
    ####################################################################
    import bar_chart
    # Load from binaries
    #sen_mth sen_wday sen_dhr sen_t  sen_d  sen_h sen_e  sen_c
    SEN = [[sen_mth[i],sen_wday[i],sen_dhr[i],sen_t[i], sen_d[i], sen_h[i], sen_e[i], sen_c[i]] for i in range(sensor_no)]
    TOTAL_SEN = np.array([sum(SEN[i]) for i in range(sensor_no)])
    idx = np.argsort(TOTAL_SEN)[-15:] # Best 15 sensors
    
    #data = [[TOTAL_SEN[i] for i in idx]] * 8
    data = [[np.array(SEN)[i,k] for i in idx] for k in range(8)] 
    labels = [[sensor_names[i] for i in idx]] * 8
    titles = ["Month", "Day", "Hour", "Temperature", "Dew Point", "Humidity", "Events", "Conditions"]
    colors = ["b" if i < 3 else "g" for i in range(8)]
    
    bar_chart.plot(data, labels, titles, colors, grid=True, savefig=dict_dir+bldg_tag+trf_tag+'bar.png', savereport=dict_dir+bldg_tag+trf_tag+'all_bar.csv')
    
    
    ####################################################################
    ## Rador Plotting for Time Weather_Sensitive_Sensors
    ####################################################################
    wtf_sstv_total=wf_sstv_total+tf_sstv_total
    arg_idx_s=np.argsort(wtf_sstv_total)[::-1]
    #arg_idx_is=np.argsort(wtf_sstv_total)
    num_of_picks=9
    print 'Most time-weather sensitive sensors'
    print '---------------------------------------------'
    WT_Sensitive_Sensors=list(np.array(sensor_names)[arg_idx_s[0:num_of_picks]])
    pprint.pprint(WT_Sensitive_Sensors)

    sensor_no = len(sensor_names)
    # convert 'inf' to 1
    SEN = [[sen_mth[i], sen_wday[i], sen_dhr[i],sen_t[i], sen_d[i], sen_h[i], sen_e[i], sen_c[i]] for i in range(sensor_no)]
    TOTAL_SEN = np.array([sum(SEN[i]) for i in range(sensor_no)])
    idx = np.argsort(TOTAL_SEN)[-num_of_picks:] # Best 9 sensors
    
    spoke_labels = ["Month", "Day", "Hour","Temperature", "Dew Point", "Humidity", "Events", "Conditions"]
    data = [SEN[i] for i in idx]
    sensor_labels = [sensor_names[i] for i in idx]
    radar_chart.subplot(data, spoke_labels, sensor_labels, saveto=dict_dir+bldg_tag+trf_tag+'time_weather_radar.png')  
    
    fig=plt.figure()
    idx = np.argsort(TOTAL_SEN)[-(min(len(TOTAL_SEN),50)):] # Best 50 sensors
    twf_sstv_tuple = np.array([SEN[i] for i in idx]).T
    sensor_labels = [sensor_names[i] for i in idx]
    #twf_sstv_tuple=np.vstack([tf_sstv_tuple,wf_sstv_tuple])
    vmax_=twf_sstv_tuple.max()
    vmin_=twf_sstv_tuple.min()
    im=plt.imshow(twf_sstv_tuple,interpolation='none',vmin=vmin_, vmax=vmax_,aspect='equal')
    y_label=['MTH', 'WD', 'HR','TemperatureC','Dew PointC','Humidity','Events', 'Conditions']
    y_ticks=range(len(y_label))
    plt.yticks(y_ticks,y_label)
    x_label=sensor_labels
    x_ticks=range(len(sensor_labels))
    plt.xticks(x_ticks,x_label,rotation=270, fontsize="small")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("top", "15%", pad="30%")
    plt.colorbar(im, cax=cax,orientation='horizontal')
    plt.savefig(dict_dir+bldg_tag+trf_tag+'time_weather_hmap.png')
    
    wtf_tuples={}
    wtf_tuples.update({'month':tf_tuple_mth})
    wtf_tuples.update({'day':tf_tuple_wday})
    wtf_tuples.update({'hour':tf_tuple_dhr})
    wtf_tuples.update({'month':tf_tuple_mth})
    wtf_tuples.update({'t':wf_tuple_t})
    wtf_tuples.update({'d':wf_tuple_d})
    wtf_tuples.update({'h':wf_tuple_h})
    wtf_tuples.update({'e':wf_tuple_e})
    wtf_tuples.update({'c':wf_tuple_c})
    
    return wtf_tuples,weather_dict



def check_cond_state(all_cond_name,cond_state):
    no_assn_key=[]
    for key in all_cond_name:
        print '------------------'
        print key
        num_cnt=0
        for k in range(len(cond_state)):
            if key in cond_state[k]:
                num_cnt=num_cnt+1
        print num_cnt
        if num_cnt==0:
            no_assn_key.append(key)
        print '------------------'
    print 'unassigned cond key ' ,no_assn_key            
    return no_assn_key
#all_cond_name=list(set(GW1_.Conditions_dict.keys()+GW2_.Conditions_dict.keys()\
#+VAK2_.Conditions_dict.keys()+VAK1_.Conditions_dict.keys()))

def check_event_state(all_event_name,event_state):
    no_assn_key=[]
    for key in all_event_name:
        print '------------------'
        print key
        num_cnt=0
        for k in range(len(event_state)):
            if key in event_state[k]:
                num_cnt=num_cnt+1
        print num_cnt
        if num_cnt==0:
            no_assn_key.append(key)
        print '------------------'
    
    print 'unassigned event key ' ,no_assn_key
    return no_assn_key
#all_event_name=list(set(GW1_.Events_dict.keys()+GW2_.Events_dict.keys()\
#+VAK2_.Events_dict.keys()+VAK1_.Events_dict.keys()))


def weather_convert(wdata_mat,wdata_name, Conditions_dict,Events_dict):
    ##########################################
    # New dictionary by state classification of weather data
    ##########################################
    weather_dict={}
    ##########################################
    # index of weather data point in previous data
    ##########################################
    try:    
        temp_idx=wdata_name.index('TemperatureC')
    except:
         temp_idx=[]
    try:    
        dewp_idx=wdata_name.index('Dew_PointC')
    except:
        dewp_idx=[]
    try:
        humd_idx=wdata_name.index('Humidity')
    except:
        humd_idx=[]
    try:
        evnt_idx=wdata_name.index('Events')
    except:
        evnt_idx=[]
    try:
        cond_idx=wdata_name.index('Conditions')
    except:
        cond_idx=[]
    ######################################################################    
    # Weather state classification
    ######################################################################    
    for class_idx in [temp_idx,dewp_idx,humd_idx]:
        obs=wdata_mat[:,class_idx][:,np.newaxis]    
        label,opt_num_cluster,model,score,score_err_sum=\
        state_retrieval(obs,max_num_cluster=30,off_set=1,est_method='kmean',PARALLEL=IS_USING_PARALLEL_OPT,PRINTSHOW=False)
        if class_idx==temp_idx:
            print 'Temp state classification...'
            weather_dict.update({'Temp':model.cluster_centers_})
        elif class_idx==dewp_idx:
            print 'Dewp state classification...'
            weather_dict.update({'Dewp':model.cluster_centers_})
        elif class_idx==humd_idx:
            print 'Humd state classification...'
            weather_dict.update({'Humd':model.cluster_centers_})
        else:
            print 'not found'
            
        for label_id in range(label.max()+1):
            label_idx=np.nonzero(label==label_id)[0]
            wdata_mat[label_idx,class_idx]=np.round(model.cluster_centers_[label_id][0])
            
    ##################################################
    # Reclassify the Condition states into clarity of the sky 
    ##################################################
    cond_state=[[]]*9    
    cond_state[8]=['Clear'] # Clear
    cond_state[7]=['Partly Cloudy','Scattered Clouds'] # 'Partly Cloudy'
    cond_state[6]=['Mostly Cloudy','Overcast'] # 'Overcast'
    cond_state[5]=['Fog','Mist', 'Shallow Fog','Patches of Fog','Light Freezing Fog'] # Light Rain
    cond_state[4]=['Drizzle', 'Heavy Drizzle','Light Drizzle','Light Freezing Drizzle']
    cond_state[3]=['Rain','Rain Showers','Thunderstorms and Rain'\
    ,'Heavy Rain','Heavy Rain Showers', 'Freezing Rain','Light Freezing Rain', \
    'Light Rain Showers','Light Rain','Light Thunderstorms and Rain'] # Heavy Rain
    cond_state[2]=['Ice Pellets', 'Ice Crystals','Light Ice Crystals','Light Ice Pellets']
    cond_state[1]=['Snow','Snow Showers','Light Snow','Light Snow Grains','Light Snow Showers'] # 'Snow'
    cond_state[0]=['Unknown']
    cond_data_array=wdata_mat[:,cond_idx].copy()
    print 'Condition state classification...'
    for k in range(len(cond_state)):
        for cond_str in cond_state[k]:
            if cond_str in Conditions_dict.keys():
                cond_val_old=Conditions_dict[cond_str]
                idx_temp=np.nonzero(cond_data_array==cond_val_old)[0]
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
    event_state[0]=[''] # No event
    event_state[1]=['Rain-Snow','Snow','Fog-Snow'] # Snow
    event_state[2]=['Rain','Thunderstorm','Rain-Thunderstorm']  # Rain
    event_state[3]=['Fog','Fog-Rain']  # Fog
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
    return wdata_mat,weather_dict
    
    
def bldg_obj_weather_convert(bldg_obj):
	#import pdb;pdb.set_trace()
    # For avg
    if 'data_weather_mat' in bldg_obj.avg.__dict__.keys():
        wdata_mat = bldg_obj.avg.data_weather_mat.copy()
        wdata_name =bldg_obj.avg.weather_names
        Conditions_dict= bldg_obj.Conditions_dict.copy()
        Events_dict= bldg_obj.Events_dict.copy()
        wdata_mat,weather_dict=weather_convert(wdata_mat,wdata_name, Conditions_dict,Events_dict)
        bldg_obj.avg.weather_dict=weather_dict
        bldg_obj.avg.data_weather_mat_=wdata_mat
    # For diff
    if 'data_weather_mat' in bldg_obj.diff.__dict__.keys():
        wdata_mat = bldg_obj.diff.data_weather_mat.copy()
        wdata_name =bldg_obj.diff.weather_names
        Conditions_dict= bldg_obj.Conditions_dict.copy()
        Events_dict= bldg_obj.Events_dict.copy()
        wdata_mat,weather_dict=weather_convert(wdata_mat,wdata_name, Conditions_dict,Events_dict)
        bldg_obj.diff.weather_dict=weather_dict
        bldg_obj.diff.data_weather_mat_=wdata_mat
    
def find_cond_lh_set(data_state_mat,cause_idx_set,effect_idx,obs_state):
    optprob_set=np.zeros(len(cause_idx_set))
    optstate_set=np.zeros(len(cause_idx_set))
    for i,cause_idx in enumerate(cause_idx_set):
        # Compute liklihoood of GW2 avg data state map among sensors
        avg_state_temp, avg_prob_temp\
        =compute_cause_likelihood(data_state_mat,[cause_idx],[[effect_idx]],[[obs_state]])
        # masking its own effect
        if cause_idx==effect_idx:
            # and its state
            max_opt_state=np.nan
            # and its probability 
            max_opt_prob=-np.inf
        else:
            # find sensor index giving the maximum likelihood 
            max_idx=np.argmax(avg_prob_temp)
            # and its state
            max_opt_state=avg_state_temp[max_idx]
            # and its probability 
            max_opt_prob=avg_prob_temp[max_idx]
            
        optprob_set[i]=max_opt_prob
        optstate_set[i]=max_opt_state
    return optstate_set, optprob_set
    
    
def create_bldg_obj(dict_dir,bldg_tag,pname_key):
    print '==================================='
    print 'create object for ', bldg_tag+'BLDG'
    print '==================================='
    cmd_=bldg_tag+'data_dict = mt.loadObjectBinaryFast(dict_dir+'+'\'data_dict.bin\')'
    exec(cmd_)
    sig_tag_set=[]
    try:
        cmd_=bldg_tag+'diffdata_dict = mt.loadObjectBinaryFast(dict_dir+'+'\'diffdata_dict.bin\')'
        exec(cmd_)
        sig_tag_set.append('diff')
    except:
        pass 
    try:    
        cmd_=bldg_tag+'avgdata_dict = mt.loadObjectBinaryFast(dict_dir+'+'\'avgdata_dict.bin\')'
        exec(cmd_)
        sig_tag_set.append('avg')
    except:
        pass
            
    ###########################################################################################
    for sig_tag in sig_tag_set:
        cmd_str=[[]]*9
        cmd_str[0]=bldg_tag+sig_tag+'data_state_mat='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_state_mat\']'
        cmd_str[1]=bldg_tag+sig_tag+'data_weather_mat='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_weather_mat\']'
        cmd_str[2]=bldg_tag+sig_tag+'data_time_mat='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_time_mat\']'
        cmd_str[3]=bldg_tag+sig_tag+'_time_slot='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'_time_slot\']'
        cmd_str[4]=bldg_tag+sig_tag+'data_exemplar='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_exemplar\']'
        cmd_str[5]=bldg_tag+sig_tag+'data_zvar=remove_dot('+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_zvar\'])'
        cmd_str[6]=bldg_tag+sig_tag+'sensor_names=remove_dot('+bldg_tag+sig_tag+'data_dict[\'sensor_names\'])'
        cmd_str[7]=bldg_tag+sig_tag+'weather_names=remove_dot('+bldg_tag+sig_tag+'data_dict[\'weather_names\'])'
        cmd_str[8]=bldg_tag+sig_tag+'time_names=remove_dot('+bldg_tag+sig_tag+'data_dict[\'time_names\'])'
        for cmd_ in cmd_str:
            exec(cmd_)

        if 'avg' in sig_tag:
            print "--*--*--*--*--*--*--*--*-- create_bldg_obj::(" + sig_tag + ") data_weather_mat --*--*--*--*--*-"
            exec("print " + bldg_tag+sig_tag+'data_weather_mat[:,4]')
            print "--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*"

        #TODO: Name correction for exemplar
        if isinstance(pname_key,list)==True:
            cmd_str_tmp=sig_tag+'p_names=pname_key'
            exec(cmd_str_tmp)
            cmd_str_tmp=bldg_tag+sig_tag+'p_idx=['+bldg_tag+sig_tag+'sensor_names.index(name_) for name_ in pname_key]'
            exec(cmd_str_tmp)
        else:
            cmd_str_tmp=bldg_tag+sig_tag+'p_idx=grep('+'\''+pname_key+'\''+','+bldg_tag+sig_tag+'sensor_names)'
            exec(cmd_str_tmp)
            cmd_str_tmp=bldg_tag+sig_tag+'p_names=list(np.array('+bldg_tag+sig_tag+'sensor_names)['+bldg_tag+sig_tag+'p_idx])'
            exec(cmd_str_tmp)
            cmd_str_tmp=sig_tag+'p_names=list(np.array('+bldg_tag+sig_tag+'sensor_names)['+bldg_tag+sig_tag+'p_idx])'
            exec(cmd_str_tmp)
        print '--------------------------------------------------------'
        print ' Power sensor selected -'+sig_tag
        print '--------------------------------------------------------'
        cmd_str_tmp='pprint.pprint('+sig_tag+'p_names)'
        exec(cmd_str_tmp)
    print '----------------------------------------'
    print 'creating '+ bldg_tag+' obj....'
    print '----------------------------------------'
    cmd_str_=bldg_tag+'=obj({'+'\'avg\''+':obj({}),'+'\'diff\''+':obj({})})'
    exec(cmd_str_)
    for sig_tag in sig_tag_set:
        print 'generating '+ sig_tag+' members....'
        cmd_str=[[]]*12
        #cmd_str[0]=bldg_tag+'.'+sig_tag+'=[]'
        cmd_str[0]='[]'
        cmd_str[1]=bldg_tag+'.'+sig_tag+'.data_state_mat='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_state_mat\']'
        cmd_str[2]=bldg_tag+'.'+sig_tag+'.data_weather_mat='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_weather_mat\']'
        cmd_str[3]=bldg_tag+'.'+sig_tag+'.data_time_mat='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_time_mat\']'
        cmd_str[4]=bldg_tag+'.'+sig_tag+'.time_slot='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'_time_slot\']'
        cmd_str[5]=bldg_tag+'.'+sig_tag+'.data_exemplar='+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_exemplar\']'
        cmd_str[6]=bldg_tag+'.'+sig_tag+'.data_zvar=remove_dot('+bldg_tag+sig_tag+'data_dict[\''+sig_tag+'data_zvar\'])'
        cmd_str[7]=bldg_tag+'.'+sig_tag+'.sensor_names=remove_dot('+bldg_tag+sig_tag+'data_dict[\'sensor_names\'])'
        cmd_str[8]=bldg_tag+'.'+sig_tag+'.weather_names=remove_dot('+bldg_tag+sig_tag+'data_dict[\'weather_names\'])'
        cmd_str[9]=bldg_tag+'.'+sig_tag+'.time_names=remove_dot('+bldg_tag+sig_tag+'data_dict[\'time_names\'])'
        cmd_str[10]=bldg_tag+'.'+sig_tag+'.p_idx='+bldg_tag+sig_tag+'p_idx'
        cmd_str[11]=bldg_tag+'.'+sig_tag+'.p_names=remove_dot('+bldg_tag+sig_tag+'p_names)'
        for cmd_ in cmd_str: exec(cmd_)
    #TODO: Name correction for exemplar
    cmd_=bldg_tag+'.'+'Conditions_dict='+bldg_tag+'data_dict[\'Conditions_dict\']'
    exec(cmd_)
    cmd_=bldg_tag+'.'+'Events_dict='+bldg_tag+'data_dict[\'Events_dict\']'
    exec(cmd_)
    cmd_='bldg_obj_weather_convert('+bldg_tag+')'
    exec(cmd_)
    # Create classs strucutre for data analysis     
    analysis={}
    for sig_tag in sig_tag_set:
        cmd_str_='p_names='+bldg_tag+'.'+sig_tag+'.p_names'
        exec(cmd_str_)
        temp1={}
        for name_ in p_names:
            temp_s=obj({'optprob_set':[],'optstate_set':[]})
            temp_t=obj({'optprob_set':[],'optstate_set':[]})
            temp_w=obj({'optprob_set':[],'optstate_set':[]})
            temp2=obj({'peak_eff_state':[],'sensor':temp_s,'time':temp_t,'weather':temp_w})
            temp1.update({remove_dot(name_):temp2})
        analysis.update({sig_tag:obj(temp1)})
    analysis=obj(analysis)
    cmd_str_=bldg_tag+'.analysis=analysis'
    exec(cmd_str_)
    print '-------------------------'
    print 'Compute LH values'
    print '-------------------------'
    for sig_tag in sig_tag_set:
        print sig_tag+'.....'
        cmd_str_='all_data_state_mat=np.vstack(('+bldg_tag+'.'+sig_tag+'.data_state_mat.T, '\
        +bldg_tag+'.'+sig_tag+'.data_time_mat.T,'+bldg_tag+'.'+sig_tag+'.data_weather_mat_.T)).T'
        exec(cmd_str_)
        cmd_str_='p_idx='+bldg_tag+'.'+sig_tag+'.p_idx'
        exec(cmd_str_)
        cmd_str_='p_names='+bldg_tag+'.'+sig_tag+'.p_names' 
        exec(cmd_str_)
        cmd_str_='len_sensor='+bldg_tag+'.'+sig_tag+'.data_state_mat.shape[1]'
        exec(cmd_str_)
        cmd_str_='len_time='+bldg_tag+'.'+sig_tag+'.data_time_mat.shape[1]'
        exec(cmd_str_)
        cmd_str_='len_weather='+bldg_tag+'.'+sig_tag+'.data_weather_mat.shape[1]'
        exec(cmd_str_)
        cmd_str_='sensor_cause_idx_set=range(len_sensor)'
        exec(cmd_str_)
        cmd_str_='time_cause_idx_set=range(len_sensor,len_sensor+len_time)'
        exec(cmd_str_)
        cmd_str_='weather_cause_idx_set=range(len_sensor+len_time,len_sensor+len_time+len_weather)'
        exec(cmd_str_)
        for k,effect_idx in enumerate(p_idx):
            print 'compute cond. prob of ' +  p_names[k]
            cmd_str_='p_name_='+bldg_tag+'.'+sig_tag+'.p_names[k]'
            exec(cmd_str_)
            # check weather it is in the set
            effect_state_set=np.array(list(set(all_data_state_mat[:, effect_idx])))
            eff_state=effect_state_set.max()
            cmd_str_=bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name_)+'.peak_eff_state=eff_state'
            exec(cmd_str_)             
            s_optstate_set_temp,s_optprob_set_temp=\
            find_cond_lh_set(all_data_state_mat,sensor_cause_idx_set,effect_idx,eff_state)
            cmd_str_=bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name_)+'.sensor.optprob_set=s_optprob_set_temp'
            exec(cmd_str_)
            cmd_str_=bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name_)+'.sensor.optstate_set=s_optstate_set_temp'
            exec(cmd_str_) 
            w_optstate_set_temp,w_optprob_set_temp=\
            find_cond_lh_set(all_data_state_mat,weather_cause_idx_set,effect_idx,eff_state)
            cmd_str_=bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name_)+'.weather.optprob_set=w_optprob_set_temp'
            exec(cmd_str_)
            cmd_str_=bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name_)+'.weather.optstate_set=w_optstate_set_temp'
            exec(cmd_str_)
            
            w_optstate_set_temp,w_optprob_set_temp=\
            find_cond_lh_set(all_data_state_mat,time_cause_idx_set,effect_idx,eff_state)
            cmd_str_=bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name_)+'.time.optprob_set=w_optprob_set_temp'
            exec(cmd_str_)
            cmd_str_=bldg_tag+'.analysis.'+sig_tag+'.'+remove_dot(p_name_)+'.time.optstate_set=w_optstate_set_temp'
            exec(cmd_str_)
    cmd_str_='mt.saveObjectBinaryFast('+bldg_tag+','+'bldg_tag+\'.bin\')'
    exec(cmd_str_)
    cmd_str_='obj_out='+bldg_tag
    exec(cmd_str_)    
    return obj_out    
    
    
def plotting_bldg_lh(bldg_,bldg_key=[],attr_class='sensor',num_picks=30):
    print 'plotting lh for '+attr_class
    print '============================================'
    sig_tag_set=['avg','diff']
    plt.ioff()
    if len(bldg_key)==0:
        bldg_set=bldg_.__dict__.keys()
        
    else :
        bldg_set=[bldg_key]
        
    for bldg_tag in bldg_set:
        print bldg_tag
        cmd_str_= bldg_tag+'=bldg_.__dict__[bldg_tag]'
        exec(cmd_str_)
        print '-------------------------'
        print bldg_tag
        print '-------------------------'
        for sig_tag in sig_tag_set:
            try:
                print sig_tag+'.....'
                cmd_str_='p_names='+bldg_tag+'.'+sig_tag+'.p_names'
                exec(cmd_str_)
                for pname_ in p_names:
                    try:
                        blank_idx=pname_.index('.')
                        pname_=pname_.replace('.','_')
                    except:
                        pass
                    cmd_str_='optprob_set='+bldg_tag+'.analysis.'+sig_tag+'.'+pname_+'.'+attr_class+'.optprob_set'
                    exec(cmd_str_)
                    cmd_str_= 's_names='+bldg_tag+'.'+sig_tag+'.'+attr_class+'_names'
                    exec(cmd_str_)
                    cmd_str_= 'optstate_set='+bldg_tag+'.analysis.'+sig_tag+'.'+pname_+'.'+attr_class+'.optstate_set'
                    exec(cmd_str_)
                    num_picks=30
                    sort_idx=np.argsort(optprob_set)[::-1]
                    sort_lh=optprob_set[sort_idx[:num_picks]].T
                    sort_state=optstate_set[sort_idx[:num_picks]].T
                    plt.figure(figsize=(20.0,15.0))
                    plt.subplot(2,1,1)
                    plt.plot(sort_lh,'-*')
                    x_label= list(np.array(s_names)[sort_idx[:num_picks]])
                    cmd_str_='key_set=bldg_.__dict__.keys()'
                    exec(cmd_str_)
                    if 'convert_name' in key_set:
                        cmd_str_='x_label=bldg_.convert_name(x_label)'
                        exec(cmd_str_)
                        cmd_str_='pname_=bldg_.convert_name(pname_)[0]'
                        exec(cmd_str_)
                    x_ticks=range(len(x_label))
                    plt.xticks(x_ticks,x_label,rotation=270, fontsize="small")
                    if sig_tag=='avg':
                        plt.title('Most relavant '+ attr_class+ ' attributes to the peak (demand) of '+pname_,fontsize=20)
                    else:
                        plt.title('Most relavant '+ attr_class+ ' attributes to the peak variations of '+pname_,fontsize=20)
                    plt.tick_params(labelsize='large')
                    plt.ylim([-0.05, 1.05])
                    plt.ylabel('Likelihood (From 0 to 1)',fontsize=18)
                    plt.savefig(fig_dir+bldg_tag+'_'+pname_+'_'+attr_class+'_'+sig_tag+'_lh_sensors.png', bbox_inches='tight')
                    plt.close()
            except:
                pass 
    plt.close()
    plt.ion()





#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&PPPPPPPPPPPPPP
def bldg_lh_sensitivity(bldg_,bldg_key=[],attr_class='sensor',sig_tag='avg'):
    print 'compute std of lh for '+attr_class+'...'
    if len(bldg_key)==0:
        bldg_set=bldg_.__dict__.keys()
    else :
        bldg_set=[bldg_key]
    bldg_lh_std_log={}    
    for bldg_tag in bldg_set:
        try:
            print bldg_tag
            cmd_str_= bldg_tag+'=bldg_.__dict__[bldg_tag]'
            exec(cmd_str_)
            print '-------------------------'
            print bldg_tag
            print '-------------------------'
            cmd_str_='p_names='+bldg_tag+'.'+sig_tag+'.p_names'
            exec(cmd_str_)
            lh_std_log={}
            for pname_ in p_names:
                try:
                    blank_idx=pname_.index('.')
                    pname_=pname_.replace('.','_')
                except:
                    pass
                cmd_str_='optprob_set='+bldg_tag+'.analysis.'+sig_tag+'.'+pname_+'.'+attr_class+'.optprob_set'
                exec(cmd_str_)
                cmd_str_= 's_names='+bldg_tag+'.'+sig_tag+'.'+attr_class+'_names'
                exec(cmd_str_)
                cmd_str_= 'optstate_set='+bldg_tag+'.analysis.'+sig_tag+'.'+pname_+'.'+attr_class+'.optstate_set'
                exec(cmd_str_)
                lh_std=np.std(np.sort(optprob_set)[1:])
                lh_std_log.update({bldg_.convert_name(pname_)[0]:lh_std})
            bldg_lh_std_log.update({bldg_tag:lh_std_log})
        except:
            pass
    return obj(bldg_lh_std_log)
            
  

###################################################&&&&&!!!!!
def bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag='avg',num_picks_bn=15,learning_alg='hc'):
    cmd_str_='s_names=bldg_obj.'+sig_tag+'.sensor_names'
    exec(cmd_str_)
    p_idx=s_names.index(p_name)        
    cmd_str_='data_state_mat=bldg_obj.'+sig_tag+'.data_state_mat'
    exec(cmd_str_)
    if not (attr=='all') :        
        cmd_str_='optprob_set=bldg_obj.analysis.'+sig_tag+'.__dict__[p_name].'+attr+'.optprob_set'
        exec(cmd_str_)
        cmd_str_='optstate_set=bldg_obj.analysis.'+sig_tag+'.__dict__[p_name].'+attr+'.optstate_set'
        sort_idx=np.argsort(optprob_set)[::-1]
        
    if (attr=='sensor') :
        print 'power - sensors...'
        cmd_str_='s_names=bldg_obj.'+sig_tag+'.sensor_names'
        exec(cmd_str_)
        idx_select=[p_idx]+ list(sort_idx[:num_picks_bn])
        cmd_str_='bndata_mat=bldg_obj.'+sig_tag+'.data_state_mat[:,idx_select]'
        exec(cmd_str_)
        cols=[s_names[k] for k in idx_select]
    elif (attr=='weather'):
        print 'power - weather...'
        cmd_str_='w_names=bldg_obj.'+sig_tag+'.weather_names'
        exec(cmd_str_)
        cmd_str_='bndata_mat=np.vstack((bldg_obj.'+sig_tag+'.data_state_mat[:,p_idx].T,bldg_obj.'+sig_tag+'.data_weather_mat_.T)).T'
        exec(cmd_str_)
        cols=[p_name]+[w_name for w_name in w_names]

        if 'avg' in sig_tag:
            print "--*--*--*--*- bn_anaylsis::sig_tag [" + sig_tag + "] data_weather_mat_ --*--*--*--*--*--*--"
            exec('print bldg_obj.'+sig_tag+'.data_weather_mat_[:,4]')
            print "--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--"

    elif (attr=='time'):
        print 'power - time...'
        cmd_str_='t_names=bldg_obj.'+sig_tag+'.time_names'
        exec(cmd_str_)
        cmd_str_='bndata_mat=np.vstack((bldg_obj.'+sig_tag+\
        '.data_state_mat[:,p_idx].T,bldg_obj.'+sig_tag+'.data_time_mat.T)).T'
        exec(cmd_str_)
        cols=[p_name]+[t_name for t_name in t_names] 
    elif (attr=='all'):
        print 'power - sensors + weather + time ...'
        s_cause_label,s_labels,s_hc,s_cp_mat,s_bndata_mat=\
        bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag=sig_tag,num_picks_bn=num_picks_bn,learning_alg=learning_alg)
        t_cause_label,t_labels,t_hc,t_cp_mat,t_bndata_mat=\
        bn_anaylsis(bldg_obj,p_name,attr='time',sig_tag=sig_tag,num_picks_bn=num_picks_bn,learning_alg=learning_alg)
        w_cause_label,w_labels,w_hc,w_cp_mat,w_bndata_mat=\
        bn_anaylsis(bldg_obj,p_name,attr='weather',sig_tag=sig_tag,num_picks_bn=num_picks_bn,learning_alg=learning_alg)
        #s_cause_label=s_labels; w_cause_label=w_labels;t_cause_label=t_labels
        cmd_str_='s_cause_idx=[bldg_obj.'+sig_tag+'.sensor_names.index(name_) for name_ in s_cause_label]'
        exec(cmd_str_)
        cmd_str_='t_cause_idx=[bldg_obj.'+sig_tag+'.time_names.index(name_) for name_ in t_cause_label]'
        exec(cmd_str_)
        cmd_str_='w_cause_idx=[bldg_obj.'+sig_tag+'.weather_names.index(name_) for name_ in w_cause_label]'
        exec(cmd_str_)
        cmd_str_='bndata_mat=np.vstack((bldg_obj.'+sig_tag+'.data_state_mat[:,p_idx].T,\
        bldg_obj.'+sig_tag+'.data_state_mat[:,s_cause_idx].T, \
        bldg_obj.'+sig_tag+'.data_weather_mat_[:,w_cause_idx].T, \
        bldg_obj.'+sig_tag+'.data_time_mat[:,t_cause_idx].T)).T'
        exec(cmd_str_)
        cmd_str_='cols=[name_ for name_ in [p_name]+s_cause_label+w_cause_label+t_cause_label]'
        exec(cmd_str_)
    else:
        print 'error'
        return 0
    if (attr=='all'):
        b_arc_list = pair_in_idx([p_name],s_cause_label+ w_cause_label+t_cause_label)+\
        pair_in_idx(s_cause_label,w_cause_label+t_cause_label)+\
        pair_in_idx(w_cause_label,t_cause_label)+\
        pair_in_idx(t_cause_label,t_cause_label)
        #import pdb;pdb.set_trace()
    elif(attr=='time'):
        b_arc_list = pair_in_idx([cols[0]],cols[1:])+pair_in_idx(cols[1:],cols[1:])
    else:
        b_arc_list = pair_in_idx([cols[0]],cols[1:])

    black_arc_frame = rbn.construct_arcs_frame(b_arc_list)
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)

    if learning_alg=='tabu':
        hc_b = rbn.bnlearn.tabu(data_frame,blacklist=black_arc_frame,score='bic')
    elif learning_alg=='mmhc':
        hc_b = rbn.bnlearn.mmhc(data_frame,blacklist=black_arc_frame,score='bic')
    else:
        hc_b = rbn.bnlearn.hc(data_frame,blacklist=black_arc_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    cause_label=list(np.array(cols)[np.nonzero(amat[:,0]==1)[0]])
    cause_idx=[cols.index(label_) for label_ in cause_label]
    return cause_label,cols, hc_b, amat,bndata_mat

def peak_analysis(cause_label,effect_label,col_labels,bndata_mat):
    if isinstance(cause_label,list)==True:
        cause_idx=[col_labels.index(label_) for label_ in cause_label]
    else:
        cause_idx=[col_labels.index(label_) for label_ in [cause_label]]
    if isinstance(effect_label,list)==True:
        effect_idx=[col_labels.index(label_) for label_ in effect_label]
    else:
        effect_idx=[col_labels.index(label_) for label_ in [effect_label]]
        
    effect_state_set=list(set(bndata_mat[:,effect_idx].T[0]))
    LOW_PEAK_STATE_EFFECT=np.min(effect_state_set)
    HIGH_PEAK_STATE_EFFECT=np.max(effect_state_set)
    high_peak_state_temp, high_peak_prob_temp=\
    compute_cause_likelihood(bndata_mat,cause_idx,[effect_idx],[[HIGH_PEAK_STATE_EFFECT]])
    low_peak_state_temp, low_peak_prob_temp=\
    compute_cause_likelihood(bndata_mat,cause_idx,[effect_idx],[[LOW_PEAK_STATE_EFFECT]])
    high_peak_state=np.array(high_peak_state_temp)
    high_peak_prob=np.array(high_peak_prob_temp)
    low_peak_state=np.array(low_peak_state_temp)
    low_peak_prob=np.array(low_peak_prob_temp)
    return low_peak_state,low_peak_prob,high_peak_state,high_peak_prob
    
def get_tick_symbol(tick_state_val,cause_labels_,Event,Cond):
    if len(cause_labels_)==1:
        iter_zip=zip(cause_labels_,tick_state_val.T[np.newaxis,:])
    else:
        iter_zip=zip(cause_labels_,tick_state_val.T)
    symbol_tuple=[]
    for cause_label_,state_val_ in iter_zip:
        symbol_out=[]
        if (isinstance(state_val_,np.ndarray)==False) and (isinstance(state_val_,list)==False):
            state_val_=[state_val_]
        temp=list(set(state_val_))
        if list(np.sort(temp))==[-1,0,1]:
            cause_label_='PEAK'
        for sval_ in state_val_:
            if cause_label_=='MTH':
                symbol_out.append(monthDict[sval_])
            elif cause_label_=='WD':
                symbol_out.append(weekDict[sval_])
            elif cause_label_=='HR':
                symbol_out.append(hourDict[sval_])
            elif cause_label_=='Dew_PointC':
                ssymbol_out.append(str(sval_)+'C')
            elif cause_label_=='Humidity':
                symbol_out.append(str(sval_)+'%')
            elif cause_label_=='Events':
                symbol_out.append([key_ for key_,val_ in Event.items() if val_==sval_])
            elif cause_label_=='Conditions':
                symbol_out.append([key_ for key_,val_ in Cond.items() if val_==sval_])
            elif cause_label_=='TemperatureC':
                symbol_out.append(str(sval_)+'C')
            elif cause_label_=='PEAK':
                symbol_out.append(stateDict[sval_])
            else:
                symbol_out.append(str(sval_))
        symbol_tuple.append(symbol_out)
    temp_=np.array(symbol_tuple)
    temp2=temp_.reshape(len(cause_labels_),np.prod(temp_.shape)/len(cause_labels_)).T
    return [tuple(symbol_) for symbol_ in temp2]
    
def bn_prob_analysis(bldg_obj,sig_tag_='avg'):
    cmd_str='Event=bldg_obj.'+sig_tag_+'.weather_dict[\'Event\']'
    exec(cmd_str)
    cmd_str='Cond=bldg_obj.'+sig_tag_+'.weather_dict[\'Cond\']'
    exec(cmd_str)
    bn_out_set={}
    cmd_str='p_name_set=bldg_obj.analysis.'+sig_tag_+'.__dict__.keys()'
    exec(cmd_str)
    for p_name in p_name_set:
        try:
            # bn analysis - Power-Sensor
            s_cause_label,s_labels,s_hc,s_cp_mat,s_bndata_mat=\
            bn_anaylsis(bldg_obj,p_name,attr='sensor',sig_tag=sig_tag_,num_picks_bn=5)
            # bn analysis -Power-Time
            t_cause_label,t_labels,t_hc,t_cp_mat,t_bndata_mat=\
            bn_anaylsis(bldg_obj,p_name,attr='time',sig_tag=sig_tag_,num_picks_bn=10)
            # bn analysis -Power-Weather
            w_cause_label,w_labels,w_hc,w_cp_mat,w_bndata_mat=\
            bn_anaylsis(bldg_obj,p_name,attr='weather',sig_tag=sig_tag_,num_picks_bn=10)
            # bn analysis -Power-Sensor+Time+Weather
            all_cause_label,all_labels,all_hc,all_cp_mat,all_bndata_mat=\
            bn_anaylsis(bldg_obj,p_name,attr='all',sig_tag=sig_tag_,num_picks_bn=20)
            # prob analysis -Power-Sensor+Time+Weather
            cause_label=all_cause_label;col_labels=all_labels;
            effect_label=p_name; bndata_mat=all_bndata_mat
            low_peak_state,low_peak_prob,high_peak_state,high_peak_prob=\
            peak_analysis(cause_label,effect_label,col_labels,bndata_mat)
            
            x_set=low_peak_state
            all_cause_symbol_xlabel=get_tick_symbol(x_set,all_cause_label,Event,Cond)
            all_cause_symbol_xtick=range(len(low_peak_state))
            # BN-PROB STORE
            bn_out={'s_cause_label':s_cause_label,'s_labels':s_labels,'s_hc':s_hc,\
            's_cp_mat':s_cp_mat, 's_bndata_mat':s_bndata_mat,'t_cause_label':t_cause_label,\
            't_labels':t_labels,'t_hc':t_hc,'t_cp_mat':t_cp_mat,'t_bndata_mat':t_bndata_mat, \
            'w_cause_label':w_cause_label,'w_labels':w_labels,'w_hc':w_hc,'w_cp_mat':w_cp_mat,\
            'w_bndata_mat':w_bndata_mat,'all_cause_label':all_cause_label,'all_labels':all_labels,\
            'all_hc':all_hc,'all_cp_mat':all_cp_mat,'all_bndata_mat':all_bndata_mat,
            'low_peak_state':low_peak_state,'low_peak_prob':low_peak_prob,\
            'high_peak_state':high_peak_state,'high_peak_prob':high_peak_prob,\
            'all_cause_symbol_xlabel':all_cause_symbol_xlabel,'all_cause_symbol_xtick':all_cause_symbol_xtick}
            bn_out_set.update({p_name:bn_out})
        except:
                print '*** Error in processing bn_prob for ', p_name, '! ****'
                pass
    return obj(bn_out_set)


def compute_bn_sensors(bldg_obj,sig_tag='avg',learning_alg='hill'):
    cmd_str_='s_names=bldg_obj.'+sig_tag+'.sensor_names'
    exec(cmd_str_)
    cmd_str_='bndata_mat=bldg_obj.'+sig_tag+'.data_state_mat'
    exec(cmd_str_)
    cols=s_names
    const_idx=np.nonzero(np.array([ len(set(col)) for col in bndata_mat.T])<2)[0]
    bndata_mat=np.delete(bndata_mat,const_idx,1)
    cols=list(np.delete(cols,const_idx,0))
    factor_data_mat = rbn.convert_pymat_to_rfactor(bndata_mat)
    data_frame = rbn.construct_data_frame(factor_data_mat,cols)
    if learning_alg=='tabu':
        hc_b = rbn.bnlearn.tabu(data_frame,score='bic')
    elif learning_alg=='mmhc':
        hc_b = rbn.bnlearn.mmhc(data_frame,score='bic')
    else:
        hc_b = rbn.bnlearn.hc(data_frame,score='bic')
    amat = rbn.py_get_amat(hc_b)
    return hc_b,cols,amat


def plotting_bldg_bn(bldg_):
    plt.ioff()
    #if 'convert_name' not in bldg_.__dict__.keys():
    #    bldg_.convert_name = lambda name_: [name_]
    for bldg_tag in bldg_.__dict__.keys():
        print 'Getting anal_out from '+ bldg_tag
        anal_out_found=True
        try:        
            cmd_str='anal_out=bldg_.__dict__[\''+bldg_tag+'\'].anal_out'
            exec(cmd_str)
        except:
            anal_out_found=False
        if anal_out_found==True:
            for sig_tag  in ['avg','diff']:
                if sig_tag in anal_out.__dict__.keys():
                    anal_out_sig=anal_out.__dict__[sig_tag]
                    p_name_sets=anal_out_sig.__dict__.keys()
                    for p_name in p_name_sets:
                        bn_out=anal_out_sig.__dict__[p_name]
                        cmd_str='pname_=bldg_.convert_name(p_name)[0]'
                        exec(cmd_str)
                        try:
                            fig_name='BN for Sensors '+pname_
                            plt.figure(fig_name,figsize=(30.0,30.0))
                            col_name=bldg_.convert_name(bn_out.s_labels)
                            rbn.nx_plot(bn_out.s_hc,col_name,graph_layout='spring',node_text_size=30)
                            plt.savefig(fig_dir+bldg_tag+'_'+pname_+'_'+sig_tag+'_bn_sensors'+get_pngid()+'.png', bbox_inches='tight')
                            plt.close()
                        except:
                            print 'error in '+fig_name
                            pass
                        try:
                            fig_name='BN for Time '+pname_
                            plt.figure(fig_name,figsize=(30.0,30.0))
                            rbn.nx_plot(bn_out.t_hc,bldg_.convert_name(bn_out.t_labels),graph_layout='spring',node_text_size=30)
                            plt.savefig(fig_dir+bldg_tag+'_'+pname_+'_'+sig_tag+'_bn_time'+get_pngid()+'.png', bbox_inches='tight')
                            plt.close()
                        except:
                            print 'error in '+fig_name
                            pass
                        try:
                            fig_name='BN for Weather '+pname_
                            plt.figure(fig_name,figsize=(30.0,30.0))
                            rbn.nx_plot(bn_out.w_hc,bldg_.convert_name(bn_out.w_labels),graph_layout='spring',node_text_size=30)
                            plt.savefig(fig_dir+bldg_tag+'_'+pname_+'_'+sig_tag+'_bn_weather'+get_pngid()+'.png', bbox_inches='tight')
                            plt.close()
                        except:
                            print 'error in '+fig_name
                            pass
                        try:
                            fig_name='BN for Sensor-Time-Weather '+pname_
                            plt.figure(fig_name,figsize=(30.0,30.0))
                            rbn.nx_plot(bn_out.all_hc,bldg_.convert_name(bn_out.all_labels),graph_layout='spring',node_text_size=30)
                            plt.savefig(fig_dir+bldg_tag+'_'+pname_+'_'+sig_tag+'_bn_sensor_time_weather'+get_pngid()+'.png', bbox_inches='tight')
                            plt.close()
                        except:
                            print 'error in '+fig_name
                            pass
                        try:
                            fig_name='BN PEAK LH Analysis for Sensor-Time-Weather '+pname_
                            plt.figure(fig_name, figsize=(30.0,30.0))
                            plt.subplot(2,1,1)
                            plt.plot(bn_out.all_cause_symbol_xtick,bn_out.high_peak_prob,'-^')
                            plt.plot(bn_out.all_cause_symbol_xtick,bn_out.low_peak_prob,'-.v')
                            plt.ylabel('Likelihood',fontsize=20)
                            plt.xticks(bn_out.all_cause_symbol_xtick,bn_out.all_cause_symbol_xlabel,rotation=270, fontsize=20)
                            plt.tick_params(labelsize=20)
                            plt.legend(('High Peak', 'Low Peak'),loc='center right', prop={'size':25})
                            plt.tick_params(labelsize=20)
                            plt.grid();plt.ylim([-0.05,1.05])
                            plt.title('Likelihood of '+ str(remove_dot(pname_))+\
                            ' given '+'\n'+str(remove_dot(bldg_.convert_name(bn_out.all_cause_label))), fontsize=20)
                            plt.savefig(fig_dir+bldg_tag+'_'+pname_+'_'+sig_tag+'_LH_sensor_time_weather'+get_pngid()+'.png', bbox_inches='tight')
                            plt.close()
                        except:
                            print 'error in '+fig_name
                            pass
    plt.ion()
        
##############################################################################
# Obslete library files
##############################################################################
"""


        plt.ioff()
        for bldg_tag in bldg_tag_set:
            print '-------------------------'
            print bldg_tag
            print '-------------------------'
            for sig_tag in sig_tag_set:
                print sig_tag+'.....'
                cmd_str_='p_names='+bldg_tag+'.'+sig_tag+'p_names'
                exec(cmd_str_)
                for pname_ in p_names:
                    try:
                        blank_idx=pname_.index('.')
                        pname_=pname_.replace('.','_')
                    except:
                        pass
                    cmd_str_='optprob_set='+bldg_tag+'.analysis.'+sig_tag+'.'+pname_+'.optprob_set'
                    exec(cmd_str_)
                    cmd_str_= 's_names='+bldg_tag+'.'+sig_tag+'sensor_names'
                    exec(cmd_str_)
                    cmd_str_= 'optstate_set='+bldg_tag+'.analysis.'+sig_tag+'.'+pname_+'.optstate_set'
                    exec(cmd_str_)
                    num_picks=30
                    sort_idx=argsort(optprob_set)[::-1]
                    sort_lh=optprob_set[sort_idx[:num_picks]].T
                    sort_state=optstate_set[sort_idx[:num_picks]].T
                    fig=figure(figsize=(20.0,15.0))
                    subplot(2,1,1)
                    plt.plot(sort_lh,'-*')
                    x_label= list(np.array(s_names)[sort_idx[:num_picks]])
                    x_ticks=range(len(x_label))
                    plt.xticks(x_ticks,x_label,rotation=270, fontsize="small")
                    if sig_tag=='avg':
                        plt.title('Most relavant '+bldg_tag +'sensors to the peak (demand) of '+pname_,fontsize=20)
                    else:
                        plt.title('Most relavant '+bldg_tag +'sensors to the peak variations of '+pname_,fontsize=20)
                    plt.tick_params(labelsize='large')
                    plt.ylabel('Likelihood (From 0 to 1)',fontsize=18)
                    #plt.get_current_fig_manager().window.showMaximized()
                    plt.savefig(fig_dir+pname_+'_'+sig_tag+'_lh_sensors.png', bbox_inches='tight')
                    plt.close()
        
        plt.ion() 
        
        
        
def interpolation_measurement_2(data_dict,input_names,err_rate=1,sgm_bnd=20):
    print 'interploattion starts....'
    measurement_point_set=[]
    num_of_discrete_val=[]
    sampling_interval_set=[]
    num_type_set=[]
    err_rate=1;sgm_bnd=20
    for i,key_name in enumerate(input_names):
        print key_name,'.....'
        start_time = time.time()
        v = mt.loadObjectBinaryFast(str(key_name) + FL_EXT)
        t_=np.array(v[2][0])

        if len(t_) == 0:
            continue

        intpl_intv=np.ceil((t_[-1]-t_[0]) /len(t_))
        sampling_interval_set.append(intpl_intv)
        val_=np.array(v[2][1])
        num_of_discrete_val_temp=len(set(val_))
        num_of_discrete_val.append(num_of_discrete_val_temp)
        # filtering outlier
        # assuming 1% of errors and 30 x standard deviation rules
        outlier_idx=outlier_detect(val_,err_rate,sgm_bnd)
        if len(outlier_idx)>0:
            print 'outlier samples are detected: ', 'outlier_idx:', outlier_idx
            t_=np.delete(t_,outlier_idx)
            val_=np.delete(val_,outlier_idx)
        t_new=np.r_[t_[0]:t_[-1]:intpl_intv]
        num_type=check_data_type(v[2][1])
        if num_type==INT_TYPE:
            val_new=fast_nearest_interp(t_new, t_,val_)
        else:
            #num_type=FLOAT_TYPE
            val_new = np.interp(t_new, t_,val_)

        c=np.vstack([t_new,val_new])
        measurement_point_set.append(c)
        num_type_set.append(num_type)

	print 'interpolation_measurement one iteration done...'
	mt.print_report(start_time)
        print '-----------------------------------------------------------------'
    #return measurement_point_set,num_type_set,num_of_discrete_val,sampling_interval_set
    return measurement_point_set,np.array(num_type_set)
    
"""
