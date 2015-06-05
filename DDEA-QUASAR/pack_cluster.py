# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:34:54 2014

@author: deokwooj
"""
from __future__ import division # To forace float point division
import numpy as np
#from numpy.linalg import norm
from sklearn import cluster
from sklearn.cluster import Ward
from sklearn.cluster import KMeans
from scipy.stats import stats 
import time
##################################################################
# Custom library
##################################################################
from data_tools import *
from shared_constants import *


def max_diff_dist_idx(dist_mat,min_dist,max_dist):
    num_nodes=dist_mat.shape[0]
    dist_diff=[]
    max_diff=-1
    max_diff_row=0
    max_diff_label=[]
    max_cluster_idx=[]
    for i,dist_vals in enumerate(dist_mat):
        # exclude its own distance
        idx_set=np.r_[np.r_[0:i:1],np.r_[i+1:num_nodes:1]]
        #print i,'th row k-mean cluster'    
        temp=dist_vals[idx_set]
        if np.min(temp)>max_dist:
            exemplar_idx=i
            max_cluster_idx=i
            #import pdb;pdb.set_trace()
            return exemplar_idx,max_cluster_idx
        
        ########################################
        # K-mean
        #_,label,_=cluster.k_means(temp[:,None],2)      
        # Herichical Binary Clutering
        ward = Ward(n_clusters=2).fit(temp[:,None])
        label=ward.labels_
        #kmean=KMeans(n_clusters=2).fit(temp[:,None])
        #label=kmean.labels_
        
        # max is default
        centroid=np.zeros(2)
        #import pdb;pdb.set_trace()
        centroid[0]=np.max(temp[label==0])
        centroid[1]=np.max(temp[label==1])
        #idx0=idx_set[np.nonzero(label==0)]
        #idx1=idx_set[np.nonzero(label==1)]
        #dist01=np.round([dist_mat[v0,v1] for v0 in idx0 for v1 in idx1],2)
        #num_min_dist_violation=len(np.nonzero(dist01<min_dist)[0])
        ########################################
        temp_1=abs(centroid[0]-centroid[1])
        cent_diff=centroid[0]-centroid[1]
        dist_diff.append(abs(cent_diff))
        if max_diff< temp_1:
        #if (max_diff< temp_1) and (num_min_dist_violation==0):
            max_idx_set=idx_set
            max_diff_row=i
            max_diff=temp_1
            max_diff_label=label
            max_cent_diff=cent_diff

    #import pdb;pdb.set_trace()
    cur_cent_idx=set([])
    if max_cent_diff>0:
        cur_cent_idx=cur_cent_idx| set(np.nonzero(max_diff_label==1)[0])
    else:
        cur_cent_idx=cur_cent_idx| set(np.nonzero(max_diff_label==0)[0])
    max_cluster_idx=list(set(max_idx_set[list(cur_cent_idx)]) |set([max_diff_row]))
    exemplar_idx=max_diff_row
    
    return exemplar_idx,max_cluster_idx

def signle_let_cluster_idx(dist_mat,max_dist):
    print max_dist
    num_nodes=dist_mat.shape[0]
    nodes_all_alone=[]
    exemplar_idx=[];
    max_cluster_idx=[]
    for i,dist_vals in enumerate(dist_mat):
        # exclude its own distance
        idx_set=np.r_[np.r_[0:i:1],np.r_[i+1:num_nodes:1]]
        temp=dist_vals[idx_set]
        #import pdb;pdb.set_trace()
        num_nodes_away_more_than_max_dist=len(np.nonzero(temp>max_dist)[0])
        #print temp
        if  num_nodes_away_more_than_max_dist==num_nodes-1:
            print '-----------------------------------------------------------'
            print i,'th node check'
            print '*** all nodes are away beyond max_dist **'
            nodes_all_alone.append(i)
            #exemplar_idx.append([i])
            exemplar_idx.append(i)
            #max_cluster_idx.append([i])
            max_cluster_idx.append(i)
    return exemplar_idx,max_cluster_idx
    

def udiag_min(a):
    return min([min(a[i,i+1:]) for i in range(a.shape[0]-1)])

def udiag_max(a):
    return max([max(a[i,i+1:]) for i in range(a.shape[0]-1)])
    
def udiag_avg(a):
    return sum([sum(a[i,i+1:]) for i in range(a.shape[0]-1)])\
    /((a.shape[0]-0)*(a.shape[0]-1)/2)


def max_pack_cluster(DIST_MAT,min_dist=0.3,max_dist=1.0):
    # minium distance for clusters set by max_dist=1.0 , min_dist=0.3
    # Initionalize
    num_nodes=DIST_MAT.shape[0]
    label=np.inf*np.ones(num_nodes)
    label_num=0
    remain_index=np.arange(num_nodes)
    dist_mat=DIST_MAT.copy()
    exemplar_list=[]
    while (dist_mat.shape[0]>2):
        #import pdb;pdb.set_trace()
        if udiag_min(dist_mat)>max_dist:
            print 'all samples are seperated further than max_dist'
            print 'remaining samples will be individual clusters' 
            # Assign different labels to all raminig samples
            inf_idx=np.nonzero(label==np.inf)[0]
            for r in inf_idx:
                exemplar_list.append(int(r))
            #label[inf_idx]=label_num+np.arange(len(inf_idx))
            label[inf_idx]=np.int_(label_num+np.arange(len(inf_idx)))
            #import pdb;pdb.set_trace()
            break
            
        elif udiag_max(dist_mat)<min_dist:
            # Assign the same label to all raminig samples
            print 'all samples are seperated within min_dist'
            print 'remaining samples will be the same' 
            inf_idx=np.nonzero(label==np.inf)[0]
            exemplar_list.append(int(inf_idx[0]))
            label[inf_idx]=int(label_num)
            #import pdb;pdb.set_trace()
            break
        else:
            exemplar_idx,max_cluster_idx=max_diff_dist_idx(dist_mat,min_dist,max_dist)
            dcluster_idx=remain_index[max_cluster_idx]
            exemplar_list.append(np.int_(remain_index[exemplar_idx]))
            #import pdb;pdb.set_trace()
            # Update dist_mat and remain_idx
            dist_mat=np.delete(dist_mat, max_cluster_idx, axis=0)
            dist_mat=np.delete(dist_mat, max_cluster_idx, axis=1)    
            remain_index=np.delete(remain_index,max_cluster_idx, axis=0)
            # Adding label info
            label[dcluster_idx]=label_num;label_num+=1
            print 'dist_mat.max()=', dist_mat.max()

    unassigned_idx=np.nonzero(label==np.inf)[0]
    if len(unassigned_idx)>0:
        label[unassigned_idx]=label_num+np.arange(len(unassigned_idx))
        exemplar_list=exemplar_list+list(unassigned_idx)
        
        #raise NameError('There exist the unassigned: '+str(unassigned_idx))
    intra_err_cnt, inter_err_cnt=check_bounded_distance_constraint_condition(DIST_MAT,label,min_dist,max_dist)        
    return np.int_(exemplar_list),np.int_(label)

def compute_cluster_err(DIST_MAT,m_labels):
    num_clusters=int(m_labels.max())+1
    # Compute Intra-Cluster Distance
    c_wgt_set=np.zeros(num_clusters)
    c_dist_w_min=np.zeros(num_clusters)
    c_dist_w_max=np.zeros(num_clusters) 
    c_dist_w_avg=np.zeros(num_clusters)
    for i in range(num_clusters):
        c_idx=np.nonzero(m_labels==i)[0]
        c_wgt=c_idx.shape[0]/DIST_MAT.shape[0]
        c_wgt_set[i]=c_wgt
        if c_idx.shape[0]>1:
            # sample weight of the cluster
            c_dist_w_min[i]=udiag_min(DIST_MAT[c_idx,:][:,c_idx])
            c_dist_w_max[i]=udiag_max(DIST_MAT[c_idx,:][:,c_idx])
            c_dist_w_avg[i]=udiag_avg(DIST_MAT[c_idx,:][:,c_idx])
        else:
            c_dist_w_min[i]=0
            c_dist_w_max[i]=0
            c_dist_w_avg[i]=0
    intra_dist_min=sum(c_dist_w_min*c_wgt_set)
    intra_dist_avg=sum(c_dist_w_avg*c_wgt_set)
    intra_dist_max=sum(c_dist_w_max*c_wgt_set)
    intra_dist_bnd=np.array([intra_dist_min, intra_dist_avg,intra_dist_max])
    
    inter_dist=[]
    # Compute Inter-Cluster Distance
    if num_clusters>1:
        for i in range(num_clusters-1):
            for j in range(i+1,num_clusters):
                i_idx=np.nonzero(m_labels==i)[0]
                j_idx=np.nonzero(m_labels==j)[0]
                temp_mat=DIST_MAT[i_idx,:][:,j_idx]
                inter_dist.append(temp_mat.min())
    
        inter_dist=np.array(inter_dist)
        inter_dist_bnd=np.array([inter_dist.min(), inter_dist.mean(),inter_dist.max()])
        validity=intra_dist_avg/inter_dist.min()
    else:
        validity=0
        inter_dist_bnd=0
        
    return validity,intra_dist_bnd,inter_dist_bnd
    
    
def show_clusters(exemplars,labels,input_names):
    n_labels = labels.max()
    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(input_names[labels == i])))
    

def plot_label(X_val,X_name,labels,exemplar,label_idx_set):
    num_label=len(label_idx_set)
    if num_label>15:
        fsize=6
    elif num_label>10:
        fsize=8
    elif num_label>5:
        fsize=10
    else:
        fsize=12
        
    for k,label_idx in enumerate(label_idx_set):
        fig = plt.figure('Label '+str(label_idx)+' Measurements')
        fig.suptitle('Label '+str(label_idx)+' Measurements',fontsize=fsize)
        idx=np.nonzero(labels==label_idx)[0]
        exemplar_idx=exemplar[label_idx]
        num_col=int(np.ceil(np.sqrt(len(idx))))
        num_row=num_col
        for k,i in enumerate(idx):    
            ax=plt.subplot(num_col,num_row,k+1)
            plt.plot(X_val[:,i])
            if exemplar_idx==i:
                plt.title('**'+X_name[i]+'**',fontsize=fsize)
            else:
                plt.title(X_name[i],fontsize=fsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fsize) 
            if (k<num_col*(num_row-1)):
                for tick in ax.xaxis.get_major_ticks():
                    ax.set_xticklabels( () )
        plt.get_current_fig_manager().window.showMaximized()


def check_bounded_distance_constraint_condition(dist_mat,labels,min_dist,max_dist):
    intra_err_cnt=0
    num_clusters=int(labels.max()+1)
    print '------------------------------------------------------------------------------------------'
    print 'Intra-Cluster distance check.....'        
    print 'Condition: inter-cluster distance is upper-bounded by',round(max_dist,2)        
    print '------------------------------------------------------------------------------------------'
    for i in range(num_clusters):
        idx_set=np.nonzero(labels==(i))[0]
        #print '----------------------------------------------------------'
        #print i,'th cluster: ',idx_set
        for idx_pair in pair_in_idx(idx_set):
            #print idx_pair, 'dist-',round(dist_mat[idx_pair[0],idx_pair[1]],2)
            dist_val_=dist_mat[idx_pair[0],idx_pair[1]]
            # Rule violation
            if dist_val_>max_dist:
                print '*** the distance of pairs :',idx_pair,'in ',i,'th cluster ~',np.round(dist_val_,2),' > max_dist=', np.round(max_dist,2),'***'          
                intra_err_cnt=intra_err_cnt+1
    print '------------------------------------------------------------------------------------------'
    print 'Inter-Cluster distance check.....'       
    print 'Condition: intra-cluster distance is lower-bounded by',round(min_dist,2)
    print '------------------------------------------------------------------------------------------'
    cluster_pairs=pair_in_idx(range(num_clusters))
    inter_err_cnt=0
    for c_pair in cluster_pairs:
        idx_set_0=np.nonzero(labels==(c_pair[0]))[0]
        idx_set_1=np.nonzero(labels==(c_pair[1]))[0]
        #print '----------------------------------------------------------'
        #print 'The pairwise distance between ',c_pair[0],'th cluster and',c_pair[1],'th cluster'
        for idx_pair in pair_in_idx(idx_set_0,idx_set_1):
            #print idx_pair, 'dist-',round(dist_mat[idx_pair[0],idx_pair[1]],2)
            dist_val_=dist_mat[idx_pair[0],idx_pair[1]]
            # Rule violation
            if dist_val_<min_dist:
                print '*** the distance of pairs :',idx_pair[0] ,'in', c_pair[0] ,' and ', idx_pair[1] ,'in',c_pair[1],'~',round(dist_val_,2),' < min_dist=', round(min_dist,2),'***'          
                inter_err_cnt=inter_err_cnt+1
    return intra_err_cnt, inter_err_cnt


def cluster_measurement_points(m_matrix,m_name,corr_bnd=[0.1,0.9],alg='aff'):
    exemplars_dict={}    
    if m_matrix.shape[1]==0:
        return [],exemplars_dict,[],[]
    elif m_matrix.shape[1]==1:
        exemplars_=[0]
        labels_=[0]
        exemplars_name=m_name
    else:
        distmat_input=find_norm_dist_matrix(m_matrix)
        # Find representative set of sensor measurements 
        min_dist_=np.sqrt(2*(1-(corr_bnd[1])))
        max_dist_=np.sqrt(2*(1-(corr_bnd[0])))
        if alg=='pack':
            print 'use pack clustering algoirthm'
            exemplars_,labels_=max_pack_cluster(distmat_input,min_dist=min_dist_,max_dist=max_dist_)
        else:
            print 'use affinity clustering algoirthm'
            SIMM_MAT=2-distmat_input
            exemplars_,labels_=cluster.affinity_propagation(SIMM_MAT,damping=0.5)

        
        num_clusters=int(labels_.max()+1)
        print '-' * 40
        print num_clusters, 'clusters out of ', len(labels_), 'measurements'
        print '-' * 40
        validity,intra_dist,inter_dist=compute_cluster_err(distmat_input,labels_)
        print 'validity:',round(validity,2),', intra_dist: ',np.round(intra_dist,2),', inter_dist: ',np.round(inter_dist,2)
        print '-' * 40
        exemplars_name=list(np.array(m_name)[exemplars_])
    
    for label_id,(m_idx,exemplar_label) in enumerate(zip(exemplars_,exemplars_name)):
        print exemplar_label
        children_set=list(set(np.nonzero(labels_==label_id)[0])-set([m_idx]))
        print 'Label ', label_id, ': ',m_idx,'<--', children_set
        exemplars_dict.update({exemplar_label:list(np.array(m_name)[children_set])})
    return m_matrix[:,exemplars_], exemplars_dict,exemplars_,labels_


def CLUSTERING_TEST(distmat_input,min_corr=0.1,max_corr=0.9):
    ################################################################################
    # Unsupervised clustering for sensors given the normalized euclidian distance
    # of sensor data
    # Find only a few represetative sensors out of many sensors
    ################################################################################
    # exemplars are a set of representative signals for each cluster
    # Smaller dampding input will generate more clusers, default is 0.5
    # 0.5 <= damping <=0.99
    ################################################################################
    print '==========================================================='
    print 'Clustering Test'
    print '==========================================================='
    print 'Pack Clustering'
    print '---------------------------'
    min_dist_=np.sqrt(2*(1-(max_corr)))
    max_dist_=np.sqrt(2*(1-(min_corr)))
    pack_exemplars,pack_labels=max_pack_cluster(distmat_input,min_dist=min_dist_,max_dist=max_dist_)
    pack_num_clusters=int(pack_labels.max()+1)
    print '-------------------------------------------------------------------------'
    print pack_num_clusters, 'clusters out of ', len(pack_labels), 'measurements'
    print '-------------------------------------------------------------------------'
    validity,intra_dist,inter_dist=compute_cluster_err(distmat_input,pack_labels)
    print 'validity:',round(validity,2),', intra_dist: ',np.round(intra_dist,2),', inter_dist: ',np.round(inter_dist,2)
    print '-------------------------------------------------------------------------'
    
    
    max_num_clusters=pack_num_clusters   
    print 'Heirachical Clustering'
    print '---------------------------'
    ward_validity_log=[];
    ward_intra_dist_log=[];
    ward_inter_dist_log=[];
    ward_num_clusters_log=[]
    for k in range(2,max_num_clusters+1):
        ward = Ward(n_clusters=k).fit(distmat_input.T)
        ward_labels=ward.labels_
        ward_validity,ward_intra_dist,ward_inter_dist=compute_cluster_err(distmat_input,ward_labels)
        ward_num_clusters=int(ward_labels.max()+1)
        ward_validity_log.append(ward_validity);
        ward_intra_dist_log.append(list(ward_intra_dist));
        ward_inter_dist_log.append(list(ward_inter_dist));
        ward_num_clusters_log.append(ward_num_clusters)
    ward_intra_dist_log=np.array(ward_intra_dist_log);
    ward_inter_dist_log=np.array(ward_inter_dist_log)
    
    

    print 'K-Mean Clustering'
    print '---------------------------'
    kmean_validity_log=[];
    kmean_intra_dist_log=[];
    kmean_inter_dist_log=[];
    kmean_num_clusters_log=[]
    for k in range(2,max_num_clusters+1):
        kmean=KMeans(n_clusters=k).fit(distmat_input.T)
        kmean_labels=kmean.labels_
        kmean_validity,kmean_intra_dist,kmean_inter_dist=compute_cluster_err(distmat_input,kmean_labels)
        kmean_num_clusters=int(kmean_labels.max()+1)
        kmean_validity_log.append(kmean_validity);
        kmean_intra_dist_log.append(list(kmean_intra_dist));
        kmean_inter_dist_log.append(list(kmean_inter_dist));
        kmean_num_clusters_log.append(kmean_num_clusters)

    kmean_intra_dist_log=np.array(kmean_intra_dist_log);
    kmean_inter_dist_log=np.array(kmean_inter_dist_log)
    
    
    
    print 'Affinity Clustering'
    print '---------------------------'
    SIMM_MAT=2-distmat_input
    aff_exemplars, aff_labels = cluster.affinity_propagation(SIMM_MAT,damping=0.5)
    aff_num_clusters=int(aff_labels.max()+1)
    aff_validity,aff_intra_dist,aff_inter_dist=compute_cluster_err(distmat_input,aff_labels)
    
    
    fig = plt.figure('Intra_dist')
    fig.suptitle('Intra_dist')
    plot(pack_num_clusters,intra_dist[0],'s',label='pack')
    plot(pack_num_clusters,intra_dist[1],'s',label='pack')
    plot(pack_num_clusters,intra_dist[2],'s',label='pack')
    plot(ward_num_clusters_log,ward_intra_dist_log[:,0],'-+',label='ward')
    plot(ward_num_clusters_log,ward_intra_dist_log[:,1],'-+',label='ward')
    plot(ward_num_clusters_log,ward_intra_dist_log[:,2],'-+',label='ward')
    plot(kmean_num_clusters_log,kmean_intra_dist_log[:,0],'-v',label='kmean')
    plot(kmean_num_clusters_log,kmean_intra_dist_log[:,1],'-v',label='kmean')
    plot(kmean_num_clusters_log,kmean_intra_dist_log[:,2],'-v',label='kmean')
    plot(aff_num_clusters,aff_intra_dist[0],'*',label='aff')
    plot(aff_num_clusters,aff_intra_dist[1],'*',label='aff')
    plot(aff_num_clusters,aff_intra_dist[2],'*',label='aff')
    plt.legend()
    
    fig = plt.figure('Inter_dist')
    fig.suptitle('Inter_dist')
    plot(pack_num_clusters,inter_dist[0],'s',label='pack')
    plot(pack_num_clusters,inter_dist[1],'s',label='pack')
    plot(pack_num_clusters,inter_dist[2],'s',label='pack')
    plot(ward_num_clusters_log,ward_inter_dist_log[:,0],'-+',label='ward')
    plot(ward_num_clusters_log,ward_inter_dist_log[:,1],'-+',label='ward')
    plot(ward_num_clusters_log,ward_inter_dist_log[:,2],'-+',label='ward')
    plot(kmean_num_clusters_log,kmean_inter_dist_log[:,0],'-v',label='kmean')
    plot(kmean_num_clusters_log,kmean_inter_dist_log[:,1],'-v',label='kmean')
    plot(kmean_num_clusters_log,kmean_inter_dist_log[:,2],'-v',label='kmean')
    plot(aff_num_clusters,aff_inter_dist[0],'*',label='aff')
    plot(aff_num_clusters,aff_inter_dist[1],'*',label='aff')
    plot(aff_num_clusters,aff_inter_dist[2],'*',label='aff')
    plt.legend()
    
    fig = plt.figure('Validity')
    fig.suptitle('Validity')
    plot(pack_num_clusters,validity,'s',label='pack')
    plot(ward_num_clusters_log,ward_validity_log,'-+',label='ward')
    plot(kmean_num_clusters_log,kmean_validity_log,'-v',label='kmean')
    plot(aff_num_clusters,aff_validity,'*',label='aff')
    plt.legend()

    aff_intra_err_cnt, aff_inter_err_cnt=check_bounded_distance_constraint_condition(distmat_input,aff_labels,min_dist_,max_dist_)        
    ward_intra_err_cnt, ward_inter_err_cnt=check_bounded_distance_constraint_condition(distmat_input,ward_labels,min_dist_,max_dist_)        
    kmean_intra_err_cnt, kmean_inter_err_cnt=check_bounded_distance_constraint_condition(distmat_input,kmean_labels,min_dist_,max_dist_)        
    pack_intra_err_cnt, pack_inter_err_cnt=check_bounded_distance_constraint_condition(distmat_input,pack_labels,min_dist_,max_dist_)        

    print 'error count'
    print '-----------------------------'
    print 'pack_intra_err_cnt:', pack_intra_err_cnt,   'pack_inter_err_cnt:', pack_inter_err_cnt
    print 'aff_intra_err_cnt:', aff_intra_err_cnt,     'aff_inter_err_cnt:', aff_inter_err_cnt
    print 'ward_intra_err_cnt:', ward_intra_err_cnt,   'ward_inter_err_cnt:', ward_inter_err_cnt
    print 'kmean_intra_err_cnt:', kmean_intra_err_cnt, 'kmean_inter_err_cnt:', kmean_inter_err_cnt
    
    print '==========================================================='
    print 'End of Clustering Test'
    print '==========================================================='