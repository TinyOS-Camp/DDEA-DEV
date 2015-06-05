# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 19:24:11 2014

@author: NGO Quang Minh Khiem
@e-mail: khiem.ngo@adsc.com.sg

"""
from __future__ import division # To forace float point division
import numpy as np
from pandas import DataFrame
# R libs
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas.rpy.common as com
from rpy2.robjects import pandas2ri
import networkx as nx
import matplotlib.pyplot as plt

#============================================================#
# Utility functions and Misc
#============================================================#
def write_to_file(filename,text):
    with open(filename,'w') as f:
        f.write(text)

# Close X11 window
def dev_off():
    r['dev.off']()
#============================================================#
# Methods for Plotting
#============================================================#
# visualize graph from adjacence matrix r_graph
# for quick usage: set simple=True (by default)
# otherwise, function allows customize some properties of the graph
def nx_plot(r_graph, cols_names, simple=True, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):
    
    #G = nx.Graph()
    dg = nx.DiGraph()
    edges = []
    np_amat = np.asarray(bnlearn.amat(r_graph))
    for ri in range(np_amat.shape[0]):
        for ci in range(np_amat.shape[1]):
            if np_amat[ri,ci] == 1:
                #G.add_edge(cols_names[ri],cols_names[ci])
                dg.add_edge(cols_names[ri],cols_names[ci])
                edges.append((cols_names[ri],cols_names[ci]))
                
    #import pdb;pdb.set_trace()
    if simple:            
        if graph_layout=='spectral':
            nx.draw_spectral(dg,font_size=node_text_size)
        elif graph_layout=='random':
            nx.draw_random(dg,font_size=node_text_size)  
        elif graph_layout=='circular':
            nx.draw_circular(dg,font_size=node_text_size)  
        elif graph_layout=='spring':
            nx.draw_spring(dg,font_size=node_text_size)  
        else:
            nx.draw(dg,font_size=node_text_size)
    else:
        draw_graph(edges,directed=True, labels=labels, graph_layout=graph_layout,
               node_size=node_size, node_color=node_color, node_alpha=node_alpha,
               node_text_size=node_text_size,
               edge_color=edge_color, edge_alpha=edge_alpha, edge_tickness=edge_tickness,
               edge_text_pos=edge_text_pos,
               text_font=text_font)
    #nxlib.draw_graph(dg,labels=cols_names)

def nx_plot2(r_graph,cols_names,is_bnlearn=True):

    G = nx.Graph()
    dg = nx.DiGraph()
    if is_bnlearn:
        np_amat = np.asarray(bnlearn.amat(r_graph))
        for ri in range(np_amat.shape[0]):
            for ci in range(np_amat.shape[1]):
                if np_amat[ri,ci] == 1:
                    G.add_edge(cols_names[ri],cols_names[ci])
                    dg.add_edge(cols_names[ri],cols_names[ci])

    else:
        np_amat = np.asarray(r_graph)
        for ri in range(np_amat.shape[0]):
            for ci in range(np_amat.shape[1]):
                if np_amat[ri,ci] >= 0:

                    #G.add_weighted_edges_from([(cols_names[ri],cols_names[ci],{'weight': np_amat[ri,ci]})])
                    G.add_edge(cols_names[ri],cols_names[ci],weight=np_amat[ri,ci])
                    #dg.add_weighted_edges_from([(cols_names[ri],cols_names[ci],np_amat[ri,ci])])
    #nx.draw(G,nx.shell_layout)
    nx.draw(G)
    #nxlib.draw_graph(dg,labels=cols_names)

# a more generic graph plotting function, using networkx lib
# graph is a list of edges
def draw_graph(graph, directed=True, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):
    
    # create networkx graph
    #G=nx.Graph()
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])
    
    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    """
    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    """
    if labels is not None:
        edge_labels = dict(zip(graph, labels))
        nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
                                     label_pos=edge_text_pos)

    # show graph
    plt.show()
#============================================================#
# bnlearn wrapper APIs
#============================================================#
###
# Construct list of arcs used for blacklisting/whitelisting
# arc list is a list of arcs. For example:
# arc_list =
#   [['A','B'] , ['A','C']]
#
# return data frame in the following format
#     from to
# 0    A  B
# 1    A  C
###
def construct_arcs_frame(arc_list):
    data_frame = DataFrame(data=np.array(arc_list),columns=['from','to'])
    return data_frame

def print_bw_rules():

    rules = """
    ============================================================
    Blacklisting Rules:
    -------------------
    1. any arc blacklisted in one of its possible directions is never present in the graph.
        if A-->B is blacklisted (but B-->A is not), A-->B and A--B are never
        present in the graph (but not B-->A)
    2. any arc blacklisted in both directions, as well as the corresponding
        undirected arc, is never present in the graph.
        B(A-->B,B-->A) => B(A--B)

    Whitelisting Rules:
    -------------------
    1. arcs whitelisted in one direction only (i.e. A-->B is whitelisted but B-->A is not)
        have the respective reverse arcs blacklisted,
        and are always present in the graph.
        W(A-->B) => B(B-->A,A--B)
    2. arcs whitelisted in both directions (i.e. both A--> B and B-->A are whitelisted)
        are present in the graph,
        but their direction is set by the learning algorithm.
    3. any arc whitelisted and blacklisted at the same time is assumed to be whitelisted,
        and is thus removed from the blacklist.

    ============================================================
    """
    print rules

def convert_pymat_to_rfactor(py_mat):
    mat_shape = py_mat.shape
    r_factor_vec = r.factor(py_mat)
    r_factor_mat = r.matrix(r_factor_vec, nrow=mat_shape[1], byrow=True)

    return np.array(r_factor_mat).reshape(mat_shape[0],mat_shape[1],order='C')

def construct_data_frame(data_mat,columns=[]):
    if len(columns) == 0:
        column_names = range(data_mat.shape[1])
    else:
        column_names = columns

    return DataFrame(data=data_mat,columns=column_names)

"""
def py_bnlearn(data_frame,method='gs',blacklist=None, whitelist=None):

    # For hill-climbing, the data must be real or factor
    #
    if method == 'hc':

        bn_structure = bnlearn.hc(data_frame)
    else:
        bn_structure = bnlearn.gs(data_frame)

    return bn_structure
"""

#============================================================#
# APIs related to bn_learn structure
#============================================================#
#=======================|
# bn structure and graph|
#=======================|
def acyclic(bn_structure):
    return bool(bnlearn.acyclic(bn_structure)[0])

def amat(bn_structure):
    return np.array(bnlearn.amat(bn_structure))

def py_get_amat(bn_structure):
    return np.array(bnlearn.amat(bn_structure))

#=======================|
# Arcs                  |
#=======================|
def narcs(bn_structure):
    return bnlearn.narcs(bn_structure)[0]

def arcs(bn_structure):
    arcs = np.array(bnlearn.arcs(bn_structure))
    ncols = 2
    nrows = len(arcs) / 2
    arcs = arcs.reshape(nrows,ncols,order='F')

    return arcs

def directed_arcs(bn_structure):
    arcs = np.array(bnlearn.directed_arcs(bn_structure))
    ncols = 2
    nrows = len(arcs) / 2
    arcs = arcs.reshape(nrows,ncols,order='F')

    return arcs

def undirected_arcs(bn_structure):
    arcs = np.array(bnlearn.undirected_arcs(bn_structure))
    ncols = 2
    nrows = len(arcs) / 2
    arcs = arcs.reshape(nrows,ncols,order='F')

    return arcs

def incoming_arcs(bn_structure, node_name):
    arcs = np.array(bnlearn.incoming_arcs(bn_structure, node_name))
    ncols = 2
    nrows = len(arcs) / 2
    arcs = arcs.reshape(nrows,ncols,order='F')
    return arcs

def outgoing_arcs(bn_structure, node_name):
    arcs = np.array(bnlearn.outgoing_arcs(bn_structure, node_name))
    ncols = 2
    nrows = len(arcs) / 2
    arcs = arcs.reshape(nrows,ncols,order='F')

    return arcs

#=======================|
# Nodes                 |
#=======================|
def nnodes(bn_structure):
    return bnlearn.nnodes(bn_structure)[0]

def degree(bn_structure, node_name):
    return bnlearn.degree(bn_structure, node_name)[0]

def in_degree(bn_structure, node_name):
    return bnlearn.in_degree(bn_structure, node_name)[0]

def out_degree(bn_structure, node_name):
    return bnlearn.out_degree(bn_structure, node_name)[0]

def root_nodes(bn_structure):
    return np.array(bnlearn.root_nodes(bn_structure))

def leaf_nodes(bn_structure):
    return np.array(bnlearn.leaf_nodes(bn_structure))

def children(bn_structure, node_name):
    return np.array(bnlearn.children(bn_structure, node_name))

def parents(bn_structure, node_name):
    return np.array(bnlearn.parents(bn_structure, node_name))

def nbr(bn_structure, node_name):
    return np.array(bnlearn.nbr(bn_structure, node_name))

#=======================|
# bn fit                |
#=======================|
###
# To fit data to bn structure, the graph must be completely directed
###
def py_bn_fit(bn_structure,data_frame):
    fit = bnlearn.bn_fit(bn_structure,data_frame)
    return fit

def py_get_node_cond_mat(fit,node_indx):
    """
        Each item in fit is a list vector with dimension attributes
        fit[node_indx] has 4 attributes ['node', 'parents', 'children', 'prob']


    """
    node_fit = fit[node_indx]
    node = node_fit[0]
    parents = node_fit[1]
    children = node_fit[2]
    prob = node_fit[3]

    """
        prob is a vector Array type in R, which contains the conditional
        probability table of this node.
        prob is a (n_0 x n_1 x ... x n_parents) matrix, where each n_i is the number
        of discrete values of each node in the list prob_dimnames
        prob_dimnames contains the name of each dimension.
    """
    prob_dimnames = np.array(prob.dimnames.names)
    prob_factors = np.array(prob.dimnames)
    prob_mat = np.array(prob)
    #prob_frame = DataFrame(data=prob_mat[0],columns=prob_dimnames)

    return prob_dimnames,prob_factors,prob_mat

def bn_fit_barchart(fit, node_idx):
    print bnlearn.bn_fit_barchart(fit[node_idx])

def bn_fit_dotplot(fit, node_idx):
    print bnlearn.bn_fit_dotplot(fit[node_idx])

#==========================================================================#
#==========================================================================#
#==========================================================================#

#============================================================#
# Use R bnlearn to learn the Bayes network structure
#============================================================#
### BN Learn
## load some R libs
r = robjects.r
utils = importr("utils")
bnlearn = importr("bnlearn")
#rgraphviz = importr("Rgraphviz")
pandas2ri.activate() ### this is important to seamlessly convert from pandas to R data frame

"""
a = com.load_data('learning.test')
#d = construct_data_frame(a)
gs = py_bnlearn(a)
amat = py_get_amat(gs)
#fit = py_bn_fit(gs,a)
"""