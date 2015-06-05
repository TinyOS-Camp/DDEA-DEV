import os
import sys
import re
import csv
from scipy.stats import kde
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl
from math import fabs
import pickle

#UUID_FILE = 'finland_ids.csv'
UUID_FILE = 'KMEG_output.txt'
DATA_FOLDER = 'VTT_year/'
DATA_EXT = '.csv'
SCRIPT_DIR = os.path.dirname(__file__)

key_list = []
key_set = set()
#key_description = {}
#description_list = []

def saveObjectBinary(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadObjectBinary(filename):
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return obj

def search(description_list, phrase):
    match_list = set()
    for description in description_list:
        if phrase in description:
            match_list.add(description)

    return list(match_list)

def save_matrix_to_file(mat, filename):
    temp = np.asarray(mat)
    np.savetxt(filename, temp, delimiter=",")

def load_matrix_from_file(filename):
    mat = np.loadtxt(open(filename, 'rb'), delimiter=',',skiprows=0)
    return mat
    
def  count(string_list):
    count = {}
    for token in string_list:
        key_word = token.rsplit(' ', 1)
        
        if key_word[0] in count.keys():
            count[key_word[0]] = count[key_word[0]] + 1
        else:
            count[key_word[0]] = 1

    count = count.items()
    count = sorted(count, key=lambda tup: tup[1])
#    print count

##    for tup in count:
##        print str(tup[0]) + ',' + str(tup[1])
    
    return count

def plot_kernel_density(x,ax=plt, title='Kernel Density'):
    np_x = np.array(x)
    density = kde.gaussian_kde(np_x)
    plt.hist(np_x, bins=16, normed=True)
    xgrid = np.linspace(np_x.min(), np_x.max(), 100)
    ax.plot(xgrid, density(xgrid), 'r-')
    #ax.set_title(title, fontsize=10)

    ax.show()
        
def load_uuid_list():
    uuid_list = []
    uuid_filepath = os.path.join(SCRIPT_DIR, UUID_FILE)
    temp_uuid_list = open(uuid_filepath).readlines()

    temp = []
    key_description = {}
    description_list = []
    
    for i in range(7):
        temp.append([])
        
    for line in temp_uuid_list:
        tokens = line.strip().split('   ')
        if len(tokens) < 2:
            continue

        uuid = tokens[0].strip()
        uuid_list.append(uuid)
        description_tokens = tokens[1].strip().lower().split(', ')

        
        for i in range(len(description_tokens)):
            temp[i].append(description_tokens[i])
            description_list.append(description_tokens[i])
            

        tokens = re.split('[._]',uuid)
        #for token in tokens:
        for i in range(len(tokens)):
            token = tokens[i]
            key_list.append(token)
            if token not in key_description.keys():
                key_description[token] = description_tokens[i]
        

    
    key_set = set(key_list)
    unique_key_list = list(key_set)

    ### print stats for each column ###
##    for i in range(len(temp)):
##        print 'Column ' + str(i) + ':' 
##        count(temp[i])
##        print
##        temp[i] = sorted(list(set(temp[i])))
        
        

    #count(temp[1])
    
    #print 'uuid list length: ' + str(len(uuid_list))
    #print 'key list length: ' + str(len(key_list))
    #print 'key set size: ' + str(len(key_set))
    
    
    #count(key_description.values())
    #count(description_list)
        
    return uuid_list, key_list, description_list, key_description

def classify_uuid(uuid, key_description, labels):
    labels_len = len(labels)
    class_vector = [0] * labels_len
    tokens = re.split('[._]',uuid)
    for key_word in tokens:
        # get description for this keyword
        description = key_description[key_word]
        # map discription to class name
        class_name = description.rsplit(' ',1)[0]
        # get index of class
        class_index = labels[class_name]

        # set class at class_index to 1
        class_vector[class_index] = 1

        #print key_word, class_name, class_index
            
    return class_vector

def construct_classification_matrix(uuid_list, key_description, labels):

    class_mat = []
    for uuid in uuid_list:
        uuid_class_vector = classify_uuid(uuid, key_description, labels)
        class_mat.append(uuid_class_vector)
        
    return np.array(class_mat)

def check_exclusive_vectors(vec1,vec2):

    vec_len = len(vec1)
    if vec_len != len(vec2):
        return False
    
    for i in range(vec_len):
        
        #if vec1[i] + vec2[i] >= 2:
        if vec1[i] + vec2[i] != 1:
            return False
        
    return True

def print_dict(my_dict):
    for k,v in my_dict.iteritems():
        print str(k) + ',' + str(v)

def save_tuple_list_csv(my_list, filename):
    writer = csv.writer(open(filename, 'wb'))
    for (k,v) in my_list:
        writer.writerow([k,v])
    
def save_dict_csv(my_dict, filename):
    writer = csv.writer(open(filename, 'wb'))
    for k,v in my_dict.iteritems():
        writer.writerow([k,v])

def compute_similarity_score(vec1,vec2,class_weights):

    score = 0.0
    if len(vec1) != len(vec2):
        return None
##    for i in range(len(vec1)):
##        score = score + (vec1[i] - vec2[i]) * class_weights[i]
    score = sum(np.absolute(np.array(vec1) - np.array(vec2)) * np.array(class_weights))
    return score

def construct_similarity_mat(uuid_list, key_description, labels, class_weights):
    class_mat = construct_classification_matrix(uuid_list, key_description, labels)
    uuid_list_len = len(uuid_list)
    sim_mat = np.zeros((uuid_list_len, uuid_list_len))
    for i in range(uuid_list_len):
        print 'row ' + str(i)
        for j in range(i+1, uuid_list_len):
            #print 'row ' + str(i) + ' col ' + str(j)
            score_ij = compute_similarity_score(class_mat[i], class_mat[j], class_weights)
            sim_mat[i,j] = score_ij
            sim_mat[j,i] = score_ij

    return sim_mat         
            
def get_sim_mat():
    uuid_list, key_list, description_list, key_description = load_uuid_list()

    phrase_count = count(description_list)
    phrases = [tup[0] for tup in phrase_count]
    counts = [tup[1] for tup in phrase_count]
    labels = {}
    for i in range(len(phrases)):
        labels[phrases[i]] = i

    # construct classification matrix
    ## class_mat = construct_classification_matrix(uuid_list, key_description, labels)
    sim_mat = construct_similarity_mat(uuid_list, key_description, labels,counts)
    #save_matrix_to_file(sim_mat, 'sim_mat.csv')

    return sim_mat, uuid_list, phrases, key_description, phrase_count
        
if __name__ == "__main__":

    sim_mat, uuid_list, phrases, key_description, phrase_count = get_sim_mat()

    ### save sim_mat into csv file ###
    save_matrix_to_file(sim_mat, 'sim_mat.csv')
    ### save sim_mat matrix into binary file ###
    saveObjectBinary(sim_mat, 'sim_mat.bin')

    ### load sim_mat matrix from binary file ###
    #sim_mat = loadObjectBinary('sim_mat.bin')
    ### load sim_mat from csv file ###
    #load_matrix_from_file('sim_mat.csv')

    ### Initial computation ###
    ##uuid_list, key_list, description_list, key_description = load_uuid_list()
    #save_dict_csv(key_description, 'key_description.csv')
    
    ### Key counts ###
##    key_counts = count(key_list)
##    #save_tuple_list_csv(key_counts, 'key_list.csv')
##    
##    phrase_count = count(description_list)
##    counts = [tup[1] for tup in phrase_count]
##    phrases = [tup[0] for tup in phrase_count]
##    labels = {}
##    for i in range(len(phrases)):
##        labels[phrases[i]] = i
    

    ### ###
    #uuid = 'GW2.CG_SYSTEM_REACTIVE_POWER_M'
    #classify_uuid(uuid, labels)
    #class_mat = construct_classification_matrix(uuid_list, key_description, labels)
    
    #sim_mat = construct_similarity_mat(uuid_list, key_description, labels, counts)
    #save_matrix_to_file(sim_mat, 'sim_mat.csv')
    
##    sim_mat = load_matrix_from_file('sim_mat.csv')
##    #sim_mat_reshape = np.absolute(sim_mat).reshape(1,sim_mat.shape[0] * sim_mat.shape[1])
##    sim_mat_reshape = np.absolute(sim_mat)[0:100][:,0:100].reshape(1,100 * 100)
##    plt.plot(sim_mat_reshape[0], [1] * sim_mat_reshape.shape[1])
##    plt.show()
    
    ### compute correlation coefficient matrix
##    corrcoef_mat = np.corrcoef(class_mat.T)
##
##    for i in range(0, corrcoef_mat.shape[0]):
##        print '=== ' + phrases[i] + ' ==='
##        for j in range(0, int(i/2)):
##            if fabs(corrcoef_mat[i,j]) >= 0.5:
##                print phrases[j] + ': ' + str(corrcoef_mat[i,j])
    #np.savetxt('corrcoef_mat.txt', corrcoef_mat)

    ### check exclusive ###
##    for i in range(class_mat.shape[1]):
##        vec1 = list(class_mat[:,i])
##        
##        for j in range(i + 1, class_mat.shape[1]):
##            vec2 = list(class_mat[:,j])
##            ret = check_exclusive_vectors(vec1,vec2)
##            if ret is True:
##                print 'found ' + str(i) + ' ' + str(j) + ': ' + phrases[i] + ',' + phrases[j]

    

    
    
##    plot_kernel_density(counts,ax=plt)
##    plt.plot(counts, 'r--')
##    plt.show()

    #print set(counts)
    #phrase = 'point'
    #print search(description_list, phrase)

