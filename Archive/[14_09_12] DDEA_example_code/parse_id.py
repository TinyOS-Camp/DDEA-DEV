# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:24:02 2014
@author: deokwoo
"""
# This script extract data point identifiers
#print(__doc__)
# Author: Deokwooo Jung deokwoo.jung@gmail.compile
import shlex, subprocess

def get_id_dict(grep_expr):
    # This get data point ids from id descriptoin file . 
    idinfo_file_name='KMEG_output.txt'
    temp = subprocess.check_output('cat  '+idinfo_file_name+ '|'+grep_expr, shell=True)
    temp2=temp.rsplit('\r\n')
    id_dict={}
    for each_temp2 in temp2:
        if len(each_temp2):
            temp3=each_temp2.rsplit(',')
            temp4=temp3[0].rsplit(' ')
            while temp4.count('') > 0:
                temp4.remove('')
            rest_desc=[]
            for desc in temp4[1:]:
                rest_desc.append(desc)
            for desc in temp3[1:]:
                rest_desc.append(desc)
            id_dict.update({temp4[0]:rest_desc})
    return id_dict


id_dict=get_id_dict('grep kW')
input_files=[]
for id_name in id_dict.keys():
    binfile_name=id_name+'.bin'
    input_files.append(binfile_name)


#id_dict.update({id_temp2:[]})
    



"""
openfile=open(filename,"r")
id_label=[]
#sensor_val=[]
#time_val=[];
for line in openfile:
    tmp=line.rstrip().rsplit(",")
    id_label.append(tmp)
    #sensor_val.append(float(tmp[1]))
    #temp=dt.datetime.strptime(tmp[0],"%Y-%m-%d %H:%M:%S")
    #temp=temp.timetuple()
    # Hour, Weekday, Day, Month
    #time_val.append([temp[3],temp[6],temp[2],temp[1]]) 
    
openfile.close()
"""

"""
plt.ion() # Interactive mode for plotting
num_files=len(input_files)
num_col_subplot=np.ceil(np.sqrt(num_files))


print 'Read sensor bin files in a single time_slots referece...'
print '----------------------------------'
sensor_list=[]
id_dict={}
for i,file_name in enumerate(input_files):
    print 'index ',i+1,': ',file_name
    # sensor value is read by time 
    sensor_name=file_name[:-4]
    for id_temp in sensor_name.split('.'):
        for id_temp2 in id_temp.split('_'):
            if id_temp2 not in id_dict.keys():
                id_dict.update({id_temp2:[]})
    sensor_list.append(sensor_name)
"""


    
    

