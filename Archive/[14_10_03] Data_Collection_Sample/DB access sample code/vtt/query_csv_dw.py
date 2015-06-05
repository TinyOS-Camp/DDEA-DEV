import httplib
import shlex
from datetime import datetime
import csv
import sys
from time import time
from multiprocessing import Process,Pipe
    
#finland = timezone('Europe/Helsinki')
server_dns = "121.78.237.160" #"keti3.oktree.com"
port = 8080 #4242


import itertools
import numpy as np
from scipy import linalg
import pylab as pl
import matplotlib as mpl

from sklearn import mixture


"""
-----query_csv------
ids = array of sensor names. eg: [sensor1,sensor2]
start_time = datetime
end_time = datetime 
returns numpy list of lists containing sensor data
"""
def query_csv(ids,start_time,end_time):
    all_data = []
    for id in ids:
        data_for_id = []
        ifile = open("%s/%s.csv"%(path,id),"r")
        reader = csv.reader(ifile)
        for row in reader:
            row_time = datetime.strptime(row[0],"%Y-%m-%d %H:%M:%S")
            if row_time >= start_time and row_time <= end_time:
                data_for_id.append(float(row[1]))
        all_data.append(data_for_id)        
    return all_data

path = "data_week"
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    ids = ["VAK2.TK4_TE40_M","GW2.HA50_MV10_KH_A","GW1.HA11_OE_FI"]
    start_time = datetime(2013,10,31)
    end_time = datetime(2013,11,07)
    all_data = query_csv(ids,start_time,end_time)
    print len(all_data), len(all_data[0]), len(all_data[1]), len(all_data[2])

X=np.asarray(all_data[0])
X=np.r_[X,X+3]
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

vbgmm=mixture.VBGMM(n_components=30, alpha=1.00,covariance_type='full')
vbgmm.fit(X)
Y_=vbgmm.predict(X)

pl.subplot(4,1,1)
pl.plot(X)
pl.ylabel('value')
pl.subplot(4,1,2)
pl.plot(vbgmm.weights_,'-s')
pl.ylabel('mixing prob')
pl.subplot(4,1,3)
pl.stem(vbgmm.means_, '-.')
pl.ylabel('means')
pl.xlabel('components')
pl.subplot(4,1,4)
for i, (mean, color) in enumerate(zip(vbgmm.means_, color_iter)):
    if not np.any(Y_==i):
        continue
    (X_idx,)=(Y_==i).nonzero()
    pl.plot(X_idx,X[Y_==i], color=color)
     
pl.ylabel('centroid')
  
 
 
        
