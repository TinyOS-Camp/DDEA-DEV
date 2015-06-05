import httplib
import shlex
from datetime import datetime
#from pytz import timezone
import csv
import sys
import math
from time import time
from multiprocessing import Process
    
#finland = timezone('Europe/Helsinki')
server_dns = "121.78.237.160" #"keti3.oktree.com"
port = 8080

def run(ids):        
    conn = httplib.HTTPConnection(server_dns,port)
    for id in ids:
        read_data(conn,id)

def read_data(conn,sensor):    
    request = "/dashboard/query/?start="+start_time+"&end="+end_time+"&m=avg:"+sensor+"&ascii"
    print "http://121.78.237.160:8080/" + request
    conn.request("GET", request)    
    response = conn.getresponse()
    data = response.read()
    #print data
    lines = data.split("\n")
    ofile= open(output_folder+sensor+".csv","w")    
    try:
        for line in lines:
            parts = shlex.split(line)
            if len(parts) > 1:            
                #print datetime.fromtimestamp(int(parts[1]),tz=finland).strftime('%Y-%m-%d %H:%M:%S'),parts[2]    
                ofile.write("%s,%s\n"%(datetime.fromtimestamp(int(parts[1])).strftime('%Y-%m-%d %H:%M:%S'),parts[2]))
    except KeyboardInterrupt:
            exit()
    except:
        pass            
    ofile.close()



def read_ids(filename):
    sensor_ids = []
    ifile = open(filename,"r")
    reader = csv.reader(ifile)
    for row    in reader:
        if len(row) > 1:
            sensor_ids.append(row[0].strip())
    ifile.close()        
    return sensor_ids        
        
program_start_time = time()
    
sensor_ids = ["GW2.HA50_MV10_KH_A"]#read_ids(sys.argv[1])
output_folder = 'csv/' #sys.argv[2] #data_day/

index = 0 # in case the script failed a certain sensor id, restart from that point sensor_ids.index("VAK1.SPR_PALO_H")
#print sensor_ids    
start_time = "2013/11/01-00:00:00"
end_time = "2013/11/07-23:59:00"

n = 6 #no of threads
factor = int(len(sensor_ids)/n)

print factor,len(sensor_ids)

all_processes = []

for i in range(n):
    if i == n-1:
        ids = sensor_ids[i*factor:]
    else:
        ids = sensor_ids[i*factor:(i+1)*factor]
    p = Process(target=run, args=(ids,)) 
    all_processes.append(p)
    p.start()        
    
for p in all_processes:
    p.join()
    
print len(all_processes)    

print "Run time (s): %f"%(time()-program_start_time)

#Multi processing: Run time (s): 6190.377293
#Serial processing: Run time (s): 25835.677658
