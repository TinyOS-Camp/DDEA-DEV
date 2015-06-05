import httplib
import shlex
from datetime import datetime
#from pytz import timezone
import csv
import sys
import math
from time import time
from multiprocessing import Process
import subprocess,shlex
    
#finland = timezone('Europe/Helsinki')
#server_dns = "keti3.oktree.com"
#port = 4242
server_dns = "121.78.237.160"
port = 4242

def run(ids):        
    conn = httplib.HTTPConnection(server_dns,port)
    for id in ids:
        read_data(conn,id)
    conn.close()

def read_data(conn,sensor):    
    request = "/q?start="+start_time+"&end="+end_time+"&m=avg:"+sensor+"%7B%7D&ascii"
    #print request
    st = time()
    conn.request("GET", request)    
    #print 'request takes: ' + str(time() - st) + ' secs'
    t_request = time() - st

    st = time()
    response = conn.getresponse()
    #print response.status,response.reason
    #print 'getresponse takes: ' + str(time() - st) + ' secs'
    t_response = time() - st

    st = time()
    data = response.read()
    #print 'response.read() takes: ' + str(time() - st) + 'secs'
    t_response_read = time() - st
    data_size = len(data)
    #print data
    st = time()
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
    t_io = time() - st
    csv_size = int(shlex.split(subprocess.check_output("stat -c %s " + output_folder+sensor+'.csv', shell=True))[0])
    print 'breakdown',sensor, t_request,t_response,t_response_read,data_size,csv_size


def read_ids(filename):
    sensor_ids = []
    ifile = open(filename,"r")
    reader = csv.reader(ifile)
    for row    in reader:
        if len(row) > 1:
            sensor_ids.append(row[0].strip())
    ifile.close()        
    return sensor_ids        

if __name__ == '__main__':        
    program_start_time = time()
        
    sensor_ids = read_ids(sys.argv[1])
    #sensor_ids = ['VAK1.CG_SYSTEM_REACTIVE_POWER_M']
    output_folder = sys.argv[2] #data_day/

    index = 0 # in case the script failed a certain sensor id, restart from that point sensor_ids.index("VAK1.SPR_PALO_H")
    #print sensor_ids    
    start_time = "2014/01/01-00:00:00"
    end_time = "2014/01/1-23:59:00"

    n = 1 #no of threads
    factor = int(len(sensor_ids)/n)

    print factor,len(sensor_ids)

    all_processes = []

    #run(['GW1.24H_LASKIN_T'])
    from random import sample
    #sensor_ids = sample(sensor_ids,10)
    sensor_ids = ['VAK1.CG_SYSTEM_REACTIVE_POWER_M']

    all_sensors = subprocess.check_output("ls -S Binfiles/*.bin", shell=True)
    all_sensors = shlex.split(all_sensors)[0:1]
    sensor_ids = [i.split('/')[-1][:-4] for i in all_sensors]

    for uuid in sensor_ids:
        for i in range(1):
            st = time()
            run([uuid])
            print "one query for %s (s): %f"%(uuid,time()-st)
    print "Run time (s): %f"%(time()-program_start_time)
"""

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
"""
    #Multi processing: Run time (s): 6190.377293
    #Serial processing: Run time (s): 25835.677658
