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

def read_data(sensor,start_time,end_time,child_pipe):    
    results = []
    conn = httplib.HTTPConnection(server_dns,port)
    request = "/dashboard/query/?start="+start_time.strftime('%Y/%m/%d-%H:%M:%S')+"&end="+end_time.strftime('%Y/%m/%d-%H:%M:%S')+"&m=avg:"+sensor+"&ascii&callback"
    #print request
    conn.request("GET", request)    
    response = conn.getresponse()
    data = response.read()
    #print json_text
    #data = json.load(json_text)
    #print data
    lines = data.split("\n")    
    #ofile = open("d1.csv","w")
    try:
        for line in lines:            
            parts = shlex.split(line)
            #print parts
            if len(parts) > 1:
                line_time = datetime.fromtimestamp(int(parts[1]))    
                #print line_time,start_time,end_time            
                if line_time >= start_time and line_time < end_time:            
                    #ofile.write(datetime.fromtimestamp(int(parts[1])).strftime('%Y-%m-%d %H:%M:%S')+","+parts[2]    +"\n")
                    
                    results.append(float(parts[2]))
    except KeyboardInterrupt:
            exit()
    
    #print results                
    conn.close()
    child_pipe.send(results)

def query_data(ids,start_time,end_time):
    all_data = []
    all_processes = []
    parent_pipes = []     
    for i in ids:            
        parent_pipe,child_pipe = Pipe()
        p = Process(target=read_data, args=(i,start_time,end_time,child_pipe,))         
        all_processes.append(p)
        parent_pipes.append(parent_pipe)
        p.start()        
        
    for i in range(len(all_processes)):
        all_data.append(parent_pipes[i].recv())
        all_processes[i].join()
    
    return all_data

if __name__ == "__main__":
    program_start_time = time()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    ids = ["VAK2.TK4_TE40_M","GW2.HA50_MV10_KH_A","GW1.HA11_OE_FI"]
    start_time = datetime(2013,11,1)
    end_time = datetime(2013,11,07)
    all_data = query_data(ids,start_time,end_time)    
    
    print "Run time (s): %f"%(time()-program_start_time)
    
    import query_csv
    all_data2 = query_csv.query_csv(ids,start_time,end_time)    
    
    print len(all_data), len(all_data[0]), len(all_data[1]), len(all_data[2])
    print len(all_data2), len(all_data2[0]), len(all_data2[1]), len(all_data2[2])                
    
    if all_data == all_data2:
        print "Arrays match from both methods"
    else:
        print "Arrays from both methods don't match"

