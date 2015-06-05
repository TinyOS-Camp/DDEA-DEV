import sys
import csv
from datetime import datetime
import numpy as np

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
            row_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            if row_time >= start_time and row_time < end_time:
                data_for_id.append(float(row[1]))
        all_data.append(data_for_id)        
    return all_data

path = "data_week"
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    ids = ["VAK2.TK4_TE40_M","GW2.HA50_MV10_KH_A", "GW1.HA11_OE_FI"]
    start_time = datetime(2013,10,31)
    end_time = datetime(2013,11,07)
    all_data = query_csv(ids,start_time,end_time)
    print len(all_data), len(all_data[0]), len(all_data[1]), len(all_data[2])
