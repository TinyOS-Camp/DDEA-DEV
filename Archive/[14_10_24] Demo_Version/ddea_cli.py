#!/adsc/DDEA_PROTO/bin/python

from ddea_main import ddea_analysis
from datetime import datetime
import traceback


import sys

if __name__ == '__main__':
    try:
        if 4 <= len(sys.argv):

            ###urls = open(sys.argv[1]).readlines()
            sub_system = sys.argv[1]
            sensor = sys.argv[2]
            start_time = sys.argv[3]
            end_time = sys.argv[4]
            stime = datetime.strptime(start_time, "%y-%m-%d")
            etime = datetime.strptime(end_time, "%y-%m-%d")

            ddea_analysis(sub_system,
                          sensor,
                          stime,
                          etime)

        else:
            raise "Invalid Arguments"

    except:
        print traceback.print_exc()
        print("Example: %s GW1,GW2,VTT1,VTT2 POWER 14-01-01 14-02-02" % sys.argv[0])
        raise SystemExit
