#!/adsc/DDEA_PROTO/bin/python

from df_data_analysis_ddea import ddea_analysis
from datetime import datetime
import traceback


import sys

if __name__ == '__main__':
    try:
        if 3 <= len(sys.argv):

            ###urls = open(sys.argv[1]).readlines()
            start_time = sys.argv[1]
            end_time = sys.argv[2]

            stime = datetime.strptime(start_time, "%y-%m-%d")
            etime = datetime.strptime(end_time, "%y-%m-%d")
            ddea_analysis('',  stime, etime)

        else:
            raise "Invalid Arguments"

    except:
        print traceback.print_exc()
        print("Example: %s 14-01-01 14-02-02" % sys.argv[0])
        raise SystemExit
