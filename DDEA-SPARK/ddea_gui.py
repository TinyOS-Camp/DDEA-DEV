#!/adsc/DDEA_PROTO/bin/python

from httpctrl import ServeGUI
from ddea_spark_main import ddea_spark_main, sc

if __name__ == '__main__':
    ServeGUI(ddea_main=ddea_spark_main, spark_context=sc)