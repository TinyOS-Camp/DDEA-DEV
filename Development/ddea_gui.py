#!/adsc/DDEA_PROTO/bin/python

from httpctrl import ServeGUI
from ddea_main import ddea_analysis

if __name__ == '__main__':

    ServeGUI(Function=ddea_analysis)