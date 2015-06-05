#!/usr/bin/python

"""
Created on Mon Mar 24 19:24:11 2014

@author: NGO Quang Minh Khiem
@e-mail: khiem.ngo@adsc.com.sg

"""

import glob, logging, logging.handlers, sys

LOG_FILENAME = 'logging_rotatingfile_example.out'
"""
# Set up a specific logger with our desired output level
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(LOG_FILENAME, maxBytes=20, backupCount=5)

my_logger.addHandler(handler)

# Log some messages
for i in range(20):
    my_logger.debug('i = %d' % i)

# See what files are created
logfiles = glob.glob('%s*' % LOG_FILENAME)

for filename in logfiles:
    print filename
"""

def get_logger():

    #l = logging.getLogger('')
    l = logging.getLogger()
    l.setLevel(logging.NOTSET)

    # add socket handler
    sh = logging.handlers.SocketHandler('localhost', logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    sh.setLevel(logging.NOTSET)
    l.addHandler(sh)

    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add stdout handler
    #ch = logging.StreamHandler(sys.stdout)
    ch = logging.StreamHandler()
    ch.setLevel(logging.NOTSET)
    ch.setFormatter(formatter)
    l.addFilter(ch)

    # create file handler which logs even debug messages
    # fh = logging.handlers.RotatingFileHandler(LOG_FILENAME, maxBytes=20, backupCount=5)
    fh = logging.FileHandler('resources/log/ddea.log')
    fh.setLevel(logging.NOTSET)
    fh.setFormatter(formatter)
    l.addFilter(fh)
    """

    return l

log = get_logger()