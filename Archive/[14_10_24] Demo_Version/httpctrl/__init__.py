""" DDEA Debug Output package. """
__version__ = '0.1'
__all__ = ['StartLogger','StopLogger', 'PRINT_LOG', 'WebSocketProc', 'ServeGUI']
__author__ = 'stkim1'

from multiprocessing import JoinableQueue
import BaseHTTPServer
from Queue import Full
import time

GET_BLOCK_TIME_SLICE = 0.005
WS_SOCKET_PORT = 9001
HTTP_PORT = 9000
BASE_SERVER_URL = "http://ddea.kmeg-os.com:9000"

BASE_JSON_PATH = '/analysis_json'

# target route lists
ANAL_ROUTES = (
        ['/bn_anal',   '/adsc/bigdata/output_data/Analysis_png_files/VTT/BN_Analysis'],
        ['/lh_anal',   '/adsc/bigdata/output_data/Analysis_png_files/VTT/LH_Analysis'],
        ['/sd_anal',   '/adsc/bigdata/output_data/Analysis_png_files/VTT/SD_Analysis']
    )

BASE_ROUTES = (
    # [url_prefix ,  directory_path]
    ['/',          '/adsc/DDEA_PROTO/html'],
    # empty string for the 'default' match
    ['',            '/adsc/DDEA_PROTO/html']
)

target_queue = JoinableQueue()
processes = []
DDEA_FUNC = None


class AsyncFactory:
    def __init__(self, func, cb_func):
        from multiprocessing import Pool
        self.func = func
        self.cb_func = cb_func
        self.pool = Pool()

    def call(self,*args, **kwargs):
        self.pool.apply_async(self.func, args, kwargs, self.cb_func)

    def wait(self):
        self.pool.close()
        self.pool.join()


def path_to_json(path, symbol):
    import os
    import json
    global BASE_SERVER_URL

    d = {'path': os.path.basename(path)}
    if os.path.isdir(path):
        p = list()
        for fn in os.listdir(path):
            if os.path.isdir(os.path.join(path, fn)):
                continue
            p.append({"url":BASE_SERVER_URL+symbol+"/"+fn,"title":fn,"thumb":BASE_SERVER_URL+symbol+"/thumbnails/"+fn})
        d['photo'] = p
    return json.dumps(d)

def StartLogger(port=WS_SOCKET_PORT):
    global processes
    processes.append(WebSocketProc(target_queue,port))
    for p in processes:
        p.start()


def StopLogger():
    global processes
    for p in processes:
        p.terminate()


def PRINT_LOG(log):
    global target_queue
    try:
        target_queue.put_nowait(log)
    except Full as e:
        pass

    return 1, 0, log


def WebSocketProc(target_in_queue, ws_socket_port):
    from wsoutlet import ClientClusterProtocol, BroadcastServerFactory
    from twisted.internet import reactor
    from multiprocessing import Process

    factory = BroadcastServerFactory("ws://localhost:" + str(ws_socket_port), target_in_queue)
    factory.protocol = ClientClusterProtocol
    factory.setProtocolOptions(allowHixie76=True)
    reactor.listenTCP(ws_socket_port, factory)
    return Process(target=reactor.run, args=(False,))

def execute_ddea(sub_system, sensor, start_date, end_date, dr_point):
    from datetime import datetime
    from multiprocessing import Pool
    global DDEA_FUNC

    #print sensor, start_date, end_date
    s_date = start_date.split('/')
    e_date = end_date.split('/')

    sd = datetime(int(s_date[0]),int(s_date[1]),int(s_date[2]),0)
    ed = datetime(int(e_date[0]),int(e_date[1]),int(e_date[2]),0)

    if DDEA_FUNC:
        try:
            if dr_point:
                DDEA_FUNC(sub_system, sensor, sd, ed, dr_point)
            else:
                DDEA_FUNC(sub_system, sensor, sd, ed)
            #AsyncFactory(DDEA_FUNC, None).call(args=(sensor, sd, ed))
            #p = Pool()
            #p.apply_async(DDEA_FUNC, args=(sensor, sd, ed))
            #p.close()
            #p.join()
        except:
            import traceback;print traceback.print_exc()

from server import RequestHandler

def ServeGUI(Function = None,
             HandlerClass = RequestHandler,
             ServerClass = BaseHTTPServer.HTTPServer,
             protocol="HTTP/1.0",
             port=HTTP_PORT):

    global DDEA_FUNC, BASE_SERVER_URL, ANAL_ROUTES
    from printhook import PrintHook

    DDEA_FUNC = Function

    StartLogger(int(port) + 1)

    phOut = PrintHook()
    phOut.Start(PRINT_LOG)

    server_address = ('', port)
    HandlerClass.protocol_version = protocol
    httpd = ServerClass(server_address, HandlerClass)

    sa = httpd.socket.getsockname()

    print time.asctime(), "Start serving HTTP on", sa[0], "port", sa[1]
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        httpd.server_close()
        StopLogger()
        phOut.Stop()
        print time.asctime(), "Stop serving HTTP on", sa[0], "port", sa[1]
        exit(0)

