#!/usr/bin/python
# To force float point division
"""
Created on Mon Mar 24 19:24:11 2014

@author: NGO Quang Minh Khiem
@e-mail: khiem.ngo@adsc.com.sg

"""
from __future__ import division

from multiprocessing import Process, JoinableQueue, Event
from SimpleHTTPServer import SimpleHTTPRequestHandler
from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer
import time, BaseHTTPServer, sys, simplejson, os
import datetime as dt
from datetime import datetime
import SimpleHTTPServer, SocketServer, logging, cgi
from shared_constants import *

from quasar_url_reader import read_sensor_data
from ddea_proc import ddea_process
import mytool as mt
import pickle, logging, logging.handlers, SocketServer, struct


user_cmd_q = JoinableQueue()
ddea_msg_q = JoinableQueue()
cmd_lock = Event()

def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii')
    return dict(map(ascii_encode, pair) for pair in data.items())


class ExecProc(Process):
    def __init__(self, cmd_q, status_q):
        Process.__init__(self)
        self.cmd_q = cmd_q
        self.status_q = status_q

    def run(self):

        from log_util import log

        try:
            while True:
                cmd = None
                try:
                    cmd = self.cmd_q.get(block=True, timeout=0.1)
                except Exception as e:
                    continue

                finally:
                    if cmd:
                        self.cmd_q.task_done()

                        try:

                            with open(META_DIR + "wip.json", 'w') as f:
                                f.write(simplejson.dumps({"wip": 1}))

                            cmdset = simplejson.loads(cmd)
                            sensor_hash = cmdset['selected-nodes']
                            s_date = datetime.strptime(cmdset['start-date'], '%Y-%m-%d')
                            e_date = datetime.strptime(cmdset['end-date'], '%Y-%m-%d')

                            if not len(sensor_hash):
                                log.critical("No sensor is selected!")
                            else:

                                log.info('****************************** Begining of DDEA *******************************')

                                bldg_key = 'SODA'
                                #exemplar by user
                                #pname_key = '_POWER_'
                                pname_key = 'POWER'

                                s_epoch = int(time.mktime(s_date.timetuple()))
                                e_epoch = int(time.mktime(e_date.timetuple()))
                                time_inv = dt.timedelta(seconds=cmdset['time-interval'])

                                log.info("Cleaning up old output...")

                                mt.remove_all_files(FIG_DIR)
                                mt.remove_all_files(JSON_DIR)
                                mt.remove_all_files(PROC_OUT_DIR)

                                log.info("start epoch : " + str(s_epoch) + " end epoch : " + str(e_epoch))
                                log.info(str(time_inv) + ' time slot interval is set for this data set !!!')
                                log.info("BLDG_KEY : " + bldg_key + " PNAME_KEY : " + pname_key)
                                log.info('*' * 80)

                                log.info("Retrieve sensor data from quasar TSDB")

                                sensor_names_hash = mt.sensor_name_uid_dict(bldg_key, sensor_hash)

                                sensor_data = read_sensor_data(sensor_names_hash, s_epoch, e_epoch)

                                if sensor_data and len(sensor_data):
                                    ddea_process(sensor_names_hash, sensor_data, s_epoch, e_epoch, time_inv, bldg_key, pname_key)
                                else:
                                    log.critical("No sensor data available for time period and sensor selected!")

                                log.info('******************************** End of DDEA **********************************')

                            os.remove(META_DIR + "wip.json")
                            cmd_lock.clear()

                            log.info("execution-lock cleared")
                            log.info('~' * 80)

                        except Exception as e:
                            os.remove(META_DIR + "wip.json")
                            cmd_lock.clear()
                            print e
                            log.error(str(e))

        except Exception as e:
            os.remove(META_DIR + "wip.json")
            cmd_lock.clear()
            print e
            log.error(str(e))

        finally:
            sys.exit(0)


class LogRecordStreamHandler(SocketServer.StreamRequestHandler):

    def __init__(self, request, client_address, server, wsserver):
        self.wsserver = wsserver
        SocketServer.StreamRequestHandler.__init__(self, request, client_address, server)

    def handle(self):
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break

            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))

            obj = pickle.loads(chunk)
            record = logging.makeLogRecord(obj)
            self.wsserver.broadcastmsg(record.msg)


class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    allow_reuse_address = 1

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):

        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None
        self.requstHandle = None
        self.wsserver = SimpleWebSocketServer('', 8081, WebSocket)

    def finish_request(self, request, client_address):
        LogRecordStreamHandler(request, client_address, self, self.wsserver)

    def cleanup(self):
        self.wsserver.close()

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            self.wsserver.servconnman()

            rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


class WSProc(Process):
    def __init__(self, status_q):
        Process.__init__(self)
        self.status_q = status_q

    def run(self):
        tcpserver = LogRecordSocketReceiver()

        try:
            tcpserver.serve_until_stopped()
        except Exception as e:
            pass

        finally:
            tcpserver.cleanup()
            sys.exit(0)


class DDEARequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    """
    def send_head(self):
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            self.send_error(404, "File not found")
            return None
        self.send_response(200)
        self.send_header("Content-type", ctype)

        if self.path == "/" or ("index.html" in self.path):
            print "- allow origin modification -"
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET POST')

        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def do_GET(self):
        f = self.send_head()
        if f:
            self.copyfile(f, self.wfile)
            f.close()
    """

    def do_GET(self):
        self.path = 'resources/' + self.path
        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)


    def do_POST(self):
        length = int(self.headers.getheader('content-length'))
        data = self.rfile.read(length)

        #print self.path, data

        if cmd_lock.is_set():
            # return unauthorized
            self.send_response(401)

        else:
            #Oh, so ungodly!
            user_cmd_q.put_nowait(data)
            # service cmd. lock
            cmd_lock.set()

            # return authorized
            self.send_response(200)

        self.send_header('Content-type', 'text/html')
        self.send_header("Content-length", 0)
        self.end_headers()
        self.wfile.write("")
        self.finish()
        self.connection.close()

if __name__ == '__main__':

    processes = list()
    processes.append(ExecProc(user_cmd_q, ddea_msg_q))
    processes.append(WSProc(ddea_msg_q))

    print time.asctime(), "Staring DDEA..."

    try:
        for p in processes:
            p.start()

        BaseHTTPServer\
            .HTTPServer(('0.0.0.0', 8080), DDEARequestHandler)\
            .serve_forever()

    except Exception as e:
        for p in processes:
            p.terminate()

    finally:
        print '\n', time.asctime(), "Stopping CPS..."
        exit(0)
