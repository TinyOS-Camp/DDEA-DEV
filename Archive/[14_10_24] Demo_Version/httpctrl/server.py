__all__ = ['RequestHandler']

import os
import posixpath
import urllib
import SimpleHTTPServer
import cgi

from . import BASE_ROUTES, PRINT_LOG, execute_ddea, path_to_json, BASE_JSON_PATH, ANAL_ROUTES

IS_EXECUTING = False

class RequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    def translate_path(self, path):
        """translate path given routes"""

        #print "1.===", path , "==="
        # set default root to cwd
        root = os.getcwd()

        # look up routes and set root directory accordingly
        for pattern, rootdir in ANAL_ROUTES + BASE_ROUTES:
            if path.startswith(pattern):
                #print "found match!", pattern
                path = path[len(pattern):]  # consume path up to pattern len
                root = rootdir
                break

        # normalize path and prepend root directory
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        path = posixpath.normpath(urllib.unquote(path))
        words = path.split('/')
        words = filter(None, words)

        path = root
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)

        return path

    def do_GET(self):

        if self.path.startswith(BASE_JSON_PATH):
            for pattern, rootdir in ANAL_ROUTES:
                if pattern in self.path:
                    flist = path_to_json(rootdir,pattern)

            if flist:
                self.send_response(200)
                self.send_header('Content-type','application/json')
                self.end_headers()
                self.wfile.write(flist)
            else:
                SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
        else:
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        global IS_EXECUTING
        # Parse the form data posted
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })


        # Begin the response
        self.send_response(200)
        self.end_headers()
        self.wfile.write('OK')
        """
        self.wfile.write('Client: %s\n' % str(self.client_address))
        self.wfile.write('User-agent: %s\n' % str(self.headers['user-agent']))
        self.wfile.write('Path: %s\n' % self.path)
        self.wfile.write('Form data:\n')
        """

        # Echo back information about what was posted in the form

        if IS_EXECUTING:
            return

        IS_EXECUTING = True
        sub_system = None
        sensor = None
        start_date = None
        end_date = None
        dr_point = None

        for field in form.keys():
            field_item = form[field]

            if field_item.filename:
                pass
            else:
                # Regular form value
                #print field, form[field].value
                if "sub_system" in field:
                    sub_system = form[field].value

                if "sensor" in field:
                    sensor = form[field].value

                if "start_date_submit" in field:
                    start_date = form[field].value

                if "end_date_submit" in field:
                    end_date = form[field].value

                if "dr_point" in field:
                    dr_point = form[field].value

        execute_ddea(sub_system, sensor, start_date, end_date, dr_point)
        IS_EXECUTING = False

        return

    """
    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })

        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
    """