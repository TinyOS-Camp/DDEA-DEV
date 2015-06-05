#!/adsc/DDEA_PROTO/bin/python
# -*- coding: utf-8 -*-

import os, os.path
from toolset import dill_load_obj

root_path = os.getcwd()


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def print_file(filepath):
    print "------- " + filepath + " --------\n"
    print dill_load_obj(filepath)


for (path, dirs, files) in walklevel(root_path, 0):
    flist = map(lambda f: os.path.join(path, f), files)
    map(lambda f: print_file(f), flist)



