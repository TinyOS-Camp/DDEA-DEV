__author__ = 'almightykim'

DEFAULT_DEBUG_OUT = "DEBUG_OUT.txt"

def out(cmd):
    import inspect
    (_, filename, linenum, funcname, _, _) = inspect.getouterframes(inspect.currentframe())[1]
    lineout = str(filename) + " | " + str(linenum) + " | (" + str(funcname) + ") : " + str(cmd) + "\n"
    with open(DEFAULT_DEBUG_OUT, "a+") as f:
        f.writelines(lineout)