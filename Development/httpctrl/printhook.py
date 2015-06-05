import sys

def DEFAULT_HOOK(text):
    return 1, 0, text

#this class gets all output directed to stdout(e.g by print statements)
#and stderr and redirects it to a user defined function
class PrintHook:
    #out = 1 means stdout will be hooked
    #out = 0 means stderr will be hooked
    def __init__(self, out=1):
        self.func = None    ##self.func is userdefined function
        self.origOut = None
        self.out = out


    def Start(self, func=DEFAULT_HOOK):
        if self.out:
            sys.stdout = self
            self.origOut = sys.__stdout__
        else:
            sys.stderr = self
            self.origOut = sys.__stderr__

        if func:
            self.func = func
        else:
            self.func = self.TestHook


    #Stop will stop routing of print statements thru this class
    def Stop(self):
        self.origOut.flush()
        if self.out:
            sys.stdout = sys.__stdout__
        else:
            sys.stderr = sys.__stderr__
        self.func = None


    #override write of stdout
    def write(self, text):
        proceed = 1
        lineNo = 0
        newText = ''

        if self.func != None:
            proceed, lineNo, newText = self.func(text)

        if proceed:
            if text.split() == []:
                self.origOut.write(text)
            else:
                #if goint to stdout then only add line no file etc
                #for stderr it is already there
                if self.out:
                    if lineNo:
                        try:
                            raise "ARTIFICIAL"
                        except:
                            newText = 'line('+str(sys.exc_info()[2].tb_frame.f_back.f_lineno)+'):'+newText
                            codeObject = sys.exc_info()[2].tb_frame.f_back.f_code
                            fileName = codeObject.co_filename
                            funcName = codeObject.co_name
                        self.origOut.write('file '+fileName+','+'func '+funcName+':')
                    self.origOut.write(newText)

    #pass all other methods to __stdout__ so that we don't have to override them
    def __getattr__(self, name):
        return self.origOut.__getattr__(name)
