#from foo import *
#from bar import *
import os
import subprocess as sp
from inspect import getmembers, isfunction

path = os.getcwd()
l = sp.check_output('ls ' + path + '/defaultfuncs', shell=True)
defaultFiles = [d for d in (l.split('\n'))[:-1] if d[-4:] != '.pyc']
defaultFiles.remove('__init__.py')
print defaultFiles

__all__ = []

for f in defaultFiles:
    #print f[:-3]
    fname = f[:-3]
    fpackage = __import__ (fname, globals(), locals(), [], -1)
    functions_list = [o for o in getmembers(fpackage) if isfunction(o[1])]
    allfuncs = []
    for fn in functions_list:
        funcname = fn[0]
        if funcname[0] != '_':
            allfuncs.append(funcname)
            #fullname = fname + '.' + funcname
            #print fullname
            #__all__.append(fullname)
            print funcname
    __import__(fname, globals(), locals(), allfuncs, -1)