"""Module for user-defined functions to be used on data.

Users can add as many files and functions per file as they please.
If they mean for some functions of methods to be hidden from the main
execution, the name of these methods should start with an underscore.

All functions and methods whose names don't start with an underscore
will be imported to the main program.

FILENAMES are not of importance, and only separate methods and functions
will be imported. Take this into consideration when naming methods.
"""

__author__ = 'fran'
import os
import subprocess as sp
import types

path = os.getcwd()
l = sp.check_output('ls ' + path + '/user', shell=True)
modules = [d[:-3] for d in (l.split('\n'))[:-1] if d[-4:] != '.pyc']
modules.remove('__init__')

for modulename in modules:
    module = __import__(modulename, globals(), locals(), [], -1)
    module = reload(module)
    for v in dir(module):
        if v[0] == '_' or isinstance(getattr(module,v), types.ModuleType):
            continue
        globals()[v] = getattr(module, v)
    del module

del modules, types
