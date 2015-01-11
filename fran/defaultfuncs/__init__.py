"""Module for default functions to be used on data.

Contains:
- Mean Filter
- Etc.
"""

import os
import subprocess as sp
import types

path = os.getcwd()
l = sp.check_output('ls ' + path + '/defaultfuncs', shell=True)
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