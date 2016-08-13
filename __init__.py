#
# dataproc - Data processing routines
#
# Copyright (C) 2013 Patricio Rojo
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of version 2 of the GNU General 
# Public License as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, 
# Boston, MA  02110-1301, USA.
#
#
"""
Data proc docstring
"""

import types
__import__('core', globals(), locals(), [], 1)

import logging
dplogger = logging.getLogger('dataproc')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
dplogger.addHandler(ch)
del ch
del formatter

#causes conflict on pycharms the following
#core = reload(core)

for v in dir(core):
    if v[0] == '_' or isinstance(getattr(core,v), types.ModuleType):
        continue
    globals()[v] = getattr(core, v)

del core
del types
del v



#from core import *


