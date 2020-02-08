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
from . import astro
from . import core
from .core import *
from . import timeseries
from . import obsrv
import logging as _log


__all__ = ['astro', 'timeseries', 'obsrv', 'dplogger']
__all__ += core.__all__

dplogger = _log.getLogger('dataproc')
_ch = _log.StreamHandler()
_formatter = _log.Formatter('%(name)s (%(module)s.%(funcName)s) %(levelname)s: %(message)s')
_ch.setFormatter(_formatter)
dplogger.addHandler(_ch)

#causes conflict on pycharms the following
#core = reload(core)

# import types
# __import__('core', globals(), locals(), [], 1)

# for v in dir(core):
#     if v[0] == '_' or isinstance(getattr(core,v), types.ModuleType):
#         continue
#     globals()[v] = getattr(core, v)

# del core
# del types
# del v

__version__ = "0.07"

#from core import *
