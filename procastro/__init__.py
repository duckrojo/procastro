#
# procastro - Data processing routines
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
Framework to easily handle multiple astronomy data files
"""
from . import astro
from . import core
from .core import *
#from . import timeseries
#from . import obsrv
import logging as _log


__all__ = ['astro', 'dplogger']
__all__ += core.__all__

dplogger = _log.getLogger('procastro')
_ch = _log.StreamHandler()
_formatter = _log.Formatter('%(name)s (%(module)s.%(funcName)s) %(levelname)s: %(message)s')
_ch.setFormatter(_formatter)
dplogger.addHandler(_ch)

__version__ = "0.2.1"
