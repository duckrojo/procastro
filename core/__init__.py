#
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

def setall(var):
    exec('tmp=%s' % var)
    _all=[m for m in dir(tmp) if m[0]!='_']
    #_all.remove(var)
    return _all


__all__ = []
from io_file import *
__all__.extend(setall('io_file'))

from io_dir import *
__all__.extend(setall('io_dir'))

from misc_arr import *
__all__.extend(setall('misc_arr'))

from misc_lists import *
__all__.extend(setall('misc_lists'))

from misc_examine import *
__all__.extend(setall('misc_examine'))

from misc_process import *
__all__.extend(setall('misc_process'))

