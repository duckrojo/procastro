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

import scipy as sp

def sortmanynsp(*arr):
    """Sort many lists following the order of the first. Returns a tuple of sorted np.ndarray"""
    return [sp.array(r) for r in sortmany(*arr)]


def sortmany(*arr):
    """Sort many lists following the order of the first."""
    tups = zip(*arr)
    tups.sort()
    pret = zip(*tups)
    return pret

