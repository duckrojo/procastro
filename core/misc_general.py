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

__all__ = ['sortmanynsp', 'sortmany', 'accept_object_name'
           ]

import numpy as np
import operator as op
import warnings
import re

def accept_object_name(name, target):
    """
    Parses an astronomical object's name and checks if its similar to a
    specific 'target'
    
    Parameters
    ----------
    name : str
        Name of object
    target : str
        Name of object to be compared with
    
    Returns
    -------
    bool
    """
    name = name.replace(r'\\', '\\\\')
    name = name.replace('+', r'\+')
    name = name.replace('{', r'\{')
    name = name.replace('}', r'\}')
    name = name.replace('[', r'\[')
    name = name.replace(']', r'\]')
    name = name.replace('*', r'\*')
    name = name.replace('?', r'\?')
    if '__' in name:
        mandatory, optional = name.split('__')
        name = f"{mandatory}(?:{optional})?"
    name = name.replace('_', '[- ]?').lower()
    return re.search(name,target.lower()) is not None


def sortmanynsp(*arr):
    """
    Sort many lists following the order of the first. 
    
    Parameters
    ----------
    arr : list
        Lists to sort
    
    Returns
    -------
    tuple 
        Sorted numpy.ndarray
    """
    return [np.array(r) for r in sortmany(*arr)]


def sortmany(*arr, **kwargs):
    """
    Sort many lists following the order of the first. Optionally using a key
    
    Parameters
    ----------
    arr : list
        Lists to sort
    key : string, optional
        Key used for sorting
        
    Returns
    -------
    list
        Sorted list 
    """

    if 'key' not in kwargs:
        key = lambda x: x
    else:
        key = kwargs['key']

    keyed=[key(a) for a in arr[0]]

    tups = list(zip(keyed,*arr))
    tups.sort(key=op.itemgetter(0))
    pret = list(zip(*tups))
    return pret[1:]


