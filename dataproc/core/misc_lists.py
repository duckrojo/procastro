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
import operator as op
import warnings

def sortmanynsp(*arr):
    """Sort many lists following the order of the first. Returns a tuple of sorted np.ndarray"""
    return [sp.array(r) for r in sortmany(*arr)]


def sortmany(*arr, **kwargs):
    """Sort many lists following the order of the first."""

    if 'key' not in kwargs:
        key = lambda x: x        
    elif  kwargs['key'] is not None:
        key = kwargs['key']
    else:
        raise ValueError("None given as Key argument for sortmany ")

    if 'postrem' not in kwargs:
        postrem = []
    elif  kwargs['postrem'] is not None:
        postrem = kwargs['postrem']
    else:
        postrem = []

    if 'prerem' not in kwargs:
        prerem = []
    elif  kwargs['prerem'] is not None:
        prerem = kwargs['prerem']
    else:
        prerem = []

    try:
        keyed=[key(a) for a in arr[0]]
    except ValueError:
        keyed = []
        for a in arr[0]:
            for p in list(postrem):
#                print ("removing %s: %s " % (p,a))
                if a and p==a[-1]:
                    a=a[:-1]
            for p in list(prerem):
                if a and p==a[0]:
                    a=a[1:]
            if a=='' and (key==float or key==int):
                a='0'
                warnings.warn("using 0 instead of empty when sorting according to key %s" % (key, ))
            keyed.append(key(a))


    tups = zip(keyed,*arr)
    tups.sort(key=op.itemgetter(0))
    pret = zip(*tups)
    return pret[1:]


