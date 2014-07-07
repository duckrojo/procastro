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

from __future__ import print_function, division
from functools import wraps as _wraps
from io_file import astrofile
import scipy as sp
import warnings
from astropy.utils.exceptions import AstropyUserWarning

class astrodir():
    """Collection of astrofile"""
    def __init__(self, path):
        """Create astrofile container from either a directory path (if given string), or directly from list of string"""
        import os
        import glob
        import os.path as pth
        files=[]

        if isinstance(path, basestring):
            filndir = glob.glob(path)
        else:
            filndir = path

        if len(filndir)==0:
            raise ValueError("invalid path to files or zero-len list given")
        for f in filndir:
            if hasattr(f,'astrofile') and f.astrofile:
                nf = f
            elif pth.isdir(f):
                for sf in os.listdir(f):
                    nf = astrofile(f+'/'+sf)
                    if nf:
                        files.append(nf)
                nf = False
            else:
                nf = astrofile(f)
            if nf:
                files.append(nf)
        self.files = files

    def readdata(self):
        import scipy as sp
        if isinstance(self,'datacube'):
            return self.datacube
        data = []
        for f in self.files():
            data.append(f.reader())
        self.datacube = sp.array(data)
        return self.datacube

    def sort(self, *args):
        """Return sorted list of files according to specified header field, use first match. It uses in situ sorting, but returns itself"""
        if len(args)==0:
            raise ValueError("At least one valid header field must be specified to sort")
        hdrfld=False
        for a in args:
            if self.getheaderval(a):
                hdrfld=a
                break
        if not hdrfld:
            raise ValueError("A valid header field must be specified to use as a sort key. None of the currently requested were found: %s" %(', '.join(args),))
        for f in self.files:
            f.sortkey=hdrfld
        self.files.sort()
        return self

    def __iter__(self):
        return iter(self.files)

    def __repr__(self):
        return "<astrofile container: %s>" % (self.files.__repr__(),)

    def __getitem__(self, item):
        if isinstance(item, sp.ndarray):
            if item.dtype=='bool':
                return astrodir([f for b,f in zip(item,self.files) if b])
        elif isinstance(item, slice):
            return astrodir(self.files.__getitem__(item))

        return self.files[item]#.__getitem__(item)


    def __len__(self):
        return self.files.__len__()

    def filter(self, *args, **kwargs):
        """Filter files according to those whose filter return True to the given arguments. What the filter does is type-dependent in each file. Check docstring of a single element."""
        from copy import copy
        new = copy(self)
        new.files = [f for f in self if f.filter(*args,**kwargs)]
        return new

    def basename(self, joinchr=', '):
        """Returns the basename of the files in object"""
        return joinchr.join([b.basename() for b in self])

    def getheaderval(self, *args, **kwargs):
        """Gets the header values specified in 'args' from each of the files. Returns a simple list if only one value is specified, or a list of tuples otherwise"""
        if 'mapout' in kwargs:
            mapout = kwargs['mapout']
        else:
            mapout = len(args)==1 and (lambda x:x[0]) or (lambda x:x)

        warnings.filterwarnings("once", "non-standard convention", AstropyUserWarning)
        ret = [f.getheaderval(*args, mapout=mapout) for f in self.files]
        warnings.resetwarnings()
        return ret

        

    def setheader(self, write=False, **kwargs):
        """Sets the header values specified in 'args' from each of the files. Returns a simple list if only one value is specified, or a list of tuples otherwise

        The special kwarg 'write' can be used to force the update of the fits with the keyword or just leave it in memory"""

        if False in [f.setheaderval(**kwargs) for f in self.files]:
            raise ValueError("Setting the header of a file returned error... panicking!")

        return self
        


