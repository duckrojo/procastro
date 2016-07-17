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
import dataproc as dp
import dataproc.combine as cm
import scipy as sp
import warnings
import copy
from astropy.utils.exceptions import AstropyUserWarning


class AstroDir(object):
    """Collection of AstroFile"""

    def __init__(self, path, mbias=None, mflat=None, mdark=None, calib_force=False,
                 hdu=0, hdud=None, hduh=None):
        """Create AstroFile container from either a directory path (if given string), or directly from list of string
        if calib is given, create one calib per directory to avoid using storing calibration files on each AstroFile

"""
        import os
        import glob
        import os.path as pth
        files = []

        if hduh is None:
            hduh=hdu
        if hdud is None:
            hdud=hdu

        if isinstance(path, basestring):
            filndir = glob.glob(path)
        else:
            filndir = path

        if len(filndir) == 0:
            raise ValueError("invalid path to files or zero-len list given")
        for f in filndir:
            if isinstance(f, dp.AstroFile):
                nf = copy.copy(f)
            elif pth.isdir(f):
                for sf in os.listdir(f):
                    nf = dp.AstroFile(f + '/' + sf, hduh=hduh, hdud=hdud)
                    if nf:
                        files.append(nf)
                nf = False
            else:
                nf = dp.AstroFile(f)
            if nf:
                files.append(nf)
        self.files = files
        calib = AstroCalib(mbias, mflat)
        for f in files:
            if calib_force or not hasattr(f, 'calib'):  # allows some of the files to keep their calibration
                f.calib = calib

        self.path = path
        self.bias = mbias
        self.dark = mdark
        self.flat = mflat

    def add_bias(self, mbias):
        unique_calibs = set([f.calib for f in self.files])
        for c in unique_calibs:
            c.add_bias(mbias)

    def add_flat(self, mflat):
        unique_calibs = set([f.calib for f in self.files])
        for c in unique_calibs:
            c.add_flat(mflat)

    def readdata(self, rawdata=True):
        """
        Reads data from AstroDir
        :param rawdata: if True returns raw data, if False returns reduced
        :return:
        """
        import scipy as sp
        # TODO datacube must be a class to be allowed as arg 2 in isinstance
        # if isinstance(self,'datacube'):
        #    return self.datacube
        data = []
        for f in self.files:
            data.append(f.reader(rawdata))
        self.datacube = sp.array(data)
        return self.datacube

    def sort(self, *args, **kwargs):
        """ Return sorted list of files according to specified header field, use first match.
            It uses in situ sorting, but returns itself"""
        if len(args) == 0:
            raise ValueError("At least one valid header field must be specified to sort")
        hdrfld = False

        for a in args:
            if self.getheaderval(a):
                hdrfld = a
                break
        if not hdrfld:
            raise ValueError(
                "A valid header field must be specified to use as a sort key. None of the currently requested were found: %s" % (
                ', '.join(args),))

        # Sorting is done using python operators __lt__, __gt__, ... who are inquired by .sort() directly.
        for f in self.files:
            f.sortkey = hdrfld
        self.files.sort()
        return self

    def __iter__(self):
        return iter(self.files)

    def __repr__(self):
        return "<AstroFile container: %s>" % (self.files.__repr__(),)

    def __getitem__(self, item):
        if isinstance(item, sp.ndarray):
            if item.dtype == 'bool':
                fdir = [f for b, f in zip(item, self.files) if b]
                return AstroDir(fdir)
        elif isinstance(item, slice):
            return AstroDir(self.files.__getitem__(item))

        return self.files[item]  # .__getitem__(item)

    def __len__(self):
        return self.files.__len__()

    def filter(self, *args, **kwargs):
        """ Filter files according to those whose filter return True to the given arguments.
            What the filter does is type-dependent in each file. Check docstring of a single element."""
        from copy import copy
        new = copy(self)
        new.files = [f for f in self if f.filter(*args, **kwargs)]
        return new

    def basename(self, joinchr=', '):
        """Returns the basename of the files in object"""
        return joinchr.join([b.basename() for b in self])

    def getheaderval(self, *args, **kwargs):
        """ Gets the header values specified in 'args' from each of the files.
            Returns a simple list if only one value is specified, or a list of tuples otherwise"""
        if 'mapout' in kwargs:
            mapout = kwargs['mapout']
        else:
            mapout = len(args) == 1 and (lambda x: x[0]) or (lambda x: x)

        warnings.filterwarnings("once", "non-standard convention", AstropyUserWarning)
        ret = [f.getheaderval(*args, mapout=mapout, **kwargs) for f in self.files]
        warnings.resetwarnings()
        return ret

    def setheader(self, write=False, **kwargs):
        """ Sets the header values specified in 'args' from each of the files.
            Returns a simple list if only one value is specified, or a list of tuples otherwise

            The special kwarg 'write' can be used to force the update of the fits
            with the keyword or just leave it in memory"""

        if False in [f.setheaderval(**kwargs) for f in self.files]:
            raise ValueError("Setting the header of a file returned error... panicking!")

        return self


class AstroCalib(object):
    def __init__(self, mbias=None, mflat=None):
        if mbias is None:
            mbias = 0.0
        if mflat is None:
            mflat = 1.0

        self.mbias = {}
        self.mflat = {}
        self.add_bias(mbias)
        self.add_flat(mflat)

    def add_bias(self, mbias):
        if isinstance(mbias, dict):
            for k in mbias.keys():
                self.mbias[k] = mbias[k]
        elif isinstance(mbias,
                        (int, float, sp.ndarray)):
            self.mbias[-1] = mbias
        elif isinstance(mbias,
                        dp.AstroFile):
            self.mbias[-1] = mbias.reader()
        elif isinstance(mbias,
                        cm.Combine):
            self.mbias[-1] = mbias.data
        else:
            raise ValueError("Master Bias supplied was not recognized.")

    def add_flat(self, mflat):
        if isinstance(mflat, dict):
            for k in mflat.keys():
                self.mflat[k] = mflat[k]
        elif isinstance(mflat,
                        dp.AstroFile):
            self.mflat[''] = mflat.reader()
        elif isinstance(mflat,
                        (int, float, sp.ndarray)):
            self.mflat[''] = mflat
        elif isinstance(mflat,
                        cm.Combine):
            self.mflat[''] = mflat.data
        else:
            raise ValueError("Master Flat supplied was not recognized.")

    def reduce(self, data, exptime=None, afilter=None):
        if exptime is None:
            exptime = -1
        if afilter is None:
            afilter = ''

        debias = data - self.mbias[exptime]
        deflat = debias / self.mflat[afilter]

        return deflat
