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

import logging

iologger = logging.getLogger('dataproc.io')

import dataproc as dp
import scipy as sp
import warnings
import copy
import sys
from astropy.utils.exceptions import AstropyUserWarning


class AstroDir(object):
    """Collection of AstroFile"""

    def __init__(self, path, mflat=None, mdark=None, calib_force=False, read_keywords=None,
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

        if isinstance(path, str):
            filndir = glob.glob(path)
            if len(filndir)==0:
                raise ValueError("Invalid path to files given")
        elif isinstance(path, dp.AstroFile):
            filndir = [path]
        else:
            filndir = path

        # if len(filndir) == 0:
        #     raise ValueError("invalid path to files or zero-len list given")
        for f in filndir:
            if isinstance(f, dp.AstroFile):
                nf = copy.deepcopy(f)
            elif pth.isdir(f):
                for sf in os.listdir(f):
                    nf = dp.AstroFile(f + '/' + sf, hduh=hduh, hdud=hdud, read_keywords=read_keywords)
                    if nf:
                        files.append(nf)
                nf = False
            else:
                nf = dp.AstroFile(f, hduh=hduh, hdud=hdud, read_keywords=read_keywords)
            if nf:
                files.append(nf)

        self.files = files
        self.props = {}
        calib = dp.AstroCalib(mdark, mflat)
        for f in files:
            if calib_force or not f.has_calib():  # allows some of the files to keep their calibration
                # AstroFile are created with an empty calib by default, which is overwritten here.
                #  Hoping that garbage collection works:D
                f.calib = calib

        self.path = path

    def add_bias(self, mbias):
        unique_calibs = set([f.calib for f in self])
        for c in unique_calibs:
            c.add_bias(mbias)

    def add_flat(self, mflat):
        unique_calibs = set([f.calib for f in self])
        for c in unique_calibs:
            c.add_flat(mflat)

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
                "A valid header field must be specified to use as a sort key."
                " None of the currently requested were found: %s" % (', '.join(args),))

        # Sorting is done using python operators __lt__, __gt__, ... who are inquired by .sort() directly.
        for f in self:
            f.sortkey = hdrfld
        self.files.sort()
        return self

    def __repr__(self):
        return "<AstroFile container: %s>" % (self.files.__repr__(),)

    def __add__(self, other):
        if isinstance(other, AstroDir):
            other_af = other.files
        elif isinstance(other, (list, tuple)) and isinstance(other[0], str):
            iologger.warning("Adding list of files to Astrodir, calib and hdu defaults"
                             " will be shared from first AstroFile in AstroDir")
            other_af = [dp.AstroFile(filename,
                                     hdud=self[0]._hdud,
                                     hduh=self[0]._hduh,
                                     mflat=self[0].calib.mflat,
                                     mbias=self[0].calib.mbias)
                        for filename in other]
        else:
            raise TypeError("Cannot add {} + {}", self.__class__, other.__class__)

        return dp.AstroDir(self.files+other_af)

    def __getitem__(self, item):
        # imitate indexing on boolean array as in scipy.
        if isinstance(item, sp.ndarray):
            if item.dtype == 'bool':
                if len(item) != len(self):
                    raise ValueError("Attempted to index AstroDir with a boolean array "
                                     "of different size (it must include all bads)")

                fdir = [f for b, f in zip(item, self) if b]
                return AstroDir(fdir)

        elif isinstance(item, slice):
            return AstroDir(self.files.__getitem__(item))

        return self.files[item]  # .__getitem__(item)

    def __len__(self):
        return len(self.files)

    def stats(self, *args, **kwargs):
        """
Return stats
        :param args: Specify the stats that want to be returned
        :param kwargs: verbose_headings is the only keyword accepted to print a heading
        :return: the stat as returned by each of the AstroFiles
        """
        verbose_heading = kwargs.pop('verbose_heading', True)
        extra_headers = kwargs.pop('extra_headers', True)
        if kwargs:
            raise SyntaxError("only the following keyword arguments for stats are accepted 'verbose_heading', 'extra_headers'")
        ret = []
        for af in self:
            ret.append(af.stats(*args, verbose_heading=verbose_heading, extra_headers=extra_headers))
            verbose_heading = False
        return ret

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
            Returns a simple list if only one value is specified, or a list of tuples otherwise
            :param cast: output function
            :param single_in_list: default False
            """
        if 'single_in_list' in kwargs:
            single_in_list = kwargs['single_in_list']
        else:
            single_in_list = False

        if 'cast' in kwargs:
            cast = kwargs['cast']
        else:
            if len(args) == 1 and not single_in_list:
                cast = lambda x: x[0]
            else:
                cast = lambda x: x

        warnings.filterwarnings("once", "non-standard convention", AstropyUserWarning)
        ret = [f.getheaderval(*args, cast=cast, **kwargs) for f in self]
        warnings.resetwarnings()
        return ret

    def setheader(self, **kwargs):
        """ Sets the header values specified in 'args' from each of the files.
            Returns a simple list if only one value is specified, or a list of tuples otherwise
"""
        if False in [f.setheaderval(**kwargs) for f in self]:
            raise ValueError("Setting the header of a file returned error... panicking!")

        return self

    def get_datacube(self, normalize_region=None, normalize=False, verbose=False,
                     check_unique=None):
        """
Returns all data from the AstroFiles in a datacube
        :param normalize_region:
        :param normalize:
        :param verbose:
        :param check_unique: receives a list of headers that need to be checked for uniqueness. Exposure time by default (it makes no sense to merge otherwise)
        :return:
        """
        if verbose:
            print("Reading {} good frames{}: ".format(len(self),
                                                      normalize and " and normalizing" or ""),
                  end='')
        if check_unique is None:
            check_unique = []
        if normalize_region is None:
            if 'normalize_region' in self.props:
                normalize_region = self.props['normalize_region']
            else:
                normalize_region = slice(None)
        all_data = sp.zeros([len(self)]+list(self[0].shape))

        # check uniqueness... first get the number of unique header values for each requested keyword
        # and then complain if any greater than 1
        if len(check_unique) > 1:
            lens = sp.array([len(set(rets)) for rets in zip(*self.getheaderval(*check_unique))])
        elif len(check_unique) == 1:
            lens = sp.array([len(set(self.getheaderval(*check_unique)))])
        else:
            lens = sp.array([])
        if True in (lens > 1):
            raise ValueError("Header(s) {} are not unique in the input dataset".
                             format(", ".join(sp.array(check_unique)[lens > 1])))

        for ad, af in zip(all_data, self):
            data = af.reader()
            if normalize:
                data /= data[normalize_region].mean()
            ad[:, :] = data
            if verbose:
                print(".", end='')
                sys.stdout.flush()
        if verbose:
            print("")
        return all_data


    def median(self, normalize_region=None, normalize=False, verbose=True, check_unique=None):
        """

        :param normalize_region: Subregion to use for normalization
        :param normalize:  Whether to Normalize
        :param verbose:   Output progress indicator
        :param check_unique: Check uniquenes in heaeder
        :return:
        """
        if check_unique is None:
            if normalize:
                check_unique = []
            else:
                check_unique = ['exptime']

        data = self.get_datacube(normalize_region=normalize_region,
                                 normalize=normalize,
                                 verbose=verbose,
                                 check_unique=check_unique)
        # todo: return AstroFile

        if verbose:
            print("Median combining: ", end="")
        sys.stdout.flush()
        ret = sp.median(data, axis=0)
        if verbose:
            print("done")
            sys.stdout.flush()

        return ret

    def mean(self, normalize_region=None, normalize=False, verbose=True, check_unique=None):
        """

        :param normalize_region: Subregion to use for normalization
        :param normalize:  Whether to Normalize
        :param verbose:   Output progress indicator
        :param check_unique: Check uniquenes in heaeder
        :return:
        """
        if check_unique is None:
            if normalize:
                check_unique = []
            else:
                check_unique = ['exptime']

        data = self.get_datacube(normalize_region=normalize_region,
                                 normalize=normalize,
                                 verbose=verbose,
                                 check_unique=check_unique)
        # todo: return AstroFile
        if verbose:
            print("Median combining: ", end="")
            sys.stdout.flush()
        ret = sp.mean(data, axis=0)
        if verbose:
            print("done")
            sys.stdout.flush()

        return ret

    def jd_from_ut(self, target='jd', source='date-obs'):
        """
Add jd in header's cache to keyword 'target' using ut on keyword 'source'
        :param target: target keyword for JD storage
        :param source: input value in UT format
        """
        for af in self:
            af.jd_from_ut(target, source)

        return self
