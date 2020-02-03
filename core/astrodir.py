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

__all__ = ['AstroDir']

import dataproc as dp
import numpy as np
import warnings
import copy
import sys
import pdb
from astropy.utils.exceptions import AstropyUserWarning

import logging
io_logger = logging.getLogger('dataproc.io')


class AstroDir(object):
    """Collection of AstroFile objects.

    Several recursive methods that are applied to each AstroFile are available

    Attributes
    ----------
    files : list
        Contains the list of all AstroFile that belong to this AstroDir.
    props : dict
        Properties related to this object

    Parameters
    ----------
    path : str or list or AstroFile
        Contains information from the file list. If str, a
        directory+wildcard format is assumed and parsed by `glob.glob`
    mbias, mflat : array_like, dict or AstroFile, optional
        Master bias and flat to associate to each AstroFile
        (in one shared AstroCalib object)
    mbias_header, mflat_header : astropy.io.fits.Header
        Headers for bias and flat
    calib_force : bool, optional
        If True, then force specified `mbias` and `mflat` to all files,
        otherwise assign it only if it doesn't have one already.
    read_keywords : list, optional
        Read all specified keywords at creation time and store them in
        the header cache
    hdu : int, optional
        default HDU
    hdud : int, optional
        default HDU for data
    hduh : int, optional
        default HDU for the header

    See Also
    --------
    dataproc.AstroCalib : Object that holds calibration information.
                          One of them can be shared by many AstroFile instances

    dataproc.AstroCalib.add_mbias()
    dataproc.AstroCalib.add_mflat()
    """

    def __init__(self, path, mflat=None, mbias=None,
                 mbias_header=None, mflat_header=None,
                 calib_force=False, read_keywords=None,
                 hdu=0, hdud=None, hduh=None, auto_trim=None,
                 jd_from_ut=None):

        import os
        import glob
        import os.path as pth
        files = []

        if hduh is None:
            hduh = hdu
        if hdud is None:
            hdud = hdu

        if isinstance(path, str):
            file_n_dir = glob.glob(path)
        elif isinstance(path, dp.AstroFile):
            file_n_dir = [path]
        else:
            file_n_dir = path
        for f in file_n_dir:
            if isinstance(f, dp.AstroFile):
                nf = copy.deepcopy(f)
            elif pth.isdir(f):
                for sf in os.listdir(f):
                    nf = dp.AstroFile(f + '/' + sf, hduh=hduh, hdud=hdud, read_keywords=read_keywords, auto_trim=auto_trim)
                    try:
                        if nf: files.append(nf)
                    except IOError:
                        warnings.warn(f"Warning: File {nf.basename()} could not be read, skipping")
                        
                nf = False
            else:
                nf = dp.AstroFile(f, hduh=hduh, hdud=hdud, read_keywords=read_keywords, auto_trim=auto_trim)
            
            try:
                if nf: files.append(nf)
            except IOError:
                warnings.warn(f"Warning: File {nf.basename()} could not be read, skipping")
        
        self.files = files
        self.props = {}
        calib = dp.AstroCalib(mbias, mflat, auto_trim=auto_trim,
                              mbias_header=mbias_header, mflat_header=mflat_header)

        for f in files:
            if calib_force or not f.has_calib():  # allows some of the files to keep their calibration
                # AstroFile are created with an empty calib by default, which is overwritten here.
                # Hoping that garbage collection works:D
                f.calib = calib

        self.path = path

        if jd_from_ut is not None:
            if len(jd_from_ut) != 2:
                raise TypeError("jd_from_ut parameter need to be a 2-element tuple: source, target. See help on method .jd_from_ut()")
            self.jd_from_ut(*jd_from_ut)

    def add_bias(self, mbias):
        """
        Updates the master bias of all AstroCalib instances included on
        the contained AstroFile instances

        Parameters
        ----------
        mbias : dict indexed by exposure time, array or AstroFile
            Master Bias to use for all frames.

        See Also
        --------
        dataproc.AstroCalib.add_bias : this function is called for each unique }
                                       AstroCalib object in AstroDir
        """
        unique_calibs = set([f.calib for f in self])
        for c in unique_calibs:
            c.add_bias(mbias)

    def add_flat(self, mflat):
        """
        Update Master Flats of all AstroCalib instances included on
        the contained AstroFile instances.

        Parameters
        ----------
        mflat : dict indexed by filter name, array or AstroFile
            Master Flat to use for all frames.

        See Also
        --------
        dataproc.AstroCalib.add_flat : This function is called for each unique
                                       AstroCalib object in AstroDir
        """
        unique_calibs = set([f.calib for f in self])
        for c in unique_calibs:
            c.add_flat(mflat)

    def sort(self, *args, **kwargs):
        """
        Sorts AstroFile instances depending on the given header fields.
        After sorting the contents the method will return a pointer to this
        AstroDir.

        Parameters
        ----------
        args : str
            A header field name to be used as comparison key

        Raises
        ------
        ValueError
            If the field given cannot be used to compare each AstroFile or
            if no header field was specified.

        """
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
            f.add_sortkey(hdrfld)
        self.files.sort()
        return self

    def __repr__(self):
        return "<AstroFile container: %s>" % (self.files.__repr__(),)

    def __add__(self, other):
        if isinstance(other, AstroDir):
            other_af = other.files
        elif isinstance(other, (list, tuple)) and isinstance(other[0], str):
            io_logger.warning("Adding list of files to Astrodir, calib and hdu defaults"
                             " will be shared from first AstroFile in AstroDir")
            other_af = [dp.AstroFile(filename,
                                     hdud=self[0].default_hdu_dh()[0],
                                     hduh=self[0].default_hdu_dh()[1],
                                     mflat=self[0].calib.mflat,
                                     mbias=self[0].calib.mbias)
                        for filename in other]
        else:
            raise TypeError("Cannot add {} + {}", self.__class__, other.__class__)

        return dp.AstroDir(self.files+other_af)

    # def __eq__(self, other):
    #     #as in scipy
    #     if isinstance(other, )

    def __getitem__(self, item):
        # imitate indexing on boolean array as in scipy.
        if isinstance(item, np.ndarray):
            if item.dtype == 'bool':
                if len(item) != len(self):
                    raise ValueError("Attempted to index AstroDir with a boolean array "
                                     "of different size (it must include all bads)")

                fdir = [f for b, f in zip(item, self) if b]
                return AstroDir(fdir)

        #if string, return as getheaderval
        if isinstance(item, str):
            return self.getheaderval(item)

        #if slice, return a new astrodir
        elif isinstance(item, slice):
            return AstroDir(self.files.__getitem__(item))

        #otherwise, use a list
        return self.files[item]  # .__getitem__(item)

    def __len__(self):
        return len(self.files)

    def stats(self, *args, **kwargs):
        """
        Obtains statistical data from each AstroFile stored in this instance

        Parameters
        ----------
        extra_headers : list, optional
            List of header items to include on the output
        verbose_heading : bool, optional
        args : Specify the stats that want to be returned

        Returns
        -------
        array_like
            The stat as returned by each of the AstroFiles

        See Also
        --------
        AstroFile.stats : for the available statistics
        """
        verbose_heading = kwargs.pop('verbose_heading', True)
        extra_headers = kwargs.pop('extra_headers', [])
        if kwargs:
            raise SyntaxError("only the following keyword arguments for stats are accepted 'verbose_heading', 'extra_headers'")
        ret = []
        for af in self:
            ret.append(af.stats(*args, verbose_heading=verbose_heading, extra_headers=extra_headers))
            verbose_heading = False
        return ret

    def filter(self, *args, **kwargs):
        """
        Filter files according to those whose filter return True to the given
        arguments.
        What the filter does is type-dependent for each file.

        Logical 'and' statements can be simulated by chaining this method
        multiple times.

        Parameters
        ----------
        **kwargs :
            Keyword arguments containing the name of the item and the expected
            value.

        Returns
        -------
        dataproc.AstroDir
            Copy containing the files which were not discarded.

        See Also
        --------
        AstroFile.filter : Specifies the syntax used by the recieved arguments.
        """
        from copy import copy
        new = copy(self)
        new.files = [f for f in self if f.filter(*args, **kwargs)]
        return new

    def basename(self, joinchr=', '):
        """
        Obtains the basename each file contained.

        Parameters
        ----------
        joinchar : str, optional
            Character used to separate the name of each file

        Returns
        -------
        str
            Each file basename separated by the specified 'joinchar'

        """
        return joinchr.join([b.basename() for b in self])

    def getheaderval(self, *args, **kwargs):
        """
        Gets the header values specified in 'args' from each file.
        A function can be specified to be used over the returned values for
        casting purposes.

        Parameters
        ----------
        cast : function, optional
            Function output used to cas the output

        Returns
        -------
        List of integers or tuples
            Type depends if a single value or multiple tuples were given

        """

        if 'cast' in kwargs:
            cast = kwargs['cast']
        else:
            cast = lambda x: x

        warnings.filterwarnings("once", "non-standard convention", AstropyUserWarning)
        ret = [f.getheaderval(*args, cast=cast, **kwargs) for f in self]
        try:
            while True:
                idx = ret.index(None)
                logging.warning("Removing file with defective header from AstroDir")
                self.files.pop(idx)
                ret.pop(idx)
        except ValueError:
            pass

        warnings.resetwarnings()
        return ret

    def setheader(self, **kwargs):
        """
        Sets the header values specified in 'args' from each of the files.
        """
        if False in [f.setheader(**kwargs) for f in self]:
            raise ValueError("Setting the header of a file returned error... panicking!")

        return self

    def get_datacube(self, normalize_region=None, normalize=False, verbose=False,
                     check_unique=None, group_by=None):
        """
        Generates a data cube with the data of each AstroFile.

        Parameters
        ----------
        normalize_region :
            Subregion to use for normalization
        normalize : bool, optional
            If True, will normalize the data inside the data cube
        verbose : bool, optional
        check_unique : array_like, optional
            Receives a list of headers that need to be checked for uniqueness.
        group_by : str, optional
            Datacube will be organized based on the header item given

        Returns
        -------
        dict
            Dictionary containing a data cube grouped by the given parameters,
            otherwise the data will be indexed to None as key
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

        grouped_data = {}

        # check uniqueness... first get the number of unique header values for each requested keyword
        # and then complain if any greater than 1
        if len(check_unique) > 1:
            lens = np.array([len(set(rets)) for rets in zip(*self.getheaderval(*check_unique))])
        elif len(check_unique) == 1:
            lens = np.array([len(set(self.getheaderval(*check_unique)))])
        else:
            lens = np.array([])
        if True in (lens > 1):
            raise ValueError("Header(s) {} are not unique in the input dataset".
                             format(", ".join(np.array(check_unique)[lens > 1])))

        for af in self:
            data = af.reader()
            if normalize:
                data /= data[normalize_region].mean()
            group_key = af[group_by]
            if group_key not in grouped_data.keys():
                grouped_data[group_key] = []
            grouped_data[group_key].append(data)

            if verbose:
                print(".", end='')
                sys.stdout.flush()

        if None in grouped_data and len(grouped_data) > 1:
            # TODO: check error message
            raise ValueError("Panic. None should have been key of grouped_data only if group_by was None. "
                            "In such case, it should have been alone.")

        for g in grouped_data:
            grouped_data[g] = np.stack(grouped_data[g])

        if verbose:
            print("")

        return grouped_data

    def pixel_xy(self, xx, yy, normalize_region=None, normalize=False,
             check_unique=None, group_by=None):
        """
        Obtains the pixel value at the ('xx','yy') position for all frames.

        Parameters
        ----------
        xx, yy: int, int
        normalize_region :
            Subregion to use for normalization
        group_by :
            If set, then group by the specified header returning an
            [n_unique, n_y, n_x] array
        normalize : bool, optional
            Whether to Normalize
        check_unique : str, optional
            Check uniqueness in header

        Returns
        -------
        array_like
            Values of the pixel on each frame
        """
        if check_unique is None:
            check_unique = []
        if group_by in check_unique:
            check_unique.pop(check_unique.index(group_by))

        data_dict = self.get_datacube(normalize_region=normalize_region,
                                      normalize=normalize,
                                      check_unique=check_unique,
                                      group_by=group_by)

        groupers = sorted(data_dict.keys())
        ret = [data_dict[g][:, yy, xx] for g in groupers]

        if len(groupers) > 1:
            return {g: r for r, g in zip(ret, groupers)}
        else:
            return ret[0]
        # todo: return AstroFile

    def median(self, normalize_region=None, normalize=False, verbose=True,
               check_unique=None, group_by=None):
        """
        Calculates the median of the contained files

        Parameters
        ----------
        normalize_region :
            Subregion to use for normalization
        normalize : bool, optional
            Whether to Normalize
        verbose : bool, optional
            Output progress indicator
        check_unique : str, optional
            Check uniqueness in header
        group_by :
            If set, then group by the specified header returning
            an [n_unique, n_y, n_x] array

        Returns
        -------
        array_like
            Median between frames
        dict
            If group_by is set, a dictionary keyed by this group will be returned
        """
        if check_unique is None:
            if normalize:
                check_unique = []
            else:
                check_unique = ['exptime']
        if group_by in check_unique:
            check_unique.pop(check_unique.index(group_by))

        data_dict = self.get_datacube(normalize_region=normalize_region,
                                      normalize=normalize,
                                      verbose=verbose,
                                      check_unique=check_unique,
                                      group_by=group_by)

        if verbose:
            print("Median combining{}".format(group_by is not None and ' grouped by "{}":'.format(group_by) or ""),
                  end='')
            sys.stdout.flush()
        groupers = sorted(data_dict.keys())
        ret = np.zeros([len(groupers)] + list(data_dict[groupers[0]][0].shape))
        for k in range(len(groupers)):
            if verbose and group_by is not None:
                print("{} {}".format(k and "," or "", groupers[k]), end='')
                sys.stdout.flush()
            ret[k, :, :] = np.median(data_dict[groupers[k]], axis=0)
        if verbose:
            print(". done.")
            sys.stdout.flush()

        if len(groupers) > 1:
            return {g: r for r, g in zip(ret, groupers)}
        else:
            return ret[0]
        # todo: return AstroFile

    def mean(self, normalize_region=None, normalize=False, verbose=True,
             check_unique=None, group_by=None):
        """
        Calculates the mean of the contained files

        Parameters
        ----------
        normalize_region :
            Subregion to use for normalization
        normalize : bool, optional
            Whether to Normalize
        verbose : bool, optional
            Output progress indicator
        check_unique : str, optional
            Check uniqueness in header
        group_by :
            If set, then group by the specified header returning an
            [n_unique, n_y, n_x] array

        Returns
        -------
        ndarray
            Mean between frames
        dict
            If group_by is set, a dictionary keyed by this group will be returned

        """
        if check_unique is None:
            if normalize:
                check_unique = []
            else:
                check_unique = ['exptime']
        if group_by in check_unique:
            check_unique.pop(check_unique.index(group_by))

        data_dict = self.get_datacube(normalize_region=normalize_region,
                                      normalize=normalize,
                                      verbose=verbose,
                                      check_unique=check_unique,
                                      group_by=group_by)

        if verbose:
            print("Mean combining{}".format(group_by is not None and ' grouped by "{}":'.format(group_by) or ""),
                  end='')
            sys.stdout.flush()
        groupers = sorted(data_dict.keys())
        ret = np.zeros([len(groupers)] + list(data_dict[groupers[0]][0].shape))
        for k in range(len(groupers)):
            if verbose and group_by is not None:
                print("{} {}".format(k and "," or "", groupers[k]), end='')
                sys.stdout.flush()
            ret[k, :, :] = np.mean(data_dict[groupers[k]], axis=0)
        if verbose:
            print(": done.")
            sys.stdout.flush()

        if len(groupers) > 1:
            return {g: r for r, g in zip(ret, groupers)}
        else:
            return ret[0]
        # todo: return AstroFile

    # todo: add support for giving variance
    def lin_interp(self, target=0,
                   verbose=True, field='EXPTIME',
                   check_unique=None, group_by=None):
        """
        Linear interpolate to given target value of the field 'field'.

        Parameters
        ----------
        target : int, optional
            Target value for the interpolation
        verbose : bool, optional
            Output progress indicator
        field : str, optional
            Name of the header item which will be used to interpolate the frames
        group_by :
            If set, then group by the specified header returning an [n_unique, n_y, n_x] array
        check_unique : str, optional
            Check uniqueness in header

        Returns
        -------
        ndarray
            Linear Interpolation between frames with the specified target
        dict
            If group_by is set, a dictionary keyed by this group will be returned

        """
        if check_unique is None:
            check_unique = []
        if group_by in check_unique:
            check_unique.pop(check_unique.index(group_by))

        data_dict = self.get_datacube(verbose=verbose,
                                      check_unique=check_unique,
                                      group_by=group_by)

        if verbose:
            print("Linear interpolating{}".format(group_by is not None and ' grouped by "{}":'.format(group_by) or ""),
                  end='')
            sys.stdout.flush()
        groupers = sorted(data_dict.keys())
        ret = np.zeros([len(groupers)] + list(data_dict[groupers[0]][0].shape))
        for k in range(len(groupers)):
            if verbose and group_by is not None:
                print("{} {}".format(k and "," or "", groupers[k]), end='')
                sys.stdout.flush()

            data_cube = data_dict[groupers[k]]
            values = self.getheaderval(field)
            values, data_cube = dp.sortmanynsp(values, data_cube)

            if not (values[0] <= target <= values[-1]):
                io_logger.warning("Target value for interpolation ({})\n outside "
                                  "the range of chosen keyword {}: [{}, {}]."
                                  "\nCannot interpolate".format(target, field,
                                                                values[0], values[-1]))
                return None

            for j in range(data_cube[0].shape[1]):
                for i in range(data_cube[0].shape[0]):
                    ret[k, j, i] = np.interp(target, values, data_cube[:, j, i])
        if verbose:
            print(". done.")
            sys.stdout.flush()

        if len(groupers) > 1:
            return {g: r for r, g in zip(ret, groupers)}
        else:
            return ret[0]

    def std(self, normalize_region=None, normalize=False, verbose=True,
             check_unique=None, group_by=None):
        """
        Get pixel by pixel standard deviation within files.

        Parameters
        ----------
        normalize_region :
            Subregion to use for normalization
        normalize : bool, optional
            Whether to Normalize
        verbose : bool, optional
            Output progress indicator
        check_unique : str, optional
            Check uniqueness in header
        group_by :
            If set, then group by the specified header returning an
            [n_unique, n_y, n_x] array

        Returns
        -------
        ndarray
            Standard deviation between frames
        dict
            If group_by is set, a dictionary keyed by this group will be returned

        """
        if check_unique is None:
            if normalize:
                check_unique = []
            else:
                check_unique = ['exptime']
        if group_by in check_unique:
            check_unique.pop(check_unique.index(group_by))

        data_dict = self.get_datacube(normalize_region=normalize_region,
                                      normalize=normalize,
                                      verbose=verbose,
                                      check_unique=check_unique,
                                      group_by=group_by)

        if verbose:
            print("Obtaining Standard Deviation{}".format(group_by is not None
                                                          and ' grouped by "{}":'.format(group_by) or ""),
                  end='')
            sys.stdout.flush()
        groupers = sorted(data_dict.keys())
        ret = np.zeros([len(groupers)] + list(data_dict[groupers[0]][0].shape))
        for k in range(len(groupers)):
            if verbose and group_by is not None:
                print("{} {}".format(k and "," or "", groupers[k]), end='')
                sys.stdout.flush()
            ret[k, :, :] = np.std(data_dict[groupers[k]], axis=0)
        if verbose:
            print(". done.")
            sys.stdout.flush()

        if len(groupers) > 1:
            return {g: r for r, g in zip(ret, groupers)}
        else:
            return ret[0]
        # todo: return AstroFile

    def jd_from_ut(self, target='jd', source='date-obs'):
        """
        Add jd in header's cache to keyword 'target' using ut on keyword
        'source'

        Parameters
        ----------
        target: str
            Target keyword for JD storage
        source: str
            Location of the input value on UT format

        See Also
        --------
        dataproc.AstroFile.jd_from_ut : For individual file implementation
        """
        for af in self:
            af.jd_from_ut(target, source)

        return self

    def merger(self, start=1, end=None):
        """
        Merges HDUImage data for all files contained in this object

        Parameters
        ----------
        start : int, optional
            HDU unit from which the method starts joining
        end ; int, optional
            Last ImageHDU unit to be included, if None will stop at the last
            hdu found

        See Also
        --------
        dataproc.AstroFile.merger : For individual file implementation
        """

        for files in self:
            files.merger(start=start, end=end)

        return self
