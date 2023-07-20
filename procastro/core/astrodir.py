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
__all__ = ['AstroDir']

from typing import Optional

import procastro as pa
from .internal_functions import trim_to_python, common_trim_fcn, extract_common, python_to_trim
import numpy as np
import warnings
import sys
from astropy.utils.exceptions import AstropyUserWarning

import logging
logging.basicConfig(level=logging.INFO)
io_logger = logging.getLogger('procastro.io')
io_logger.propagate = False


class AstroDir(object):
    """Collection of AstroFile objects.

    Collection can be indexed, filtered, and has several recursive methods that are applied to each
     AstroFile are available

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
    hdu : int, optional
        default HDU
    hdud : int, optional
        default HDU for data
    hduh : int, optional
        default HDU for the header

    See Also
    --------
    procastro.AstroCalib : Object that holds calibration information.
                          One of them can be shared by many AstroFile instances

    procastro.AstroCalib.add_bias
    procastro.AstroCalib.add_flat
    """

    def __new__(cls, *args, **kwargs):
        """
        If passed an AstroDir, then do not create a new instance, just pass
        that one
        """
        if args and isinstance(args[0], AstroDir) and len(kwargs) == 0:
            return args[0]

        return super(AstroDir, cls).__new__(cls)

    def __init__(self, path, mflat=None, mbias=None,
                 mbias_header=None, mflat_header=None,
                 mbias_filter: Optional[str] = None,
                 mflat_filter: Optional[str] = None,
                 raise_calib_filter = True,
                 calib_force=False,
                 hdu=0, hdud=None, hduh=None, auto_trim=None,
                 jd_from_ut=None):

        import os
        import glob
        import os.path as pth
        files = []

        if hasattr(self, "files"):
            return

        if hduh is None:
            hduh = hdu
        if hdud is None:
            hdud = hdu

        if isinstance(path, str):
            file_n_dir = glob.glob(path)
        elif isinstance(path, pa.AstroFile):
            file_n_dir = [path]
        else:
            file_n_dir = path
        for f in file_n_dir:
            if isinstance(f, str) and pth.isdir(f):
                for sf in os.listdir(f):
                    nf = pa.AstroFile(f + '/' + sf,
                                      hduh=hduh,
                                      hdud=hdud,
                                      auto_trim=auto_trim)
                    try:
                        if nf:
                            files.append(nf)
                    except IOError:
                        io_logger.warning(f"Warning: File {nf.basename()} could"
                                          f"not be read, skipping")

                nf = False
            else:
                nf = pa.AstroFile(f, hduh=hduh,
                                  hdud=hdud,
                                  auto_trim=auto_trim)

            try:
                if nf:
                    files.append(nf)
            except IOError:
                io_logger.warning(f"Warning: File {nf.basename()} could not "
                                  f"be read, or HDU {hdud} empty... skipping")

        self.files = files
        self.props = {}

        calib = pa.AstroCalib(mbias, mflat,
                              auto_trim=auto_trim,
                              mbias_header=mbias_header,
                              mflat_header=mflat_header)

        # check that the given calibrator complies with its given filters
        filter_calib = {}
        calibrators = ["bias", "flat"]
        for calibrator in calibrators:
            if vars()[f"m{calibrator}_filter"] is not None and getattr(calib, f"m{calibrator}_header") is not None:
                for value in vars()[f"m{calibrator}_filter"]:
                    if (value in filter_calib and
                            getattr(calib, f"m{calibrator}_header")[value] != filter_calib[value]):
                        raise ValueError(f"{'/'.join(calibrators)} have conflictive filter requests")
                    filter_calib[value] = getattr(calib, f"m{calibrator}_header")

        for f in files:
            # Allows files with preexistent calibrations to keep it
            if calib_force or not f.has_calib():
                # AstroFile are created with an (0, 1) calib by default, which
                # is overwritten here if they pass the calib filters
                for k, v in filter_calib.items():
                    if f[k] != v:
                        message = f"File {f} does not comply with calibration filter '{f[k]}' = {v}"
                        if raise_calib_filter:
                            raise IOError(message)
                        else:
                            io_logger.warning(message)
                f.set_calib(calib)

        self.path = path

        if jd_from_ut is not None:
            if len(jd_from_ut) != 2:
                raise TypeError("jd_from_ut parameter need to be a 2-element "
                                "tuple: source, target."
                                "See help on method .jd_from_ut()")
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
        procastro.AstroCalib.add_bias : this function is called for each unique
                                       AstroCalib object in AstroDir
        """
        unique_calibs = set([f._calib for f in self])
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
        procastro.AstroCalib.add_flat : This function is called for each unique
                                       AstroCalib object in AstroDir
        """
        unique_calibs = set([f._calib for f in self])
        for c in unique_calibs:
            c.add_flat(mflat)

    def sort(self, *args):
        """
        Sorts AstroFile instances inplace depending on the given header fields.
        After sorting the contents the method will return None to avoid problems

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
            raise ValueError("At least one valid header field must be "
                             "specified to sort")
        hdrfld = False

        for a in args:
            if None not in self.values(a):
                hdrfld = a
                break
        if not hdrfld:
            raise ValueError(
                "A valid header field must be specified to use as a sort key."
                " None of the currently requested "
                "were found: {0:s}".format(', '.join(args),))

        # Sorting is done using python operators __lt__, __gt__
        # which are inquired by .sort() directly.
        for f in self:
            f.add_sortkey(hdrfld)
        self.files.sort()
        return None

    def __repr__(self):
        return "<AstroFile container: {0:s}>".format(self.files.__repr__(),)

    def __add__(self, other):
        if isinstance(other, AstroDir):
            other_af = other.files
        elif isinstance(other, (list, tuple)) and isinstance(other[0], str):
            io_logger.warning("Adding list of files to Astrodir, calib and hdu"
                              "defaults will be shared from first "
                              "AstroFile in AstroDir")
            other_af = [pa.AstroFile(filename,
                                     hdud=self[0].default_hdu_dh()[0],
                                     hduh=self[0].default_hdu_dh()[1],
                                     mflat=self[0]._calib.mflat,
                                     mbias=self[0]._calib.mbias)
                        for filename in other]
        else:
            raise TypeError(
                "Cannot add {0:s} + {1:s}".format(self.__class__.__name__,
                                                  other.__class__.__name__))

        return pa.AstroDir(self.files + other_af)

    def __getitem__(self, item):

        # if an integer, return that astrofile
        if isinstance(item, (int, np.integer)):
            return self.files[item]  # .__getitem__(item)

        # if string, return as values()
        if isinstance(item, str):
            return self.values(item)

        # imitate indexing on boolean array as in scipy.
        if isinstance(item, np.ndarray):
            if item.dtype == 'bool':
                if len(item) != len(self):
                    raise ValueError("Attempted to index AstroDir with "
                                     "a boolean array of different size"
                                     "(it must include all bads)")

                fdir = [f for b, f in zip(item, self) if b]

                ad = AstroDir(fdir)
                ad.props = self.props.copy()
                ad.path = self.path

                return ad

        # if slice, return a new astrodir
        if isinstance(item, slice):
            return AstroDir(self.files.__getitem__(item))

        # else asume list of integers, then return an Astrodir with those indices
        try:
            return AstroDir([self[i] for i in item])
        except TypeError:
            pass

        raise TypeError(f"item ({item}) is not of a valid type: np.array of booleans, int, str, iterable of ints")

    def __len__(self):
        return len(self.files)

    def stats(self, *args, **kwargs):
        """
        Obtains statistical data from each AstroFile stored in this instance

        Parameters
        ----------
        args : Specify the stats that want to be returned

        Returns
        -------
        array_like
            The stat as returned by each of the AstroFiles

        See Also
        --------
        procastro.AstroFile.stats : for the available statistics
        """
        verbose_heading = kwargs.pop('verbose_heading', True)
        extra_headers = kwargs.pop('extra_headers', [])
        if kwargs:
            raise SyntaxError(
                "Only the following keyword arguments for stats"
                "are accepted: 'verbose_heading', 'extra_headers'")

        ret = []
        for af in self:
            ret.append(af.stats(*args, verbose_heading=verbose_heading,
                                extra_headers=extra_headers))
            verbose_heading = False
        return np.array(ret)

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
        procastro.AstroDir
            Copy containing the files which were not discarded.

        See Also
        --------
        procastro.AstroFile.filter :
            Specifies the syntax used by the recieved arguments.

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
        joinchr : str, optional
            Character used to separate the name of each file

        Returns
        -------
        str
            Each file basename separated by the specified 'joinchar'

        """
        return joinchr.join([b.basename() for b in self])

    def values(self, *args, cast=None, single_in_list=False, hdu=None):
        """
        Gets the header values specified in 'args' from each file.
        A function can be specified to be used over the returned values for
        casting purposes.

        Parameters
        ----------
        single_in_list
        hdu
        cast : function, optional
            Function output used to cas the output

        Returns
        -------
        List of integers or tuples
            Type depends on whether a single value or multiple tuples were given

        """

        if cast is None:
            def cast(x): return x

        warnings.filterwarnings("once",
                                "non-standard convention",
                                AstropyUserWarning)
        ret = [f.values(*args, cast=cast, single_in_list=single_in_list, hdu=hdu) for f in self]

        warnings.resetwarnings()
        return np.array(ret)

    def setheader(self, **kwargs):
        """
        Sets the header values specified in 'args' from each of the files.
        """
        if False in [f.set_values(**kwargs) for f in self]:
            raise ValueError("Setting the header of a file returned error")

        return self

    def get_datacube(self, normalize_region=None, normalize=False,
                     verbose=False, check_unique=None, group_by=None,
                     trim_area_warn=0.25):
        """
        Generates a data cube with the data of each AstroFile.

        Parameters
        ----------
        trim_area_warn
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
                                                      normalize
                                                      and " and normalizing"
                                                      or ""),
                  end='')
        if check_unique is None:
            check_unique = []
        if normalize_region is None:
            if 'normalize_region' in self.props:
                normalize_region = self.props['normalize_region']
            else:
                normalize_region = slice(None)

        grouped_data = {}
        grouped_trims = {}

        # check uniqueness... first get the number of unique header values for
        # each requested keyword and then complain if any greater than 1
        if len(check_unique) > 1:
            lens = np.array([len(set(rets)) for rets
                             in zip(*self.values(*check_unique))])
        elif len(check_unique) == 1:
            lens = np.array([len(set(self.values(*check_unique)))])
        else:
            lens = np.array([])
        if True in (lens > 1):
            raise ValueError(
                "Header(s) {} are not unique in the input dataset"
                .format(", ".join(np.array(check_unique)[lens > 1])))

        correct = []
        for af in self:
            try:
                data = af.reader().astype(float)
            except IOError:
                correct.append(False)
                continue
            correct.append(True)

            if normalize:
                data /= data[normalize_region].mean()
            group_key = af[group_by]
            if group_key not in grouped_data.keys():
                grouped_data[group_key] = []
                grouped_trims[group_key] = []
            grouped_data[group_key].append(data)
            grouped_trims[group_key].append(af.get_trims())

            if verbose:
                print(".", end='')
                sys.stdout.flush()

        try:
            while True:
                idx = correct.index(False)
                logging.warning(f"Removing file with defective data "
                                f"from AstroDir: {self.files[idx].basename}")
                self.files.pop(idx)
                correct.pop(idx)
        except ValueError:
            pass

        if None in grouped_data and len(grouped_data) > 1:
            # TODO: check error message
            raise ValueError("Panic. None should have been key of "
                             "grouped_data only if group_by was None. "
                             "In such case, it should have been alone.")

        # first loop for warnings only
        for g in grouped_data:
            unique_trims = set(grouped_trims[g])
            common_trim = common_trim_fcn(grouped_trims[g])
            if common_trim is None:
                io_logger.warning("Auto-Trim section not defined or not found in all objects, using full frames only")
                break
            common_trim_area = (common_trim[1] - common_trim[0])*(common_trim[3] - common_trim[2])

            if len(unique_trims) > 1:
                for trim in unique_trims:
                    trim_area = (trim[1] - trim[0])*(trim[3] - trim[2])
                    if trim_area > (1 + trim_area_warn)*common_trim_area:
                        io_logger.warning(f"Trim section ({trim}) is"
                                          f" {100*(common_trim/common_trim_area-1)}% larger than"
                                          f" common area({common_trim})")

        # second loop applies trim
        for g in grouped_data:
            common_trim = common_trim_fcn(grouped_trims[g])

            if common_trim is None:
                grouped_data[g] = np.array(grouped_data[g])
            else:
                trim_data = []

                for data, trim in zip(grouped_data[g], grouped_trims[g]):

                    out, trimmed = extract_common(data, trim, common_trim)
                    if trimmed and verbose:
                        logging.info(f"Adjusting to minimum common trim: "
                                     f"[({trim}) -> ({common_trim})]")
                    trim_data.append(out)

                grouped_data[g] = np.stack(trim_data)

        if verbose:
            print("")

        return grouped_data
    
    def common_trim(self,
                    group_by=None,
                    return_as_fits=False,
                    ):
        """Computes minimum common trim area

        Parameters
        ----------
        return_as_fits: bool
            If True return string in standard FITS format, otherwise a np.array.
        group_by : str
            Header field for grouping the files
        """
        trims = {}        
        common_trim = {}
        for af in self:
            try:
                group_key = af[group_by]
            except IOError:
                continue
            if group_key not in trims.keys():
                trims[group_key] = []
            trims[group_key].append(af.get_trims())
        for group in trims.keys():
            common_trim[group] = common_trim_fcn(trims[group])

        if return_as_fits:
            ret = {}
            for key, val in common_trim.items():
                ret[key] = python_to_trim(val)
            return ret
        return common_trim

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
            If group_by is set, a dictionary keyed by this group will be
            returned
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
            print("Median combining{}".format(group_by
                                              is not None
                                              and f' grouped by "{group_by}":'
                                              or ""),
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
            If group_by is set, a dictionary keyed by this group will be
            returned

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
            print("Mean combining{}".format(group_by
                                            is not None
                                            and f' grouped by "{group_by}":'
                                            or ""),
                  end='')
            sys.stdout.flush()
        groupers = sorted(data_dict.keys())
        ret = np.zeros([len(groupers)]
                       + list(data_dict[groupers[0]][0].shape))
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
            Name of the header item which will be used to interpolate
            each frame
        group_by :
            If set, then group by the specified header returning an [n_unique,
            n_y, n_x] array
        check_unique : str, optional
            Check uniqueness in header

        Returns
        -------
        ndarray
            Linear Interpolation between frames with the specified target
        dict
            If group_by is set, a dictionary keyed by this group will be
            returned

        """
        if check_unique is None:
            check_unique = []
        if group_by in check_unique:
            check_unique.pop(check_unique.index(group_by))

        data_dict = self.get_datacube(verbose=verbose,
                                      check_unique=check_unique,
                                      group_by=group_by)

        if verbose:
            print("Linear interpolating{}"
                  .format(group_by
                          is not None
                          and f' grouped by "{group_by}":'
                          or ""),
                  end='')

            sys.stdout.flush()
        groupers = sorted(data_dict.keys())
        ret = np.zeros([len(groupers)]
                       + list(data_dict[groupers[0]][0].shape))
        for k in range(len(groupers)):
            if verbose and group_by is not None:
                print("{} {}".format(k and "," or "", groupers[k]), end='')
                sys.stdout.flush()

            data_cube = data_dict[groupers[k]]
            values = self.values(field)
            values, data_cube = pa.sortmanynsp(values, data_cube)

            if not (values[0] <= target <= values[-1]):
                io_logger.warning("Target value for interpolation ({})\n"
                                  "outside the range of chosen keyword {}: "
                                  "[{}, {}].\nCannot "
                                  "interpolate".format(target,
                                                       field,
                                                       values[0],
                                                       values[-1]))
                return None

            for j in range(data_cube[0].shape[1]):
                for i in range(data_cube[0].shape[0]):
                    ret[k, j, i] = np.interp(target,
                                             values,
                                             data_cube[:, j, i])
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
            If group_by is set, a dictionary keyed by this group will be
            returned

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
            print("Obtaining "
                  "Standard Deviation{}".format(group_by
                                                is not None
                                                and f"grouped by '{group_by}':"
                                                or ""),
                  end='')
            sys.stdout.flush()
        groupers = sorted(data_dict.keys())
        ret = np.zeros([len(groupers)]
                       + list(data_dict[groupers[0]][0].shape))
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

    def jd_from_ut(self, target='jd', source='date-obs', return_only_clean=False):
        """
        Add jd in header's cache to keyword 'target' using ut on keyword
        'source'

        Parameters
        ----------
        return_only_clean
        target: str
            Target keyword for JD storage
        source: str or list
            Location of the input value on UT format

        See Also
        --------
        procastro.AstroFile.jd_from_ut : For individual file implementation
        """
        corrupt = np.ones(len(self)) == 0
        for i, af in enumerate(self):
            try:
                af.jd_from_ut(target, source)
            except ValueError as msg:
                if not return_only_clean:
                    raise ValueError("You might want to try return_only_clean=True in jd_from_ut()\n"+str(msg))
                io_logger.info(f"Removing file {af.filename} with corrupt timestamp ({af[source]})")
                corrupt[i] = True

        return self[~corrupt]

    def merger(self, start=1, end=None, verbose=None):
        """
        Merges HDUImage data for all files contained in this object

        Parameters
        ----------
        verbose: str
            a single character to be repeated unbuffered for every AstroFile
        start : int, optional
            HDU unit from which the method starts joining
        end ; int, optional
            Last ImageHDU unit to be included, if None will stop at the last
            hdu found

        See Also
        --------
        procastro.AstroFile.merger : For individual file implementation

        """
        if isinstance(verbose, str):
            verbose = verbose[0]
        else:
            verbose = None

        with warnings.catch_warnings():
            warnings.filterwarnings("once", category=UserWarning, message=".+Merged image stored in HDU 0.+")
            for files in self:
                print(verbose, flush=True, end='')
                files.merger(start=start, end=end)

        return self
