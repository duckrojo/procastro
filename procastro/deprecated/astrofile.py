# procastro - general data processing routines
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

__all__ = ['AstroFile']

from functools import wraps as _wraps
import warnings
from pathlib import PurePath, Path

import procastro as pa
import procastro.astrofile.spec
from procastro.statics import trim_to_python
from procastro.config import pa_logger

from procastro.calib.raw2d import CalibRaw2D
from procastro.cache.cache import astrofile_cache

import astropy.time as apt
import numpy as np
import astropy.io.fits as pf


def _numerize_other(method):
    """
    When operating between two AstroFile objects, this decorator will try to
    read the second AstroFile first.
    """

    @_wraps(method)
    def wrapper(instance, other, *args, **kwargs):
        if isinstance(other, AstroFile):
            other = other.reader()
        return method(instance, other, *args, **kwargs)

    return wrapper


#######################
#
# FITS handling of AstroFile
#
##################################


def _fits_reader(filename, hdu=0):
    """
    Read fits files.

    Parameters
    ----------
    hdu : int, optional
        HDU slot to be read, if -1 returns all hdus

    Returns
    -------
    array_like
    """
    if hdu < 0:
        return pf.open(filename)

    hdu_list = pf.open(filename)
    fl = hdu_list[hdu].data
    hdu_list.close()
    return fl


def _fits_writer(filename, data, header=None):
    """
    Writes 'data' to specified file

    Parameters
    ----------
    filename : str
    data : array_like
    header : Header Object, optional
    """
    if header is None:
        header = pf.Header()
        pa_logger.warning(
            "No header provided to save on disk, using a default empty header"
            "for '{}'".format(filename))
    header['history'] = "Saved by procastro v{} on {}".format(pa.__version__,
                                                              apt.Time.now())
    return pf.writeto(filename, data, header,
                      overwrite=True, output_verify='silentfix')


def _fits_verify(filename, ffilter=None, hdu=0):
    """
    Verifies that the given file has the expected extensions of fit files
    Accepted extensions: 'fits', 'fit', 'ftsc', 'fts' with or without the 'gz' compression

    Parameters
    ----------
    filename : str
    ffilter : dict, list, optional
        If given, the method checks if the file also has the specified values
        on their header
    hdu : int, optional
        HDU to be checked

    Returns
    -------
    bool
    """
    try:
        filename = Path(filename)
    except TypeError:
        return False
    if not isinstance(filename, PurePath):
        return False

    suffixes = filename.suffixes
    if suffixes[-1] in ['.gz']:
        suffixes.pop()
    if suffixes[-1].lower() in ['.fits', '.fit', '.ftsc', '.fts']:
        if ffilter is None:
            return True
        h = pf.getheader(filename, hdu)
        if isinstance(ffilter, dict):
            return False not in [(f in h and h[f] == ffilter[f])
                                 for f in ffilter.keys()]
        if isinstance(ffilter, (list, tuple)):
            return False not in [f in h for f in ffilter]
    return False


def _fits_getheader(_filename, *args, **kwargs):
    """
    Obtains data from headers

    Parameters
    ----------
    _filename : str

    Returns
    -------
    list of dicts
        Each dictionary contains the header data from a hdu unit inside
        the file
    """
    ret = []
    hdu_list = pf.open(_filename)
    for h in hdu_list:
        ret.append({k.lower(): v for k, v in h.header.items()})
    hdu_list.close()

    return ret


def _fits_read_data_header(_filename, hdu=0):
    """
    Obtains data from headers

    Parameters
    ----------
    _filename : str

    Returns
    -------
    list of dicts
        Each dictionary contains the header data from a hdu unit inside
        the file
    """
    ret = []
    hdu_list = pf.open(_filename)
    for h in hdu_list:
        ret.append({k.lower(): v for k, v in h.header.items()})
    fl = hdu_list[hdu].data
    hdu_list.close()

    return fl, ret


#############################
#
# Startnparray type
#
####################################


def _array_reader(array, **kwargs):
    return array


def _array_read_data_header(array, hdu=0):
    return array, [{}]


def _array_write(filename, **kwargs):
    raise NotImplementedError("Write array into file is not implemented yet")


def _array_getheader(array, **kwargs):
    return [{}]


def _array_verify(array, **kwargs):
    if isinstance(array, np.ndarray):
        return True
    return False


#############################
#
# End Defs
#
####################################


def return_none(*args, **kwargs):
    return None


class AstroFile(object):
    """
    Representation of an astronomical data file, contains information related
    to the file's data. Currently, supports usage of numpy.ndarray and .fits files.

    Parameters
    ----------
    filename : str, procastro.AstroFile, numpy.ndarray
        If AstroFile, then it does not read header again, but all new kwargs are applied.
    mbias : dict indexed by exposure time, array or AstroFile, optional
        Default Master bias
    mflat : dict indexed by filter name, array or AstroFile, optional
        Default Master flat
    exists: boolean, optional
        Represents if the given pathname has an existing file
    mbias_header : astropy.io.fits.Header
    mflat_header : astropy.io.fits.Header
    hdu : int, optional
        Default hdu slot used for collection, if "hduh" or "hdud" are not
        specified, this parameter will override them.
    hduh : int, optional
        hdu slot containing the header information to be collected
    hdud : int, optional
        hdu slot containing astronomical data to be collected
    read_keywords : list, optional
        Read all specified keywords at creation time and store them in
        the header cache
    auto_trim : str, optional
        Header field name which is used to cut a portion of the data. This
        field should contain the dimensions of the section to be cut.
        (i.e. TRIMSEC = '[1:1992,1:1708]')
    ron : float, str, astropy quantity, optional
        Read-out-noise value or header keyword. Only consulted if either
        mbias or mflat are not CCDData. If unit not specified, photon/ADU
        is used as default unit; if not set at all, uses 1.0 (with warning)
    gain : float, str, astropy quantity, optional
        Gain value or header keyword. Only consulted if either
        mbias or mflat are not CCDData. If unit not specified, photon/ADU
        is used as default unit; if not set at all, uses 1.0 (with warning)
    unit : astropy.unit, optional
        Specifies the unit for the data

    Attributes
    ----------
    header_cache : dict
        Data contained on the most recently read header
    _sort_key : str
        Header Item used when sorting this object
    type : str
        File extension

    """

    # Interface between public and private methods for each supported type
    _reads = {'fits': _fits_reader, 'nparray': _array_reader}
    _getdh = {'fits': _fits_read_data_header, 'nparray': _array_read_data_header}
    _ids = {'fits': _fits_verify, 'nparray': _array_verify}
    _writes = {'fits': _fits_writer, 'nparray': _array_write}
    _geth = {'fits': _fits_getheader, 'nparray': _array_getheader}

    # def __repr__(self):
    #     if isinstance(self.filename, str):
    #         filename = self.filename
    #     elif isinstance(self.filename, np.ndarray):
    #         filename = f"Array {'x'.join([str(i) for i in self.filename.shape])}"
    #     else:
    #         filename = str(self.filename)
    #     return '<AstroFile{}: {}>'.format(self.has_calib()
    #                                       and "({}{})"
    #                                       .format(self._calib.has_bias
    #                                               and "B" or "",
    #                                               self._calib.has_flat
    #                                               and "F" or "", )
    #                                       or "",
    #                                       filename, )
    #
    # def __new__(cls, *args, **kwargs):
    #     """
    #     If passed an AstroFile, then do not create a new instance, just pass
    #     that one. If passed None, return None
    #     """
    #     if args and ((isinstance(args[0], AstroFile) and len(kwargs) == 0) or args[0]) is None:
    #         return args[0]
    #
    #     return super(AstroFile, cls).__new__(cls)
    #
    def __init__(self, filename=None,
                 mbias=None, mflat=None,
                 mbias_header=None, mflat_header=None,
                 hdu=None, hduh=None, hdud=None,
                 auto_trim=None,
                 header: dict = None,
                 sort_key="filename",
                 *args, **kwargs):

        self.auto_trim = auto_trim
        self._corrupt = False
        self._calib = None
        self._sort_key = None

        self.add_sortkey(sort_key)

        if isinstance(filename, AstroFile):
            self._hduh = filename._hduh
            self._hdud = filename._hdud
            if hdu is not None:
                self._hduh = hduh
                self._hdud = hdud
            if hduh is not None:
                self._hduh = hduh
            if hdud is not None:
                self._hdud = hdud
            self.filename = filename.filename
            self.type = filename.type
            self.header_cache = filename.header_cache

            if all(v is None for v in [mbias, mbias_header, mflat, mflat_header, auto_trim]):
                self._calib = filename._calib
        else:
            if hdu is None:
                hdu = 0
            self._hdud = self._hduh = hdu
            if hduh is not None:
                self._hduh = hduh
            if hdud is not None:
                self._hdud = hdud
            self.filename = filename

            self.type = self.checktype(*args, **kwargs)
            try:
                if pa.astrofile_cache.available():
                    _, self.header_cache = self.read_data_headers()
                else:
                    self.header_cache = self.read_headers()
            except OSError:
                self.type = None
                pa_logger.warning("Omitting corrupt file")
                return

        # if custom header is added, then overwrite/append that in the first HDU
        if isinstance(header, (dict, pf.Header)):
            for k, v in header.items():
                self.header_cache[0][k] = v

        if self._calib is None:
            self._calib = CalibRaw2D(procastro.astrofile.astrofile.AstroFile(mbias, header=mbias_header),
                                     procastro.astrofile.astrofile.AstroFile(mflat, header=mflat_header),
                                     auto_trim=auto_trim)

    def read_headers(self):
        return self._geth[self.type](self.filename)

    def read_data_headers(self):
        return self._getdh[self.type](self.filename)
    #
    # def __hash__(self):
    #     """If calib changes, then the astrofile hash should change as well. ."""
    #     # todo: manually changing some of the header's keywords might affect the output as well losing uniqueness.
    #     return hash((self.filename, self._calib))

    def set_calib(self,
                  calib: "CalibRaw2D",
                  ):
        if not isinstance(calib, CalibRaw2D):
            raise TypeError("calib needs to be AstroCalib in .set_calib()")
        self._calib = calib

    def get_calib(self,
                  ) -> CalibRaw2D:
        return self._calib

    def get_trims(self):
        """"Get trim limits, returning in python-style YX"""
        if self.auto_trim is None:
            return None

        # TODO: include minimum common area with AstroCalib

        return trim_to_python(self[self.auto_trim])

    def add_bias(self, mbias, header=None):
        """
        Includes a Master Bias to this instance

        Parameters
        ----------
        header:
           force new header
        mbias:
           new calibration array or astrofile
        """
        if header is None:
            header = self._calib.bias_header

        # Creating of new AstroCalib upon change of calibration is on purpose as it will
        # prevent using the cached value when reading
        calib = CalibRaw2D(mbias, self._calib.flat,
                           bias_header=header,
                           flat_header=self._calib.flat_header,
                           auto_trim=self._calib.auto_trim_keyword)
        self._calib = calib

    def add_flat(self, mflat, header=None):
        """
        Includes a Master Bias to this instance

        Parameters
        ----------
        master_flat: dict indexed by filter name time, array or AstroFile
        """
        if header is None:
            header = self._calib.flat_header

        # Creating of new AstroCalib upon change of calibration is on purpose as it will
        # prevent using the cached value when reading
        calib = CalibRaw2D(self._calib.bias, mflat,
                           flat_header=header,
                           bias_header=self._calib.flat_header,
                           auto_trim=self._calib.auto_trim_keyword)
        self._calib = calib

    def default_hdu_dh(self):
        """
        Returns default HDU for data and header collection
        """
        return [self._hdud, self._hduh]

    def checktype(self, *args, **kwargs):
        """
        Verifies if the filename given corresponds to an existing file

        Parameters
        ----------

        Returns
        -------
        str: identified by its own filetype identifier
        """
        if not hasattr(self, 'filename'):
            return None

        for k in self._ids.keys():
            if self._ids[k](self.filename, *args, **kwargs):
                return k
        return None

    def plot(self, *args, **kwargs):
        """
        Generates a 1D plot based on a cross-section of the current data.

        Parameters
        ----------
        args: list
            unnamed arguments passed to procastro.plot()
        kwargs: dict
            named arguments passed to procastro.plot()

        Returns
        -------

        """
        return pa.plot_accross(self.reader(), *args, **kwargs)

    def __nonzero__(self):
        return hasattr(self, 'filename') and (self.type is not None)

    # def filter(self, **kwargs):
    #     """
    #     Compares if the expected value of the given header item matches
    #     the value stored in this file header.
    #
    #     Filtering can be customized by including one of the following options
    #     after the item name, separated by an underscore.
    #
    #     - Strings:
    #         + BEGIN:     True if value starts with the given string
    #         + END:       True if value ends with the given string
    #         + ICASE:     Case unsensitive match
    #         + MATCH:     Case sensitive match
    #
    #     - Numeric values:
    #         + LT:        True if current value is lower than the given number
    #         + GT:        True if current value is greater than the given number
    #         + EQUAL:     True if both values are the same
    #     - Other:
    #         + NOT:       Logical Not
    #
    #     Its possible to include multiple options, this statements count as
    #     a logical "or" statements.
    #
    #     Parameters
    #     ----------
    #     kwargs : Keyword Arguments or unpacked dictionary
    #         item_name_option : value
    #
    #     Returns
    #     -------
    #     bool
    #
    #     Notes
    #     -----
    #     If a header item has a "-" character on its name, use two underscores
    #     to represent it.
    #
    #     Examples
    #     --------
    #     NAME_BEGIN_ICASE_NOT = "WASP"   (False if string starts with wasp)
    #     NAXIS1_GT_EQUAL = 20                  (True if NAXIS1 >= 20)
    #
    #     """
    #     ret = []
    #
    #     for filter_keyword, request in kwargs.items():
    #         functions = []
    #         # By default, it is not comparing match, but rather equality
    #         match = False
    #         exists = True
    #
    #         filter_keyword = filter_keyword.replace('__', '-')
    #         if '_' in filter_keyword:
    #             tmp = filter_keyword.split('_')
    #             filter_keyword = tmp[0]
    #             functions.extend(tmp[1:])
    #         header_val = self.values(filter_keyword)
    #
    #         # Treat specially the not-found and list as filter_keyword
    #         if header_val is None:
    #             ret.append(False)
    #             continue
    #
    #         if isinstance(request, str):
    #             request = [request]
    #         elif isinstance(request, dict):
    #             raise TypeError("Filter string cannot be dict anymore. ")
    #         elif isinstance(request, (tuple, list)):
    #             pass
    #         else:
    #             request = [request]
    #
    #         less_than = greater_than = False
    #         for f in functions:
    #             f = f.lower()
    #             if f[:5] == 'begin':
    #                 header_val = header_val[:len(request[0])]
    #             elif f[:3] == 'end':
    #                 header_val = header_val[-len(request[0]):]
    #             elif f[:5] == 'icase':
    #                 header_val = header_val.lower()
    #                 request = [a.lower() for a in request]
    #             elif f[:3] == 'not':
    #                 exists = False
    #             elif f[:5] == 'match':
    #                 match = True
    #             elif f[:5] == 'equal':
    #                 match = False
    #             elif f[:3] == 'lt':
    #                 less_than = True
    #             elif f[:3] == 'gt':
    #                 greater_than = True
    #             else:
    #                 io_logger.warning(f"Function '{f}' not recognized in "
    #                                   f"filtering, ignoring")
    #
    #         if greater_than:
    #             ret.append((True in [r < header_val
    #                                  for r in request]) == exists)
    #         elif less_than:
    #             ret.append((True in [r > header_val
    #                                  for r in request]) == exists)
    #         elif match:
    #             ret.append((True in [r in header_val
    #                                  for r in request]) == exists)
    #         else:
    #             ret.append((True in [r == header_val
    #                                  for r in request]) == exists)
    #
    #     # Returns whether the filter existed (or not if _not function)
    #     return True in ret

    # def set_values(self, **kwargs):
    #     """
    #     Set header values from kwargs. They can be specified as tuple to add
    #     comments as in pyfits.
    #
    #     Parameters
    #     ----------
    #     **kwargs : Keyword argument or unpacked dict
    #
    #     Examples
    #     --------
    #     setheader(item1 = data1, item2 = data2)
    #     ``setheader(**{"item1" : data1, "item2" : data2})``
    #
    #     Returns
    #     -------
    #     True if header was edited properly
    #
    #     Notes
    #     -----
    #     Setting a header value to None will remove said item from the header
    #     """
    #
    #     tp = self.type
    #     hdu = kwargs.pop('hduh', self._hduh)
    #     for k, v in kwargs.items():
    #         if v is None:
    #             del self.header_cache[hdu][k]
    #         else:
    #             self.header_cache[hdu][k] = v

    # def values(self, *args, single_in_list=False, cast=None, hdu=None):
    #     """
    #     Get header value for each of the fields specified in args.
    #
    #     Parameters
    #     ----------
    #     single_in_list: bool
    #       returns a 1 element list if given one element.
    #     args : list, tuple
    #       One string per argument, or a list as a single argument:
    #       .values(field1,field2,...) or
    #       .values([field1,field2,...]).
    #       In addition to return headers from the file, it can return "basename"
    #       and "dirname" from the filename itself if available.
    #     cast : function, optional
    #       Function to use for casting each element of the result.
    #     hdu :  int, optional
    #       HDU to read.
    #
    #     Returns
    #     -------
    #     list
    #         If multiple values are found
    #     string
    #         If only one result was found
    #     """
    #
    #     tp = self.type
    #
    #     if self._corrupt:
    #         if single_in_list:
    #             return [None]
    #         else:
    #             return None
    #
    #     if hdu is None:
    #         hdu = self._hduh
    #     if cast is None:
    #         def cast(x): return x
    #
    #     if len(args) == 1:
    #         # If first argument is tuple use those values as searches
    #         if isinstance(args[0], (list, tuple)):
    #             args = args[0]
    #         # If it is a string, separate by commas
    #         elif isinstance(args[0], str):
    #             args = args[0].split(',')
    #
    #     hdr = self.header_cache[hdu]
    #     ret = []
    #     for k in args:
    #         k = k.strip()
    #         k_lc = k.lower()
    #         if k_lc in hdr:
    #             ret.append(cast(hdr[k_lc]))
    #         elif k in hdr:
    #             ret.append(cast(hdr[k]))
    #         elif k_lc == "filename":
    #             ret.append(self.filename)
    #         elif k_lc == "basename":
    #             ret.append(path.basename(self.filename))
    #         elif k_lc == "dirname":
    #             ret.append(path.dirname(self.filename))
    #         else:
    #             ret.append(None)
    #
    #     if len(ret) == 1 and not single_in_list:
    #         return ret[0]
    #     else:
    #         return ret

    @astrofile_cache
    def reader(self,
               skip_calib=False, hdud=None, hdu=None, verbose=True,
               **kwargs):
        """
        Read astro data and return it calibrated if provided

        TODO: Respond to different exposure times or filters

        Parameters
        ----------
        hdu, hdud: int, optional
            hdu slot to be read
        skip_calib : bool, optional
            If True returns data without calibration

        """
        tp = self.type

        if hdud is not None:
            hdu = hdud
        if hdu is None:
            hdu = self._hdud

        if self._corrupt:
            raise IOError("File data corrupt")

        if not tp:
            return False

        data = self._reads[tp](self.filename, hdu=hdu, **kwargs)

        if data is None:  # File with no data / corrupt
            self._corrupt = True
            raise IOError(f"Empty or corrupt file {self.filename} or HDU {hdu}")

        if 'hdu' in kwargs and kwargs['hdu'] < 0:
            raise ValueError("Invalid HDU specification ({hdu})"
                             "in file {file}?\n available: {hdus}"
                             .format(hdu=hdu,
                                     file=self.filename,
                                     hdus=self.reader(hdu=-1, skip_calib=True)))

        if self.has_calib() and not skip_calib:
            data = self._calib.reduce(data,
                                      data_trim=self.get_trims(),  # header=self.read_headers()[self._hduh],
                                      verbose=verbose)

        return data

    def has_calib(self):
        """
        Checks if a bias or flat has been associated with this file

        Returns
        -------
        bool
        """
        return self._calib.has_flat or self._calib.has_bias

    def writer(self, data, *args, **kwargs):
        """
        Writes given data to file
        """
        tp = self.type
        # TODO: Save itself if data exists. Now it only saves explicit
        #       array given by user
        return tp and self._writes[tp](self.filename, data, *args, **kwargs)

    def basename(self):
        """
        Obtains the file basename

        Returns
        -------
        str
        """
        import os.path as path
        if not hasattr(self, 'filename'):  # TODO ojo aca
            return None
        return path.basename(self.filename)

    # def __getitem__(self, key):
    #     """
    #     Read data and return key
    #
    #     Returns
    #     -------
    #     If key is string, it returns either a one element or a list of headers.
    #     Otherwise, it returns an index to its data content
    #     """
    #     if self._corrupt:
    #         raise IOError("File data declared as corrupt")
    #     if key is None:
    #         return None
    #     elif isinstance(key, str):
    #         return self.values(key)
    #     else:
    #         return self.reader()[key]

    # # Object Comparison
    # def __lt__(self, other):
    #     return self.values(self._sort_key) < \
    #         other.values(self._sort_key)
    #
    # def __le__(self, other):
    #     return self.values(self._sort_key) <= \
    #         other.values(self._sort_key)
    #
    # def __gt__(self, other):
    #     return self.values(self._sort_key) > \
    #         other.values(self._sort_key)
    #
    # def __eq__(self, other):
    #     return self.values(self._sort_key) == \
    #         other.values(self._sort_key)
    #
    # def __ne__(self, other):
    #     return self.values(self._sort_key) != \
    #         other.values(self._sort_key)

    # # Object Arithmetic
    # @_numerize_other
    # def __add__(self, other):
    #     return self.reader() + other
    #
    # @_numerize_other
    # def __sub__(self, other):
    #     return self.reader() - other
    #
    # @_numerize_other
    # def __floordiv__(self, other):
    #     return self.reader() // other
    #
    # @_numerize_other
    # def __truediv__(self, other):
    #     return self.reader() / other
    #
    # @_numerize_other
    # def __mul__(self, other):
    #     return self.reader() * other
    #
    # @_numerize_other
    # def __radd__(self, other):
    #     return other + self.reader()
    #
    # @_numerize_other
    # def __rsub__(self, other):
    #     return other - self.reader()
    #
    # @_numerize_other
    # def __rfloordiv__(self, other):
    #     return other // self.reader()
    #
    # @_numerize_other
    # def __rtruediv__(self, other):
    #     return other / self.reader()
    #
    # @_numerize_other
    # def __rmul__(self, other):
    #     return self.reader() * other
    #
    def __len__(self):
        return len(self.reader(skip_calib=True))

    def __bool__(self):
        return self.type is not None

    # @property
    # def shape(self):
    #     return self.reader(skip_calib=True).shape
    #
    # def stats(self, *args,
    #           verbose_heading = True,
    #           extra_headers = None,
    #           ):
    #     """
    #     Calculates statistical data from the file, a request can include header
    #     keywords to be included in the response.
    #
    #     Parameters
    #     ----------
    #     *args : str
    #         Statistic to be extracted, Possible values are: min, max, mean,
    #         mean3sclip, std and median
    #
    #     **kwargs : dict
    #         Options available: verbose_heading and extra_headers
    #
    #     Returns
    #     -------
    #     list
    #         List containing the requested data
    #
    #     """
    #     extra_headers = extra_headers or []
    #
    #     if not args:
    #         args = ['min', 'max', 'mean', 'mean3sclip', 'median', 'std']
    #     if verbose_heading:
    #         print("Computing the following stats: {} {}"
    #               .format(", ".join(args),
    #                       len(extra_headers)
    #                       and "\nand showing the following headers: {}"
    #                       .format(", ".join(extra_headers))
    #                       or ""))
    #     ret = []
    #     data = self.data
    #     for stat in args:
    #         if stat == 'min':
    #             ret.append(data.min())
    #         elif stat == 'max':
    #             ret.append(data.max())
    #         elif stat == 'mean':
    #             ret.append(data.mean())
    #         elif stat == 'mean3sclip':
    #             clip = data.copy().astype("float64")
    #             clip -= data.mean()
    #             std = data.std()
    #             ret.append(clip[np.absolute(clip) < 3 * std].mean())
    #         elif stat == 'std':
    #             ret.append(data.std())
    #         elif stat == 'median':
    #             ret.append(np.median(data))
    #         else:
    #             raise ValueError("Unknown stat '{}' was requested for "
    #                              "file {}".format(stat, self.filename))
    #     for h in extra_headers:
    #         ret.append(self[h])
    #
    #     if len(ret) == 1:
    #         return ret[0]
    #     else:
    #         return ret
    #
    # def jd_from_ut(self, target='jd', source='date-obs'):
    #     """
    #     Add jd in header's cache to keyword 'target' using ut on keyword
    #     'source'
    #
    #     Parameters
    #     ----------
    #     target : str
    #         Target keyword for JD storage
    #     source : str or list
    #         Input value in UT format, if a tuple is given it will join the
    #         values obtained from both keywords following the UT format
    #         (day+T+time)
    #
    #     """
    #     newhd = {}
    #     target = target.lower()
    #     if isinstance(source, list) is True:
    #         newhd[target] = apt.Time(self[source[0]] + "T" + self[source[1]]).jd
    #     else:
    #         try:
    #             newhd[target] = apt.Time(self[source]).jd
    #         except ValueError:
    #             raise ValueError(f"File {self.filename} has invalid time specificiation")
    #     self.set_values(**newhd)

    def merger(self, start=1, end=None):
        """
        Merges multiple ImageHDU objects into a single composite image
        , saves the new ImageHDU into a new hdu slot.

        Parameters
        ----------
        start: int, optional
            HDU slot from which the algorithm starts concatenating images
        end: int, optional
            Last HDU to be merged, if None will continue until the end of the
            HDUList
        """
        try:
            f = pf.open(self.filename, mode='update', memmap=False)
            read_only = False
        except IOError:
            f = pf.open(self.filename)
            read_only = True

        with f as fit:
            # This file already has a composite image
            if fit[-1].header.get('COMP') is not None:
                fit.close()
                return self

            if end is None:
                end = len(fit)

            dataset = []
            for i in range(start, end):
                if isinstance(fit[i], (pf.ImageHDU, pf.PrimaryHDU)):
                    mat = fit[i].data
                    if mat is None:
                        warnings.warn("Warning: {} contains empty data at "
                                      "hdu = {}, skipping hdu"
                                      .format(self.basename, i))
                    else:
                        dataset.append(mat)
                else:
                    raise (TypeError,
                           "Unable to merge hdu of type: " + type(fit[i]))

            comp = np.concatenate(dataset, axis=1)

            composite = pf.ImageHDU(comp, header=fit[0].header)
            composite.header.set('COMP', True)
            if read_only:
                self.filename = comp
                self.type = self.checktype()
                self.header_cache = [composite.header]
                self._hduh = 0
                self._hdud = 0
                warnings.warn("Merged image stored in hdu 0 and previous info discarded as system was read-only",
                              UserWarning,
                              )

            else:
                fit.append(composite)
                fit.flush()

        return self

    #################
    #
    # WISHLIST FROM HERE. BASIC IMPLEMENTATION OF METHODS
    # TODO : Improve
    #
    #############################
    def spplot(self,
               axes=None, title=None, xtitle=None, ytitle=None,
               *args, **kwargs):

        """
        Plots spectral data contained on this file
        """
        fig, ax = pa.figaxes(axes, title)
        pa.set_plot_props(ax, xlabel=xtitle, ylabel=ytitle)

        data = self.reader()
        dim = len(data.shape)

        if dim == 2:
            if data.shape[0] < data.shape[1]:
                wav = data[0, :]
                flx = data[1, :]
            else:
                wav = data[:, 0]
                flx = data[:, 1]
        elif dim == 1:
            raise NotImplementedError(
                "Needs to add reading of wavelength from headers")
        else:
            raise NotImplementedError("Spectra not understood")

        ax.plot(wav, flx, *args, **kwargs)
