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

__all__ = ['AstroFile', 'AstroCalib']

import logging
from functools import wraps as _wraps
import warnings
from pathlib import PurePath, Path

import procastro as pa
from .internal_functions import trim_to_python, common_trim_fcn, extract_common
import astropy.time as apt
import numpy as np
import os.path as path
import astropy.io.fits as pf

logging.basicConfig(level=logging.INFO)
io_logger = logging.getLogger('procastro.io')
io_logger.propagate = False


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

# def _fits_header(filename, hdu=0):
#     """
#     Read fits header.
#
#     Parameters
#     ----------
#     hdu: int, optional
#         HDU slot to be read
#
#     Returns
#     -------
#     Header Object of the specified hdu
#     """
#
#     hdu_list = pf.open(filename)
#     ret = hdu_list[hdu].header
#     hdu_list.close()
#     return ret


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
        io_logger.warning(
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
    filename = Path(filename)
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


def _fits_setheader(_filename, **kwargs):
    """
    Sets value of a header item

    Parameters
    ----------
    _filename : str
    hdu : int, optional
    write : bool, optional
    **kwargs :
        Set of key value pairs containing the data to be read

    Returns
    -------
    bool
        True if successful
    """
    hdu = ('hdu' in kwargs and [kwargs['hdu']] or [0])[0]

    if 'write' in kwargs and kwargs['write']:
        save = True
        try:
            # TODO: if file does not exist, create it
            hdulist = pf.open(_filename, 'update')
            fits = hdulist[hdu]
        except IOError:
            io_logger.warning(
                "Read-only filesystem or file not found. "
                "Header update of '{0:s}' will not remain "
                "on disk.".format(', '.join(kwargs.keys())))
            return
    else:
        hdulist = pf.open(_filename)
        fits = hdulist[hdu]
        save = False
    if 'write' in kwargs:
        del kwargs['write']

    h = fits.header
    for k, v in kwargs.items():
        # If new value is None, then delete it
        if v is None:
            del h[k]
        else:
            h[k] = v
    if save:
        fits.flush()
    hdulist.close()
    return True

#############################
#
# Startnparray type
#
####################################


def _array_reader(array, **kwargs):
    return array


def _array_write(filename, **kwargs):
    raise NotImplementedError("Write array into file is not implemented yet")


def _array_setheader(filename, **kwargs):
    warnings.warn("Saving an np.array type is not implemented yet: only saving in memory for now ",
                   UserWarning)
    return None


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


def _checksortkey(f):
    """
    Verifies that the given sortkey is valid
    """
    @_wraps(f)
    def ret(self, *args, **kwargs):
        if not hasattr(self, 'sortkey'):
            raise ValueError(
                "Sortkey must be defined before trying to sort AstroFile")
        if not isinstance(self.sortkey, str) or "," in self.sortkey:
            raise ValueError(
                "Invalid value for sortkey ({}), it must be a single header"
                "specification (without commas)".format(self.sortkey))
        return f(self, *args, **kwargs)

    return ret


def _checkfilename(f):
    """
    Verifies that the given filename is valid
    """
    @_wraps(f)
    def isfiledef(inst, *args, **kwargs):
        if hasattr(inst, 'filename'):
            if inst.type is None:
                # raise ValueError("File %s not a supported astro-type." % (f))
                raise ValueError(
                    "Please specify filename with setFilename first.")
            return f(inst, *args, **kwargs)
        else:
            raise ValueError(
                "Filename not defined. Must give valid filename to AstroFile")

    return isfiledef


class AstroFile(object):
    """
    Representation of an astronomical data file, contains information related
    to the file's data. Currently supports usage of arrays and .fit files

    Parameters
    ----------
    filename : str or procastro.AstroFile
        If AstroFile, then it does not duplicate the information,
        but uses that object directly.
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
    shape : tuple
        Shape of the data contained on the file
    header_cache : dict
        Data contained on the most recently read header
    sortkey : str
        Header Item used when sorting this object
    type : str
        File extension

    """

    # Interface between public and private methods for each supported type
    _reads = {'fits': _fits_reader, 'nparray': _array_reader}
    # _readhs = {'fits': _fits_header, 'nparray': array_header}
    _ids = {'fits': _fits_verify, 'nparray': _array_verify}
    _writes = {'fits': _fits_writer, 'nparray': _array_write}
    _geth = {'fits': _fits_getheader, 'nparray': _array_getheader}
    _seth = {'fits': _fits_setheader, 'nparray': _array_setheader}

    def __repr__(self):
        if isinstance(self.filename, str):
            filename = self.filename
        elif isinstance(self.filename, np.ndarray):
            filename = f"Array {'x'.join([str(i) for i in self.filename.shape])}"
        else:
            filename = str(self.filename)
        return '<AstroFile{}: {}>'.format(self.has_calib()
                                          and "({}{})"
                                              .format(self.calib.has_bias
                                                      and "B" or "",
                                                      self.calib.has_flat
                                                      and "F" or "",)
                                          or "",
                                          filename,)

    def __new__(cls, *args, **kwargs):
        """
        If passed an AstroFile, then do not create a new instance, just pass
        that one. If passed None, return None
        """
        if args and (isinstance(args[0], AstroFile) or args[0] is None):
            return args[0]

        return super(AstroFile, cls).__new__(cls)

    def __init__(self, filename=None,
                 mbias=None, mflat=None,
                 mbias_header=None, mflat_header=None,
                 hdu=0, hduh=None, hdud=None,
                 auto_trim=None, header=None,
                 *args, **kwargs):

        if isinstance(filename, AstroFile):
            return

        self.filename = filename
        self.type = self.checktype(*args, **kwargs)
        try:
            self.header_cache = self._geth[self.type](self.filename)
        except OSError:
            self.type = None
            io_logger.warning("Omitting corrupt file")
            return
        if isinstance(header, dict):
            for k, v in header.items():
                self.header_cache[0][k] = v

        if hduh is None:
            hduh = hdu
        if hdud is None:
            hdud = hdu
        self._hduh = hduh
        self._hdud = hdud

        self.auto_trim = auto_trim

        self.calib = AstroCalib(pa.AstroFile(mbias, header=mbias_header),
                                pa.AstroFile(mflat, header=mflat_header),
                                auto_trim=auto_trim)

    def get_trims(self):
        """"Get trim limits, returning in python-style YX"""
        if self.auto_trim is None:
            return None

        return trim_to_python(self[self.auto_trim])

    def add_bias(self, mbias):
        """
        Includes a Master Bias to this instance

        Parameters
        ----------
        mbias: dict indexed by exposure time, array or AstroFile
        """
        self.calib.add_bias(mbias)

    def add_flat(self, master_flat):
        """
        Includes a Master Bias to this instance

        Parameters
        ----------
        master_flat: dict indexed by filter name time, array or AstroFile
        """
        self.calib.add_flat(master_flat)

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
        str
            "array" : If AstroFile represents a scipy array
            "fits" : If AstroFile points to an existing .fits file
            None : Otherwise
        """
        if not hasattr(self, 'filename'):
            return None
        # if not isinstance(self.filename, str):
        #     return None
        # elif self.filename is None:
        #     return "array"
        # if exists and not path.isfile(self.filename):
        #     return None
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

    def add_sortkey(self, key):
        """
        Sets a header item used for sorting comparisons

        Parameters
        ----------
        key: str
        """
        self.sortkey = key

    def imshowz(self, *args, **kwargs):
        """
        Plots frame after being processed using a zscale algorithm

        See Also
        --------
        misc_graph.imshowz :
            For details on what keyword arguments are available
        """
        hdu = kwargs.pop('hdu', self._hdud)

        return pa.imshowz(self.reader(hdu=hdu), *args, **kwargs)

    def __nonzero__(self):
        return hasattr(self, 'filename') and (self.type is not None)

    @_checkfilename
    def filter(self, **kwargs):
        """
        Compares if the expected value of the given header item matches
        the value stored in this file header.

        Filtering can be customized by including one of the following options
        after the item name, separated by an underscore.

        - Strings:
            + BEGIN:     True if value starts with the given string
            + END:       True if value ends with the given string
            + ICASE:     Case unsensitive match
            + MATCH:     Case sensitive match

        - Numeric values:
            + LT:        True if current value is lower than the given number
            + GT:        True if current value is greater than the given number
            + EQUAL:     True if both values are the same
        - Other:
            + NOT:       Logical Not

        Its possible to include multiple options, this statements count as
        a logical "or" statements.

        Parameters
        ----------
        kwargs : Keyword Arguments or unpacked dictionary
            item_name_option : value

        Returns
        -------
        bool

        Notes
        -----
        If a header item has a "-" character on its name, use two underscores
        to represent it.

        Examples
        --------
        NAME_BEGIN_ICASE_NOT = "WASP"   (False if string starts with wasp)
        NAXIS1_GT_EQUAL = 20                  (True if NAXIS1 >= 20)

        """
        ret = []

        for filter_keyword, request in kwargs.items():
            functions = []
            # By default, it is not comparing match, but rather equality
            match = False
            exists = True

            def cast(x):
                return x
            filter_keyword = filter_keyword.replace('__', '-')
            if '_' in filter_keyword:
                tmp = filter_keyword.split('_')
                filter_keyword = tmp[0]
                functions.extend(tmp[1:])
            header_val = self.getheaderval(filter_keyword)

            # Treat specially the not-found and list as filter_keyword
            if header_val is None:
                ret.append(False)
                continue

            if isinstance(request, str):
                request = [request]
            elif isinstance(request, (tuple, list)):
                raise TypeError("Filter string cannot be tuple/list anymore. "
                                "It has to be a dictionary with the casting "
                                "function as key (e.g. 'str')")
            elif isinstance(request, dict):
                keys = list(request.keys())
                if len(keys) != 1:
                    raise NotImplementedError(
                        "Only a single key (casting) per filtering function "
                        "has been implemented for multiple alternatives")
                try:
                    request = list(request[keys[0]])
                    if 'begin' in functions or 'end' in functions:
                        raise ValueError("Cannot use '_begin' or '_end'"
                                         "if comparing to a list")
                except TypeError:
                    request = [request[keys[0]]]
                cast = eval(keys[0])
                if not callable(cast):
                    raise ValueError(
                        "Dictionary key (casting) has to be a callable "
                        "function accepting only one argument")
            else:
                cast = type(request)
                request = [request]

            less_than = greater_than = False
            for f in functions:
                f = f.lower()
                if f[:5] == 'begin':
                    header_val = header_val[:len(request[0])]
                elif f[:3] == 'end':
                    header_val = header_val[-len(request[0]):]
                elif f[:5] == 'icase':
                    header_val = header_val.lower()
                    request = [a.lower() for a in request]
                elif f[:3] == 'not':
                    exists = False
                elif f[:5] == 'match':
                    match = True
                elif f[:5] == 'equal':
                    match = False
                elif f[:3] == 'lt':
                    less_than = True
                elif f[:3] == 'gt':
                    greater_than = True
                else:
                    io_logger.warning(f"Function '{f}' not recognized in "
                                     f"filtering, ignoring")

            # print("r:%s h:%s m:%i f:%s" %(request, header_val,
            #                               match, functions))
            if greater_than:
                ret.append((True in [cast(r) < cast(header_val)
                                     for r in request]) == exists)
            elif less_than:
                ret.append((True in [cast(r) > cast(header_val)
                                     for r in request]) == exists)
            elif match:
                ret.append((True in [cast(r) in cast(header_val)
                                     for r in request]) == exists)
            else:
                ret.append((True in [cast(r) == cast(header_val)
                                     for r in request]) == exists)

        # Returns whether the filter existed (or not if _not function)
        return True in ret

    @_checkfilename
    def setheader(self, **kwargs):
        """
        Set header values from kwargs. They can be specified as tuple to add
        comments as in pyfits.

        Parameters
        ----------
        **kwargs : Keyword argument or unpacked dict

        Examples
        --------
        setheader(item1 = data1, item2 = data2)
        ``setheader(**{"item1" : data1, "item2" : data2})``

        Returns
        -------
        True if header was edited properly

        Notes
        -----
        Setting a header value to None will remove said item from the header
        """

        tp = self.type
        hdu = kwargs.pop('hduh', self._hduh)
        for k, v in kwargs.items():
            if v is None:
                del self.header_cache[hdu][k]
            else:
                self.header_cache[hdu][k] = v
        return self._seth[tp](self.filename, hdu=hdu, **kwargs)

    @_checkfilename
    def getheaderval(self, *args, single_in_list=False, **kwargs):
        """
        Get header value for each of the fields specified in args.

        Parameters
        ----------
        single_in_list: bool
          returns a 1 element list if given one element.
        args : list, tuple
          One string per argument, or a list as a single argument:
          getheaderval(field1,field2,...) or
          getheaderval([field1,field2,...]).
          In addition to return headers from the file, it can return "basename"
          and "dirname" from the filename itself if available.
        cast : function, optional
          Function to use for casting each element of the result.
        hdu :  int, optional
          HDU to read.

        Returns
        -------
        list
            If multiple values are found
        string
            If only one result was found
        """

        tp = self.type

        cast = kwargs.pop('cast', lambda x: x)
        hdu = kwargs.pop('hdu', self._hduh)

        if len(args) == 1:
            # If first argument is tuple use those values as searches
            if isinstance(args[0], (list, tuple)):
                args = args[0]
            # If it is a string, separate by commas
            elif isinstance(args[0], str):
                args = args[0].split(',')

        if self.header_cache is None:
            self.header_cache = self._geth[tp](self.filename)

        hdr = self.header_cache[hdu]
        ret = []
        for k in args:
            k = k.strip()
            k_lc = k.lower()
            if k_lc in hdr:
                ret.append(cast(hdr[k_lc]))
            elif k in hdr:
                ret.append(cast(hdr[k]))
            elif k_lc == "basename":
                ret.append(path.basename(self.filename))
            elif k_lc == "dirname":
                ret.append(path.dirname(self.filename))
            else:
                ret.append(None)

        if len(ret) == 1 and not single_in_list:
            return ret[0]
        else:
            return ret

    @_checkfilename
    def reader(self, *args,
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

        if not tp:
            return False

        data = self._reads[tp](self.filename, *args, hdu=hdu, **kwargs)

        if data is None:  # File with no data / corrupt
            raise IOError(f"Empty or corrupt file {self.filename} or HDU {hdu}")

        if 'hdu' in kwargs and kwargs['hdu'] < 0:
            raise ValueError("Invalid HDU specification ({hdu})"
                             "in file {file}?\n available: {hdus}"
                             .format(hdu=hdu,
                                     file=self.filename,
                                     hdus=self.reader(hdu=-1, skip_calib=True)))

        if self.has_calib() and not skip_calib:
            data = self.calib.reduce(data,
                                     exptime=self[self.calib.exptime_keyword],
                                     ffilter=self[self.calib.filter_keyword],
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
        return self.calib.has_flat or self.calib.has_bias

    @_checkfilename
    def read_headers(self, *args, **kwargs):
        """
        Reads header from the file

        Parameters
        ----------
        hdu: int, optional
            Specific hdu slot to be read
        """
        tp = self.type
        hdu = kwargs.pop('hdu', self._hduh)

        if not tp:
            return False
        return self.header_cache

    @_checkfilename
    def writer(self, data, *args, **kwargs):
        """
        Writes given data to file
        """
        tp = self.type
        # TODO: Save itself if data exists. Now it only saves explicit
        #       array given by user
        return tp and self._writes[tp](self.filename, data, *args, **kwargs)

    @_checkfilename
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

    @_checkfilename
    def __getitem__(self, key):
        """
        Read data and return key

        Returns
        -------
        If key is string, it returns either a one element or a list of headers.
        Otherwise, it returns an index to its data content
        """

        if key is None:
            return None
        elif isinstance(key, str):
            return self.getheaderval(key)
        else:
            return self.reader()[key]

    # Object Comparison
    @_checksortkey
    def __lt__(self, other):
        return self.getheaderval(self.sortkey) < \
            other.getheaderval(self.sortkey)

    @_checksortkey
    def __le__(self, other):
        return self.getheaderval(self.sortkey) <= \
            other.getheaderval(self.sortkey)

    @_checksortkey
    def __gt__(self, other):
        return self.getheaderval(self.sortkey) > \
            other.getheaderval(self.sortkey)

    @_checksortkey
    def __eq__(self, other):
        return self.getheaderval(self.sortkey) == \
            other.getheaderval(self.sortkey)

    @_checksortkey
    def __ne__(self, other):
        return self.getheaderval(self.sortkey) != \
            other.getheaderval(self.sortkey)

    # Object Arithmetic
    @_numerize_other
    def __add__(self, other):
        return self.reader() + other

    @_numerize_other
    def __sub__(self, other):
        return self.reader() - other

    @_numerize_other
    def __floordiv__(self, other):
        return self.reader() // other

    @_numerize_other
    def __truediv__(self, other):
        return self.reader() / other

    @_numerize_other
    def __mul__(self, other):
        return self.reader() * other

    @_numerize_other
    def __radd__(self, other):
        return other + self.reader()

    @_numerize_other
    def __rsub__(self, other):
        return other - self.reader()

    @_numerize_other
    def __rfloordiv__(self, other):
        return other // self.reader()

    @_numerize_other
    def __rtruediv__(self, other):
        return other / self.reader()

    @_numerize_other
    def __rmul__(self, other):
        return self.reader() * other

    def __len__(self):
        return len(self.reader(skip_calib=True))

    def __bool__(self):
        return self.type is not None

    @property
    def shape(self):
        return self.reader(skip_calib=True).shape

    def stats(self, *args, **kwargs):
        """
        Calculates statistical data from the file, a request can include header
        keywords to be included in the response.

        Parameters
        ----------
        *args : str
            Statistic to be extracted, Possible values are: min, max, mean,
            mean3sclip, std and median

        **kwargs : str
            Options available: verbose_heading and extra_headers

        Returns
        -------
        list
            List containing the requested data

        """
        verbose_heading = kwargs.pop('verbose_heading', True)
        extra_headers = kwargs.pop('extra_headers', [])
        if kwargs:
            raise SyntaxError(
                "Only the following keyword argument for stats are"
                " permitted 'verbose_heading', 'extra_headers'")

        if not args:
            args = ['min', 'max', 'mean', 'mean3sclip', 'median', 'std']
        if verbose_heading:
            print("Computing the following stats: {} {}"
                  .format(", ".join(args),
                          len(extra_headers)
                          and "\nand showing the following headers: {}"
                              .format(", ".join(extra_headers))
                          or ""))
        ret = []
        data = self.reader()
        for stat in args:
            if stat == 'min':
                ret.append(data.min())
            elif stat == 'max':
                ret.append(data.max())
            elif stat == 'mean':
                ret.append(data.mean())
            elif stat == 'mean3sclip':
                clip = data.copy().astype("float64")
                clip -= data.mean()
                std = data.std()
                ret.append(clip[np.absolute(clip) < 3 * std].mean())
            elif stat == 'std':
                ret.append(data.std())
            elif stat == 'median':
                ret.append(np.median(data))
            else:
                raise ValueError("Unknown stat '{}' was requested for "
                                 "file {}".format(stat, self.filename))
        for h in extra_headers:
            ret.append(self[h])

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def jd_from_ut(self, target='jd', source='date-obs'):
        """
        Add jd in header's cache to keyword 'target' using ut on keyword
        'source'

        Parameters
        ----------
        target : str
            Target keyword for JD storage
        source : str or list
            Input value in UT format, if a tuple is given it will join the
            values obtained from both keywords following the UT format
            (day+T+time)

        """
        newhd = {}
        target = target.lower()
        if isinstance(source, list) is True:
            newhd[target] = apt.Time(self[source[0]]+"T"+self[source[1]]).jd
        else:
            try:
                newhd[target] = apt.Time(self[source]).jd
            except ValueError:
                raise ValueError(f"File {self.filename} has invalid time specificiation")
        self.setheader(**newhd)

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
                    raise(TypeError,
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
        fig, ax = pa.prep_canvas(axes, title, xtitle, ytitle)

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


class AstroCalib(object):
    """
    Object to hold calibration frames.

    Since several AstroFiles might use the same calibration frames, one
    AstroCalib object might be shared by more than one AstroFile. For instance,
    all the files initialized through AstroDir share a single calibration
    object.

    Attributes
    ----------
    has_bias : bool
    has_flat : bool

    Parameters
    ----------
    mbias : dict indexed by exposure time, array or AstroFile
        Master Bias
    mflat : dict indexed by filter name, array or AstroFile
        Master Flat
    exptime_keyword : str, optional
        Header item name containing the exposure time of te frames
    mbias_header : astropy.io.fits.Header, optional
        Header of the master bias
    mflat_header : astropy.io.fits.Header, optional
        Header of the master bias
    filter_keyword : str, optional
    auto_trim : str
        Header field name which is used to cut a portion of the data. This
        field should contain the dimensions of the section to be cut.
        (i.e. TRIMSEC = '[1:1992,1:1708]')
    """

    def __init__(self, mbias=None, mflat=None,
                 exptime_keyword='exptime',
                 mbias_header=None, mflat_header=None,
                 filter_keyword='filter',
                 auto_trim=None):

        # Its always created false, if the add_*() has something, then its
        # turned true.
        self.has_bias = self.has_flat = False

        self.mbias = {}
        self.mflat = {}
        self.mbias_header = None
        self.mflat_header = None

        self.exptime_keyword = exptime_keyword
        self.filter_keyword = filter_keyword
        self.auto_trim_keyword = None if auto_trim is None else auto_trim.lower()

        self.add_bias(mbias, mbias_header=mbias_header)
        self.add_flat(mflat, mflat_header=mflat_header)

    def add_bias(self, mbias, mbias_header=None):
        """
        Add Master Bias to Calib object.

        Parameters
        ----------
        mbias : dict indexed by exposure time, array_like or AstroFile
            Master bias to be included
        mbias_header: astropy.io.fits.Header
            Header to be included

        Raises
        ------
        ValueError
            If the bias type is invalid
        """
        self.mbias_header = mbias_header
        if mbias is None:
            self.mbias[-1] = 0.0
            return
        self.has_bias = True
        if isinstance(mbias, dict):
            for k in mbias.keys():
                self.mbias[k] = mbias[k]
        elif isinstance(mbias,
                        (int, float, np.ndarray)):
            self.mbias[-1] = mbias
        elif isinstance(mbias,
                        pa.AstroFile):
            self.mbias[-1] = mbias.reader()
            self.mbias_header = mbias.read_headers()
        else:
            raise ValueError("Master Bias supplied was not recognized.")

    def add_flat(self, mflat, mflat_header=None):
        """
        Add master flat to Calib object

        Parameters
        ----------
        mflat: dict indexed by filter name, array_like, AstroFile
            Master flat to be included
        mflat_header: astropy.io.fits.Header, optional
            Master flat header

        Raises
        ------
        ValueError
            If the bias type is invalid
        """

        self.mflat_header = mflat_header
        if mflat is None:
            self.mflat[''] = 1.0
            return
        self.has_flat = True
        if isinstance(mflat, dict):
            for k in mflat.keys():
                self.mflat[k] = mflat[k]
        elif isinstance(mflat,
                        pa.AstroFile):
            self.mflat[''] = mflat.reader()
            self.mflat_header = mflat.read_headers()
        elif isinstance(mflat,
                        (int, float, np.ndarray)):
            self.mflat[''] = mflat
        else:
            raise ValueError("Master Flat supplied was not recognized.")

    def reduce(self, data, exptime=None, ffilter=None, data_trim=None, verbose=True):
        """
        Process given "data" using the bias and flat contained in this instance

        Parameters
        ----------
        data : array_like
            Data to be reduced
        exptime : int, optional
            Exposure time for bias
        ffilter : str, optional
            Filter used by the flat
        header : astropy.io.fits.Header, optional

        Returns
        -------
        array_like
            Reduced data

        """

        if exptime is None or exptime not in self.mbias:
            exptime = -1
        if ffilter is None or ffilter not in self.mflat:
            ffilter = ''

        if exptime not in self.mbias:
            raise ValueError(
                "Requested exposure time ({}) is not available "
                "for mbias, only: {}".format(exptime,
                                             ", "
                                             .join(map(str,
                                                       self.mbias.keys()))))
        if ffilter not in self.mflat:
            raise ValueError(
                "Requested filter ({}) is not available for mflat, only: {}"
                .format(ffilter, ", ".join(self.mflat.keys())))

        flat = self.mflat[ffilter]
        bias = self.mbias[exptime]

        mintrim = np.array([0, data.shape[0], 0, data.shape[1]])

        in_data = [data]
        if data_trim is not None:
            trim = [data_trim]
            for label in ['bias', 'flat']:
                tdata = vars()[label]
                in_data.append(tdata)

                theader = getattr(self, f'm{label}_header')
                if theader is not None:
                    theader = theader[0]

                if theader is None or self.auto_trim_keyword not in theader.keys():
                    if not isinstance(tdata, (int, float)):
                        io_logger.warning(f"Trim info {self.auto_trim_keyword} found on"
                                          f" science frames but not in {label} frames... ignoring")
                    trim.append(trim[0])
                else:
                    trim.append(trim_to_python(theader[self.auto_trim_keyword]))

            common_trim = common_trim_fcn(trim)

            out_data = []
            for label, tdata, trim in zip(['data', 'bias', 'flat'], in_data, trim):
                if isinstance(tdata, (int,float)): # if tdata is bias = 0 or flat = 1.0, dont trim
                    out_data.append(tdata)
                else:
                    out, trimmed = extract_common(tdata, trim, common_trim)
                    out_data.append(out)

                if trimmed and verbose:
                    logging.info(f"Adjusting {label} shape to minimmum common trim [{self.auto_trim_keyword}: "
                                 f"({str(trim)}) -> ({str(common_trim)})]")
            data, bias, flat = out_data

        debias = data - bias
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            deflat = debias / flat

        return deflat
