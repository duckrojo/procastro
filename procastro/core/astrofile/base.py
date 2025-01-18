from pathlib import Path
from random import random
from typing import Any

import numpy as np
from astropy.table import Table

import procastro as pa
from procastro.core.astrofile.static_identify import static_identify
from procastro.core.astrofile.static_read import static_read
from procastro.core.astrofile.static_write import static_write
from procastro.core.cache import astrofile_cache
from procastro.core.calib.base import CalibBase
from procastro.core.logging import io_logger

# TYPING CONSTRUCTS
PADataReturn = np.ndarray | Table  # returns an array if there is no spectral information, otherwise table

__all__ = ['AstroFile']


def _identity(x):
    return x


def _check_first_astrofile(fcn):
    """
Decorator that raises error if first argument is not AstroFile

    Parameters
    ----------
    fcn

    Returns
    -------

    """
    def wrapper(self, arg1, *args, **kwargs):
        if not isinstance(arg1, AstroFile):
            raise TypeError(f"First argument of function {fcn.__name__} needs to be an AstroFile instances.")
        return fcn(self, arg1, *args, **kwargs)

    return wrapper


def _numerize_other(fcn):
    def wrapper(self, other):
        if isinstance(other, AstroFile):
            other = other.data
        return fcn(self, other)

    return wrapper


class AstroFile:
    def __init__(self, filename, format_spec=None, *args, **kwargs):
        self._random = random()

        self._sort_key = None
        self._corrupt = False
        self._calib = None

        self._data_file = filename
        self._format = static_identify(filename, options=format_spec)

        self._meta = None
        first_read = self.data

    ########################################################
    #
    # .meta and .data might be overriden for complex types that want to secure the existence of
    # particular headers or some extra mathematical operations in data (mosaics at least)
    #
    ##################################

    @property
    def meta(self):
        return self._meta

    @property
    @astrofile_cache
    def data(self) -> PADataReturn:
        """
    Returns ndarray with the data if single spectral channel (filter)... otherwise, it returns a Table with each
    wavelength channel
        """
        data = self.read()

        if self._calib is not None:
            data = self._calib(self, data=data)

        return data

    ##########################################
    #
    # The following methods are unlikely to need modifications in subclasses
    #
    ##########################################

    def read(self) -> PADataReturn:
        """This function should always re-read from file updating meta and forcing cache update"""

        data, meta = static_read(self._format, self._data_file)

        self._meta = {k.upper(): v for k, v in meta.items()}
        self._random = random()

        return data

    def write(self, filename=None, backup_extension=".bak"):
        if self._format != "FITS":
            if static_identify(filename) != "FITS":
                raise ValueError("Filename must be FITS for .write()... use .write_as() for alternative formats")
            if filename is None:
                raise ValueError("Filename must be provided explicitly when original type was not FITS")
        else:
            filename = self._data_file

        filename = str(filename)

        if Path(filename + backup_extension).exists():
            io_logger(f"Backup file {filename + backup_extension} already exists, not overwriting it")
        else:
            io_logger(f"Backing up in: {filename + backup_extension}")
            static_write("FITS", filename + backup_extension, self.data, self.meta,
                         overwrite=True)

        static_write("FITS", filename, self.data, self.meta,
                     overwrite=True)

    def write_as(self, filename, overwrite=False):
        file_type = static_identify(filename)

        io_logger(f"Saving in {filename} using file type {file_type}")
        static_write(file_type, filename, self.data, self.meta,
                     overwrite=overwrite)

    def forced_data(self):
        return self.data(force=True)

    def add_calib(self, calib):
        if not isinstance(calib, CalibBase):
            raise TypeError("'calib' must be a CalibBase instance")
        self._calib = calib

    def __hash__(self):
        """This is important for cache. If calib changes, then the astrofile hash should change as well. self._random
         provides a method to force reloading of cache."""

        return hash((self._data_file, self._calib, self._random))

    def set_values(self, **kwargs):
        """
        Set header values from kwargs. They can be specified as tuple to add
        comments as in pyfits.

        Parameters
        ----------
        **kwargs : Keyword argument or unpacked dict

        Examples
        --------
        ``setheader(item1 = data1, item2 = data2)``
        ``setheader(**{"item1" : data1, "item2" : data2})``

        Returns
        -------
        True if header was edited properly

        Notes
        -----
        Setting a header value to None will remove said item from the header
        """

        for k, v in kwargs.items():
            if v is None:
                del self._meta[k]
            else:
                self._meta[k] = v

        self._random = random()

    def values(self,
               *args,
               single_in_list=False,
               cast=None,
               ):
        """
        Get header value for each of the fields specified in args.

        Parameters
        ----------
        single_in_list: bool
          returns a 1 element list if given one element.
        args : list, tuple
          One string per argument, a comma separated string, or a list as a single argument:
          .values(field1,field2,...) or
          .values("field1,field2,...") or
          .values([field1,field2,...]).
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

        if self._corrupt:
            if single_in_list:
                return [None]
            else:
                return None

        if cast is None:
            cast = _identity

        if len(args) == 1:
            # If first argument is tuple use those values as searches
            if isinstance(args[0], (list, tuple)):
                args = args[0]
            # If it is a string, separate by commas
            elif isinstance(args[0], str):
                args = args[0].split(',')

        hdr = self.meta()
        ret = []
        for k in args:
            k = k.strip()
            k_lc = k.lower()
            if k_lc in hdr:
                ret.append(cast(hdr[k_lc]))
            elif k in hdr:
                ret.append(cast(hdr[k]))
            elif k_lc == "filename":
                ret.append(self._data_file)
            elif k_lc == "basename":
                ret.append(str(Path(self._data_file).name))
            elif k_lc == "dirname":
                ret.append(str(Path(self._data_file).parent))
            else:
                ret.append(None)

        if len(ret) == 1 and not single_in_list:
            return ret[0]
        else:
            return ret

    def __getitem__(self, key):
        """
        Read single meta value

        Returns
        -------
        If key is string, it returns either a one element or a list of headers if it contains commas.

        If None, then it will get the value defined as sort_order

        """
        if self._corrupt:
            raise IOError("File data declared as corrupt")

        if key is None:
            key = self._sort_key

        return self.values(key, single_in_list=True)

    def filter(self, **kwargs) -> bool:
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

        It is possible to include multiple options, this statements count as
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

            filter_keyword = filter_keyword.replace('__', '-')
            if '_' in filter_keyword:
                tmp = filter_keyword.split('_')
                filter_keyword = tmp[0]
                functions.extend(tmp[1:])
            header_val = self.values(filter_keyword)

            # Treat specially the not-found and list as filter_keyword
            if header_val is None:
                ret.append(False)
                continue

            if isinstance(request, str):
                request = [request]
            elif isinstance(request, dict):
                raise TypeError("Filter string cannot be dict anymore. ")
            elif isinstance(request, (tuple, list)):
                pass
            else:
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

            if greater_than:
                ret.append((True in [r < header_val
                                     for r in request]) == exists)
            elif less_than:
                ret.append((True in [r > header_val
                                     for r in request]) == exists)
            elif match:
                ret.append((True in [r in header_val
                                     for r in request]) == exists)
            else:
                ret.append((True in [r == header_val
                                     for r in request]) == exists)

        # Returns whether the filter existed (or not if _not function)
        return True in ret

    # Object Comparison is done according to the sort_key if defined
    @_check_first_astrofile
    def __lt__(self, other):
        return self[None] < other[None]

    @_check_first_astrofile
    def __le__(self, other):
        return self[None] <= other[None]

    @_check_first_astrofile
    def __gt__(self, other):
        return self[None] > other[None]

    @_check_first_astrofile
    def __eq__(self, other):
        return self[None] == other[None]

    @_check_first_astrofile
    def __ne__(self, other):
        return self[None] != other[None]

    # Object Arithmetic is applied on data
    @_numerize_other
    def __add__(self, other):
        return self.data + other

    @_numerize_other
    def __sub__(self, other):
        return self.data - other

    @_numerize_other
    def __floordiv__(self, other):
        return self.data // other

    @_numerize_other
    def __truediv__(self, other):
        return self.data / other

    @_numerize_other
    def __mul__(self, other):
        return self.data * other

    @_numerize_other
    def __radd__(self, other):
        return other + self.data

    @_numerize_other
    def __rsub__(self, other):
        return other - self.data

    @_numerize_other
    def __rfloordiv__(self, other):
        return other // self.data

    @_numerize_other
    def __rtruediv__(self, other):
        return other / self.data

    @_numerize_other
    def __rmul__(self, other):
        return self.data * other

    @property
    def shape(self):
        return self.data.shape

    def add_sortkey(self, key):
        """
        Sets a header item used for comparisons

        Parameters
        ----------
        key: str
        """
        self._sort_key = key

    def imshowz(self, *args, **kwargs):
        """
        Plots frame after being processed using a zscale algorithm

        See Also
        --------
        misc_graph.imshowz :
            For details on what keyword arguments are available
        """

        return pa.imshowz(self.data, *args, **kwargs)

    def stats(self, *args,
              verbose_heading = True,
              extra_headers = None,
              ):
        """
        Calculates statistical data from the file, a request can include header
        keywords to be included in the response.

        Parameters
        ----------
        verbose_heading
        extra_headers
        *args : str
            Statistic to be extracted, Possible values are: min, max, mean,
            mean3sclip, std and median

        Returns
        -------
        list
            List containing the requested data

        """
        extra_headers = extra_headers or []

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
        data = self.data
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

    def __repr__(self):
        if isinstance(self._data_file, str):
            filename = self._data_file
        elif isinstance(self._data_file, np.ndarray):
            filename = f"Array {'x'.join([str(i) for i in self._data_file.shape])}"
        else:
            filename = str(self._data_file)

        calib = "" if self._calib is None else self._calib.short()
        return '<AstroFile{}: {}>'.format(calib, filename, )

    def __new__(cls, *args, **kwargs):
        """
        If passed an AstroFile, then do not create a new instance, just pass
        that one. If passed None, return None
        """
        if args and ((isinstance(args[0], AstroFile) and len(kwargs) == 0) or args[0]) is None:
            return args[0]

        return super().__new__(cls)


if __name__ == "__main__":
    af = pa.AstroFile("../../../sample_files/image.fits.gz")
    mb = pa.AstroFile("../../../sample_files/image_mbias.fits")
    mf = pa.AstroFile("../../../sample_files/image_mflat.fits")
    cal = pa.CalibRaw2D(bias=mb, flat=mf)
    af.add_calib(cal)
    af.imshowz()
