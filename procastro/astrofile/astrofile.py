
__all__ = ['AstroFile']

import os
import pathlib
import shutil
from random import random
from typing import Any

import numpy as np
import astropy.time as apt
from matplotlib import pyplot as plt, axes

from procastro.astrofile import static_identify, static_read, static_guess, static_write
from procastro.cache.cache import astrofile_cache
from procastro.astrofile.meta import CaseInsensitiveMeta
from procastro.interfaces import IAstroFile, IAstroCalib
from procastro.logging import io_logger
from procastro.statics import PADataReturn, identity, dict_from_pattern


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


class AstroFile(IAstroFile):
    _initialized = False
    _combinators = []

    def __init__(self,
                 filename,
                 file_options: dict | None = None,
                 astrocalib: IAstroCalib | None = None,
                 spectral: bool = None,
                 meta: dict | None = None,
                 meta_from_name: str = "",
                 *args, **kwargs):
        if self._initialized:
            return

        if len(kwargs) > 0 or len(args) > 0:
            raise TypeError(f"Extra unknown arguments {args} or kwargs {kwargs} passed to AstroFileBase")
        self._corrupt = False

        self._calib = []
        self._meta = {}
        self._last_processed_meta = None
        self._random = random()
        self._data_file = filename
        self._sort_key = None

        self._format = static_identify.identify(filename, options=file_options)

        self._calib = []
        self.add_calib(astrocalib)

        if file_options is None:
            file_options = {}
        self._data_file_options: dict = file_options

        self._meta = CaseInsensitiveMeta({k: v for k, v
                                          in meta.items()}) if meta is not None else CaseInsensitiveMeta({})
        if meta_from_name:
            self._meta |= dict_from_pattern(meta_from_name, filename)
        self.meta_from_name = meta_from_name

        self._spectral = spectral

        self._initialized = True

    @property
    def spectral(self):
        """
        If ._spectral attribute is not set, then call .data for a first reading which will set this
         boolean as well

        Returns
        -------
        bool
        whether the AstroFile contains spectral information or not.

        """
        if self._spectral is None:
            identity(self.data)
        return self._spectral

    def read(self) -> PADataReturn:
        """This function should always re-read from file updating meta and forcing cache update"""

        data, meta = static_read.read(self._format, self._data_file)

        if self._spectral is None:
            self._spectral = static_guess.is_spectral(data, meta)

        if self.spectral:
            table = static_read.ndarray_to_table(data,
                                                 file_options=self._data_file_options)
            meta['infochn'] = [col for col in table.colnames if col not in ['pix', 'wav']]
            return table

        self._meta |= {k: v for k, v in meta.items()}   # only actualizes read fields. Does not touch otherwise
        self._random = random()

        return data

    ##########################################
    #
    # The following methods are unlikely to need modifications in subclasses
    #
    ##########################################

    def write(self, filename=None, data=None, backup_extension=".bak"):
        if self._format != "FITS":
            if static_identify.identify(filename) != "FITS":
                raise ValueError("Must provide a FITS filename for .write()... "
                                 "use .write_as() for alternative formats")
            if filename is None:
                raise ValueError("Filename must be provided explicitly when original type was not FITS")
        else:
            filename = self._data_file

        filename = str(filename)
        self.write_as(filename,
                      data=data, overwrite=True, backup_extension=backup_extension)

    def write_as(self, filename, data=None, overwrite=False,
                 channel=None,
                 backup_extension=".bak"):

        meta = self.meta
        meta['channel'] = channel

        filename = str(filename).format(**meta)
        file_type = static_identify.identify(filename)

        msg = ""
        if pathlib.Path(filename).exists():
            backup = pathlib.Path(str(filename) + backup_extension)
            if backup.exists():
                os.remove(backup)
            msg = f". Back-up in: {str(filename) + backup_extension}"
            shutil.move(filename, backup)

        io_logger.warning(f"Saving in {filename} using file type {file_type}{msg}")
        if data is None:
            if self.spectral and channel is not None:
                data = self.data[str(channel)].data
            else:
                data = self.data

        static_write.write(file_type, filename, data, meta,
                           overwrite=overwrite)

    def plot(self, ax=None, channels=0, title="", ncols=2, epochs=None):
        if not self.spectral:
            io_logger.warning("Cannot plot image, use imshowz instead")
            return

        if channels is None:
            channels = self.meta['infochn']
        if not isinstance(channels, list):
            channels = [channels]

        if ax is None:
            f, ax = plt.subplots(ncols=ncols if len(channels) > 1 else 1,
                                 nrows=int(np.ceil(len(channels)/ncols)))
        if not isinstance(ax, list):
            ax = [ax]
        if len(ax) < len(channels):
            ax = ax + [ax[-1]]*(len(channels)-len(ax))

        ax: list[axes.Axes]
        xlabel = ""

        for axx, channel in zip(ax, channels):
            data = self.data[str(channel)].transpose()
            if 'wav' in self.data.colnames:
                x = self.data['wav']
                xlabel = "Wavelength (AA)"
            elif 'pix' in self.data.colnames:
                x = self.data['pix']
                xlabel = "Pixel"
            else:
                x = np.arange(len(self.data))
                xlabel = "Raw Pixel"

            if len(data.shape) > 1:
                if epochs is None:
                    epochs = [0, np.argmin, np.argmax, -1]

                # numpy gives a warning here about ignoring masks in masked array.
                data = data.data
                if isinstance(data, np.ma.MaskedArray):
                    medians = [np.median(x[m]) for x, m in zip(data.data, ~data.mask)]
                else:
                    medians = np.median(data, axis=1)

                for epoch in epochs:
                    axx.plot(x, data[epoch if isinstance(epoch, int) else epoch(medians)],
                             label=f'{epoch if isinstance(epoch, int) else str(epoch).split()[1]}')
            else:
                axx.plot(x, data)

        ax[-1].set_xlabel(xlabel)
        ax[-1].legend()
        ax[0].set_title(title)

    def add_sortkey(self, key):
        """
        Sets a header item used for comparisons

        Parameters
        ----------
        key: str
        """
        self._sort_key = key

    @property
    def meta(self):
        """It will always return a copy"""
        if getattr(self, '_last_processed_meta') is None:
            identity(self.data)
        return CaseInsensitiveMeta(self._last_processed_meta)

    @meta.setter
    def meta(self, value):
        self._meta = CaseInsensitiveMeta(value)
        self._random = random()

    @property
    @astrofile_cache
    def data(self) -> PADataReturn:
        """
        Returns the data in AstroFile by calling .read() the first time and then applying calibration,
        but caching afterward until caching update
        """

        data = self.read()
        meta = self._meta

        for calibration in self._calib:
            data, meta = calibration(data, meta)

        self._last_processed_meta = meta
        return data

    @property
    def filename(self):
        return self._data_file

    @classmethod
    def __init_subclass__(cls):
        if hasattr(cls, "get_combinators") and cls.get_combinators():
            cls._combinators.extend(cls.get_combinators())
        super().__init_subclass__()

    @classmethod
    def get_combinators(cls) -> list[tuple[int, type]]:
        return sorted(cls._combinators, key=lambda c: c[0])

    def get_format(self):
        return self._format

    def forced_data(self):
        return self.data(force=True)

    def add_calib(self, astrocalib):
        if astrocalib is None:
            return self

        if not isinstance(astrocalib, list):
            astrocalib = [astrocalib]

        for calib in astrocalib:
            if not isinstance(calib, IAstroCalib):
                raise TypeError(f"'calib' must be a AstroCalib instance, not {type(calib)}: {calib}")
            self._calib.append(calib)

        return self

    def get_calib(self) -> tuple:
        return tuple(f"{i}: {repr(cal)}" for i, cal in enumerate(self._calib))

    def del_calib(self, position: int):
        """delete calibration at position (as ordered by .get_calib())"""

        self._calib.pop(position)

    def __hash__(self):
        """This is important for cache. If calib changes, then the astrofile hash should change as well. self._random
         provides a method to force reloading of cache."""

        return hash((self._data_file, self.get_calib(), self._random))

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
            cast = identity

        if len(args) == 1:
            # If first argument is tuple use those values as searches
            if isinstance(args[0], (list, tuple)):
                args = args[0]
            # If it is a string, separate by commas
            elif isinstance(args[0], str):
                args = args[0].split(',')

        hdr = self.meta
        ret = []
        for k in args:
            k = k.strip()
            if k in hdr:
                ret.append(cast(hdr[k]))
            elif k == "filename":
                ret.append(self._data_file)
            elif k == "basename":
                ret.append(str(pathlib.Path(self._data_file).name))
            elif k == "dirname":
                ret.append(str(pathlib.Path(self._data_file).parent))
            else:
                ret.append(None)

        if len(ret) == 1 and not single_in_list:
            return ret[0]
        else:
            return ret

    def __contains__(self, item):
        """

        Parameters
        ----------
        item: any, list
        if it is a list then search for the presence of any elements.  Stops searching as it finds one

        Returns
        -------

        """
        if not isinstance(item, (list, tuple)):
            item = [item]

        for found in item:
            if found in self.meta.keys():
                return True

        return False

    def __getitem__(self, key) -> Any:
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

        if isinstance(key, int):
            raise ValueError("AstroFile can only be indexed with strings to meta keys")

        return self.values(key, single_in_list=False)

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

    def stats(self, *args,
              column=None,
              verbose_heading=True,
              extra_headers=None,
              ):
        """
        Calculates statistical data from the file, a request can include header
        keywords to be included in the response.

        Parameters
        ----------
        column
        verbose_heading
        extra_headers
        *args : str
            Statistic to be extracted, Possible values are: min, max, mean,
            mean3sclip, std and median

        Returns
        -------
        list:
            A List containing the requested data in the requested order

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
        if column is None:
            data = self.data
        else:
            data = self.data[column].data
        for stat in args:
            if stat == 'min':
                ret.append(data.min())
            elif stat == 'max':
                ret.append(data.max())
            elif stat == 'delta':
                ret.append(data.max() - data.min())
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
                                 "file {}".format(stat, self._data_file))
        for h in extra_headers:
            ret.append(self[h])

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _get_calib_filename_str(self):
        if isinstance(self._data_file, np.ndarray):
            filename = f"Array {'x'.join([str(i) for i in self._data_file.shape])}"
        else:
            filename = self.filename

        astrocalib = "".join([calibrator.short() for calibrator in self._calib])
        if astrocalib:
            astrocalib = f"({astrocalib})"

        return astrocalib, filename

    def __repr__(self):
        astrocalib, filename = self._get_calib_filename_str()
        return '<{}{}: {}>'.format(self.__class__.__name__,
                                   astrocalib, filename,
                                   )

    def __new__(cls, *args, **kwargs):
        """
        If passed an AstroFile, then do not create a new instance, just pass
        that one. If passed None, return None
        """
        if args:
            same_params = True
            if isinstance(args[0], AstroFile):
                if 'spectral' in kwargs and args[0].spectral != kwargs['spectral']:
                    same_params = False
                elif 'calib' in kwargs and args[0].get_calib() != kwargs['calib']:
                    same_params = False
            elif args[0] is not None:
                same_params = False

            if same_params:
                return args[0]

        return super().__new__(cls)

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
        if isinstance(source, tuple) is True:  # a 2-element tuple with date and time keywords are eexpected.
            ut_time = f"{self[source[0]]}T{self[source[1]]}"
        else:
            ut_time = self[source]

        try:
            newhd[target] = apt.Time(ut_time).jd
        except ValueError:
            raise ValueError(f"File {self._data_file} has invalid time specification: {ut_time}")
        self.set_values(**newhd)

    def imshowz(self, *args, **kwargs):
        """
        Plots frame after being processed using a zscale algorithm

        See Also
        --------
        misc_graph.imshowz :
            For details on what keyword arguments are available
        """

        return pa.imshowz(self.data, *args, **kwargs)
