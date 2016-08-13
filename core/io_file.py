# dataproc - general data processing routines
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

import scipy as sp
from functools import wraps as _wraps
import warnings
import dataproc as dp
import astropy.time as apt

def _numerize_other(method):
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

def _fits_header(filename, hdu=0):
    """Read fits files.
    """
    import pyfits as pf
    return pf.open(filename)[hdu].header


def _fits_reader(filename, hdu=0):
    """Read fits files.

    :param hdu: if -1 return all hdus
    """
    import pyfits as pf
    if hdu < 0:
        return pf.open(filename)

    fl = pf.open(filename)[hdu]
    return fl.data


def _fits_writer(filename, data, header=None):
    import pyfits as pf
    raise NotImplemented("Cuek. More work here. header not save, no history. ")
    return pf.writeto(filename, data, header, clobber=True, output_verify='silentfix')


def _fits_verify(filename, ffilter=None, hdu=0):
    import pyfits as pf
    single_extension = filename.lower().split('.')[-1] in ['fits', 'fit', 'ftsc']
    double_extension = ''.join(filename.lower().split('.')[-2:]) in ['fitsgz', 'fitgz', 'fitszip', 'fitzip']
    if single_extension or double_extension:
        if ffilter is None:
            return True
        h = pf.getheader(filename, hdu)
        if isinstance(ffilter, dict):
            return False not in [(f in h and h[f] == ffilter[f]) for f in ffilter.keys()]
        if isinstance(ffilter, (list, tuple)):
            return False not in [f in h for f in ffilter]
    return False


def _fits_getheader(_filename, *args, **kwargs):
    import pyfits as pf
    hdu = ('hdu' in kwargs and [kwargs['hdu']] or [0])[0]
    h = pf.getheader(_filename, hdu)
    return tuple([(a in h and [h[a]] or [None])[0] for a in args])


def _fits_setheader(_filename, **kwargs):
    import pyfits as pf
    hdu = ('hdu' in kwargs and [kwargs['hdu']] or [0])[0]
    if 'write' in kwargs and kwargs['write']:
        save = True
        try:
            # todo: if file does not exist, create it
            fits = pf.open(_filename, 'update')[hdu]
        except IOError:
            iologger.warning(
                "Read-only filesystem or file not found. Header update of '%s' will not remain on disk." % ', '.join(
                    kwargs.keys()))
            return
    else:
        fits = pf.open(_filename)[hdu]
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
    return True


#############################
#
# Start sparray type
#
####################################

def array_reader(filename, **kwargs):
    return filename


def array_verify(filename, **kwargs):
    if isinstance(filename, sp.ndarray):
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
    @_wraps(f)
    def ret(self, *args, **kwargs):
        if not hasattr(self, 'sortkey'):
            raise ValueError("sortkey must be defined before trying to sort AstroFile")
        return f(self, *args, **kwargs)

    return ret


def _checkfilename(f):
    @_wraps(f)
    def isfiledef(inst, *args, **kwargs):
        if hasattr(inst, 'filename'):
            if inst.type is None:
                # raise ValueError("File %s not a supported astro-type." % (f))
                raise ValueError("Please specify filename with setFilename first.")
            return f(inst, *args, **kwargs)
        else:
            raise ValueError("Filename not defined. Must give valid filename to AstroFile")

    return isfiledef


class AstroFile(object):
    """Valid astronomy data file format. Only filename is checked so far"""

    _reads = {'fits': _fits_reader, 'sparray': array_reader}
    _readhs = {'fits': _fits_header, 'sparray': return_none}
    _ids = {'fits': _fits_verify, 'sparray': array_verify}
    _writes = {'fits': _fits_writer, 'sparray': return_none}
    _geth = {'fits': _fits_getheader, 'sparray': return_none}
    _seth = {'fits': _fits_setheader, 'sparray': return_none}

    def __repr__(self):
        return '<AstroFile{}: {}>'.format(self.has_calib() and "({}{})".format(self.calib.has_bias and "B" or "",
                                                                               self.calib.has_flat and "F" or "",)
                                          or "",
                                          self.filename,)

    def __new__(cls, *args, **kwargs):
        """If passed an AstroFile, then do not create a new instance, just pass that one"""
        if args and isinstance(args[0], AstroFile):
            return args[0]
        else:
            return super(AstroFile, cls).__new__(cls, *args)

    def __init__(self, filename=None,
                 mbias=None, mflat=None, exists=False,
                 hdu=0, hduh=None, hdud=None,
                 *args, **kwargs):
        """

        :param filename: Filename, magic var
        :param mbias: Default Master bias
        :param mflat:
        :param exists:
        :param hdu:
        :param hduh:
        :param hdud:
        :param args:
        :param kwargs:
        """

        import os.path as path

        if isinstance(filename, AstroFile):
            return

        if isinstance(filename, str):
            self.filename = filename
            self.header_cache = {'basename': path.basename(filename),
                                 'dirname':  path.dirname(filename),
                                 }
        else:
            self.filename = None
            self.header_cache = {}
        self.type = self.checktype(exists, *args, **kwargs)

        if hduh is None:
            hduh = hdu
        if hdud is None:
            hdud = hdu
        self._hduh = hduh
        self._hdud = hdud

        self.calib = AstroCalib(mbias, mflat)

    def add_bias(self, mbias):
        self.calib.add_bias(mbias)

    def add_flat(self, mflat):
        self.calib.add_flat(mflat)

    # To load data to an empty AstroFile
    def load(self, filename, exists=False, *args, **kwargs):
        if self.filename is None:
            import os.path as path
            self.filename = filename
            self.type = self.checktype(exists, *args, **kwargs)
            self.header_cache = {'basename': path.basename(filename)}
            # print("ok")
        else:
            raise ValueError("The current Astrofile already has data in it.\n \
                             Data must be loaded to an empty AstroFile.")

    # Para ponerle filename a los AstroFile que se inicializaron vacios
    # Para setHeader, puedo querer poner header antes de cargar datos
    def setFilename(self, filename, exists=False, *args, **kwargs):
        if self.filename is None:
            self.filename = filename
            import os.path as path
            self.type = self.checktype(exists, *args, **kwargs)
            self.header_cache = {'basename': path.basename(filename)}
        else:
            raise ValueError("Existing file cannot be renamed.")

    def checktype(self, exists, *args, **kwargs):
        import os.path as path
        if not hasattr(self, 'filename'):
            return None
        if not isinstance(self.filename, str):
            return None
        elif self.filename is None:
            return "array"
        if exists and not path.isfile(self.filename):
            return None
        for k in self._ids.keys():
            if self._ids[k](self.filename, *args, **kwargs):
                return k
        return None

    def spplot(self,
               axes=None, title=None, xtitle=None, ytitle=None,
               *args, **kwargs):
        fig, ax = dp.prep_canvas(axes, title, xtitle, ytitle)

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
            raise NotImplemented("Needs to add reading of wavelength from headers")
        else:
            raise NotImplemented("Spectra not understood")

        ax.plot(wav, flx)

    def plot(self, *args, **kwargs):
        """ Calls plot_accross(data, axes=None, title=None,
                 ytitle=None, xtitle=None, xlim=None, ylim=None,
                 ticks=True, colorbar=False, hdu=0, rotate=0,
                 pos=0, forcenew=False, **kwargs)

                 

                 """

        return dp.plot_accross(self.reader(), *args, **kwargs)

    def imshowz(self, *args, **kwargs):
        return dp.imshowz(self.reader(), *args, **kwargs)

    def __nonzero__(self):
        return hasattr(self, 'filename') and (self.type is not None)

    @_checkfilename
    def filter(self, **kwargs):
        """True if the header given by the keyword argument matches its value in the fits header.
            Multiple alternatives can be specified with a dict whose keys are casting function (e.g. filter(exptime={'int':300,10})). Multiple keywords are 'or' alternatives.
            If you want 'and' filtering then filter in chain (e.g. filter(exptime=300).filter(object='star'))
        """
        ret = []

        #       print (kwargs.items())
        for filter_keyword, request in kwargs.items():
            functions = []
            # by default is not comparing match, but rather equality
            match = False
            exists = True

            cast = lambda x: x
            filter_keyword = filter_keyword.replace('__', '-')
            if '_' in filter_keyword:
                tmp = filter_keyword.split('_')
                filter_keyword = tmp[0]
                functions.extend(tmp[1:])
            header_val = self.getheaderval(filter_keyword)[0]

            # treat specially the not-found and list as filter_keyword
            if header_val is None:
                ret.append(False)
                continue

            if isinstance(request, str):
                request = [request]
            elif isinstance(request, (tuple, list)):
                raise TypeError("Filter string cannot be tuple/list anymore.  "
                                "It has to be a dictionary with the casting function as key (e.g. 'str')")
            elif isinstance(request, dict):
                keys = request.keys()
                if len(keys) != 1:
                    raise NotImplemented("Only a single key (casting) per filtering function has been implemented "
                                         "for multiple alternatives")
                try:
                    request = list(request[keys[0]])
                    if 'begin' in functions or 'end' in functions:
                        raise ValueError("Cannot use '_begin' or '_end' if comparing to a list")
                except TypeError:
                    request = [request[keys[0]]]
                cast = eval(keys[0])
                if not callable(cast):
                    raise ValueError("Dictionary key (casting) has to be a callable function "
                                     "accepting only one argument")
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
                if f[:5] == 'icase':
                    header_val = header_val.lower()
                    request = [a.lower() for a in request]
                if f[:3] == 'not':
                    exists = False
                if f[:5] == 'match':
                    match = True
                if f[:5] == 'equal':
                    match = False
                if f[:3] == 'lt':
                    less_than = True
                if f[:3] == 'gt':
                    greater_than = True

            # print("r:%s h:%s m:%i f:%s" %(request, header_val,
            #                               match, functions))
            if greater_than:
                ret.append((True in [cast(r) < cast(header_val) for r in request]) == exists)
            elif less_than:
                ret.append((True in [cast(r) > cast(header_val) for r in request]) == exists)
            elif match:
                ret.append((True in [cast(r) in cast(header_val) for r in request]) == exists)
            else:
                ret.append((True in [cast(r) == cast(header_val) for r in request]) == exists)

        # returns whether the filter existed (or not if _not function)
        return (True in ret)

    @_checkfilename
    def setheader(self, **kwargs):
        """Set header values from kwargs. They can be specfied as tuple to add comments as in pyfits"""
        tp = self.type
        hdu = kwargs.pop('hduh', self._hduh)
        for k, v in kwargs.items():
            if v is None:
                del self.header_cache[k]
            else:
                self.header_cache[k] = v
        return self._seth[tp](self.filename, hdu=hdu, **kwargs)

    @_checkfilename
    def getheaderval(self, *args, **kwargs):
        """Get header value for each of the fields specified in args.  It can also accept a list
        as a single argument: getheaderval(field1,field2,...) or getheaderval([field1,field2,...])"""

        tp = self.type

        mapout = kwargs.pop('mapout', lambda x: x)
        hdu = kwargs.pop('hdu', self._hduh)

        if len(args) == 1:
            if isinstance(args[0], (list, tuple)):  # if first argument is tuple use those values as searches
                args = args[0]
            # if only 1 already-read header is requested, use a shortcut
            elif args[0] in self.header_cache.keys():
                return mapout([self.header_cache[args[0]]])

# read some typical keywords on first pass just in case to speed up
        args_n_def = args + ('exptime', 'filter', 'date-obs')

# only open fits file if there are non-cached keywords.
        new_keys = [k for k in args_n_def if k not in self.header_cache.keys()]
        new_values = new_keys and self._geth[tp](self.filename, *new_keys, hdu=hdu) or []

        for k, v in zip(new_keys, new_values):
            self.header_cache[k] = v

        ret = [self.header_cache[k] for k in args]

        return mapout(ret)

    # emulating arithmetic
    def __add__(self, other):
        return self.reader() + other

    def __mul__(self, other):
        return self.reader() * other

    def __sub__(self, other):
        return self.reader() - other

    def __div__(self, other):
        import warnings
        warnings.warn('Better use "from __future__ import division" for future compability')
        return self.reader() / other

    def __truediv__(self, other):
        return self.reader() / other

    @_checkfilename
    def reader(self, *args, **kwargs):
        """Read astro data and return it calibrated if provided
            :param rawdata: if True return raw instead of calibrated
            TODO: Respond to different exposure times or filters
        """
        tp = self.type
        hdu = kwargs.pop('hdud', None)
        if hdu is None:
            hdu = kwargs.pop('hdu', self._hdud)
        kwargs.pop('hdu', None)

        if not tp:
            return False

        data = self._reads[tp](self.filename, *args, hdu=hdu, **kwargs)

        # Just print the HDUs if requested hdu=-1
        if hdu == -1:
            return data

        if data is None:
            raise ValueError("HDU {hdu} empty at file {file}?\n "
                             "available: {hdus}".format(hdu=hdu, file=self.filename,
                                                        hdus=self.reader(hdu=-1)))

        if ('hdu' in kwargs and kwargs['hdu'] < 0) or ('rawdata' in kwargs and kwargs['rawdata']):
            return data

        if self.has_calib():
            return self.calib.reduce(data,
                                     exptime=self[self.calib.exptime_keyword],
                                     filter=self[self.calib.filter_keyword]
                                     )

        return data

    def has_calib(self):
        return self.calib.has_flat or self.calib.has_bias

    @_checkfilename
    def readheader(self, *args, **kwargs):
        """Read astro header
        """
        tp = self.type
        hdu = kwargs.pop('hdu', self._hduh)

        if not tp:
            return False
        return self._readhs[tp](self.filename, *args, hdu=hdu, **kwargs)

    @_checkfilename
    def writer(self, *args, **kwargs):
        """Write astro data"""
        tp = self.type
        # todo: save itself if data exists. Now it only saves explicit array given by user
        data = args[0]
        return tp and self._writes[tp](self.filename, data, *args, **kwargs)

    @_checkfilename
    def basename(self):
        """Returns file basename"""
        import os.path as path
        if not hasattr(self, 'filename'):  # TODO ojo aca
            return None
        return path.basename(self.filename)

    @_checkfilename
    def __getitem__(self, key):
        """Read data and return key"""

        ###TODO: return header
        if isinstance(key, str):
            return self.getheaderval(key)[0]
        else:
            return self.reader()[key]

    @_checksortkey
    def __lt__(self, other):
        return self.getheaderval(self.sortkey)[0] < \
               other.getheaderval(self.sortkey)[0]

    @_checksortkey
    def __le__(self, other):
        return self.getheaderval(self.sortkey)[0] <= \
               other.getheaderval(self.sortkey)[0]

    @_checksortkey
    def __gt__(self, other):
        return self.getheaderval(self.sortkey)[0] > \
               other.getheaderval(self.sortkey)[0]

    @_checksortkey
    def __eq__(self, other):
        return self.getheaderval(self.sortkey)[0] == \
               other.getheaderval(self.sortkey)[0]

    @_checksortkey
    def __ne__(self, other):
        return self.getheaderval(self.sortkey)[0] != \
               other.getheaderval(self.sortkey)[0]

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
        return len(self.reader())

    @property
    def shape(self):
        return self.reader().shape

    def stats(self, *args, **kwargs):
        verbose_heading = kwargs.pop('verbose_heading', True)
        extra_headers = kwargs.pop('extra_headers', [])
        if kwargs:
            raise SyntaxError("only the following keyword argument for stats are permitted 'verbose_heading', 'extra_headers'")

        if not args:
            args = ['min', 'max', 'mean', 'mean3sclip', 'median', 'std']
        if verbose_heading:
            print("Computing the following stats: {} {}".format(", ".join(args),
                                                                len(extra_headers) and "\nand showing the following headers: {}".format(", ".join(extra_headers)) or ""))
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
                clip = data.copy()
                clip -= data.mean()
                std = data.std()
                ret.append(clip[sp.absolute(clip) < 3 * std].mean())
            elif stat == 'std':
                ret.append(data.std())
            elif stat == 'median':
                ret.append(sp.median(data))
            else:
                raise ValueError("Unknown stat '{}' was requested for "
                                 "file {}".format(stat, self.filename))
        for h in extra_headers:
            ret.append(self[h])

        return ret

    def jd_from_ut(self, target='jd', source='date-obs'):
        """
Add jd in header's cache to keyword 'target' using ut on keyword 'source'
        :param target: target keyword for JD storage
        :param source: input value in UT format
        """
        newhd = {}
        newhd[target] = apt.Time(self[source]).jd
        self.setheader(**newhd)


class AstroCalib(object):
    """Object to hold calibration frames. It is a separate entity than AstroFile, since if is set by AstroDir, there will be no duplication of calibration frames, but just a common object."""
    def __init__(self, mbias=None, mflat=None,
                 exptime_keyword='exptime',
                 filter_keyword='filter'):
        # it is always created false, if the add_*() has something, then it is turned true.
        self.has_bias = self.has_flat = False

        self.mbias = {}
        self.mflat = {}
        self.exptime_keyword = exptime_keyword
        self.filter_keyword = filter_keyword
        self.add_bias(mbias)
        self.add_flat(mflat)

    def add_bias(self, mbias):
        """
Add Master Bias to Calib object.

        :param mbias: It can be either a dictionary indexed by exposure
        time or an array/AstroFile for generic times
        """
        if mbias is None:
            self.mbias[-1] = 0.0
            return
        self.has_bias = True
        if isinstance(mbias, dict):
            for k in mbias.keys():
                self.mbias[k] = mbias[k]
        elif isinstance(mbias,
                        (int, float, sp.ndarray)):
            self.mbias[-1] = mbias
        elif isinstance(mbias,
                        dp.AstroFile):
            self.mbias[-1] = mbias.reader()
        else:
            raise ValueError("Master Bias supplied was not recognized.")

    def add_flat(self, mflat):
        """
Add master flat to Calib object

        :param mflat: it can be either a dictionary
         indexed by filter name, or an AstroFile/array for generic filter.
        """
        if mflat is None:
            self.mflat[''] = 1.0
            return
        self.has_flat = True
        if isinstance(mflat, dict):
            for k in mflat.keys():
                self.mflat[k] = mflat[k]
        elif isinstance(mflat,
                        dp.AstroFile):
            self.mflat[''] = mflat.reader()
        elif isinstance(mflat,
                        (int, float, sp.ndarray)):
            self.mflat[''] = mflat
        else:
            raise ValueError("Master Flat supplied was not recognized.")

    def reduce(self, data, exptime=None, filter=None):
        """
Process flat & bias
        :param data: science sp.array()
        :param exptime: exposure time for master bias
        :param afilter: filter for flat
        :return: reduced data
        """
        if exptime is None or exptime not in self.mbias:
            exptime = -1
        if filter is None or filter not in self.mflat:
            filter = ''

        if exptime not in self.mbias:
            raise ValueError("Requested exposure time ({}) is not available "
                             "for mbias, only: {}".format(exptime, ", ".join(map(str,
                                                                                 self.mbias.keys()))))
        if filter not in self.mflat:
            raise ValueError("Requested filter ({}) is not available "
                             "for mflat, only: {}".format(filter, ", ".join(self.mflat.keys())))

        debias = data - self.mbias[exptime]
        deflat = debias / self.mflat[filter]

        return deflat
