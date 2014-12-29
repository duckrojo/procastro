#
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
from functools import wraps as _wraps
import warnings
import combine as cm


def _numerize_other(method):
    @_wraps(method)
    def wrapper(instance, other, *args, **kwargs):
        if isinstance(other, AstroFile):
            other = other.reader()
        if isinstance(other, cm.Combine):
            other = other.data
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
    raise NotImplemented("Cuek. More work here. header not ssave, no history. ")
    return pf.writeto(filename, data, header, clobber=True, output_verify='silentfix')


def _fits_verify(filename, ffilter=None, hdu=0):
    import pyfits as pf
    nc = filename.lower().split('.')[-1] in ['fits', 'fit'] 
    cmpr = ''.join(filename.lower().split('.')[-2:]) in ['fitsgz', 'fitgz'] 
    if nc or cmpr:
        if ffilter is None:
            return True
        h = pf.getheader(filename, hdu)
        if isinstance (ffilter,dict):
            return False not in [(f in h and h[f] == ffilter[f]) for f in ffilter.keys()]
        if isinstance (ffilter, (list,tuple)):
            return False not in [f in h for f in ffilter]
    return False


# def _fits_filter(_filename, hdu=0, **kwargs):
#     import pyfits as pf
#     h=pf.getheader(_filename, hdu)
#     return True in [k in h and (isinstance(kwargs[k], (tuple, list)) 
#                                 and [kwargs[k][0](h[k]) in map(kwargs[k][0],kwargs[k][1:])]
#                                 or [h[k] == kwargs[k]])[0]
#                     for k in kwargs.keys()]


def _fits_getheader(_filename, *args, **kwargs):
    import pyfits as pf
    hdu = ('hdu' in kwargs and [kwargs['hdu']] or [0])[0]
    h = pf.getheader(_filename, hdu)
    return tuple([(a in h and [h[a]] or [None])[0] for a in args])


def _fits_setheader(_filename, *args, **kwargs):
    import pyfits as pf
    hdu = ('hdu' in kwargs and [kwargs['hdu']] or [0])[0]
    if 'write' in kwargs and kwargs['write']:
        save = True
        try:
            #todo: if file does not exists, create it
            fits = pf.open(_filename, 'update')[hdu]
        except IOError:
            warnings.warn("Read only filesystem or file not found. Header update of '%s' will not remain on disk." % ', '.join(kwargs.keys()))
            return
    else:
        fits = pf.open(_filename)[hdu]
        save = False
    if 'write' in kwargs:
        del kwargs['write']

    h = fits.header
    for k, v in kwargs.items():
        #If new value is None, then delete it
        if v is None:
            del h[k]
        else:
            h[k] = v
    if save:
        fits.flush()
    return True

#############################
#
# End FITS
#
####################################


class AstroFile(object):
    """Valid astronomy data file format. Only filename is checked so far"""

    _reads = {'fits': _fits_reader}
    _readhs = {'fits': _fits_header}
    _ids = {'fits': _fits_verify}
    _writes = {'fits': _fits_writer}
    _geth = {'fits': _fits_getheader}
    _seth = {'fits': _fits_setheader}

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
                    #raise ValueError("File %s not a supported astro-type." % (f))
                    raise ValueError("Please specify filename with setFilename first.")
                return f(inst, *args, **kwargs)
            else:
                raise ValueError("Filename not defined. Must give valid filename to AstroFile")
        return isfiledef

#     def __call__(self):
#         import copy
#         import os.path as path
#         new = copy.copy(self)
#         new.filename = filename
#         new.type = new.checktype(exists,*args, **kwargs)
#         new.reqheads={'basename':path.basename(filename)}
# #        new.filter.__func__.__doc__=new._flts[new.type].__doc__
#         return new

    def __repr__(self):
        return '<astrofile: %s>' % (self.filename,)

    def __init__(self, filename = None,
                 mbias=None, mflat=None, exists=False,
                 *args, **kwargs):
        import os.path as path
        # agregado aca para inicializar sin nombre, AstroFile "vacio"
        if not hasattr(self, 'filename'):
            self.filename = None
            self.type = None
            self.header_cache = None
        else:
            self.filename = filename
            self.type = self.checktype(exists, *args, **kwargs)
            self.header_cache = {'basename': path.basename(filename)}
            if mbias is not None:
                self.add_bias(mbias)
            if mflat is not None:
                self.add_flat(mflat)

    def add_bias(self, mbias):
        if not hasattr(self, 'calib'):
            self.calib = dp.AstroCalib()
        self.calib.add_bias(mbias)

    def add_flat(self, mflat):
        if not hasattr(self, 'calib'):
            self.calib = dp.AstroCalib()
        self.calib.add_flat(mflat)

    # Para cargar data a un AstroFile vacio
    def load(self, filename, exists=False, *args, **kwargs):
        if self.filename is None:
            import os.path as path
            self.filename = filename
            self.type = self.checktype(exists, *args, **kwargs)
            self.header_cache = {'basename': path.basename(filename)}
            #print("ok")
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

        #todo: following is only necessary because isinstance is not working on combine module!!
#        self.astrofile=True

    # def register(self,name,identify, reader, writer, 
    #              getheader=None, setheader=None):
    #     """register(name, id_fcn, read_fcn, write_fcn): Register the read-, write-, and identify-able functions for a new type of astro filename (name)"""
    #     self._ids[name]=identify
    #     self._reads[name]=reader
    #     self._writes[name]=writer
    #     self._seth[name]=setheader
    #     self._geth[name]=getheader

    def checktype(self, exists, *args, **kwargs):
        import os.path as path
        if not hasattr(self, 'filename'):
            #raise ValueError("Must define file name")
            return None
        if not isinstance(self.filename, basestring):
            return None
        if exists and not path.isfile(self.filename):
            return None
        for k in self._ids.keys():
            if self._ids[k](self.filename, *args, **kwargs):
                return k
        return None

    def imshowz(self, *args, **kwargs):
        return dp.imshowz(self.reader(), *args, **kwargs)

    def __nonzero__(self):
        return hasattr(self,'filename') and (self.type is not None)

    @_checkfilename
    def filter(self, *args, **kwargs):
        """True if the header given by the keyword argument matches its value in the fits header.
            Multiple alternatives can be specified with a dict whose keys are casting function (e.g. filter(exptime={'int':300,10})). Multiple keywords are 'or' alternatives.
            DEPRECATED/NOT ANYMORE: You can also make a substring match search "'needle' in keyword" by specifying None as the first element in the tuple: basename=(None,'dflat').
            If you want 'and' filtering then filter in chain (e.g. filter(exptime=300).filter(object='star'))
        """
        ret = []

#       print (kwargs.items())
        for filter_keyword, request in kwargs.items():
            functions = ['equal']
            exists = True
            cast = lambda x: x
            filter_keyword = filter_keyword.replace('__', '-')
            if '_' in filter_keyword:
                tmp = filter_keyword.split('_')
                filter_keyword = tmp[0]
                functions.extend(tmp[1:])
            header_val = self.getheaderval(filter_keyword)[0]
#            print ("hv:%s fk:%s rq:%s" % (header_val, filter_keyword, request))

            #treat specially the not-found and list as filter_keyword
            if header_val is None:
                ret.append(False)
                continue

            if isinstance(request, basestring):
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
#                warnings.warn("Attempting auto-casting the filter's value. Found '%s' for '%s'." % (cast,request))

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
                ret.append(True in [cast(r) < cast(header_val) for r in request])
            elif less_than:
                ret.append(True in [cast(r) > cast(header_val) for r in request])
            elif match:
                ret.append(True in [cast(r) in cast(header_val) for r in request])
            else:
                ret.append(True in [cast(r) == cast(header_val) for r in request])

        #returns whether the filter existed (or not if _not function)
        return (True in ret) == exists


    @_checkfilename
    def setheader(self, *args, **kwargs):
        """Set header values from kwargs. They can be specfied as tuple to add comments as in pyfits"""
        tp = self.type
        for k, v in kwargs.items():
            if v is None:
                del self.header_cache[k]
            else:
                self.header_cache[k] = v
        return self._seth[tp](self.filename, **kwargs)


    @_checkfilename
    def getheaderval(self, *args, **kwargs):
        """Get header value for each of the fields specified in args.  It can also accept a list
        as a single argument: getheaderval(field1,field2,...) or getheaderval([field1,field2,...])"""
        tp = self.type
        mapout = ('mapout' in kwargs and [kwargs['mapout']] or [lambda x:x])[0]
        if len(args) == 1:
            if isinstance(args[0], (list, tuple)):  # if first argument is tuple use those values as searches
                args = args[0]
#if only 1 already-read header is requested, use a shortcut
            elif args[0] in self.header_cache.keys():
                return mapout([self.header_cache[args[0]]])

        nkeys = [k for k in args if k not in self.header_cache.keys()]
        nhds = nkeys and self._geth[tp](self.filename, *nkeys, **kwargs) or []
        for k, v in zip(nkeys,nhds):
            self.header_cache[k] = v
        ret = [self.header_cache[k] for k in args]
        return mapout(ret)

    #emulating arithmetic
    def __add__(self, other):
        return self.reader()+other

    def __mul__(self, other):
        return self.reader()*other

    def __sub__(self, other):
        return self.reader()-other

    def __div__(self, other):
        import warnings
        warnings.warn('Better use "from __future__ import division" for future compability')
        return self.reader()/other

    def __truediv__(self, other):
        return self.reader()/other

    @_checkfilename
    def reader(self, *args, **kwargs):
        """Read astro data and return it calibrated if provided
            :param rawdata: if True return raw instead of calibrated
            TODO: Respond to different exposure times or filters
        """
        tp = self.type
        if not tp:
            return False
        data = self._reads[tp](self.filename, *args, **kwargs)

        if not hasattr(self, 'calib'):
            self.calib = AstroCalib()

        if ('hdu' in kwargs and kwargs['hdu']<0) or ('rawdata' in kwargs and kwargs['rawdata']):
            return data
        return self.calib.reduce(data)

    @_checkfilename
    def readheader(self, *args, **kwargs):
        """Read astro header
        """
        tp = self.type
        if not tp:
            return False
        return self._readhs[tp](self.filename, *args, **kwargs)


    @_checkfilename
    def writer(self, *args, **kwargs):
        """Write astro data"""
        tp = self.type
        #todo: save itself if data exists. Now it only saves explicit array given by user
        data = args[0]
        return tp and  self._writes[tp](self.filename, data, *args, **kwargs)

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
        return  other + self.reader()

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

    @property
    def shape(self):
        return self.reader().shape
