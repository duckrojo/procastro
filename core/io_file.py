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

from functools import wraps as _wraps



class _astrofile():
    """Valid astronomy data file format. Only filename is checked so far"""

    def _checksortkey(f):
        @_wraps(f)
        def ret(self, *args, **kwargs):
            if not hasattr(self, 'sortkey'):
                raise ValueError("sortkey must be defined before trying to sort AstroFile")
            return f(self,*args,**kwargs)
        return ret

    def _checkfilename(f):
        @_wraps(f)
        def isfiledef(inst, *args, **kwargs):
            if hasattr(inst,'filename'):
                if (inst.type is None):
                    raise ValueError("File %s not a supported astro-type." % (self.filename))
                return f(inst, *args, **kwargs)
            else:
                raise ValueError("Filename not defined. Must give valid filename to AstroFile")
        return isfiledef

    def __call__(self,filename, exists=False, *args, **kwargs):
        import copy
        new = copy.copy(self)
        new.filename = filename
        new.type = new.checktype(exists,*args, **kwargs)
        new.reqheads={}
#        new.filter.__func__.__doc__=new._flts[new.type].__doc__
        return new

    def __repr__(self):
        return '<astrofile: %s>' % (self.filename,)

    def __init__(self):
        self._reads={}
        self._ids={}
        self._writes={}
        self._flts={}
        self._geth={}

    def register(self,name,identify, reader, writer, getheader=None):
        """register(name, id_fcn, read_fcn, write_fcn): Register the read-, write-, and identify-able functions for a new type of astro filename (name)"""
        self._ids[name]=identify
        self._reads[name]=reader
        self._writes[name]=writer
#        self._flts[name]=filter
        self._geth[name]=getheader

    def checktype(self, exists, *args, **kwargs):
        import os.path as path
        if not hasattr(self,'filename'):
            raise ValueError("Must define file name")
        if (not isinstance(self.filename, basestring)):
            return None
        if(exists and  not path.isfile(self.filename)):
            return None
        for k in self._ids.keys():
            if self._ids[k](self.filename, *args, **kwargs):
                return k
        return None

    def __nonzero__(self):
        return hasattr(self,'filename') and (self.type is not None)

    @_checkfilename
    def filter(self, *args, **kwargs):
        """True if the header given by the keyword argument matches its value in the fits header. Multiple alternatives can be specified with a tuple/list that starts with a casting function (e.g. filter(exptime=(int,300,10))). Multiple keywords are 'or' alternatives. If you want 'and' filtering then filter in chain (e.g. filter(exptime=300).filter(object='star')) """
        keys = kwargs.keys()
        val = self.getheaderval(*keys)
        return True in [v is not None and 
                        (isinstance(kwargs[k], (tuple, list)) 
                         and [kwargs[k][0](v) in map(kwargs[k][0],kwargs[k][1:])]
                         or [v == kwargs[k]])[0]
                        for k,v in zip(keys,val)]

    @_checkfilename
    def getheaderval(self, *args, **kwargs):
        """Get header value for each of the fields specified in args. It can also accept a list as a single argument: getheaderval(field1,field2,...) or getheaderval([field1,field2,...])"""
        tp = self.type
        mapout = 'mapout' in kwargs and kwargs['mapout'] or None
        if len(args)==1:
            if isinstance(args[0],(list,tuple)): #if first argument is tuple use those values as searches
                args = args[0]
#if only 1 already-read header is requested, use a shortcut
            elif args[0] in self.reqheads.keys():
                return [self.reqheads[args[0]]]
                    
        nkeys = [k for k in args if k not in self.reqheads.keys()]
        nhds = nkeys and self._geth[tp](self.filename, *nkeys, **kwargs) or []
        for k,v in zip(nkeys,nhds):
            self.reqheads[k] = v
        ret = [self.reqheads[k] for k in args]
        return mapout is None and ret or mapout(ret)

    @_checkfilename
    def reader(self, *args, **kwargs):
        """Read astro data"""
        tp = self.type
        return tp and self._reads[tp](self.filename, *args, **kwargs)

    @_checkfilename
    def writer(self, *args, **kwargs):
        """Write astro data"""
        tp = self.type
        return tp and  self._writes[tp](self.filename, *args, **kwargs)

    @_checkfilename
    def basename(self):
        """Returns file basename"""
        import os.path as path
        return path.basename(self.filename)

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

AstroFile = _astrofile()


#######################
#
# FITS handling of AstroFile
#
##################################

def _fits_reader(filename, hdu=0, datahead=False):
    import pyfits as pf
    fl = pf.open(filename)[hdu]
    if datahead:
        return fl.data, fl.header
    else:
        return fl.data
def _fits_writer(filename, data, header=None):
    import pyfits as pf
    return pf.writeto(filename, data, header, clobber=True)
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
    hdu=('hdu' in kwargs and [kwargs['hdu']] or [0])[0]
    h=pf.getheader(_filename, hdu)
    return tuple([a in h and h[a] or None for a in args])

AstroFile.register('fits',_fits_verify,_fits_reader,_fits_writer, getheader=_fits_getheader)

#############################
#
# End FITS
#
####################################

