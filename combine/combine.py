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
from functools import wraps as _wraps
import scipy as sp
import warnings
import dataproc as dp
#import core as dp # TODO OJO CON ESTO"!"!!!1
import os.path as path
import pyfits as pf


###########################################
#
# Decorators
#
##############


def _combine_first(method):
    """Decorator to check whether data has been combined first"""
    @_wraps(method)
    def wrapper(instance, *args, **kwargs):
        if not hasattr(instance, 'ncombine'):
            return UnboundLocalError("To operate a Combine class, the combination method must have been applied before")
        return method(instance, *args, **kwargs)
    return wrapper


def _normalize(method):
    """Decorator to normalize the return"""
    import scipy as sp
    @_wraps(method)
    def donorm(instance, *args, **kwargs):
        ret = method(instance, *args, **kwargs)
        if 'normalize' in kwargs and  kwargs['normalize']:
            y1 = x1 = 0
            y2, x2 = ret.data.shape
            if 'normalize_region' in kwargs and kwargs['normalize_region']:
                region = kwargs['normalize_region']
                if isinstance(region, (list,tuple)) and len(region)==4:
                    y1, y2, x1, x2 = region
            elif hasattr(instance, "_normalize_region"):
                y1, y2, x1, x2 = instance._normalize_region
            if kwargs['normalize']=='mean':
                norm = ret.data[y1:y2,x1:x2].mean()
                norm_name = 'mean'
            else: #use median as default
                norm = sp.median(ret.data[y1:y2,x1:x2])
                norm_name = 'median'
            print("Normalizing by method '%s'" % (norm_name,))
        else:
            norm=1
        ret.norm_fct=norm
        ret.data = instance.data/norm
        return ret  #[y1:y2, x1:x2]
    return donorm


def _trim(f):
    """Decorator to trim the return"""
    import scipy as sp
    @_wraps(f)
    def dotrim(instance, *args, **kwargs):
        ret = f(instance, *args, **kwargs)
        if instance._trim:
            y1 = x1 = 0
            y2, x2 = instance.data.shape
            if hasattr(instance, "_normalize_region"):
                y1, y2, x1, x2 = instance._normalize_region
            instance.data = instance.data[y1:y2, x1:x2]
        return instance 
    return dotrim


def _checksig(f):
    """Decorator to check standard appropriate standard deviation without zeros"""
    @_wraps(f)
    def sigmaok(instance, *args, **kwargs):
        import warnings
        var=instance.sigmas[instance.okframes]**2
        if 0 in var:
            if (var!=0).any():
                warnings.warn("Confusing as some sigmas were 0 and some not. Ignoring them.")
        var=instance.arrays*0+1
        return f(instance, *args, var=var, **kwargs)

    return sigmaok


def _ignore_if_load(f):
    """Decorator to make ignore function if loaded from file instead of being given as arrays"""
    @_wraps(f)
    def ignoring(instance, *args, **kwargs):
        import warnings
        if hasattr(instance,'loaded') and instance.loaded:
            return instance
        return f(instance, *args, **kwargs)

    return ignoring


#########################################
#
# Combine Class
#
##############


class Combine(object):#_gr_examine):
    """"Many ways to combine data"""

    def __init__(self,astrodata,
                 header=None, flagfield='ORIGHEAD', 
                 interact=False,
                 doauto=False,
                 saveto=None,
                 load_saved=False,
                 bias=None,
                 descriptor='frame',
                 normalize_region=None,
                 normalize_pre = False,
                 trim=False,
                 force_fits=True,
                 **kwargs):
        """astrodata can be a list of astrofile or a list of array. Uses header of first astrofile element if not supplied. Bias to be substracted can be specified with 'bias' keyword.  

:param saveto: Store filename to be later used with .write(). TODO: decide whether it is better to load from this file if it exists.
:param bias: Bias to be subtracted from each frame
:param normalize_pre: Pre-normalize the data before combining using median
:param normalize_region: [y1,y2,x1,x2] of the region before normalization.
"""

        import scipy as sp
        import sys
        self.flagfield = flagfield
        arrays=[]
        heads = []
        shape1 = None

        if saveto is not None:
            if force_fits and '.fit' not in saveto.lower()[-5:]:
                saveto = saveto + '.fits'
            self.saveto = saveto

            if load_saved and path.isfile(saveto):
                #TODO: use dataproc instead of pyfits to open
                loaded_file = pf.open(saveto)[0]
                self.headers = [loaded_file.header]
                if not self.headers[0]["COMBINE"]:
                    raise ValueError("File '%s' does not contain a valid 'combine' output" % (saveto))
                self.data = loaded_file.data

                self.loaded = True
                #TODO: read sigma
                self.sigma = None
                self.lastmet = [self.headers[0]["COMBMETH"]]
                for prm in self.headers[0]["COMBPRM*"]:
                    self.lastmet.append(self.headers[0][prm])
                self.ncombine = self.headers[0]["NCOMBINE"]
                print ("Loaded result file from '%s'" % (saveto,))

                return


        sys.stdout.write (" reading %i %s%s" % (len(astrodata),descriptor,len(astrodata)>1 and "s" or ""))
        for d in astrodata:
            #Check supported type of element
#            print ("\n%s: %s/%s\n%s" %(d, d.__class__, dp.astrofile.__class__, dir(d)))
            #todo: find out why isinstance does not work!!!
            #if isinstance(d, dp.AstroFile):
            if isinstance(d, dp.AstroFile):
                dt = d.reader()
                hd = d.readheader()
                #use the first read header to fill all the data without headers
                if header is None:
                    header = hd.copy()
                    header[flagfield] = (False, 'Is this header original?')
                    #chck back if any data was without header
                    for h in heads:
                        if h is None:
                            h = header
            elif isinstance(d, Combine):
                dt = d.data
                hd = d.outhdr
            elif isinstance(d, sp.ndarray):
                dt = d
                hd = header
            else:
                raise TypeError("combine()  initialization only with a list of astrofile or  ndarray is currently supported")

            #check equal size and initialize normalization region
            if shape1 is None:
                shape1=dt.shape
                y1 = x1 = 0
                y2, x2 = dt.shape
                if normalize_region is not None:
                    if isinstance(normalize_region, (list, tuple)) and len(normalize_region)==4:
                        y1, y2, x1, x2 = normalize_region
                    self._normalize_region = normalize_region

            elif shape1 != dt.shape:
                raise ValueError("All arrays passed to combine should have the same dimension")

            if bias is not None:
                if isinstance(bias, Combine):
                    bias = bias.data
                if bias.shape != shape1:
                    raise ValueError("Bias shape (%s) does not match data's (%s)" % (bias.shape, shape1))
                dt -= bias
    
            if normalize_pre:
                dt /= sp.median(dt[y1:y2, x1:x2])
            arrays.append(dt)
            heads.append(hd)
            sys.stdout.write('.')
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        self.arrays = sp.array(arrays)

        ##TD: Get real sigmas given poisson!
        self.sigmas = sp.zeros(self.arrays.shape)
        if self.sigmas.any():
            raise ValueError("NaN are found self.sigmas!")

        self._trim = trim
        self.okframes = sp.zeros(len(arrays))==0
        if interact:
            print(" selecting good frames")
            self.showeach('Select valid frames', interact=True)

        if doauto:
            self.auto(**kwargs)

        self.headers = heads


    def _updatehdr(self, **kwargs):
        """Returns a header with info about the combination and any other indicated header. It choses the type of the first file."""

        import dataproc as dp
        from copy import copy
        if not hasattr(self,'lastmet'):
            raise ValueError("_updatehdr must be run after a combination method")
        #TD: it assumes fits header. Not always true
        if not hasattr(self,'outhdr'):
            self.outhdr = copy(self.headers[0])
            self.outhdr["COMBINE"] = (True,"Is this a combination of files?")
            self.outhdr["NCOMBINE"] = (self.ncombine,"Number of frames combined")
            self.outhdr["COMBMETH"] = (self.lastmet[0],"Method used to combine")
            for i, prm in zip(range(len(self.lastmet)-1), self.lastmet[1:]):
                self.outhdr["COMBPRM%i" %i] = (prm, "Parameter %i to method" % i)
        for k in kwargs.keys():
            self.outhdr[k] = kwargs[k]


    def save(self, *args, **kwargs):
        """Alias to write()"""
        return self.write(*args, **kwargs)


    def write(self, filename=None, verbose=True, **kwargs):
        """Write combined array to file. Add specified headers"""

        if (filename is None):
            if (not self.saveto):
                raise ValueError("Filename to store combined file was not specified with .write, nor in the object initialization")
            filename = self.saveto
        if not hasattr(self,'lastmet'):
            raise ValueError("write must be run after a combination method")
        self._updatehdr(**kwargs)
        #TODO: include sigma
        if verbose:
            print("Saving combined array to '%s'" % filename)
        warnings.filterwarnings("ignore", "Overwriting", UserWarning)
        dp.AstroFile(filename).writer(self.data, self.outhdr)
        return self


    @_ignore_if_load
    def best(self, field='EXPTIME'):
        """Finds the best function to use, return a tuple (fcnname, fcn, required_arguments)"""
        import scipy as sp
        hd = [h for h,o in zip(self.headers,self.okframes) if o]
        fv = sp.array([(field in h and h[field]) for h in hd])

        if (fv.all()) and (fv[0] != fv).any():
            return ("linear interpolation vs '%s' (%.1f - %.1f)" %
                    (field,  min(fv), max(fv))
                    , self.lininterp, ('target=%.1f',))
        if len(hd)>2:
            return ("median", self.median, ())
        else:
            return ("mean", self.mean, ())



    @_ignore_if_load
    def auto(self, **kwargs):
        """Executes the best normalization found by best()"""

        name,fcn,reqarg=self.best()
        for k in reqarg:
            if k.split('=')[0] not in kwargs:
                raise ValueError("Auto selected function <%s> needs mandatory argument (%s), but it was not specified" % (name, k))

        print(" Auto-combining with %s. %s" %(name,
                                              ', '.join([k%kwargs[k.split('=')[0]] for k in reqarg if k.split('=')[0] in kwargs])))
        return fcn( **kwargs)


############################
#
# Here goes combine methods
#
##############

    @_ignore_if_load
    @_trim
    @_normalize
    @_checksig
#    @_debias
    def mean(self, sigclip=False, var=None, **kwargs):
        """Combines the data using mean. Optionally sigma-clipping beyond the specified sigclip

        :param normalize: Normalizes the data before combination. Accepts 'mean' or any other True to default to median
        """
        if var is None:
            var = self.sigmas[self.okframes]**2
        arr = self.arrays[self.okframes]
        sig = self.sigmas[self.okframes]
        if sigclip:
            #TODO sigmask!!!
            mask = sigmask(arr,sigclip,0)
            comb = (arr*mask).sum(0)/mask.sum(0, dtype=float)
        else:
            comb = arr.mean(0)
        self.data = comb
        self.lastmet = ('mean',sigclip)
        ##TODO: Sigma for mean
        self.sigma=None
        self.ncombine=len(arr)
        self._updatehdr()
        return self


    @_ignore_if_load
    @_trim
    @_checksig
    @_normalize
#    @_debias
    def lininterp(self, target=0, 
                  field='EXPTIME', sigma=False, doerr=True,
                  var=None, multi=1):
        """Linear interpolate to given target value of the field 'field'. Returns optionally array with 1-sigma values. Optionally the variance info can be given (same size as data array)"""

        import scipy as sp
        def pvns(pfret,target):
            """Reconstructs polyfit to target value and sigma"""
            p,cov=pfret
            val = p[0]*target + p[1]
            var = cov[0,0]*target**2 + cov[1,1] + 2*target*cov[1,0]
            return val,var


        def getlinterp(params, pool=None):
            """Evaluate line interpolation """
            ##TD: being called for wgt and nwgt
            dowgt = lambda fv,target,dat,wgt: sp.array([pvns(sp.polyfit(fv, dt, 1,
                                                                        w=w, cov=True),
                                                             target)
                                                        for dt,w in zip(dat,wgt)]
                         )
            donwgt = lambda fv,target,dat: sp.array([pvns(sp.polyfit(fv, dt, 1), target)
                          for dt in dat])
            dofcn = (len(params)==4) and dowgt or donwgt
            if pool is None:  #Can't use ...and...or... or it will execute twice!
                return dofcn(*params)
            else:
                return pool.apply_async(dofcn,params)

        if not target:
            raise ValueError("keyword target=value must be specified for lineinterp(), where value cannot be zero.")

        arr=self.arrays[self.okframes]
        if var is None:
            var=self.sigmas[self.okframes]**2
        hd = [h for h,o in zip(self.headers,self.okframes) if o]
        fv = sp.array([h[field] for h in hd])

        import warnings
        from numpy import __version__ as vrs
        if float(vrs[0:3])<1.7:
            raise ValueError("numpy version must be at least 1.7.  Otherwise polyfit() cannot handle weights")
        if (fv==fv[0]).all():
            raise ValueError("Cannot linear-interpolate since all values of field '%s' are identical in the data" % field)
        if len([h for h in hd if (self.flagfield in h)]):
            warnings.warn("Interpolating in a list of frames who did not all had header with the target field")
        if target<min(fv) or target>max(fv):
            warnings.warn("Requested interpolation target (%f) is beyond the range of the '%s' field in the frames (%f - %f)" %(target,field,min(fv),max(fv)))

        print ("  sorting data%s" 
               % (doerr and  " and weights" or "",))
        sh = arr.shape
        fldata = arr.reshape(sh[0],sp.array(sh[1:]).prod())
        flwgt = 1.0/var.reshape(sh[0],sp.array(sh[1:]).prod())
        fv,fldata,flwgt = sortmanynsp(fv,fldata,flwgt)

            
        from multiprocessing.pool import ThreadPool
        print("  evaluating linear fit to %i pixels" 
              % (fldata.shape[1],))
        if doerr:
            pool=multi>1 and ThreadPool(processes=multi) or None
            print ("   using %i processors" % (multi,) )
            delta = int(fldata.shape[1]/multi+0.999)
            multistart = [[delta*i,delta*(i+1)] for i in range(multi)]
            multistart[-1][1] = fldata.shape[1]

            mpool = [getlinterp((fv, target, 
                                 fldata.T[i0:i1,:],
                                 flwgt.T[i0:i1,:]),
                                pool=pool)
                     for i0,i1 in multistart]
            if(len(mpool)>1):
                tmp=[]
                for mp in mpool:
                    tmp.extend(mp.get())
                tmp=sp.array(tmp)
            else:
                tmp=mpool[0]
                    
            # else:
            #     tmp = getlinterp((fv,target,fldata.T,flwgt.T))
        else:
            print(" - Requested to avoid error estimation")
            tmp = sp.array([sp.polyval(sp.polyfit(fv, dt, 1), target)
                            for dt in zip(fldata.T)])
        comb = tmp[:,0].reshape(sh[1:])
        sig = tmp[:,1].reshape(sh[1:])
        self.data=comb
        self.sigma=sig
        self.lastmet=('lininterp',target,field,sigma, doerr)
        self.ncombine = len(arr)
        self._updatehdr(exptime=target)
        return self



    @_ignore_if_load
    @_trim
    @_normalize
    @_checksig
#    @_debias
    def concatenate(self, axis, **kwargs):
        """Concatenates the data along given axis

        :param axis: axis along which to concatenate
        :param normalize: Normalizes the data before combination. Accepts 'mean' or any other True to default to median
        """
        arr = self.arrays[self.okframes]
        sig = self.sigmas[self.okframes]

        self.data  = sp.concatenate(arr, axis)
        self.sigma = sp.concatenate(arr, axis)
        self.lastmet = ('concatenate',axis)
        self.ncombine=len(arr)
        self._updatehdr()
        return self


    @_ignore_if_load
    @_trim
    @_normalize
    @_checksig
#    @_debias
    def median(self, var=None, **kwargs):
        """Combines the data using median

        :param normalize: Normalizes the data before combination. Accepts 'mean' or any other True to default to median
        """
        import scipy as sp
        arr=self.arrays[self.okframes]
        comb= sp.median(arr, 0)
        self.data=comb
        ##TD: Sigma for median
        self.ncombine=len(arr)
        self.sigma=None
        self.lastmet=('median',)
        self._updatehdr()
        return self


######################################
#
# Here comes some arithmethic magic
#
################

    @_combine_first
    def __radd__(self, other):
        if isinstance(other, Combine):
            other = other.data
        return self.data + other

    @_combine_first
    def __rsub__(self, other):
        if isinstance(other, Combine):
            other = other.data
        return self.data - other

    @_combine_first
    def __rtruediv__(self, other):
        if isinstance(other, Combine):
            other = other.data
        return self.data / other

    @_combine_first
    def __rdiv__(self, other):
        if isinstance(other, Combine):
            other = other.data
        return self.data / other

    @_combine_first
    def __rfloordiv__(self, other):
        if isinstance(other, Combine):
            other = other.data
        return self.data // other

    @_combine_first
    def __rmul__(self, other):
        if isinstance(other, Combine):
            other = other.data
        return self.data * other

    @_combine_first
    def __add__(self, other):
        if isinstance(other, Combine):
            other = other.data
        self.data += other
        return self

    @_combine_first
    def __sub__(self, other):
        if isinstance(other, Combine):
            other = other.data
        self.data -= other
        return self

    @_combine_first
    def __truediv__(self, other):
        if isinstance(other, Combine):
            other = other.data
        self.data /= other
        return self

    @_combine_first
    def __div__(self, other):
        if isinstance(other, Combine):
            other = other.data
        self.data /= other
        return self

    @_combine_first
    def __floordiv__(self, other):
        if isinstance(other, Combine):
            other = other.data
        self.data //= other
        return self

    @_combine_first
    def __mul__(self, other):
        if isinstance(other, Combine):
            other = other.data
        self.data *= other
        return self




