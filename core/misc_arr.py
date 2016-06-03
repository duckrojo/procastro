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
import scipy as sp
import inspect
import scipy.signal as sg
import pyfits as pf
import copy


def sigmask(arr, sigmas, axis=None, kernel=0, algorithm='median', npass=1, mask=None, full=False):
    """Returns a mask with those values that are 'sigmas'-sigmas beyond the mean value of arr.

:param sigmas: int
    Variation beyond this number of sigmas will be masked.
:param axis: int, optional
    Look for the condition along an axis, mark those. None is the full array.
:param kernel: int, optional (some algorithms accepts ndarray)
    Size of the kernel to build the comparison. If 0, then obtain just an scalar from the whole array for comparison. Note that the borders are likely to contain useless data.
:param algorithm: str, optional
    Algorithm to build the comparison. 
    If kernel==0, then any scipy function that receives a single  array argument and returns an scalar works.  
    Otherwise, the following kernels are implemented: 'median' filter, or convolution with a 'gaussian' (total size equals 5 times the specified sigma),  'boxcar'.
:param mask: ndarray, optional
    Initial mask
:param npass: int, optional
    Number of passes this function is run, the mask-out pixels are cumulative.  However, only the standard deviation to find sigmas is recomputed, the comparison is not.
:param full: bool, optional
    Return full statistics
:rtype: boolean ndarray
    A boolean sp.array with True on good pixels and False otherwise.
    If full==True, then it also return standard deviation and residuals
"""
    if not isinstance(arr,sp.ndarray):
        arr = sp.array(arr)

    krnfcn = {'mean': lambda n: sg.boxcar(n)/n,
              'gaussian': lambda n: sg.gaussian(n*5, n)}

    if kernel:
        if algorithm=='median':
            comparison = sg.medfilt(arr, kernel)
        else:
            comparison = sg.convolve(arr, krnfcn[algorithm](kernel), mode='same')
    else:
        try:
            comparison = getattr(sp, algorithm)(arr)
            if hasattr(comparison,'__len__'):
                raise AttributeError()
        except AttributeError:
            print("In function '%s', algorithm requested '%s' is not available from scipy to receive an array and return a comparison scalar " % (inspect.stack()[0][3], algorithm))

    residuals = arr - comparison

    if mask is None:
        mask = (arr*0 == 0)
    for i in range(npass):
        std = (residuals*mask).std(axis)
        mask *= sp.absolute(residuals) < sigmas*std

    if full:
        return mask, std, residuals
    return mask            




def zscale(img,  trim = 0.05, contr=1, mask=None):
    """Returns lower and upper limits found by zscale algorithm for improved contrast in astronomical images.

:param mask: bool ndarray
    True are good pixels, pixels marked as False are ignored
:rtype: (min, max)
    Minimum and maximum values recommended by zscale
"""

    if not isinstance(img, sp.ndarray):
        img = sp.array(img)
    if mask is None:
        mask = (sp.isnan(img) == False)

    itrim = int(img.size*trim)
    x = sp.arange(mask.sum()-2*itrim)+itrim

    sy = sp.sort(img[mask].flatten())[itrim:img[mask].size-itrim]
    a, b = sp.polyfit(x, sy, 1)

    return b, a*img.size/contr+b




def expandlims(xl,yl,offset=0):
    """Find x1,x2,y1,and y2 from the 2-item pairs xl and yl including some offset (negative is inwards, positive outwards)"""
    if (not isinstance(xl,(list,tuple))) or (not isinstance(xl,(list,tuple))) or len(xl)!=2 or len(yl)!=2:
        raise ValueError("xl and yl must each be 2-element list or tuple")
    dx = xl[1]-xl[0]
    dy = yl[1]-yl[0]
    return xl[0]-offset*dx,xl[1]+offset*dx, \
        yl[0]-offset*dy,yl[1]+offset*dy


def irafwav(h, axis=1):
    """Returns a wavelength array from the Iraf header keywords

:param h: header or primary HDU
:param axis: axis along the wavelength dispersion. If positive, then return data with the same size as FITS, otherwise just the desired dimension
"""

    if isinstance(h, pf.PrimaryHDU):
        header = h.header
    else:
        header = h

    nax = header["NAXIS"]
    dims = [header["NAXIS%i" % (ax+1,)] for ax in range(nax)]

    if axis<0:
        axis = -axis
        grid = sp.arange(dims[axis-1])
    elif nax == 1:
        grid = sp.arange(dims[0])
    elif nax == 2:
        grid = sp.mgrid[0:dims[0], 0:dims[1]][axis-1]
    elif nax == 3:
        grid = sp.mgrid[0:dims[0], 0:dims[1], 0:dims[2]][axis-1]

    wav_offset = header["crval%i" % (axis,)]
    wav_reference = header["crpix%i" % (axis,)]
    try: 
        delt = header["cdelt%i" % (axis,)]
    except KeyError:
        delt = header["cd%i_%i" %(axis, axis)]

    wav = delt*(grid + 1 - wav_reference) + wav_offset

    return wav



def fluxacross(diameter, seeing,
               shape='slit', psf='gauss', 
               nseeing=10, nsamp=300,
               show=False):
    """Return the fraction of flux that passes considering a particular block

    :param diameter: Diameter of the block
    :param seeing: seeing
    :param shape: shape of the block. Currently slit, square, circle 
    :param psf: PSF shape. Currently gauss, cube
    :param nseeing: how many times the seeing is the considered stamp size
    :param nsamp: divide the stamp in this many samples
    :param show: show a plot
"""
    import scipy as sp

    hseeing=seeing/2.0

    gtof=(sp.sqrt(2.0*sp.log(2.0)))
    gaussigma = hseeing/gtof
    
    psfshape={'gauss': lambda hseeing, ygrid, xgrid: (gtof/hseeing)**2/(2.0*sp.pi)*sp.exp(-(xgrid**2 + ygrid**2)/2.0/(hseeing/gtof)**2),
              'cube': lambda hseeing, ygrid, xgrid: (-hseeing<ygrid<hseeing)*(-hseeing<xgrid<hseeing)/4.0/hseeing**2,
              }

    rad= diameter/2.0
    block = {'slit': lambda rad, ygrid, xgrid: (-rad<xgrid)*(xgrid<rad),
             'square': lambda rad, ygrid, xgrid: (-rad<xgrid)*(xgrid<rad)*(-rad<ygrid)*(ygrid<rad),
             'circle': lambda rad, ygrid, xgrid: (xgrid**2+ygrid**2)<rad**2,
             }

    dy = dx = nseeing*seeing/2.0/nsamp
    y,x = sp.mgrid[-nsamp/2.0:nsamp/2.0,-nsamp/2.0:nsamp/2.0]
    y*=dy
    x*=dx

    psf = psfshape[psf](hseeing, y, x)
    blk = block[shape](rad, y, x)
    if show:
        import pylab as pl
        pl.clf()
        pl.imshow(psf)
        pl.contour(blk)

    return dy*dx*(psf*blk).sum()


def subarray(data, cyx, rad):
    """Reurns a subarray centered on cxy with radius rad
    :param arr: original array
    :type arr: array
    :param y: vertical center
    :type y: int
    :param x: horizontal center
    :type x: int
    :param rad: radius
    :type rad: int
    :rtype: array
    """

    return data[cyx[0]-rad:cyx[0]+rad, cyx[1]-rad:cyx[1]+rad]


def centroid(orig_arr, medsub=True):
    """Find centroid of small array
    
    :param arr: array
    :type arr: array
    :rtype: [float,float]
    """

    arr = copy.copy(orig_arr)
    if medsub:
        med = sp.median(arr)
        arr = arr - med
    arr = arr * (arr > 0)

    iy, ix = sp.mgrid[0:len(arr), 0:len(arr)]

    cy = sp.sum(iy * arr) / sp.sum(arr)
    cx = sp.sum(ix * arr) / sp.sum(arr)

    return cy, cx


def subcentroid(arr, cyx, stamprad, medsub=True, iters=1):
    """Returns the centroid after a number of iterations"""

    sub_array = arr
    cy, cx = cyx

    for i in range(iters):
        scy, scx = centroid(subarray(sub_array, [cy, cx], stamprad),
                            medsub=medsub)
        cy += scy - stamprad
        cx += scx - stamprad

    return cy, cx



def radial(data, cxy):
    """Return a same-dimensional array with the pixel distance to cxy"""

    ndim = data.ndim
    if len(cxy) != ndim:
        raise ValueError("Number of central coordinates (%i) does not match the data dimension (%i)" % (len(cxy), ndim))

    grid = sp.meshgrid(*[sp.arange(l) for l in data.shape])

    return sp.sqrt(sp.array([(dgrid-c)**2
                             for dgrid,c
                             in zip(grid,cxy)]
                            ).sum(0)
                   )





