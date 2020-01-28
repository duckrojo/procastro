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

__all__ = ['sigmask', 'zscale', 'expandlims', 'axis_from_fits',
           'fluxacross', 'subarray',
           'centroid', 'subcentroid', 'subcentroidxy',
           'azimuth', 'radial', 'radial_profile',
           ]

import numpy as np
import inspect
import scipy.signal as sg
import astropy.io.fits as pf
import copy
from IPython.core.debugger import Tracer

from .misc_general import sortmanynsp

def subarray(data, cyx, rad, padding=False, return_origpos=False):
    """
    Returns a subarray centered on 'cxy' with radius 'rad'
    
    Parameters
    ----------
    arr : array_like
        Original array
    cxy : tuple
        Center coordinates
    rad : int 
        radius
    padding: bool, optional
        Returns a zero padded subarray when the requested stamp does not 
        completely fit inside the array.
    
    Returns
    -------
    origpos: tuple
        Returns the original position of the subarray in the original array, 
        it could be negative when padding if reached the origin of array
    """

    icy = int(cyx[0])
    icx = int(cyx[1])

    orig_y = (icy-rad>0)*(icy-rad)
    orig_x = (icx-rad>0)*(icx-rad)
    
    ret0 = data[orig_y:icy+rad+1,
                orig_x:icx+rad+1]
    if 0 in ret0.shape:
        print("Warning: subarray with an empty dimension: {ret0.shape}")
    ny, nx = data.shape
    stamp = 2*rad+1
    y1 = (rad - icy > 0) * (rad - icy)
    x1 = (rad - icx > 0) * (rad - icx)
    if padding and (ret0.shape[0] < stamp or ret0.shape[1] < stamp):
        ret = np.zeros([2*rad+1, 2*rad+1])
        y2 = stamp + (ny - icy - rad - 1 < 0) * (ny - icy - rad - 1)
        x2 = stamp + (nx - icx - rad - 1 < 0) * (nx - icx - rad - 1)
        ret[y1:y2, x1:x2] = ret0
        orig_y -= y1
        orig_x -= x1
    else:
        ret = ret0


    if return_origpos:
        return ret, (orig_y, orig_x)
    else:
        return ret


def centroid(orig_arr, medsub=True):
    """
    Find centroid of small array
    
    Parameters
    ----------
    orig_arr : array_like
    med_sub : bool, optional
        If set, substracts the median from the array
    
    Returns
    -------
    float, float :
        Center coordinates
    """

    arr = copy.copy(orig_arr)
    if medsub:
        med = np.median(arr)
        arr = arr - med
    arr = arr * (arr > 0)

    iy, ix = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]

    cy = np.sum(iy * arr) / np.sum(arr)
    cx = np.sum(ix * arr) / np.sum(arr)

    return cy, cx


def subcentroid(arr, cyx, stamprad, medsub=True, iters=1):
    """
    Returns the centroid after a number of iterations
    
    Parameters
    ----------
    arr : array_like
    cyx : tuple
        Center coordinates
    stamprad : int
        Stamp radius
    medsub : bool, optional
        If True, substracts median from array
    iters : int, optional
        Number of times this procedure is repeated
        
    Returns
    -------
    float, float
        Subcentroid coordinates
    """

    sub_array = arr
    cy, cx = cyx

    for i in range(iters):
        sub_array, (offy, offx) = subarray(sub_array, [cy, cx], stamprad,
                                           padding=False, return_origpos=True)
        
        scy, scx = centroid(sub_array, medsub=medsub)
        cy = scy + offy
        cx = scx + offx
        
    return cy, cx

def subcentroidxy(arr, cxy, stamprad, medsub=True, iters=1):
    """
    Returns the centroid after a number of iterations, order by xy
    
    Parameters
    ----------
    arr : array_like
    cyx : tuple
        Center coordinates
    stamprad : int
        Stamp radius
    medsub : bool, optional
        If True, substracts median from array
    iters : int, optional
        Number of times this procedure is repeated
        
    Returns
    -------
    float, float
        Subcentroid coordinates
    
    """
    cy,cx = subcentroid(arr,[cxy[1],cxy[0]], stamprad,
                        medsub=medsub, iters=iters)
    return cx,cy

def radial(data, cyx):
    """
    Return a same-dimensional array with the pixel distance to cxy
    
    Parameters
    ----------
    data : array_like 
        Data to get the shape from
    cyx: tuple  
        Center in native Y-X coordinates
    
    Returns
    -------
    array_like
    """

    ndim = data.ndim
    if len(cyx) != ndim:
        raise ValueError("Number of central coordinates (%i) does not match the data dimension (%i)" % (len(cyx), ndim))

    grid = np.meshgrid(*[np.arange(l) for l in data.shape], indexing='ij')

    return np.sqrt(np.array([(dgrid-c)**2
                             for dgrid, c
                             in zip(grid, cyx)]
                            ).sum(0)
                   )


def radial_profile(data, cnt_xy=None, stamp_rad=None, recenter=False):
    """
    Returns the x&y arrays for radial profile

    Parameters
    ----------
    data : array_like
    cnt_xy : tuple
        Center coordinates
    stamp_rad : int, optional
        Stamp radius
    recenter: bool, optional
        Whether to recenter
    
    Returns
    -------
    array_like :
        (x-array,y-array, [x,y] center)
    """

    ny, nx = data.shape

    #use data's center if not given explicit
    if cnt_xy is None:
        cx, cy = nx//2, ny//2
    else:
        cx, cy = cnt_xy


    #use the whole data range if no stamp radius is given
    if stamp_rad is None:
        to_show = data
    else:
        to_show = subarray(data, [cy, cx], stamp_rad)

    if recenter:
        cy, cx = centroid(to_show)
        if stamp_rad is not None:
            to_show = subarray(data, [cy, cx], stamp_rad)

    d = radial(to_show, [cy, cx])
    x, y = sortmanynsp(d.flatten(), to_show.flatten())

    return x, y, (cx, cy)

def zscale(img,  trim = 0.05, contr=1, mask=None):
    """
    Returns lower and upper limits found by zscale algorithm for improved 
    contrast in astronomical images.

    Parameters
    ----------
    img : array_like
        Image to scale
    trim : float, optional
    contr : int, optional
    mask: bool ndarray
        True are good pixels, pixels marked as False are ignored

    Returns
    -------
    tuple :
        Minimum and maximum values recommended by zscale
    """

    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if mask is None:
        mask = (np.isnan(img) == False)

    itrim = int(img.size*trim)
    x = np.arange(mask.sum()-2*itrim)+itrim

    sy = np.sort(img[mask].flatten())[itrim:img[mask].size-itrim]
    a, b = np.polyfit(x, sy, 1)

    return b, a*img.size/contr+b
    
###############################################################################
# Unused methods
######

def expandlims(xl,yl,offset=0):
    """
    Find x1,x2,y1,and y2 from the 2-item pairs xl and yl including some offset 
    (negative is inwards, positive outwards)
    
    Parameters
    ----------
    xl, yl : int or tuples, optional
    
    Returns
    -------
    """
    if (not isinstance(xl,(list,tuple))) or (not isinstance(xl,(list,tuple))) or len(xl)!=2 or len(yl)!=2:
        raise ValueError("xl and yl must each be 2-element list or tuple")
    dx = xl[1]-xl[0]
    dy = yl[1]-yl[0]
    return xl[0]-offset*dx,xl[1]+offset*dx, \
        yl[0]-offset*dy, yl[1]+offset*dy


def axis_from_fits(h, axis=1):
    """
    Returns a wavelength array from the standard FITS header keywords

    Parameters
    ----------
    h : int 
        Header or primary HDU
    axis : int, optional
        Axis along the wavelength dispersion. If positive, then return data 
        with the same size as FITS, otherwise just the desired dimension
    
    Returns
    -------
    array_like
    """

    if isinstance(h, pf.PrimaryHDU):
        header = h.header
    else:
        header = h

    nax = header["NAXIS"]
    dims = [header["NAXIS%i" % (ax+1,)] for ax in range(nax)]

    if axis<0:
        axis = -axis
        grid = np.arange(dims[axis-1])
    elif nax == 1:
        grid = np.arange(dims[0])
    elif nax == 2:
        grid = np.mgrid[0:dims[0], 0:dims[1]][axis-1]
    elif nax == 3:
        grid = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]][axis-1]

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
    """
    Return the fraction of flux that passes considering a particular block

    Parameters
    ----------
    diameter : 
        Diameter of the block
    seeing : 
        Seeing
    shape : str, optional
        Shape of the block. Currently slit, square, circle 
    psf: str, optional
        PSF shape. Currently gauss, cube
    nseeing : int, optional
        How many times the seeing is the considered stamp size
    nsamp : int, optional 
        Divide the stamp in this many samples
    show : bool, optional 
        Show a plot
        
    Returns
    -------
    
    """
    import scipy as sp

    hseeing=seeing/2.0

    gtof=(np.sqrt(2.0*np.log(2.0)))
    gaussigma = hseeing/gtof
    
    psfshape={'gauss': lambda hseeing, ygrid, xgrid: (gtof/hseeing)**2/(2.0*np.pi)*np.exp(-(xgrid**2 + ygrid**2)/2.0/(hseeing/gtof)**2),
              'cube': lambda hseeing, ygrid, xgrid: (-hseeing<ygrid<hseeing)*(-hseeing<xgrid<hseeing)/4.0/hseeing**2,
              }

    rad= diameter/2.0
    block = {'slit': lambda rad, ygrid, xgrid: (-rad<xgrid)*(xgrid<rad),
             'square': lambda rad, ygrid, xgrid: (-rad<xgrid)*(xgrid<rad)*(-rad<ygrid)*(ygrid<rad),
             'circle': lambda rad, ygrid, xgrid: (xgrid**2+ygrid**2)<rad**2,
             }

    dy = dx = nseeing*seeing/2.0/nsamp
    y,x = np.mgrid[-nsamp/2.0:nsamp/2.0,-nsamp/2.0:nsamp/2.0]
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




def azimuth(data, cyx):
    """
    Return a same-dimensional array with the azimuth value of each pixel with 
    respect to 'cxy'
    
    Parameters
    ----------
    data: data to get the shape from. It has to be 2D
    cyx:  Center in native Y-X coordinates
    
    Returns
    -------
    float
        Azimuthal values in radians, setting 0 upwards.
    """

    ndim = data.ndim
    if ndim != 2:
        raise ValueError("Input array must be 2-D in order to get azimuthal values. Shape: {data}".format(data=data.shape))
    if len(cyx) != ndim:
        raise ValueError("Number of central coordinates (%i) does not match the data dimension (%i)" % (len(cyx), ndim))

    yy, xx = np.mgrid[0:data.shape[0],0:data.shape[1]]

    return np.arctan2(yy-cyx[0], xx-cyx[1])

###############################################################################
# Deprecated methods
######


def sigmask(arr, sigmas, axis=None, kernel=0, algorithm='median', npass=1, mask=None, full=False):
    """
    Returns a mask with those values that are 'sigmas'-sigmas beyond the mean 
    value of arr.

    Parameters
    ----------
    arr : array_like
    sigmas: int
        Variation beyond this number of sigmas will be masked.
    axis : int, optional
        Look for the condition along an axis, mark those. 
        None is the full array.
    kernel : int, optional (some algorithms accepts ndarray)
        Size of the kernel to build the comparison. If 0, then obtain just 
        an scalar from the whole array for comparison. 
        Note that the borders are likely to contain useless data.
    algorithm : str, optional
        Algorithm to build the comparison. 
        If kernel==0, then any scipy function that receives a single array 
        argument and returns an scalar works.  
        Otherwise, the following kernels are implemented: 'median' filter, 
        or convolution with a 'gaussian' (total size equals 5 times the 
        specified sigma),  'boxcar'.
    mask : ndarray, optional
        Initial mask
    npass : int, optional
        Number of passes this function is run, the mask-out pixels are 
        cumulative. However, only the standard deviation to find sigmas is 
        recomputed, the comparison is not.
    full : bool, optional
        Return full statistics
        
    Returns
    -------
    bool ndarray
        A boolean np.array with True on good pixels and False otherwise.
        If full==True, then it also return standard deviation and residuals
    """
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)

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
        mask *= np.absolute(residuals) < sigmas*std

    if full:
        return mask, std, residuals
    return mask            
