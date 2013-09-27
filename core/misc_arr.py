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




def sigmask(arr, sigmas, axis=None):
    """Returns a mask with those values that are 'sigmas'-sigmas beyond the mean value of arr. Optionally when they accomplish that condition along an axis 'axis'. Returns True for good elements"""
    import scipy as sp
    if not isinstance(arr,ndarray):
        arr = sp.array(arr)

    mn = arr.std(axis)
    st = arr.std(axis)
    mask = sp.absolute(arr-mn) > sigmas*st

    return mask            




def zscale(img,  trim = 0.05, contr=1):
    """Returns lower and upper limits found by zscale algorithm for improved contrast in astronomical images."""

    import scipy as sp

    itrim = int(img.size*trim)
    a,b = sp.polyfit(sp.arange(img.size-2*itrim)+itrim,sp.sort(sp.array(img).flatten())[itrim:img.size-itrim],1)

    return b,a*img.size/contr+b




def expandlims(xl,yl,offset=0):
    """Find x1,x2,y1,and y2 from the 2-item pairs xl and yl including some offset (negative is inwards, positive outwards)"""
    if (not isinstance(xl,(list,tuple))) or (not isinstance(xl,(list,tuple))) or len(xl)!=2 or len(yl)!=2:
        raise ValueError("xl and yl must each be 2-element list or tuple")
    dx = xl[1]-xl[0]
    dy = yl[1]-yl[0]
    return xl[0]-offset*dx,xl[1]+offset*dx, \
        yl[0]-offset*dy,yl[1]+offset*dy


def fluxacross(diameter, seeing,
               shape='slit', psf='gauss', 
               nseeing=10, nsamp=300,
               show=False):
    """Return the fraction of flux that passes across a block"""
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
