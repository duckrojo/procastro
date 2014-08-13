#
#
# Copyright (C) 2014 Patricio Rojo
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


from __future__ import division, print_function
import dataproc as dp
import matplotlib.pyplot as plt
import pyfits as pf
import scipy as sp

def imshowz(data, 
            axes=None, title=None,
            ytitle=None, xtitle=None,
            minmax=None, xlim=None, ylim=None,
            cxy=None, plot_rad=None,
            ticks=True, colorbar=False,
            hdu=0,
            origin='lower', forcenew=False,
            **kwargs):
    """Plots data using zscale algorithm to fix the min and max values

:param data: Data to  be plotted
:type data: string, HDU, HDUList, sp.array
:param axes: Where to plot
:type axes: Figure, Axes, integer (figure in which to plot)
:param title: Figure Title
:type title: string
:param ytitle: Figure YTitle
:type ytitle: string
:param xtitle: Figure XTitle
:type xtitle: string
:param minmax: Force this value of contrast
:type minmax: 2-element list or tuple
:param ticks: whether to display the ticks
:type ticks: boolean
:param colorbar: whether to use a colorbar
:type colorbar: boolean 
:param hdu: Which hdu to plot (only relevant if data is string or HDUList
:type hdu: int
:param origin: Option of the same name given to imshow
:type origin: string
:param forcenew: whether to create a new plot if no axis has been specified
:type forcenew: boolean
:param cxy: Center at this coordinate and use plot_rad as radius.  If both xlim and cxy are specified, then only cxy is considered.
:type cxy: 2-element tuple or list
:param plot_rad: Radius of plotting area. Only relevant if cxy is not None. If None then it is the minimum distance to an image border.
:type plot_rad: integer
:param kwargs: passed to matplotlib.pyplot.imshow()
"""

    #set the data to plot
    if isinstance(data, pf.hdu.base._BaseHDU):
        data = data.data
        avail = ""
    elif isinstance(data, dp.AstroFile):
        data = data.reader(hdu)
    elif isinstance(data, pf.HDUList):
        avail = "available: %s" % (data,)
        data = data[hdu].data
    elif isinstance(data, basestring):
        open_file  = pf.open(data)
        data = open_file[hdu].data
        avail = "available: %s" % (open_file,)
    if data is None:
        raise ValueError("Nothing to print. HDU %i empty?\n %s" % (hdu, avail))

    if xlim is  None:
        xlim = [0,data.shape[1]]
    if ylim is None:
        ylim = [0,data.shape[0]]

    if cxy is not None:
        border_distance = [data.shape[1]-cxy[0], cxy[0], data.shape[0]-cxy[1], cxy[1]]
        if plot_rad is None:
            plot_rad = min(border_distance)
        xlim = [cxy[0]-plot_rad, cxy[0]+plot_rad]
        ylim = [cxy[1]-plot_rad, cxy[1]+plot_rad]
        xlim[0] = xlim[0]*(xlim[0]>0)
        xlim[1] = (xlim[1]>data.shape[1]) and data.shape[1] or xlim[1]
        ylim[0] = ylim[0]*(ylim[0]>0)
        ylim[1] = (ylim[1]>data.shape[0]) and data.shape[0] or ylim[1]


    #Find the contrast
    if minmax is None:
        mn,mx = dp.zscale(data[ylim[0]:ylim[1],xlim[0]:xlim[1]])
    else:
        mn,mx = minmax


    #set the canvas
    fig, ax = dp.axesfig(axes, forcenew)

    #draw in the canvas
    if title is not None:
        ax.set_title(title)
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    if xtitle is not None:
        ax.set_xlabel(xtitle)
    imag = ax.imshow(data, vmin=mn, vmax=mx, origin=origin, **kwargs)
    if not ticks:
        ax.xaxis.set_ticklabels([' ']*20)
        ax.yaxis.set_ticklabels([' ']*20)
    if colorbar:
        fig.colorbar(imag)


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.show()
    return mn,mx



def axesfig(axes=None, forcenew=True):
    """Function that accepts a variety of canvas formats and returns the output ready for use with matplotlib 
    :param axes:
    :type axes: int, plt.Figure, plt.Axes
    :param forcenew: If true starts a new axes when axes=None instead of using last figure
    :type forcenew: boolean
"""
    if axes is None:
        if forcenew:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(num=plt.gcf().number)
    elif isinstance(axes, int):
        fig, ax = plt.subplots(num=axes)
    elif isinstance(axes, plt.Figure):
        ax = axes.add_subplot(111)
        fig = axes
    elif isinstance(axes, plt.Axes):
        ax = axes
        fig = axes.figure
    else:
        raise ValueError("Given value for axes (%s) is not recognized" 
                         % (axes,))

    ax.cla()

    return fig, ax



def polygonxy(cxy, rad, npoints=20):
    angles = sp.arange(npoints+1)*2*3.14159/npoints
    xx = cxy[0] + rad*sp.cos(angles)
    yy = cxy[1] + rad*sp.sin(angles)

    return xx,yy
