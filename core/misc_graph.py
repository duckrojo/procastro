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
#import dataproc as dp
import astropy.time as apt
import datetime
import dataproc as dp
import dataproc.combine as cm
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pyfits as pf
import scipy as sp

def plot_accross(data,
                 axes=None, title=None,
                 ytitle=None, xtitle=None,
                 xlim=None, ylim=None,
                 ticks=True, colorbar=False,
                 hdu=0, 
                 rotate=0,
                 pos=0,
                 forcenew=False,
                 **kwargs):
    print(ylim)

    data = prep_data_plot(data, hdu)

    dt = data
    if not isinstance(pos, (list, tuple)):
        pos = [None] + [0]*(len(data.shape)-2) + [pos] 
    if len(pos) != len(data.shape):
        raise TypeError("pos (size: %i) must have the same size as data array dimension (%i)" % (len(pos), len(data.shape)))
    pos = tuple([p is None and slice(None, None) or p for p in pos])

    print(pos)

    accross = data[pos]

    fig, ax = prep_canvas(axes, title, xtitle, ytitle)

    if xlim is not  None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.plot(accross, label="ll")

    return accross, data


def prep_data_plot(indata, hdu=0):
    #set the data to plot
    if isinstance(indata, pf.hdu.base._BaseHDU):
        data = indata.data
        error_msg = ""
    elif isinstance(indata, dp.AstroFile):
        error_msg = "HDU %i empty?\n available: %s" % (hdu, indata.reader(hdu=-1),)
        data = indata.reader(hdu)
    elif isinstance(indata, cm.Combine):
        error_msg = ""
        data = indata.data
    elif isinstance(indata, pf.HDUList):
        error_msg = "HDU %i empty?\n available: %s" % (hdu, indata,)
        data = indata[hdu].data
    elif isinstance(indata, basestring):
        open_file  = pf.open(indata)
        data = open_file[hdu].data
        error_msg = "HDU %i empty?\n available: %s" % (hdu, open_file,)
    elif isinstance(indata, sp.ndarray):
        data = indata
    else:
        raise TypeError("Unrecognized type for input data: %s" % (indata.__class__,))
    if data is None:
        raise ValueError("Nothing to plot for HDU %i. %s" % (hdu, error_msg))

    return data



def prep_canvas(axes=None, title=None,
                ytitle=None, xtitle=None,
                forcenew=False,
                ):


    #set the canvas
    fig, ax = figaxes(axes, forcenew)

    if title is not None:
        ax.set_title(title)
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    if xtitle is not None:
        ax.set_xlabel(xtitle)

    return fig, ax



def imshowz(data, 
            axes=None, title=None,
            ytitle=None, xtitle=None,
            minmax=None, xlim=None, ylim=None,
            cxy=None, plot_rad=None,
            ticks=True, colorbar=False,
            hdu=0, 
            rotate=0, invertx=False, inverty=False,
            origin='lower', forcenew=False,
            trim_data=False,
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

    data = prep_data_plot(data, hdu)

    if xlim is None:
        xlim = [0, data.shape[1]]
    if ylim is None:
        ylim = [0, data.shape[0]]


    if rotate:
        times = rotate/90
        if times % 1 != 0:
            raise ValueError("rotate must be a multiple of 90")
        data = sp.rot90(data, int(times))

    if invertx:
        data = data[:, ::-1]

    if inverty:
        data = data[::-1, :]

    if cxy is not None:
        border_distance = [data.shape[1]-cxy[0], cxy[0], data.shape[0]-cxy[1], cxy[1]]
        if plot_rad is None:
            plot_rad = min(border_distance)
        xlim = [cxy[0]-plot_rad, cxy[0]+plot_rad]
        ylim = [cxy[1]-plot_rad, cxy[1]+plot_rad]
        xlim[0] = xlim[0]*(xlim[0]>0)
        xlim[1] = (xlim[1] > data.shape[1]) and data.shape[1] or xlim[1]
        ylim[0] = ylim[0]*(ylim[0]>0)
        ylim[1] = (ylim[1] > data.shape[0]) and data.shape[0] or ylim[1]

    if trim_data:
        data = data[ylim[0]:ylim[1],xlim[0]:xlim[1]]
        xlim = [0, data.shape[1]-1]
        ylim = [0, data.shape[0]-1]

    #Find the contrast
    if minmax is None:
        mn, mx = dp.zscale(data[ylim[0]:ylim[1], xlim[0]:xlim[1]])
    else:
        mn, mx = minmax

    fig, ax = prep_canvas(axes, title, ytitle, xtitle)

    #draw in the canvas
    imag = ax.imshow(data, vmin=mn, vmax=mx, origin=origin, **kwargs)
    if not ticks:
        ax.xaxis.set_ticklabels([' ']*20)
        ax.yaxis.set_ticklabels([' ']*20)
    if colorbar:
        fig.colorbar(imag)


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()
    return mn, mx


def axesfig_xdate(axes, x, overwrite=False):
    """Returns the figure and axes with a properly formatted date X-axis"""

    f,ax = dp.figaxes(axes, overwrite=overwrite)
    if isinstance(x, (apt.Time, datetime.datetime)):
        if isinstance(x, apt.Time):
            x = x.plot_date
        ax.xaxis_date()
        tdelta = (x[-1]-x[0])*24*60
        if tdelta < 4: #if range is smaller than 4 minutes
            fmt = '%H:%M:%S'
        elif tdelta < 4*60: #if range is smaller than 4 hours
            fmt = '%H:%M'
        elif tdelta < 2*60*24: #if range is smaller than 2 days
            fmt = '%Y-%b-%d %H:%M'
        elif tdelta < 1*60*24*365: #if range is smaller than 1 years
            fmt = '%Y %b'
        else:
            fmt = '%Y'
        ax.xaxis.set_major_formatter(md.DateFormatter(fmt))

    return f,ax,x


def figaxes(axes=None, forcenew=True, overwrite=False):
    """Function that accepts a variety of canvas formats and returns the output ready for use with matplotlib 
    :param axes:
    :type axes: int, plt.Figure, plt.Axes
    :param forcenew: If true starts a new axes when axes=None instead of using last figure
    :type forcenew: boolean
    :rtype: figure, axes
"""
    if axes is None:
        if forcenew:
            fig, ax = plt.subplots()
        else:
            plt.gcf().clf()
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

    if not overwrite:
        ax.cla()

    return fig, ax



def polygonxy(cxy, rad, npoints=20):
    angles = sp.arange(npoints+1)*2*3.14159/npoints
    xx = cxy[0] + rad*sp.cos(angles)
    yy = cxy[1] + rad*sp.sin(angles)

    return xx,yy
