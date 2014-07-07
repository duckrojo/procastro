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

from __future__ import print_function,division

import astrocalc
import astroplot
import matplotlib.pyplot as plt
astrocalc=reload(astrocalc)
astroplot=reload(astroplot)

from astropy.time import Time

import copy
import scipy as sp
import dataproc as dp
import warnings


class Timeseries(astrocalc.AstroCalc):

    """Timeseries class inherited from AstroCalc class.The purpose of this class is to organize all the calculation tasks related to the timeseries analysis.

    """

    def __init__(
            self,
            data,
            coordsxy=None,
            labels=None,
            stamprad=20,
            maxskip=6,
            epoch='JD',
            exptime='EXPTIME',
            keytype='IMAGETYP',
            ingain='GTGAIN11',
            inron='GTRON11',
            mastermode='median'):
        """Timeseries object constructor.

        :param data: image data for the timeseries
        :type data: str (path to a folder), astrodir, ndarray list (list of images)
        :param coordsxy: list of x,y coordinates of stars in the first image of the timeseries (if the value is None an interface for coordinates selection will be displayed)
        :type coordsxy: list of 2-tuples or 2-list
        :param labels: labels for the stars
        :type labels: list of strings
        :param stamprad: radius of the (square) stamp that contains a star
        :type stamprad: int (default value: 20)
        :param maxskip: maximum skip considered (without warning) between the positions of a star in two sucesive images
        :type maxskip: int (default value: 6)
        :param epoch: header key of observation date in the images (allows image sorting in terms of time) or list
        :type epoch: str or list
        :param keytype: header key of image type ('OBJECT','BIAS','DOME FLAT',etc)
        :type keytype: str
        :param mastermode: mode for master (BIAS, DARK, FLAT, etc) image obtention. Options: 'mean' and 'median' (default).
        :type mastermode: str
        :rtype: Timeseries
        """

        # data type check
        if isinstance(data, str):  # data is a string (path to a directory)
            self.files = dp.astrodir(data)
            if isinstance(epoch,basestring):
                self.files = self.files.sort(epoch)
            self.isAstrodir = True

        elif isinstance(data, dp.astrodir):  # data is an astrodir object
            self.files = data
            if isinstance(epoch,basestring):
                self.files = self.files.sort(epoch)
            self.isAstrodir = True

        elif isinstance(data, list):  # data is a list of images (ndarray)
            if len(data) == 0:
                raise TypeError('data is an empty list')
            for img in data:
                if not isinstance(img, sp.ndarray):
                    raise TypeError(
                        'data is a list but not all the elements are ndarray')
            if isinstance(epoch,(list,tuple,sp.ndarray)):
                self.epoch = epoch
            else:
                warnings.warn("Epochs were not specified and list of data was given")
            if isinstance(exptime,(list,tuple)):
                self.exptime = exptime
            else:
                warnings.warn("Exposure time was not specified and list of data was given")
            self.files = data
            self.isAstrodir = False

        else:
            raise TypeError('data should be str, astrodir or ndarray list')

        # mjd list and science image filtering
        if self.isAstrodir:

            # time_file_list = []
            # bias_files = []
            # dark_files = []
            # flat_files = []

            self.epoch = self.files.getheaderval(epoch)
            self.exptime = self.files.getheaderval(exptime)

#             for astrofile in self.files:
# #                type_str = astrofile.getheaderval(keytype)[0]
# #                if 'OBJECT' in type_str:
#                 date_str = astrofile.getheaderval(keydate)[0]
#                 date_format = "iso"
#                 if "T" in date_str or "t" in date_str:
#                     date_format += "t"
#                 mjd_val = Time(
#                     date_str,
#                     format=date_format,
#                     scale='utc').mjd
#                 time_file_list.append((mjd_val, astrofile))
#                 # elif 'BIAS' in type_str:
#                 #     data, head = astrofile.reader(datahead=True)
#                 #     bias_files.append(data)
#                 #     print("%s is used for MASTERBIAS" % astrofile)
#                 # elif 'DARK' in type_str:
#                 #     data, head = astrofile.reader(datahead=True)
#                 #     dark_files.append(data)
#                 #     print("%s is used for MASTERDARK" % astrofile)
#                 # elif 'FLAT' in type_str:
#                 #     data, head = astrofile.reader(datahead=True)
#                 #     flat_files.append(data)
#                 #     print("%s is used for MASTERFLAT" % astrofile)
#                 # else:
#                 #     print("%s is not used" % astrofile)

# #            time_file_list.sort()  # Bug correction of dataproc (BUG????? TODO)
#             print("\n%s is the FIRST file of the timeseries (initial star coordinates should consider this file)\n" % time_file_list[0][1])
#             self.mjd = [mjd for mjd, astrofile in time_file_list]
#             self.exptime = [expt for expt, astrofile in expt_file_list]
#             self.files = [astrofile for mjd, astrofile in time_file_list]
#             self.masterbias = self.masterimage(bias_files, mode=mastermode)
#             self.masterdark = self.masterimage(dark_files, mode=mastermode)
#             self.masterflat = self.masterimage(flat_files, mode=mastermode)

        if not hasattr(self,'epoch'):
            self.epoch = sp.arange(len(data))
        if not hasattr(self, 'exptime'):
            self.exptime = sp.ones(len(data))

        print (" Data list ready for %i elements" % (len(self.files),))

        # # coordsxy check
        # if coordsxy is None:
        #     from astrointerface import AstroInterface
        #     print("No coordinates provided: beginning interface mode")
        #     if self.isAstrodir:
        #         astrofile = self.files[0]
        #         data, head = astrofile.reader(datahead=True)
        #     else:
        #         data = self.files[0]
        #     coordsxy = AstroInterface(data, maxsize=650).execute()
        #     print("Selected coordinates: %s" % str(coordsxy))
        #     print("Interface mode finished\n")

        # label list
        try:
            if labels is None:
                labels = []
            nstars = len(coordsxy)
            if len(labels) > nstars:
                labels = labels[:nstars]
            elif len(labels) < nstars:
                labels = list(
                    labels) + sp.arange(len(labels),
                                        nstars).astype(str).tolist()
            targetsxy = {lab: coo
                         for coo, lab in zip(coordsxy, labels)}
        except:
            raise ValueError("Coordinates of target stars need to be "+
                             "specified as a list of 2 elements, not: %s" %
                             (str(coordsxy),))
        print (" Initial guess received for %i targets: %s" %
               (len(coordsxy),
                ", ". join(["%s %s" % (lab,coo) 
                            for lab,coo in zip(labels,coordsxy)])
                ))

        # instance variables (in addition to self.files, self.isAstrodir and
        # self.mjd,self.masterbias, self.masterbias, self.masterflat)
        self.labels = labels
        self.targetsxy = targetsxy
        self.stamprad = stamprad
        self.maxskip = maxskip
        self.skydata = None
        self.gain = ingain
        self.ron = inron



    def perform_phot(
            self,
            aperture,
            sky=None,
            deg=1):
        """Perform apperture photometry with the images of the Timeseries object.

        :param aperture: aperture photometry radius
        :type aperture: int
        :param sky: inner and outer radius for sky annulus
        :type sky: [int,int]
        :param deg: Degre to sky polynomial
        :type deg: int
        :rtype: TimeseriesResults
        """

        if sky is None and self.skydata is None:
            raise ValueError(
                'No sky information available: sky and self.skydata are None')

        flx = {lab: [] for lab in self.labels}
        err = {lab: [] for lab in self.labels}
        fwhms = {lab: [] for lab in self.labels}

        skydata = []
        self.skip = []
        targetsxy = {lab:[coo] for lab, coo in self.targetsxy.items()}

        for rdata, i in zip(self.files, range(len(self.files))):
            if sky is not None:
                sky_img_dict = {lab: [] for lab in self.labels}

            gain = self.gain
            ron = self.ron
            if isinstance(rdata, dp.astrodir):
                if isinstance(gain,basestring):
                    gain = rdata.getheaderval(self.gain)
                if isinstance(ron, basestring):
                    ron  = rdata.getheaderval(self.ron)
                data, head = rdata.reader(datahead=True)
                ##todo: offer datareduction?
                # data = self.imagereduction(
                #     data,
                #     self.masterbias,
                #     self.masterflat,
                #     self.masterdark)
            else:
                data = rdata

            if isinstance(gain, basestring) or isinstance(ron, basestring):
                raise ValueError("Gain or RON specified as header (%s/%s), but not found: %s/%s" % (ingain,inron,str(gain),str(ron)))

            for lab, cooxy in targetsxy.items():
                cx, cy = cooxy[-1]
                sarr = self.subarray(data, cy, cx, self.stamprad)
                scy, scx = self.centroid(sarr)

                skip = sp.sqrt((self.stamprad - scy) ** 2 +
                               (self.stamprad - scx) ** 2)
                self.skip.append(skip)
                if skip > self.maxskip:
                    print(("Jump of %f pixels has occurred on"+
                           " frame %i for star %s") %
                          (skip, i, lab))

                if sky is not None:
                    phot, phot_err, fwhm, sky_out = self.apphot(
                        sarr, [scy, scx], aperture, sky, gain=gain, ron=ron, deg=deg)
                    sky_img_dict[lab] = sky_out
                else:
                    sky_info = self.skydata[i][lab]
                    phot, phot_err, fwhm, sky_out = self.apphot(
                        sarr, [scy, scx], aperture, sky_info, gain=gain, ron=ron, deg=deg)

                flx[lab].append(phot)
                err[lab].append(phot_err)
                fwhms[lab].append(fwhm)
                cooxy.append([cx + scx - self.stamprad,
                              cy + scy - self.stamprad])

            if sky is not None:
                skydata.append(sky_img_dict)

        if sky is not None:
            self.skydata = skydata

        return TimeseriesResults(self.epoch, flx, err, targetsxy, self.exptime, fwhms)


class TimeseriesExamine(astroplot.AstroPlot, astrocalc.AstroCalc):

    def __init__(self, timeseries):
        """GRaphical utilities for timeseries
        :param timeseries: timeserie object to examine
        :type timeseries: Timeserie object
        """
        self.ts = timeseries

        
    def plot_radialprofile(self, targets=None, xlim=None, axes=1,
                           legend_size=None,
                           **kwargs):
        """Plot Radial Profile from data using radialprofile() function
        :param target: Target spoecification for recentering. Either an integer for specifc target, or a 2-element list for x/y coordinates.
        :type target: integer/string or 2-element list
        """


        colors = ['rx', 'b^', 'go', 'r^', 'bx', 'g+']
        fig, ax = dp.axesfig(axes)

        ax.cla()
        ax.set_xlabel('distance')
        ax.set_ylabel('ADU')
        if targets is None:
            targets = self.ts.targetsxy.keys()
        elif isinstance(targets, basestring):
            targets = [targets]
        elif isinstance(targets, (list,tuple)) and \
                not isinstance(targets[0], (basestring, list, tuple)):
                #Assume that it is a coordinate
            targets = [targets]

        trgcolor = {str(trg): color for trg, color in zip(targets, colors)}
        for trg in targets:
            distance, value, center = self.radialprofile(trg, **kwargs)
            ax.plot(distance, value, trgcolor[str(trg)],
                    label = "%s: (%.1f, %.1f)" % (trg, 
                                                  center[0],
                                                  center[1]),
                    )
        prop={}
        if legend_size is not None:
            prop['size'] = legend_size
        ax.legend(loc=1, prop=prop)

        if xlim is not None:
            if isinstance(xlim, (int,float)):
                ax.set_xlim([0,xlim])
            else:
                ax.set_xlim(xlim)


        
    def radialprofile(self, target, frame=0, recenter=True, stamprad=20):
        """Returns the x&y arrays for radial profile

        :param target: Target spoecification for recentering. Either an integer for specifc target, or a 2-element list for x/y coordinates.
        :type target: integer/string or 2-element list
        :param frame: which frame to show
        :type frame: integer
        :param recenter: whether to recenter
        :type recenter: bool
        :rtype: (x-array,y-array, [x,y] center) 
"""
        if isinstance(target, (int, str)):
            try:
                cx, cy = self.ts.targetsxy[target]
            except KeyError:
                raise KeyError("Invalid target specification. Choose from '%s'" % ', '.join(self.ts.targetsxy.keys()))
        elif isinstance(target, (list,tuple)):
            cx, cy = target
        else:
            print("Invalid coordinate specification '%s'" % (target,))

        stamp = self.ts.files[frame][cy-stamprad: cy+stamprad, 
                                     cx-stamprad: cx+stamprad]

        if recenter:
            cy, cx = self.ts.centroid(stamp) + sp.array([cy,cx]) - stamprad
            stamp = self.ts.files[frame][int(cy-stamprad): int(cy+stamprad), 
                                         int(cx-stamprad): int(cx+stamprad)]

        d = self.centraldistances(stamp, 
                                  sp.array(stamp.shape)//2 
                                  + sp.array(cy%1, cx%1)).flatten()
        x,y =  dp.sortmanynsp(d, stamp.flatten())

        return x,y,(cx,cy)




class TimeseriesResults(astroplot.AstroPlot):

    """TimeseriesResults class inherited from AstroPlot class.The purpose of this class is to centralize the data output (mainly the plotting routines).

    """

    def __init__(self, epoch, flx, err, targets, exptime, fwhms):
        """AstroPlot object constructor.
        
        :param mjd: date array
        :type mjd: array
        :param flx: flux array dictionary
        :type flx: dict
        :param err: flux error array dictionary
        :type err: dict
        :param targets: coordinates array dictionary
        :type targets: dict
        :param fwhms: FWHMs of each photometry
        :type fwhms: dict
        :rtype: AstroPlot
        """

        self.epoch = sp.array(epoch)
        self.flx = flx
        self.err = err
        self.cooxy = targets
        self.fwhms = fwhms
        self.exptime = sp.array(exptime)
        self.ratio = None



    def plot_ratio(self, trg=None, ref=None, normframes=None, 
                   color='g', axes=None, 
                   legend_size=None,
                   **kwargs):
        """Display the ratio of science and reference

        :param trg: label name of target star
        :type trg: string (key of flx and coo)
        :param ref: list of stars to be used as reference
        :type ref: None or list of strings. If None, then all stars except target
        :param normframes: list of frames to normalize (for example, in case only out of transit want to be considered)
        :type normframes: None or boolean array. If None, then normalize by all frames
        :rtype: None (and plot display)
        """

        if self.ratio is None:
            if trg is None:
                trg = self.flx.keys()[0]
            self.doratio(trg=trg, ref=ref, normframes=normframes)

        if self.ratio is None:
            print("No ratio computed computed")
            return

        if axes is None:
            axes = plt

        axes.errorbar(self.epoch, self.ratio,
                      color=color, ls='none',
                      yerr=self.ratio_error,
                      capsize=0, **kwargs)
#        plt.plot(self.epoch, self.ratio, 'r')
#        axes.title("Ratio for target = " + str(trg))
        return


    def plot_timeseries(self):
        """Display the timeseries data: flux (with errors) as function of mjd

        :rtype: None (and plot display)
        """

        fig = plt.figure(figsize=(5, 5), num=5)
        ax = fig.add_subplot(1, 1, 1)

        for lab in self.flx.keys():
            if self.err is None:
                yerr = None
            else:
                yerr = self.err[lab]

            ax.errorbar(self.epoch,
                        self.flx[lab],
                        yerr=yerr,
                        marker="o",
                        label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        plt.show()
        return


    def plot_fwhm(self, axes=None, label=None, 
                  colors=None, omit_legend=False,
                  title=None, xtitle="MJD", ytitle="FWHMs",
                  legend_size=None):
        """Plot the calculated fwhm
        :param axes: multi-choice for axes
        :type axes: see dp.axesfig()
        :param method: Method to compute the fwhm, currently supported: stdev
        :type method: string
        :param label: Target's label or None if plotting both
        :type label: string or None
"""

        f, ax = dp.axesfig(axes)

        if label is None:
            label = self.flx.keys()
        else:
            label = [label]

        if colors is None:
            colors = ['r','b','g','k','y']

        for lab, col in zip(label,colors):                            
            ax.plot(self.epoch, self.fwhms[lab], color=col,
                    label=lab)

        if title is not None:
            ax.set_title(title)
        if xtitle is not None:
            ax.set_xlabel(xtitle)
        if ytitle is not None:
            ax.set_ylabel(ytitle)

        prop={}
        if legend_size is not None:
            prop['size'] = legend_size
        if len(lab)>1 and not omit_legend:
            ax.legend(prop=prop)
        f.show()
        return



    def plot_drift(self, axes=None):
        """Show the drift of the stars using the coordinates obtained for every image in the timeseries.

        :rtype: None (and plot display)
        """

        f, ax = dp.axesfig(axes)

        lines = []
        leg = []
        for k in self.flx.keys():
            cooxy = self.cooxy[k]
            xd, yd = sp.array(cooxy).transpose()
            l = ax.plot(xd - xd[0], yd - yd[0],
                        label='%8s. X,Y: %-7.1f,%-7.1f' % (k, xd[0], yd[0]))
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .302), loc=3,
                   ncol=1, mode="expand", borderaxespad=0.,
                   prop={'size': 6})
        #f.show()
        return

    def tofile(self, filename, all=False, comments=None):
        out = []
        for k in self.flx.keys():
            out.append(['flux_%s' % (k,),      self.flx[k]])
            out.append(['error_%s' % (k,), self.err[k]])
            if all:
                out.append(['fwhm_%s' % (k,),  self.fwhms[k]])
                out.append(['Cx_%s' % (k,),    map(lambda x:x[0],
                                                   self.cooxy[k][1:])])
                out.append(['Cy_%s' % (k,),    map(lambda x:x[1],
                                                   self.cooxy[k][1:])])

        f = open(filename, 'w')
        if comments is not None:
            f.write(comments+'\n')
        f.write('#%s\n' % ('   '.join(map(lambda x:x[0], out)),))
        lines = ['  '.join(map(str,line)) 
                 for line 
                 in zip(*map(lambda x:x[1], out))]
        f.write('%s' % ('\n'.join(lines)))
