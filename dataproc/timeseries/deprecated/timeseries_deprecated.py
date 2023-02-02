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


raise DeprecationWarning("Use timeseries_alsodeprecated.py instead.")


import sys
import astrocalc
import astroplot
import matplotlib.pyplot as plt
import astropy.time as apt
astrocalc=reload(astrocalc)
astroplot=reload(astroplot)

#from astropy.time import Time

import copy
import scipy as sp
import dataproc as dp
import warnings


# def plot_apnsky(cxy, apnsky, 
#                 apcolor='w', skycolor='r', alpha=0.6):
#     """Plot aperture and sky at specified coordinates"""

#     x,y = sp.mgrid[-apnsky[2]:apnsky[2],
#                     -apnsky[2]:apnsky[2]]
#     d = sp.sqrt(x*x + y*y)
#     plt.imshow

    
#     xap,yap = dp.polygon(cxy, apert)
#     xsk1,ysk1 = dp.polygon(cxy, apnsky)
#     xsk2,ysk2 = dp.polygon(cxy, apnsky)
#     plt.fill(xap, yap, apcolor, alpha=alpha)
#     plt.fill_between(xsk1, ysk1, apcolor, alpha=alpha)
#     plt.fill_between(xsk1, ysk1, apcolor, alpha=alpha)


class TimeSeries(astrocalc.AstroCalc):

    """Timeseries class inherited from AstroCalc class.The purpose of this class is to organize all the calculation tasks related to the timeseries analysis.

    """

    def __init__(
        self,
        data,
        coordsxy=None,
        labels=None,
        maxskip=6,
        epoch='JD',
        epoch_format='jd',
        exptime='EXPTIME',
        ingain='GTGAIN11',
        inron='GTRON11',
        offsetxy=None,
        masterbias = None,
        masterflat = None,
            hdu=0,
            hdud=None,
            hduh=None,
        #            keytype='IMAGETYP',
        #            mastermode='median',
        ):
        """Timeseries object constructor.

        :param data: image data for the timeseries
        :type data: str (path to a folder), AstroDir, ndarray list (list of images)
        :param coordsxy: list of x,y coordinates of stars in the first image of the timeseries. If dictionary is given with the right 2-tuple as values, the keys will be used as labels
        :type coordsxy: dictionary or list of 2-tuples or 2-list
        :param labels: labels for the stars
        :type labels: list of strings
        :param maxskip: maximum skip considered (without warning) between the positions of a star in two sucesive images
        :type maxskip: int (default value: 6)
        :param epoch: header key of observation date in the images (allows image sorting in terms of time) or list
        :type epoch: str or list
        :param ingain: header or value for the gain (e/ADU)
        :param inron: header or value for the readout noise (e)
        :rtype: Timeseries
        """
        # :param keytype: header key of image type ('OBJECT','BIAS','DOME FLAT',etc)
        # :type keytype: str
        # :param mastermode: mode for master (BIAS, DARK, FLAT, etc) image obtention. Options: 'mean' and 'median' (default).
        # :type mastermode: str

        if hduh is None:
            hduh = hdu
        if hdud is None:
            hdud = hdu    
        
        # data type check
        if isinstance(data, str):  # data is a string (path to a directory)
            self.files = dp.AstroDir(data)
            if isinstance(epoch,basestring):
                self.files = self.files.sort(epoch, hduh=hduh)
            self.isAstrodir = True

        elif isinstance(data, dp.AstroDir):  # data is an astrodir object
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
                self.epoch, self.files = dp.sortmany(epoch, self.files)
            else:
                warnings.warn("Epochs were not specified and list of data was given")
            if isinstance(exptime,(list,tuple)):
                self.exptime = exptime
            else:
                warnings.warn("Exposure time was not specified and list of data was given")
            self.files = data
            self.isAstrodir = False

        else:
            raise TypeError('data should be str, AstroDir or ndarray list')

        if offsetxy is None:
            offsetxy = {}

        # mjd list and science image filtering
        if self.isAstrodir:

            self.epoch = self.files.getheaderval(epoch)
            self.exptime = self.files.getheaderval(exptime)

            # time_file_list = []
            # bias_files = []
            # dark_files = []
            # flat_files = []

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
            epoch = sp.arange(len(data))
        if not hasattr(self, 'exptime'):
            self.exptime = sp.ones(len(data))

        self.epoch = apt.Time(self.epoch, format=epoch_format, scale='utc')

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
        if isinstance(coordsxy, dict):
            labels = coordsxy.keys()
            coordsxy = coordsxy.values()
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

        self._shape = self.files[0].shape

        # instance variables (in addition to self.files, self.isAstrodir and
        # self.mjd,self.masterbias, self.masterbias, self.masterflat)
        self.labels = labels
        self.targetsxy = targetsxy
        self.offsetxy = offsetxy
        self.maxskip = maxskip
        self.skydata = None
        self.gain = ingain
        self.ron = inron

        if masterbias is None:
            masterbias = sp.zeros(self._shape)
        if masterflat is None:
            masterflat = sp.ones(self._shape)

        self.masterbias = masterbias
        self.masterflat = masterflat


    def perform_phot(self,
                     aperture,
                     sky=None,
                     stamprad=25,
                     sky_store=True,
                     deg=1,
                     quiet=False,
                     ):
        """Perform apperture photometry with the images of the Timeseries object.

        :param aperture: aperture photometry radius
        :type aperture: int
        :param sky: inner and outer radius for sky annulus
        :type sky: [int,int]
        :param deg: Degre to sky polynomial
        :type deg: int
        :param stamprad: radius of the (square) stamp that contains a star
        :type stamprad: int (default value: 20)
        :rtype: TimeseriesResults
        """

        if sky is None and self.skydata is None:
            raise ValueError(
                'No sky information available: sky and self.skydata are None')

        flx = {lab: [] for lab in self.labels}
        err = {lab: [] for lab in self.labels}
        fwhms = {lab: [] for lab in self.labels}

        skydata = []
        targetsxy = {lab:[coo] for lab, coo in self.targetsxy.items()}
        nframes = len(self.files)

        for rdata, i in zip(self.files, range(nframes)):
            if sky is not None:
                sky_img_dict = {lab: [] for lab in self.labels}

            gain = self.gain
            ron = self.ron
            if isinstance(rdata, dp.AstroFile):
                if isinstance(gain,basestring):
                    gain = rdata.getheaderval(self.gain)[0]
                if isinstance(ron, basestring):
                    ron  = rdata.getheaderval(self.ron)[0]
                data = rdata.reader()
                head = rdata.readheader()
            else:
                data = rdata
                
            data = (data - self.masterbias) / self.masterflat

            if isinstance(gain, basestring) or isinstance(ron, basestring):
                raise ValueError("Gain or RON specified as header (%s/%s), but not found: %s/%s" % (self.gain,self.ron,str(gain),str(ron)))

            for lab, cooxy in targetsxy.items():
                cx, cy = cooxy[-1]
                offx=offy=0
                frame_number = len(cooxy)-1
                if frame_number in self.offsetxy.keys():
                    offx = self.offsetxy[frame_number][0]
                    offy = self.offsetxy[frame_number][1]
                cx += offx
                cy += offy

                scy, scx = dp.subcentroid(data, [cy, cx], stamprad)
                sarr = dp.subarray(data, [scy, scx], stamprad)
                relscy, relscx = [stamprad + scy%1, stamprad + scx%1]

                skip = sp.sqrt((cy - scy) ** 2 +
                               (cx - scx) ** 2)
                if skip > self.maxskip:
                    print(("Unexpected jump of %f pixels has occurred on"+
                           " frame %i for star %s") %
                          (skip, i, lab))


                if sky is None:
                    if not hasattr(self, 'skydata'):
                        raise ValueError('Sky not specified, but no  previous sky data was stored.')
                    sky_info = self.skydata[i][lab]
                    phot, phot_err, fwhm, sky_out = self.apphot(sarr, [relscy, relscx], 
                                                                aperture, sky_info,
                                                                gain=gain, ron=ron, deg=deg)
                else:
                    if (sky[1] > 0.8*stamprad):
                        warnings.warn("Stamp radius (%i) might be too small given outer sky radius (%.1f)" %(stamprad,sky[1]))
                    phot, phot_err, fwhm, sky_out = self.apphot(sarr, [relscy, relscx],
                                                                aperture, sky,
                                                                gain=gain, ron=ron, deg=deg,
                                                                raisee=lab=='Target')
                    sky_img_dict[lab] = sky_out


                flx[lab].append(phot)
                err[lab].append(phot_err)
                fwhms[lab].append(fwhm)
                cooxy.append([scx, scy])


            if sky is not None and sky_store:
                skydata.append(sky_img_dict)

            if not quiet:
                sys.stdout.write(("#%%0%ii: %%f +- %%f (%%.1f, %%.1f). %%f\n" 
                                  % (int(sp.log10(nframes))+1,) )
                                 % (i, phot, phot_err, cooxy[-1][0], cooxy[-1][1], fwhm))
                sys.stdout.flush()

        if sky is not None and sky_store:
            self.skydata = skydata

        tsr = TimeseriesResults(self.epoch, flx, err, targetsxy, 
                                self.exptime, fwhms, [aperture]+ list(sky))

        self.lastphotometry = tsr
        self.stamprad = stamprad

        return tsr


class TimeseriesExamine(astroplot.AstroPlot, astrocalc.AstroCalc):

    def __init__(self, timeseries):
        """GRaphical utilities for timeseries
        :param timeseries: timeserie object to examine
        :type timeseries: Timeserie object
        """
        self.ts = timeseries

    def imshowz(self, frame=0,
                apcolor='w', skcolor='LightCyan',
                alpha=0.6, axes=None,
                annotate=False,
                npoints=30, **kwargs):
        """Plot image"""

        dp.imshowz(self.ts.files[frame],
                   axes=axes, **kwargs)
        if (hasattr(self.ts, 'lastphotometry') and 
            isinstance(self.ts.lastphotometry, TimeseriesResults)):
            print(" Using coordinates from photometry for frame %i" 
                  % (frame,))
            tsr = self.ts.lastphotometry
            apnsky = tsr.apnsky
            for lab in tsr.cooxy.keys():
                cxy = tsr.cooxy[lab][frame+1]
                # circle = plt.Circle((xx,yy), radius=tsr.apnsky[0],
                #                     fc=apcolor, alpha=alpha)
                theta = sp.linspace(0, 2*sp.pi, npoints, endpoint=True)
                xs = cxy[0] + apnsky[0]*sp.cos(theta)
                ys = cxy[1] + apnsky[0]*sp.sin(theta)
                plt.fill(xs, ys,
                         edgecolor=apcolor, color=apcolor,
                         alpha=alpha)

                xs = cxy[0] + sp.outer(apnsky[1:3], sp.cos(theta))
                ys = cxy[1] + sp.outer(apnsky[1:3], sp.sin(theta))
                xs[1,:] = xs[1,::-1]
                ys[1,:] = ys[1,::-1]
                plt.fill(sp.ravel(xs), sp.ravel(ys),
                         edgecolor=skcolor, color=skcolor,
                         alpha=alpha)
                if annotate:
                    outer_sky = self.ts.lastphotometry.apnsky[2]
                    plt.gca().annotate(lab, 
                                       xy=(cxy[0],cxy[1]+outer_sky),
                                       xytext=(cxy[0]+1*outer_sky,
                                               cxy[1]+1.5*outer_sky),
                                       fontsize=20)
            # xx, yy = zip(*cooxy)


            # plt.plot(xx, yy, 'w+', 
            #          markersize=10, 
            #          markeredgewidth=2)

    def showstamp(self, target=None, stamprad=30, 
                  first=0, last=-1, figure=None, ncol=None):
        """Show the star at the same position for the different frames

        :param target: None for the first key()
        :param stamprad: Plotting radius
        :param first: First frame to show
        :param last: Last frame to show. It can be onPython negative format
        :param figure: Specify figure number
        :param ncol: Number of columns
"""
        if last<0:
            nimages = len(self.ts.files)+1+last-first
        else:
            nimages = last-first

        if target is None:
            target = self.ts.targetsxy.keys()[0]

        if ncol is None:
            ncol = int(sp.sqrt(nimages))
        nrow = int(sp.ceil(nimages/ncol))

        f, ax = plt.subplots(nrow, ncol, num=figure, 
                             sharex=True, sharey=True)
        f.subplots_adjust(hspace=0, wspace=0)
        ax1 = list(sp.array(ax).reshape(-1))

        cx,cy = self.ts.targetsxy[target]

        for n, a in zip(range(nimages), ax1):
            frame_number = n+first
            if frame_number in self.ts.offsetxy.keys():
                cx += self.ts.offsetxy[frame_number][0]
                cy += self.ts.offsetxy[frame_number][1]
            frame = (self.ts.files[n]-self.ts.masterbias) / self.ts.masterflat

            dp.imshowz(frame,
                       axes=a, 
                       cxy = [cx, cy],
                       plot_rad = stamprad,
                       ticks=False,
                       trim_data=True,
                       )




    def plot_radialprofile(self, targets=None, xlim=None, axes=1,
                           legend_size=None, save=None, show=None,
                           **kwargs):
        """Plot Radial Profile from data using radialprofile() function
        :param target: Target spoecification for recentering. Either an integer for specifc target, or a 2-element list for x/y coordinates.
        :type target: integer/string or 2-element list
        """


        colors = ['rx', 'b^', 'go', 'r^', 'bx', 'g+']
        fig, ax = dp.figaxes(axes)

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
                                                  center[1],
                                                  center[0]),
                    )
        prop={}
        if legend_size is not None:
            prop['size'] = legend_size
        ax.legend(loc=1, prop=prop)

        if xlim is not None:
            if isinstance(xlim, (int, float)):
                ax.set_xlim([0, xlim])
            else:
                ax.set_xlim(xlim)

        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()


        
    def radialprofile(self, target, frame=0, recenter=False, stamprad=20):
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

        if (frame > len(self.ts.files)):
            raise ValueError("Specified frame (%i) is too large (there are %i frames)" % (frame,len(self.ts.files)))

        if recenter:
            image = (self.ts.files[frame]-self.ts.masterbias)/self.ts.masterflat
            cy, cx = dp.subcentroid(image, [cy, cx], stamprad) #+ sp.array([cy,cx]) - stamprad
            print(" Using coordinates from recentering (%.1f, %.1f) for frame %i" 
                  % (cx, cy, frame))
        else:
            if (hasattr(self.ts, 'lastphotometry') and 
                isinstance(self.ts.lastphotometry, TimeseriesResults)):
                cx, cy = self.ts.lastphotometry.cooxy[target][frame+1]
                print(" Using coordinates from photometry (%.1f, %.1f) for frame %i" 
                      % (cx, cy, frame))
                

        image = (self.ts.files[frame]-self.ts.masterbias)/self.ts.masterflat
        stamp = image[int(cy-stamprad): int(cy+stamprad), 
                      int(cx-stamprad): int(cx+stamprad)]

        d = self.centraldistances(stamp, 
                                  sp.array(stamp.shape)//2 
                                  + sp.array([cy%1, cx%1])).flatten()
        x,y =  dp.sortmanynsp(d, stamp.flatten())

        return x,y,(cy,cx)




class TimeseriesResults(astroplot.AstroPlot):

    """TimeseriesResults class inherited from AstroPlot class.The purpose of this class is to centralize the data output (mainly the plotting routines).

    """

    def __init__(self, epoch, flx, err, targets, exptime, fwhms, apnsky):
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

        self.epoch = epoch
        self.flx = flx
        self.err = err
        self.cooxy = targets
        self.fwhms = fwhms
        self.exptime = sp.array(exptime)
        self.ratio = None
        self.apnsky = apnsky
    



    def plot_ratio(self, trg=None, ref=None, normframes=None, 
                   color='g', axes=None, 
                   legend_size=None,
                   recompute=True, overwrite=True,
                   label=None,
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

        if self.ratio is None or recompute:
            if trg is None:
                trg = self.flx.keys()[0]
            self.doratio(trg=trg, ref=ref, normframes=normframes)

        if self.ratio is None:
            print("No ratio computed computed")
            return


        f, ax, epoch = dp.axesfig_xdate(axes, self.epoch,
                                        overwrite=overwrite)

        ax.plot(epoch, self.ratio, '+',
                color=color, label=label)
        ax.errorbar(epoch, self.ratio,
                    color=color, ls='none',
                    yerr=self.ratio_error,
                    capsize=0, **kwargs)


        f.show()
#        plt.plot(self.epoch, self.ratio, 'r')
#        axes.title("Ratio for target = " + str(trg))
        return


    def plot_flux(self, label=None, axes=None):
        """Display the timeseries data: flux (with errors) as function of mjd

        :param label: Specify a single star to plot
        :rtype label: basestring

        :rtype: None (and plot display)
        """

        fig, ax, epoch = dp.axesfig_xdate(axes, self.epoch)

        if label is None:
            disp = self.flx.keys()
        else:
            disp = [label]

        for lab in disp:
            if self.err is None:
                yerr = None
            else:
                yerr = self.err[lab]

            ax.errorbar(epoch,
                        self.flx[lab],
                        yerr=yerr,
                        marker="o",
                        label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        fig.show()
        return


    def plot_normflux(self, label=None, axes=None, lowylim=0):
        """Display the timeseries data: normalized flux (with errors) as function of epoch

        :param label: Specify a single star to plot
        :rtype label: basestring

        :rtype: None (and plot display)
        """

        fig, ax, epoch = dp.axesfig_xdate(axes, self.epoch)

        if label is None:
            disp = self.flx.keys()
        else:
            disp = [label]

        for lab in disp:
            flx = self.flx[lab]
            if self.err is None:
                yerr = None
            else:
                yerr = self.err[lab]

            ax.errorbar(epoch,
                        flx/sp.median(flx),
                        yerr=yerr/sp.median(flx),
                        marker="o",
                        label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Normalized Flux")
        yl1, yl2 = ax.get_ylim()
        ax.set_ylim([lowylim,yl2])

        ax.legend()
        fig.show()
        return


    def plot_fwhm(self, axes=None, label=None, 
                  colors=None, omit_legend=False,
                  title=None, xtitle="MJD", ytitle="FWHMs",
                  legend_size=None):
        """Plot the calculated fwhm
        :param axes: multi-choice for axes
        :type axes: see dp.figaxes()
        :param method: Method to compute the fwhm, currently supported: stdev
        :type method: string
        :param label: Target's label or None if plotting both
        :type label: string or None
"""

        f, ax, epoch = dp.figaxes(axes, self.epoch)

        if label is None:
            label = self.flx.keys()
        else:
            label = [label]

        if colors is None:
            colors = ['r','b','g','k','y']

        for lab, col in zip(label,colors):                            
            ax.plot(epoch, self.fwhms[lab], color=col,
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

        f, ax = dp.figaxes(axes)

        lines = []
        leg = []
        nlab = len(self.flx)
        print ("nc: %s" % (int(nlab//2.5),))
        for k in self.flx.keys():
            cooxy = self.cooxy[k]
            xd, yd = sp.array(cooxy).transpose()
            l = ax.plot(xd - xd[0], yd - yd[0],
                        label='%8s. X,Y: %-7.1f,%-7.1f' % (k, xd[0], yd[0]))
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .302), loc=3,
                   ncol=int(nlab//2.5), mode="expand", borderaxespad=0.,
                   prop={'size': 6})
        f.show()
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
