from __future__ import print_function

from IPython.core.debugger import Tracer

import dataproc as dp
import copy
import scipy as sp
import sys
import numpy as np
import warnings

import pdb
import timeserie
import matplotlib.pyplot as plt


class Photometry(object):
    def __init__(self, sci_files, aperture=None, sky=None, mdark=None, mflat=None,
                 target_coords_xy=None, stamp_rad=None,
                 maxskip=8, max_counts = 50000,
                 new_coords=None, stamp_coords=None, epoch=None, labels=None,
                 deg=1, gain=None, ron=None):

        if isinstance(epoch, str):
            self.epoch = sci_files.getheaderval(epoch)
        elif hasattr(epoch, '__iter__'):
            self.epoch = epoch
        else:
            raise ValueError(
                "Epoch must be an array of dates in julian date, or a a header's keyword for the Julian date of the observation")

        self.target_coords_xy = target_coords_xy
        self.aperture = aperture
        self.sky = sky
        self.deg = deg
        self.gain = gain
        self.ron = ron
        self.stamp_rad = stamp_rad
        self.max_counts = max_counts

        # label list
        if isinstance(target_coords_xy, dict):
            labels = target_coords_xy.keys()
            coordsxy = target_coords_xy.values()
        try:
            if labels is None:
                labels = []
            nstars = len(target_coords_xy)
            if len(labels) > nstars:
                labels = labels[:nstars]
            elif len(labels) < nstars:
                labels = list(
                    labels) + sp.arange(len(labels),
                                        nstars).astype(str).tolist()
            targetsxy = {lab: coo
                         for coo, lab in zip(target_coords_xy, labels)}
        except:
            raise ValueError("Coordinates of target stars need to be " +
                             "specified as a list of 2 elements, not: %s" %
                             (str(target_coords_xy),))
        print(" Initial guess received for %i targets: %s" %
              (len(target_coords_xy),
               ", ".join(["%s %s" % (lab, coo)
                          for lab, coo in zip(labels, target_coords_xy)])
               ))

        self.labels = labels
        self.targetsxy = targetsxy

        self.sci_stamps, self.new_coords_xy = self.get_stamps(sci_files, target_coords_xy, stamp_rad,
                                                              maxskip,
                                                              mdark=mdark, mflat=mflat)
        self.mdark = mdark
        self.mflat = mflat

    def photometry(self, aperture=None, sky=None, deg=None):
        if aperture is not None:
            self.aperture = aperture
        if sky is not None:
            self.sky = sky
        if deg is not None:
            self.deg = deg

        if self.aperture is None or self.sky is None:
            raise ValueError("ERROR: aperture photometry parameters are incomplete. Either aperture "
                             "photometry radius or sky annulus were not giving. Please call photometry "
                             "with the following keywords: photometry(aperture=a, sky=s) or define aperture "
                             "and sky when initializing Photometry object.")

        ts = self.CPUphot()
        return ts

    def phot_error(self, phot, sky_std, n_pix_ap, n_pix_sky, gain, ron=None):
        """Calculates the photometry error

        :param phot: star flux
        :type phot: float
        :param sky: sky flux
        :type sky: float
        :param n_pix_ap: number of pixels in the aperture
        :type n_pix_ap: int
        :param n_pix_sky: number of pixels in the sky annulus
        :type n_pix_sky: int
        :param gain: gain
        :type gain: float
        :param ron: read-out-noise
        :type ron: float (default value: None)
        :rtype: float
        """

        # print("f,s,npa,nps,g,ron: %f,%f,%i,%i,%f,%f" %
        #       (phot, sky_std, n_pix_ap, n_pix_sky, gain, ron))

        if ron is None:
            print("Photometric error calculated without RON")
            ron = 0.0

        if gain is None:
            print("Photometric error calculated without Gain")
            gain = 1.0

        var_flux = phot / gain
        var_sky = sky_std ** 2 * n_pix_ap * (1 + float(n_pix_ap) / n_pix_sky)

        var_total = var_sky + var_flux + ron * ron * n_pix_ap

        return sp.sqrt(var_total)

    @staticmethod
    def centroid(orig_arr, medsub=True):
        """Find centroid of small array
        :param orig_arr:
        :param medsub:
        :return:
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

    def get_stamps(self, sci_files, target_coords_xy, stamp_rad, maxskip,
                   mdark=None, mflat=None):
        """

        :param sci_files:
        :type sci_files: AstroDir
        :param target_coords_xy: [[t1x, t1y], [t2x, t2y], ...]
        :param stamp_rad:
        :return:
        """

        all_cubes = []
        # epoch = sci_files.getheaderval('DATE-OBS')
        # epoch = sci_files.getheaderval('MJD-OBS')
        # labels = sci_files.getheaderval('OBJECT')
        new_coords = []
        stamp_coords = []

        skipcalib = False
        if mdark is None and mflat is None:
            skipcalib = True
        if mdark is None:
            mdark = sp.zeros(sci_files[0].shape)
        if mflat is None:
            mflat = sp.ones(sci_files[0].shape)

        frame_id = []
        all_cubes = [[] for i in target_coords_xy]
        center_xy = [[[xx, yy]] for xx, yy in target_coords_xy]
        print("Obtaining stamps for {0} files".format(len(sci_files)))

        for astrofile in sci_files:
            print('.', end='')
            sys.stdout.flush()

            d = astrofile.reader()
            frame_id.append(astrofile.filename)
            if not skipcalib:
                d = (d - mdark) / mflat

            for tc_xy, cube, lab in zip(center_xy, all_cubes, self.labels):
                cx, cy = tc_xy[-1][0], tc_xy[-1][1]
                ncy, ncx = dp.subcentroid(d, [cy, cx], stamp_rad)
                stamp = dp.subarray(d, [ncy, ncx], stamp_rad)

                skip = sp.sqrt((cx-ncx)**2+(cy-ncy)**2)
                if skip>maxskip:
                    warnings.warn("\nUnexpected jump of {skip} pixels has occurred on frame"
                                  " {frame} for star '{name}'".format(skip=skip, frame=i, name=lab))

                cube.append(stamp)
                tc_xy.append([ncx, ncy])

        print('')

        # get rid of user-provided first guesses for centers
        dummy = [stamp_cnt.pop(0) for stamp_cnt in center_xy]
        self.frame_id = frame_id

        return all_cubes, center_xy

    def CPUphot(self):
        all_phot = []
        all_err = []

        print("Processing CPU photometry for {0} targets: ".format(len(self.sci_stamps)), end='')
        sys.stdout.flush()
        for label, target, centers_xy in zip(self.labels, self.sci_stamps, self.new_coords_xy):  # For each target
            t_phot, t_err = [], []

            for data, center_xy, frame_id in zip(target, centers_xy, self.frame_id):
                cx, cy = center_xy

                # Stamps are already centered, only decimals could be different
                cstamp = [self.stamp_rad + cy % 1, self.stamp_rad + cx % 1]

                # Preparing arrays for photometry
                d = self.centraldistances(data, cstamp)
                dy, dx = data.shape
                y, x = sp.mgrid[-cstamp[0]:dy - cstamp[0], -cstamp[1]:dx - cstamp[1]]

                # Compute sky correction
                # Case 1: sky = [fit, map_of_sky_pixels]
                if isinstance(self.sky[0], sp.ndarray):
                    fit = self.sky[0]
                    idx = self.sky[1]

                # Case 2: sky = [inner_radius, outer_radius]
                else:
                    import scipy.optimize as op
                    idx = (d > self.sky[0]) * (d < self.sky[1])
                    errfunc = lambda coef, x, y, z: (self.bipol(coef, x, y) - z).flatten()
                    coef0 = sp.zeros((self.deg, self.deg))
                    coef0[0, 0] = data[idx].mean()
                    fit, cov, info, mesg, success = op.leastsq(errfunc, coef0.flatten(),
                                                               args=(x[idx], y[idx], data[idx]), full_output=1)

                # Apply sky subtraction
                n_pix_sky = idx.sum()
                sky_fit = self.bipol(fit, x, y)
                sky_std = (data - sky_fit)[idx].std()
                res = data - sky_fit  # minus sky

                # Following to compute FWHM by fitting gaussian
                res2 = res[d < self.aperture * 4].ravel()
                d2 = d[d < self.aperture * 4].ravel()
                tofit = lambda d, h, sig: h * dp.gauss(d, sig, ndim=1)
                try:
                    sig, cov = op.curve_fit(tofit, d2, res2, sigma=1 / sp.sqrt(sp.absolute(res2)),
                                            p0=[max(res2), self.aperture / 3])
                except RuntimeError:
                    sig = sp.array([0, 0, 0])
                fwhmg = 2.355 * sig[1]

                # now photometry
                psf = res[d<self.aperture]
                if (psf>self.max_counts).any():
                    warnings.warn("Object {} on frame {} has count above the "
                                  "threshold ({})".format(label, frame_id, self.max_counts))
                phot = float(psf.sum())
                # print("phot: %.5d" % (phot))

                # now the error
                if self.gain is None:
                    error = None
                else:
                    n_pix_ap = (d < self.aperture).sum()
                    error = self.phot_error(phot, sky_std, n_pix_ap, n_pix_sky, self.gain, ron=self.ron)

                t_phot.append(phot)
                t_err.append(error)

            all_phot.append(t_phot)
            all_err.append(t_err)
            print('X', end='')
            sys.stdout.flush()

        print('')
        return timeserie.TimeSeries(all_phot, all_err,
                                    labels=self.labels, epoch=self.epoch,
                                    extras={'centers_xy': self.new_coords_xy})

    def centraldistances(self, data, c):
        """Computes distances for every matrix position from a central point c.
        :param data: array
        :type data: sp.ndarray
        :param c: center coordinates
        :type c: [float, float]
        :rtype: sp.ndarray
        """
        dy, dx = data.shape
        y, x = sp.mgrid[0:dy, 0:dx]
        return sp.sqrt((y - c[0]) * (y - c[0]) + (x - c[1]) * (x - c[1]))

    def bipol(self, coef, x, y):
        """Polynomial fit for sky subtraction

        :param coef: sky fit polynomial coefficients
        :type coef: sp.ndarray
        :param x: horizontal coordinates
        :type x: sp.ndarray
        :param y: vertical coordinates
        :type y: sp.ndarray
        :rtype: sp.ndarray
        """
        plane = sp.zeros(x.shape)
        deg = sp.sqrt(coef.size).astype(int)
        coef = coef.reshape((deg, deg))

        if deg * deg != coef.size:
            print("Malformed coefficient: " + str(coef.size) + "(size) != " + str(deg) + "(dim)^2")

        for i in sp.arange(coef.shape[0]):
            for j in sp.arange(i + 1):
                plane += coef[i, j] * (x ** j) * (y ** (i - j))

        return plane

    def plot_radialprofile(self, targets=None, xlim=None, axes=1,
                           legend_size=None,
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
            targets = self.targetsxy.keys()
        elif isinstance(targets, basestring):
            targets = [targets]
        elif isinstance(targets, (list, tuple)) and \
                not isinstance(targets[0], (basestring, list, tuple)):
            # Assume that it is a coordinate
            targets = [targets]

        trgcolor = {str(trg): color for trg, color in zip(targets, colors)}

        for trg in targets:
            distance, value, center = self.radialprofile(trg, stamprad=self.stamp_rad, **kwargs)
            ax.plot(distance, value, trgcolor[str(trg)],
                    label="%s: (%.1f, %.1f)" % (trg,
                                                center[1],
                                                center[0]),
                    )
        prop = {}
        if legend_size is not None:
            prop['size'] = legend_size
        ax.legend(loc=1, prop=prop)

        if xlim is not None:
            if isinstance(xlim, (int, float)):
                ax.set_xlim([0, xlim])
            else:
                ax.set_xlim(xlim)

        plt.show()

    def radialprofile(self, target, stamprad=None, frame=0, recenter=False):
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
                cx, cy = self.targetsxy[target]
                target = self.target_coords.index([cx, cy])
            except KeyError:
                raise KeyError("Invalid target specification. Choose from '%s'" % ', '.join(self.targetsxy.keys()))
        elif isinstance(target, (list, tuple)):
            cx, cy = target
        else:
            print("Invalid coordinate specification '%s'" % (target,))

        slen = len(self.sci_stamps[0])

        if (frame > slen):
            raise ValueError("Specified frame (%i) is too large (there are %i frames)"
                             % (frame, slen))

        if recenter:
            # image = (self.ts.files[frame]-self.ts.masterbias)/self.ts.masterflat
            image = self.sci_stamps[target][frame]
            cy, cx = dp.subcentroid(image, [cy, cx], stamprad)  # + sp.array([cy,cx]) - stamprad
            print(" Using coordinates from recentering (%.1f, %.1f) for frame %i"
                  % (cx, cy, frame))
        else:
            # if (hasattr(self.ts, 'lastphotometry') and
            #    isinstance(self.ts.lastphotometry, TimeSerie)):
            cx, cy = self.new_coords[target][frame + 1][0], self.new_coords[target][frame + 1][1]
            print(" Using coordinates from photometry (%.1f, %.1f) for frame %i"
                  % (cx, cy, frame))

        stamp = self.sci_stamps[target][frame]  # -self.ts.masterbias)/self.ts.masterflat

        d = self.centraldistances(stamp, self.stamp_coords[target][frame]).flatten()
        x, y = dp.sortmanynsp(d, stamp.flatten())

        return x, y, (cy, cx)

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
        if last < 0:
            nimages = len(self.sci_stamps) + 1 + last - first
        else:
            nimages = last - first

        if target is None:
            target = self.targetsxy.keys()[0]

        if ncol is None:
            ncol = int(sp.sqrt(nimages))
        nrow = int(sp.ceil(nimages / ncol))

        f, ax = plt.subplots(nrow, ncol, num=figure,
                             sharex=True, sharey=True)
        f.subplots_adjust(hspace=0, wspace=0)
        ax1 = list(sp.array(ax).reshape(-1))

        cx, cy = self.targetsxy[target]
        target = self.target_coords.index([cx, cy])
        cx_s, cy_s = self.stamp_coords[target][0]

        # for n, a in zip(range(nimages), ax1):
        ik = 0
        for n, a in zip(self.sci_stamps, ax1):
            frame_number = ik + first
            ik += 1
            frame = (self.sci_stamps[target][frame_number])  # - self.ts.masterbias) / self.ts.masterflat

            dp.imshowz(frame,
                       axes=a,
                       cxy=[cx_s, cy_s],
                       plot_rad=self.stamp_rad,
                       ticks=False,
                       trim_data=False,
                       )
