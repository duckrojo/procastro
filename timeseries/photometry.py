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

from __future__ import print_function

from IPython.core.debugger import Tracer
import dataproc as dp
import scipy as sp
import scipy.optimize as op
import sys
import os.path
import timeserie
import matplotlib.pyplot as plt
import logging

tmlogger = logging.getLogger('dataproc.timeseries')
for hnd in tmlogger.handlers:
    tmlogger.removeHandler(hnd)
PROGRESS = 35
handler_console = logging.StreamHandler()
handler_console.setLevel(PROGRESS)
formatter_console = logging.Formatter('%(message)s')
handler_console.setFormatter(formatter_console)
tmlogger.addHandler(handler_console)


def _get_stamps(sci_files, target_coords_xy, stamp_rad, maxskip,
                mdark=None, mflat=None, labels=None, recenter=True,
                offsets_xy=None, logger=None, ignore=None):
    """

    :param sci_files:
    :type sci_files: AstroDir
    :param target_coords_xy: [[t1x, t1y], [t2x, t2y], ...]
    :param stamp_rad:
    :return:
    """

    if ignore is None:
        ignore = []

    ngood = len(sci_files)-len(ignore)

    if labels is None:
        labels = range(len(target_coords_xy))

    if offsets_xy is None:
        offsets_xy = sp.zeros(len(sci_files))

    if logger is None:
        logger = tmlogger

    skipcalib = False
    if mdark is None and mflat is None:
        skipcalib = True
    if mdark is None:
        mdark = 0.0
    if mflat is None:
        mflat = 1.0

    all_cubes = sp.zeros([len(target_coords_xy), ngood,
                          stamp_rad*2+1, stamp_rad*2+1])

    center_xy = [[[xx, yy]] for xx, yy in target_coords_xy]
    logger.log(PROGRESS, " Obtaining stamps for {0} files: ".format(ngood))

    to_store = 0
    for astrofile, idx, off in zip(sci_files, range(len(sci_files)), offsets_xy):
        if idx in ignore:
            continue

        d = astrofile.reader()
        if not skipcalib:
            if astrofile.has_calib():
                logger.warning("Skipping calibration given to Photometry() because "
                               "calibration files\nwere included in AstroFile: {}".format(astrofile))
            d = (d - mdark) / mflat

        stat = 0
        for tc_xy, cube, lab in zip(center_xy, all_cubes, labels):
            # todo: make fall backs when star is closer than stamp_Rad to a border
            cx, cy = tc_xy[-1][0]+off[0], tc_xy[-1][1]+off[1]
            if recenter:
                ncy, ncx = dp.subcentroid(d, [cy, cx], stamp_rad)
            else:
                ncy, ncx = cy, cx
            if ncy < 0 or ncx < 0 or ncy > d.shape[0] or ncx > d.shape[1]:
                # todo: undo af.is_bad :S
                # following is necessary to keep indexing that does not consider skipped bad frames
                af_names = [af.filename for af in sci_files]
                raise ValueError("Centroid for frame #{} falls outside data for target {}. "
                                 " Initial/final center was: [{:.2f}, {:.2f}]/[{:.2f}, {:.2f}]"
                                 " Offset: {}\n{}".format(af_names.index(astrofile.filename), lab, cx, cy, ncx, ncy, off, astrofile))
            cube[to_store] = dp.subarray(d, [ncy, ncx], stamp_rad, padding=True)

            skip = sp.sqrt((cx - ncx) ** 2 + (cy - ncy) ** 2)
            if skip > maxskip:
                stat = 1
                if idx == 0:
                    logger.warning(
                        "Position of user coordinates adjusted by {skip:.1f} pixels on first frame for target '{name}'".format(
                            skip=skip, name=lab))
                else:
                    logger.warning(
                        "Large jump of {skip:.1f} pixels for {name} has occurred on {frame}".format(
                            skip=skip, frame=astrofile, name=lab))

            tc_xy.append([ncx, ncy])

        if stat == 1:
            print('J', end='')
        else:
            print('.', end='')
        sys.stdout.flush()

        to_store+=1

    print('')

    # get rid of user-provided first guesses for centers
    dummy = [stamp_cnt.pop(0) for stamp_cnt in center_xy]

    return all_cubes, center_xy


def _phot_error(phot, sky_std, n_pix_ap, n_pix_sky, gain=None, ron=None):
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
        logging.warning("Photometric error calculated without read-out-noise")
        ron = 0.0

    if gain is None:
        logging.warning("Photometric error calculated without Gain")
        gain = 1.0

    var_flux = phot / gain
    var_sky = sky_std ** 2 * n_pix_ap * (1 + float(n_pix_ap) / n_pix_sky)

    var_total = var_sky + var_flux + ron * ron * n_pix_ap

    return sp.sqrt(var_total)



class Photometry(object):

    def __init__(self, sci_files, target_coords_xy,
                 aperture=None, sky=None, mdark=None, mflat=None,
                 stamp_rad=30, offsets_xy=None,
                 maxskip=8, max_counts=50000, recenter=True,
                 epoch='JD', labels=None,
                 deg=1, gain=None, ron=None,
                 logfile=None, ignore=None):

        if isinstance(epoch, str):
            self.epoch = sci_files.getheaderval(epoch)
        elif hasattr(epoch, '__iter__'):
            self.epoch = epoch
        else:
            raise ValueError(
                "Epoch must be an array of dates in julian date, or a a header's keyword for the Julian date of the observation")

        #following as default but are not used until .photometry(), which can override them
        self.aperture = aperture
        self.sky = sky

        self.deg = deg
        self.gain = gain
        self.ron = ron
        self.stamp_rad = stamp_rad
        self.set_max_counts(max_counts)
        self.recenter = recenter

        if logfile is None:
            tmlogger.propagate = True
            self._logger = tmlogger
        else:
            tmlogger.propagate = False

            #use logger instance that includes filename to allow different instance of photometry with different loggers as long as they use different files
            self._logger = logging.getLogger('dataproc.timeseries.{}'.format(os.path.basename(logfile).replace('.', '_')))
            self._logger.setLevel(logging.INFO)
            #in case of using same file name start new with loggers
            for hnd in self._logger.handlers:
                self._logger.removeHandler(hnd)

            handler = logging.FileHandler(logfile, 'w')
            formatter = logging.Formatter('%(asctime)s: %(name)s %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            self._logger.addHandler(handler)

            print ("Detailed logging redirected to {}".format(logfile))

        self._logger.info("dataproc.timeseries.Photometry execution on: {}".format(sci_files))
        #Tracer()()


        sci_files = dp.AstroDir(sci_files)

        offset_list = sp.zeros([len(sci_files), 2])
        if offsets_xy is not None:
            for k, v in offsets_xy.items():
                offset_list[k, :] = sp.array(v)

        # label list
        if isinstance(target_coords_xy, dict):
            coords_user_xy = target_coords_xy.values()
            labels = target_coords_xy.keys()
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
        except:
            raise ValueError("Coordinates of target stars need to be " +
                             "specified as dictionary or as a list of 2 elements, not: %s" %
                             (str(target_coords_xy),))

        self._logger.log(PROGRESS,
                         " Initial guess received for {} targets".format(len(target_coords_xy)))
        tmlogger.info(": {}".format(", ".join(["%s %s" % (lab, coo)
                                               for lab, coo in zip(labels, coords_user_xy)])
                                    ))

        self.labels = labels
        self.coords_user_xy = coords_user_xy

#coords_new_xy[frame][target]
        self.sci_stamps, self.coords_new_xy = _get_stamps(sci_files, self.coords_user_xy,
                                                          self.stamp_rad, maxskip=maxskip,
                                                          mdark=mdark, mflat=mflat,
                                                          recenter=recenter,
                                                          labels=labels,
                                                          offsets_xy=offset_list,
                                                          logger=self._logger,
                                                          ignore=ignore)
        self.frame_id = ["{}/{}".format(d, f) for d, f, i in zip(sci_files.getheaderval('dirname'),
                                                                 sci_files.getheaderval('basename'),
                                                                 range(len(sci_files)))
                         if i not in ignore]
        self.mdark = mdark
        self.mflat = mflat
        self.maxskip = maxskip
        if isinstance(sci_files, dp.AstroDir):
            self._input_astrodir = sci_files

    def imshowz(self, frame=0,
                apcolor='w', skcolor='LightCyan',
                alpha=0.6, axes=None,
                annotate=False,
                npoints=30, **kwargs):

        dp.imshowz(self.frame_id[frame],
                   axes=axes, **kwargs)

        aperture = self.aperture
        if self.aperture is None:
            logging.warn("Plotting a default aperture of 10 pixels")
            aperture = 10
        if self.sky is None:
            logging.warn("Using default sky of 15-20 pixels")
            sky = [15,20]

        for cooxy, lab in zip(zip(*self.coords_new_xy)[frame], self.labels):
            cx, cy = cooxy
            # circle = plt.Circle((xx,yy), radius=tsr.apnsky[0],
            #                     fc=apcolor, alpha=alpha)
            theta = sp.linspace(0, 2 * sp.pi, npoints, endpoint=True)
            xs = cx + aperture * sp.cos(theta)
            ys = cy + aperture * sp.sin(theta)
            plt.fill(xs, ys,
                     edgecolor=apcolor, color=apcolor,
                     alpha=alpha)

            xs = cx + sp.outer(sky, sp.cos(theta))
            ys = cy + sp.outer(sky, sp.sin(theta))
            xs[1, :] = xs[1, ::-1]
            ys[1, :] = ys[1, ::-1]
            plt.fill(sp.ravel(xs), sp.ravel(ys),
                     edgecolor=skcolor, color=skcolor,
                     alpha=alpha)

            if annotate:
                outer_sky = sky[1]
                plt.gca().annotate(lab,
                                   xy=(cx, cy + outer_sky),
                                   xytext=(cx + 1 * outer_sky,
                                           cy + 1.5 * outer_sky),
                                   fontsize=20)


    def set_max_counts(self, counts):
        self.max_counts = counts


    def photometry(self, aperture=None, sky=None, deg=None, max_counts=None):
        if aperture is not None:
            self.aperture = aperture
        if sky is not None:
            self.sky = sky
        if deg is not None:
            self.deg = deg
        if max_counts is not None:
            self.set_max_counts(self.max_counts)

        if self.aperture is None or self.sky is None:
            raise ValueError("ERROR: aperture photometry parameters are incomplete. Either aperture "
                             "photometry radius or sky annulus were not giving. Please call photometry "
                             "with the following keywords: photometry(aperture=a, sky=s) or define aperture "
                             "and sky when initializing Photometry object.")

        ts = self.CPUphot()
        return ts

    def append(self, sci_files, ignore=None, offsets_xy=None):
        """
Adds more files to photometry
        :param sci_files:
        :param ignore: Keeps the same zero from original serie
        """

        sci_files = dp.AstroDir(sci_files)
        start_frame = len(self.frame_id)

        offset_list = sp.zeros([len(sci_files), 2])
        if offsets_xy is not None:
            for k, v in offsets_xy.items():
                offset_list[k - start_frame, :] = sp.array(v)

        last_coords = [coords[-1] for coords in self.coords_new_xy]

        sci_stamps, coords_new_xy = _get_stamps(sci_files, last_coords,
                                                self.stamp_rad, maxskip=self.maxskip,
                                                mdark=self.mdark, mflat=self.mflat,
                                                recenter=self.recenter,
                                                labels=self.labels,
                                                offsets_xy=offset_list,
                                                logger=self._logger,
                                                ignore=ignore)
        self.sci_stamps = sp.concatenate((self.sci_stamps, sci_stamps), axis=1)
        self.coords_new_xy = sp.concatenate((self.coords_new_xy, coords_new_xy))

        self.frame_id += ["{}/{}".format(d, f) for d, f, i in zip(sci_files.getheaderval('dirname'),
                                                                  sci_files.getheaderval('basename'),
                                                                  range(len(sci_files)))
                          if i not in ignore]
        if isinstance(sci_files, dp.AstroDir):
            self._input_astrodir += sci_files

    def CPUphot(self):
        all_phot = []
        all_err = []
        all_fwhm = []

        print("Processing CPU photometry for {0} targets: ".format(len(self.sci_stamps)), end='')
        sys.stdout.flush()
        for label, target, centers_xy in zip(self.labels, self.sci_stamps, self.coords_new_xy):  # For each target
            t_phot, t_err, t_fwhm = [], [], []

            for data, center_xy, frame_id in zip(target, centers_xy, self.frame_id):
                cx, cy = center_xy

                # Stamps are already centered, only decimals could be different
                cstamp = [self.stamp_rad + cy % 1, self.stamp_rad + cx % 1]

                # Preparing arrays for photometry
                d = dp.radial(data, cstamp)
                dy, dx = data.shape
                y, x = sp.mgrid[-cstamp[0]:dy - cstamp[0], -cstamp[1]:dx - cstamp[1]]

                # Compute sky correction
                # Case 1: sky = [fit, map_of_sky_pixels]
                if isinstance(self.sky[0], sp.ndarray):
                    fit = self.sky[0]
                    idx = self.sky[1]

                # Case 2: sky = [inner_radius, outer_radius]
                else:
                    idx = (d > self.sky[0]) * (d < self.sky[1])
                    if self.deg == -1:
                        fit = sp.median(data[idx])
                    elif self.deg>=0:
                        errfunc = lambda coef, x, y, z: (dp.bipol(coef, x, y) - z).flatten()
                        coef0 = sp.zeros((self.deg, self.deg))
                        coef0[0, 0] = data[idx].mean()
                        fit, cov, info, mesg, success = op.leastsq(errfunc, coef0.flatten(),
                                                                   args=(x[idx], y[idx], data[idx]), full_output=True)
                    else:
                        raise ValueError("invalid degree '{}' to fit sky".format(self.deg))

                # Apply sky subtraction
                n_pix_sky = idx.sum()
                if self.deg ==-1:
                    sky_fit = fit
                elif self.deg>=0:
                    sky_fit = dp.bipol(fit, x, y)
                else:
                    raise ValueError("invalid degree '{}' to fit sky".format(self.deg))

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
                psf = res[d < self.aperture]
                if (psf > self.max_counts).any():
                    logging.warning("Object {} on frame {} has counts above the "
                                    "threshold ({})".format(label, frame_id, self.max_counts))
                phot = float(psf.sum())

                # now the error
                if self.gain is None:
                    error = None
                else:
                    n_pix_ap = (d < self.aperture).sum()
                    error = _phot_error(phot, sky_std, n_pix_ap, n_pix_sky, self.gain, ron=self.ron)

                #                Tracer()()
                t_phot.append(phot)
                t_err.append(error)
                t_fwhm.append(fwhmg)

            all_phot.append(t_phot)
            all_err.append(t_err)
            all_fwhm.append(t_fwhm)
            print('X', end='')
            sys.stdout.flush()

        print('')
        return timeserie.TimeSeries(all_phot, all_err,
                                    labels=self.labels, epoch=self.epoch,
                                    extras={'centers_xy': self.coords_new_xy, 'fwhm': all_fwhm})

    def plot_radialprofile(self, targets=None, xlim=None, axes=1,
                           legend_size=None, frame=0,
                           recenter=True):
        """Plot Radial Profile from data using radialprofile() function
        :param targets: Target specification for re-centering. Either an integer for specific target,
        or a 2-element list for x/y coordinates.
        :type targets: integer/string or 2-element list
        :param xlim:
        :param axes:
        :param legend_size:
        :param frame:
        :param recenter:
    """

        colors = ['rx', 'b^', 'go', 'r^', 'bx', 'g+']
        fig, ax = dp.figaxes(axes)

        ax.cla()
        ax.set_xlabel('distance')
        ax.set_ylabel('ADU')
        if targets is None:
            targets = self.labels
        elif isinstance(targets, str):
            targets = [targets]
        elif isinstance(targets, (list, tuple)) and isinstance(targets[0], (int, )):
            targets = [self.labels[a] for a in targets]
        elif isinstance(targets, int):
            targets = [self.labels[targets]]

        stamp_rad = self.sci_stamps.shape[0]

        for stamp, coords_xy, color, lab in zip(self.sci_stamps, self.coords_new_xy,
                                                colors, targets):
            cx, cy = coords_xy[frame]
            distance, value, center = dp.radial_profile(stamp[frame],
                                                        [stamp_rad+cx % 1, stamp_rad+cy % 1],
                                                        stamp_rad=self.stamp_rad,
                                                        recenter=recenter)
            ax.plot(distance, value, color,
                    label="%s: (%.1f, %.1f)" % (lab,
                                                coords_xy[frame][1],
                                                coords_xy[frame][0]),
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

    def showstamp(self, target=None, stamp_rad=None,
                  first=0, last=-1, figure=None, ncol=None, annotate=True):
        """Show the star at the same position for the different frames

        :param stamp_rad:
        :param annotate:
        :param target: None for the first key()
        :param first: First frame to show
        :param last: Last frame to show. It can be onPython negative format
        :param figure: Specify figure number
        :param ncol: Number of columns
"""
        if target is None:
            target = 0
        elif isinstance(target, str):
            target = self.labels.index(target)

        if last < 0:
            nimages = len(self.sci_stamps[target]) + 1 + last - first
        else:
            nimages = last - first

        if stamp_rad is None or stamp_rad > self.stamp_rad:
            stamp_rad = self.stamp_rad

        if ncol is None:
            ncol = int(sp.sqrt(nimages))
        nrow = int(sp.ceil(float(nimages) / ncol))

        f, ax = plt.subplots(nrow, ncol, num=figure,
                             sharex=True, sharey=True)
        f.subplots_adjust(hspace=0, wspace=0)
        ax1 = list(sp.array(ax).reshape(-1))

        if annotate:
            if not hasattr(self, '_input_astrodir'):
                tmlogger.error(
                    "input to photometry was not a list (AstroDir or Filenames), cannot annotate list position")
            inputs = self._input_astrodir.getheaderval('basename')

        for data, a, frame in zip(self.sci_stamps[target], ax1, self.frame_id):
            #todo: take provisions about star near the border
            dp.imshowz(data,
                       axes=a,
                       cxy=[stamp_rad, stamp_rad],
                       plot_rad=stamp_rad,
                       ticks=False,
                       trim_data=False,
                       force_show=False
                       )

            if annotate:
                idx = inputs.index(os.path.basename(frame))
                a.text(1.5*stamp_rad, 0.2*stamp_rad, idx)

        plt.show()
