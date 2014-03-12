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


from astrocalc import AstroCalc
from astroplot import AstroPlot

from astropy.time import Time

import copy
import scipy as sp
import dataproc as dp


class Timeseries(AstroCalc):

    """
    Timeseries class inherited from AstroCalc class.
    The purpose of this class is to organize all the calculation tasks related to the timeseries analysis.
    """

    def __init__(
            self,
            data,
            coordsxy=None,
            labels=None,
            stamprad=20,
            maxskip=6,
            keydate='DATE-OBS',
            keytype='IMAGETYP'):
        """
        Timeseries object constructor.
        :param data: image data for the timeseries
        :type data: str (path to a folder), astrodir, ndarray list (list of images)
        :param coordsxy: list of x,y coordinates of stars in the first image of the timeseries
        :type coordsxy: list of 2-tuples or 2-list
        :param labels: labels for the stars
        :type labels: list of strings
        :param stamprad: radius of the (square) stamp that contains a star
        :type stamprad: int (default value: 20)
        :param maxskip: maximum skip considered (without warning) between the positions of a star in two sucesive images
        :type maxskip: int (default value: 6)
        :param keydate: header key of observation date in the images (allows image sorting in terms of time)
        :type keydate: str
        """

        # data type check
        if isinstance(data, str):  # data is a string (path to a directory)
            self.files = dp.astrodir(data).sort(keydate)
            self.isAstrodir = True

        elif isinstance(data, dp.astrodir):  # data is an astrodir object
            self.files = data.sort(keydate)
            self.isAstrodir = True

        elif isinstance(data, list):  # data is a list of images (ndarray)
            if len(data) == 0:
                raise TypeError('data is an empty list')
            for img in data:
                if not isinstance(img, sp.ndarray):
                    raise TypeError(
                        'data is a list but not all the elements are ndarray')
            self.files = data
            self.isAstrodir = False

        else:
            raise TypeError('data should be str, astrodir or ndarray list')

        # label list
        try:
            if labels is None:
                labels = []
            nstars = len(coordsxy)
            labels = list(
                labels) + sp.arange(len(labels),
                                    nstars).astype(str).tolist()
            targets = {lab: [[coo[0], coo[1]], ]
                       for coo, lab in zip(coordsxy, labels)}
        except:
            raise ValueError(
                "Coordinates of target stars need to be specified as a list of 2 elements, not: %s" %
                (str(coordsxy)))

        # mjd list and science image filtering
        if self.isAstrodir:
            self.mjd = []
            science_files = []
            for astrofile in self.files:
                date_str, type_str = astrofile.getheaderval(keydate, keytype)
                if type_str == 'OBJECT':
                    date_format = "iso"
                    if "T" in date_str or "t" in date_str:
                        date_format += "t"
                    self.mjd.append(
                        Time(date_str,
                             format=date_format,
                             scale='utc').mjd)
                    science_files.append(astrofile)
                else:
                    print "%s is not a science image" % astrofile
            self.files = science_files
        else:
            self.mjd = sp.arange(len(data))

        # instance variables (in addition to self.files, self.isAstrodir and
        # self.mjd)
        self.labels = labels
        self.targets = targets
        self.stamprad = stamprad
        self.maxskip = maxskip
        self.skydata = None

    def perform_phot(
            self,
            aperture,
            sky=None,
            keygain='GTGAIN11',
            keyron='GTRON11'):
        """
        Perform apperture photometry with the images of the Timeseries object.
        :param aperture: aperture photometry radius
        :type aperture: int
        :param sky: inner and outer radius for sky annulus
        :type sky: [int,int]
        :param keygain: header key of gain
        :type keygain: str
        :param keyron: header key of read-out-noise (RON)
        :type keyron: str
        """

        flx = {lab: [] for lab in self.labels}
        err = {lab: [] for lab in self.labels}

        skydata = []
        targets = copy.deepcopy(self.targets)

        if self.isAstrodir:
            for filename, i in zip(self.files, range(len(self.files))):

                if sky is not None:
                    sky_img_dict = {lab: [] for lab in self.labels}

                data, head = filename.reader(datahead=True)
                gain, ron = filename.getheaderval(keygain, keyron)

                for lab, cooxy in targets.items():

                    cx, cy = cooxy[-1]
                    sarr = self.subarray(data, cy, cx, self.stamprad)
                    scy, scx = self.centroid(sarr)

                    skip = sp.sqrt(
                        (self.stamprad - scy) ** 2 + (self.stamprad - scx) ** 2)
                    if skip > self.maxskip:
                        print(
                            "Jump of %f pixels has occurred on frame %s for star %s" %
                            (skip, filename, lab))

                    if sky is not None:
                        phot, phot_err, sky_out = self.apphot(
                            sarr, [scy, scx], aperture, sky, gain=gain, ron=ron)
                        sky_img_dict[lab] = sky_out
                    else:
                        sky_info = self.skydata[i][lab]
                        phot, phot_err, sky_out = self.apphot(
                            sarr, [scy, scx], aperture, sky_info, gain=gain, ron=ron)

                    flx[lab].append(phot)
                    err[lab].append(phot_err)
                    cooxy.append(
                        [cx + scx - self.stamprad, cy + scy - self.stamprad])

                if sky is not None:
                    skydata.append(sky_img_dict)

            if sky is not None:
                self.skydata = copy.deepcopy(skydata)
            return TimeseriesResults(self.mjd, flx, err, targets)

        else:
            for data, i in zip(self.files, range(len(self.files))):

                if sky is not None:
                    sky_img_dict = {lab: [] for lab in self.labels}

                for lab, cooxy in targets.items():

                    cx, cy = cooxy[-1]
                    sarr = self.subarray(data, cy, cx, self.stamprad)
                    scy, scx = self.centroid(sarr)

                    skip = sp.sqrt(
                        (self.stamprad - scy) ** 2 + (self.stamprad - scx) ** 2)
                    if skip > self.maxskip:
                        print(
                            "Jump of %f pixels has occurred on frame %s for star %s" %
                            (skip, filename, lab))

                    if sky is not None:
                        phot, phot_err, sky_out = self.apphot(
                            sarr, [scy, scx], aperture, sky, gain=None, ron=None)
                        sky_img_dict[lab] = sky_out
                    else:
                        sky_info = self.skydata[i][lab]
                        phot, phot_err, sky_out = self.apphot(
                            sarr, [scy, scx], aperture, sky_info, gain=None, ron=None)

                    flx[lab].append(phot)
                    # err[lab].append(phot_err) No info for error calculation
                    cooxy.append(
                        [cx + scx - self.stamprad, cy + scy - self.stamprad])

                if sky is not None:
                    skydata.append(sky_img_dict)

            if sky is not None:
                self.skydata = copy.deepcopy(skydata)
            return (
                TimeseriesResults(self.mjd, flx, None, targets)  # err = None
            )

import matplotlib.pyplot as plt


class TimeseriesResults(AstroPlot):

    """
    TimeseriesResults class inherited from AstroPlot class.
    The purpose of this class is to centralize the data output (mainly the plotting routines).
    """

    def __init__(self, mjd, flx, err, targets):
        """
        TimeseriesResult object constructor (inherited from AstroPlot).
        :param mjd: date array
        :type mjd: array
        :param flx: flux array dictionary
        :type flx: dict
        :param err: flux error array dictionary
        :type err: dict
        :param targets: coordinates array dictionary
        :type targets: dict
        """

        super(TimeseriesResults, self).__init__(mjd, flx, err, targets)

    def plot_ratio(self, trg=None, ref=None, normframes=None):
        """
        Display the ratio of science and reference
        :param trg: label name of target star
        :type trg: string (key of flx and coo)
        :param ref: list of stars to be used as reference
        :type ref: None or list of strings. If None, then all stars except target
        :param normframes: list of frames to normalize (for example, in case only out of transit want to be considered)
        :type normframes: None or boolean array. If None, then normalize by all frames
        """

        if self.ratio is None:
            if trg is None:
                trg = self.flx.keys()[0]
            self.doratio(trg=trg, ref=ref, normframes=normframes)

        plt.plot(self.ratio)
        plt.title("Ratio for target = " + str(trg))
        plt.show()
        return

    def plot_timeseries(self):
        """
        Display the timeseries data: flux (with errors) as function of mjd
        """

        fig = plt.figure(figsize=(4, 5), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        for lab in self.flx.keys():
            if self.err is None:
                ax.errorbar(
                    self.mjd,
                    self.flx[lab],
                    yerr=None,
                    marker="o",
                    label=lab)
            else:
                ax.errorbar(
                    self.mjd,
                    self.flx[lab],
                    yerr=self.err[lab],
                    marker="o",
                    label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        plt.show()
        return

    def plot_drift(self):
        """
        Show the drift of the stars using the coordinates obtained for every image in the timeseries.
        """

        lines = []
        leg = []
        for k in self.flx.keys():
            cooxy = self.cooxy[k]
            xd, yd = sp.array(cooxy).transpose()
            l = plt.plot(xd - xd[0], yd - yd[0],
                         label='%8s. X,Y: %-7.1f,%-7.1f' % (k, xd[0], yd[0]))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .302), loc=3,
                   ncol=1, mode="expand", borderaxespad=0.,
                   prop={'size': 6})
        plt.show()
        return
