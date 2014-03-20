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

import scipy as sp


class AstroCalc(object):

    # TO DO: REVIEW AND CORRECT !!
    def imagereduction(self, image, bias, flat, dark=None):
        """Returns a reduced image.

        :param image: image to be reduced
        :type image: array
        :param bias: bias image for reduction
        :type bias: array
        :param flat: flat image for reduction
        :type flat: array
        :param dark: dark image for reduction
        :type dark: array
        :rtype: array
        """

        if dark is None:
            return (image - bias) / flat
        else:
            return (image - dark) / (flat - bias)

    def masterimage(self, image_list, mode='median'):
        """Returns a calibration master image.

        :param image_list: list of calibration images
        :type image_list: list of arrays
        :param mode: obtention mode for the master image (the possible values are 'median' (default) and 'mean').
        :type mode: str
        :rtype: array
        """

        if len(image_list) == 0:
            return None
        if mode == 'median':
            return sp.median(image_list, axis=0)  # MEDIAN
        elif mode == 'mean':
            return sp.mean(image_list, axis=0)  # MEAN
        else:
            raise ValueError(
                "Unknown mode '%s' for master image obtention" %
                mode)

    @staticmethod
    def centraldistances(data, c):
        """Computes distances for every matrix position from a central point c.

        :param data: array
        :type data: array
        :param c: center coordinates
        :type c: [float,float]
        :rtype: array
        """

        dy, dx = data.shape
        y, x = sp.mgrid[0:dy, 0:dx]
        return sp.sqrt((y - c[0]) * (y - c[0]) + (x - c[1]) * (x - c[1]))

    @staticmethod
    def bipol(coef, x, y):
        """Polynomial fit for sky sustraction

        :param coef: sky fit polynomial coefficientes
        :type coef: array
        :param x: horizontal coordinates
        :type x: array
        :param y: vertical coordinates
        :type y: array
        :rtype: array
        """

        plane = sp.zeros(x.shape)
        deg = sp.sqrt(coef.size).astype(int)
        coef = coef.reshape((deg, deg))
        if (deg * deg != coef.size):
            print "Malformed coefficient: " + str(coef.size) + \
                "(size) != " + str(deg) + "(dim)^2"
        for i in sp.arange(coef.shape[0]):
            for j in sp.arange(i + 1):
                plane += coef[i, j] * (x ** j) * (y ** (i - j))

        return plane

    @staticmethod
    def phot_error(phot, sky, n_pix_ap, n_pix_sky, gain, ron=None):
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

        if ron is None:
            print "Photometric error calculated without RON"
            ron = 0.0

        N_e_star = gain * phot
        N_e_sky = gain * sky
        SNR = N_e_star / \
            sp.sqrt(N_e_star + n_pix_ap *
                    (1.0 + float(n_pix_ap) / n_pix_sky) * (N_e_sky + ron))
        return phot / SNR

    def subarray(self, arr, y, x, rad):
        """Returns a subarray of arr (with size 2*rad+1 and centered in (y,x))

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

        return arr[(y - rad):(y + rad + 1), (x - rad):(x + rad + 1)]

    def apphot(self, data, cs, sap, skydata, deg=3, gain=None, ron=None):
        """Do aperture photometry on data array

        :param data: image
        :type data: array
        :param cs: coordinates
        :type cs: [float,float]
        :param sap: apperture
        :type sap: float
        :param skydata: inner and outer radius for sky annulus or sky fit (coefficientes) and number of sky pixels
        :type skydata: [float,float] or [array,int]
        :param deg: degree of the polynomial for the sky fit
        :type deg: int
        :param gain: gain
        :type gain: float
        :param ron: read-out-noise
        :type ron: float (default value: None)
        :rtype: [float,float,[coeff_list,int]]
        """

        d = self.centraldistances(data, cs)
        dy, dx = data.shape
        y, x = sp.mgrid[-cs[0]:dy - cs[0], -cs[1]:dx - cs[1]]

        # Case 1: skydata = [fit,number_of_sky_pixels]
        if isinstance(skydata[0], sp.ndarray):
            fit = skydata[0]
            n_pix_sky = skydata[1]

        else:  # Case 2: skydata = [inner_radius,outer_radius]
            import scipy.optimize as op
            idx = (d > skydata[0]) * (d < skydata[1])
            errfunc = lambda coef, x, y, z: (
                self.bipol(coef, x, y) - z).flatten()
            coef0 = sp.zeros((deg, deg))
            coef0[0, 0] = data[idx].mean()
            fit, cov, info, mesg, success = op.leastsq(
                errfunc, coef0.flatten(), args=(x[idx], y[idx], data[idx]), full_output=1)
            n_pix_sky = idx.sum()

        sky = self.bipol(fit, x, y)
        res = data - sky  # minus sky
        res = res * (res > 0)
        phot = float(res[d < sap].sum())

        if gain is None:
            print "Not GAIN provided for error calculation"
            return phot, None, [fit, n_pix_sky]

        n_pix_ap = (d < sap).sum()
        error = self.phot_error(
            phot,
            sky.mean(),
            n_pix_ap,
            n_pix_sky,
            gain,
            ron=ron)
        return phot, error, [fit, n_pix_sky]

    def centroid(self, arr):
        """Find centroid of small array

        :param arr: array
        :type arr: array
        :rtype: [float,float]
        """

        med = sp.median(arr)
        arr = arr - med
        arr = arr * (arr > 0)

        iy, ix = sp.mgrid[0:len(arr), 0:len(arr)]

        cy = sp.sum(iy * arr) / sp.sum(arr)
        cx = sp.sum(ix * arr) / sp.sum(arr)

        return cy, cx
