__author__ = 'fran'

import CPUmath
import pyfits as pf
from os import listdir
from os.path import isfile, join
import warnings
from dataproc.core import io_file


class Reduction(object):
    def __init__(self, raw, bias, dark, flat, combine_mode=CPUmath.mean_combine):
        """
        :param raw: nparray of sci images
        :param bias: nparray of bias images
        :param dark: nparray of dark images
        :param flat: nparray of flat images
        :param combine_modes: combine function. default is CPUmath.mean
        :return:
        """
        self.bias = bias
        self.dark = dark
        self.flat = flat
        self.raw = raw
        self.combine_type = combine_mode
        self.mb = None
        self.mf = None
        self.md = None

    def get_masterbias(bias, save=False):
        """ Returns masterbias, combining all bias files on the bias
            path using mean or median, if established. If not, uses
            mean as default. Returns masterbias file and brings option
            to save it to .fits file.
        :return: np.ndarray (master bias)
        """

        warnings.warn('Combine type not defined for bias. Will use mean as default.')
        master_bias = CPUmath.mean_combine(self.bias)

        if save:
            hdu = pf.PrimaryHDU(master_bias)
            hdu.writeto('MasterBias.fits')

        self.mb = master_bias


    def get_masterflat(self, save=False):

        warnings.warn('Combine type not defined for flats. Will use mean as default.')
        master_flat = CPUmath.mean_combine(self.flat)

        if save:
            hdu = pf.PrimaryHDU(master_flat)
            hdu.writeto('MasterFlat.fits')

        self.mf = master_flat

    def get_masterdark(self, save=False):
        warnings.warn('Combine type not defined for darks. Will use mean as default.')
        master_dark = CPUmath.mean_combine(self.dark)

        if save:
            hdu = pf.PrimaryHDU(master_dark)
            hdu.writeto('MasterDark.fits')

        self.md = master_dark

    def reduce(self):
        for f in self.sci:
            f = (f - self.md - self.mb) / self.mf

        return self.sci
