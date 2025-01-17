from typing import Any
from astropy.io import fits as pf
import astropy.time as apt

from procastro.core.astrofile.base import AstroFileBase
from procastro.core.cache import astrofile_cache
import procastro as pa


class AstroFileFITS(AstroFileBase):
    def __init__(self, filename):
        super().__init__(filename)

    def writer(self, filename=None, overwrite=False):
        """
        Writes 'data' to specified file

        Parameters
        ----------
        filename : str
        data : array_like
        """

        if filename is None:
            filename = self._data_file

        header = pf.Header(self._meta)
        header['history'] = "Saved by procastro v{} on {}".format(pa.__version__,
                                                                  apt.Time.now())

        return pf.writeto(filename, self.data, header,
                          overwrite=overwrite)  # , output_verify='silentfix')

    @astrofile_cache
    @property
    def data(self, update_meta=False):
        elements = self._data_file.split(":")
        if len(elements) == 1:
            hdu = 0
        else:
            hdu = int(elements[1])
        unit = pf.open(self._data_file)[hdu]

        if update_meta:
            self._meta = dict(unit.header)

        return unit.data
