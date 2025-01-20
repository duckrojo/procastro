from pathlib import Path

import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt

from procastro.core.astrofile.base import PADataReturn, AstroFile
from procastro.core.logging import io_logger


__all__ = ['AstroFileSpec']


class AstroFileSpec(AstroFile):
    def __init__(self, filename, **kwargs):
        AstroFile.__init__(self, filename, **kwargs)

        self._spectral_axis = True

    def read(self) -> PADataReturn:
        data = super().read()

        n_axes = len(data.shape)
        nx = data.shape[-1]

        if n_axes == 1:
            data = data.reshape(1, nx)
        else:
            for remove_ax in range((n_axes - 2 > 0) * (n_axes - 2)):
                data = data[0]

        n_channels = data.shape[0]

        try:
            column_names = self._data_file_options['colnames']
        except KeyError:
            column_names = [f"{i}" for i in range(n_channels)]

        table = Table()
        for name, column in zip(column_names, data):
            table[name] = column
        return table

    def add_calib(self, calib):

        from procastro import CalibRaw2D

        if isinstance(calib, CalibRaw2D) and self._spectral_axis:
            io_logger.warning("Cannot add CalibRaw2D calibration to spectral file")
            return self

        super().add_calib(calib)

        return self

    def plot(self, channel=0, dispersion='pix'):
        data = self.data

        if dispersion in data.colnames:
            x = data[dispersion]
            x_label = dispersion
        else:
            x = np.arange(len(data))
            x_label = ""

        if isinstance(channel, int):
            channel = data.colnames[channel]
        elif not isinstance(channel, str):
            raise ValueError("channel with spectral data can only the name of the column to plot (str),"
                             " or the position in .colnames (int)")

        plt.plot(x, data[channel])
        plt.xlabel(x_label)


if __name__ == '__main__':
    import procastro as pa

    sp = pa.AstroFileSpec("../../../sample_files/arc.fits")
    sp.plot()



