from pathlib import Path

import numpy as np
from astropy.io import fits as pf
from astropy.table import Table
from numpy.ma.core import MaskedArray

from .meta import CaseInsensitiveMeta


def read(file_type,
         filename):

    match file_type:
        case "FITS":
            elements = Path(filename).name.split(":")

            if len(elements) == 1:
                hdu = 0
            else:
                hdu = int(elements[1])

            with pf.open(filename) as hdulist:
                unit = hdulist[hdu]
                data, meta = unit.data, unit.header

            return data, CaseInsensitiveMeta(meta)

        case "ARRAY":
            return filename, CaseInsensitiveMeta({})

        case "TABLE":
            return filename, CaseInsensitiveMeta(filename.meta)

        case "TXT":
            data = np.loadtxt(filename, unpack=True)

            return data, CaseInsensitiveMeta({})

        case "ECSV":
            table = Table().read(filename)

            return table, CaseInsensitiveMeta(table.meta)

    raise TypeError(f"File type {file_type} cannot be read.")


def ndarray_to_table(data, file_options=None):
    if isinstance(data, Table):
        return data
    elif isinstance(data, np.ndarray):
        n_axes = len(data.shape)
        nx = data.shape[-1]

        if n_axes == 1:
            data = data.reshape(1, nx)

        n_channels = data.shape[len(data.shape) - 2]

        try:
            column_names = file_options['colnames']
        except KeyError:
            column_names = [f"{i}" for i in range(n_channels)]

        table = Table()
        axis = len(data.shape) - 2
        newshape = list(data.shape)[::-1]
        newshape.pop(1)
        for idx, name in enumerate(column_names):
            data_channel = np.take(data, idx, axis=axis).transpose()
            table[name] = MaskedArray(data_channel, mask=np.isnan(data_channel))
        if 'pix' not in table.colnames:
            table['pix'] = np.reshape(np.arange(len(table)),
                                      [-1] + [1]*(len(newshape)-1)
                                      ) * np.ones(newshape)

        return table
    else:
        raise TypeError(f"data type not supported: {type(data)}")
