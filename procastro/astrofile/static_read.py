from pathlib import Path

import numpy as np
from astropy.io import fits as pf
from astropy.table import Table
from numpy.ma.core import MaskedArray


def static_read(file_type,
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

            return data, meta

        case "ARRAY":
            return filename

        case "ECSV":
            table = Table().read(filename)

            return table, table.meta

    raise TypeError(f"File type {file_type} cannot be read.")


def static_ndarray_to_table(data, file_options=None):
    if isinstance(data, Table):
        return data
    elif isinstance(data, np.ndarray):
        n_axes = len(data.shape)
        nx = data.shape[-1]

        if n_axes == 1:
            data = data.reshape(1, nx)
        # else:
        #     for remove_ax in range((n_axes - 2 > 0) * (n_axes - 2)):
        #         data = data[0]

        n_channels = data.shape[len(data.shape) - 2]

        try:
            column_names = file_options['colnames']
        except KeyError:
            column_names = [f"{i}" for i in range(n_channels)]

        table = Table()
        axis = len(data.shape) - 2
        for idx, name in enumerate(column_names):
            data_channel = np.take(data, idx, axis=axis).transpose()
            table[name] = MaskedArray(data_channel, mask=np.isnan(data_channel))
        return table
    else:
        raise TypeError(f"data type not supported: {type(data)}")
