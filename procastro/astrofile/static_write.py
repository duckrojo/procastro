from astropy.io import fits as pf
from astropy import time as apt
from pandas.io.pytables import Table

import procastro as pa


def static_write(file_type, filename, data, meta,
                 overwrite=False,
                 ):
    match file_type:
        case "FITS":
            header = pf.Header(meta)
            header['history'] = "Saved by procastro v{} on {}".format(pa.__version__,
                                                                      apt.Time.now())

            return pf.writeto(filename, data, header,
                              overwrite=overwrite)

        case "ARRAY":
            raise TypeError("File type 'ARRAY' cannot save to file. Specify file_type explicitly")

        case "ECSV":
            data: Table
            data.meta = meta
            return data.write(filename, format='ascii.ecsv')

    raise TypeError(f"File type {file_type} cannot be written.")
