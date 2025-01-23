from pathlib import Path

from astropy.io import fits as pf
from astropy.table import Table


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
