import re
import numpy as np
from astropy import table

import procastro as pa


def identify(filename, options=None):

    # Check if AstroFile
    if isinstance(filename, pa.AstroFile):
        return filename.get_format()

    # Check if np.ndarray
    if isinstance(filename, np.ndarray):
        return "ARRAY"

    # Check if Table
    if isinstance(filename, table.Table):
        return "TABLE"

    # Check if TXT file
    match = re.search(r'\.txt(\.gz)?(:.+)?$', str(filename), re.IGNORECASE)
    if match is not None:
        return "TXT"

    # Check if FITS file
    match = re.search(r'\.fits(\.gz)?(:.+)?$', str(filename), re.IGNORECASE)
    if match is not None:
        return "FITS"

    # check if ECSV
    match = re.search(r'\.ecsv(\.gz)?(:.+)?$', str(filename), re.IGNORECASE)
    if match is not None:
        return "ECSV"

    raise ValueError(f"Filename '{filename}' does not match any known format:"
                     f"ARRAY, "
                     f"FITS, "
                     f"ECSV"
                     f"")
