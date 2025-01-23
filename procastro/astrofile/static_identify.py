import re

import numpy as np


import procastro as pa


def static_identify(filename, options=None):

    # Check if AstroFile
    if isinstance(filename, pa.AstroFile):
        return filename.get_format()

    # Check if np.ndarray
    if isinstance(filename, np.ndarray):
        return "ARRAY"

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
