import re

import numpy as np


def static_identify(filename, options=None):
    # Check if np.ndarray

    if isinstance(filename, np.ndarray):
        return "ARRAY"

    # Check if FITS file

    match = re.search(r'\.fits(\.gz)?(:.+)?$', str(filename), re.IGNORECASE)
    if match is not None:
        return "FITS"

    raise ValueError(f"Filename '{filename}' does not match any known format:"
                     f"ARRAY,"
                     f"FITS,"
                     f"")
