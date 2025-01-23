import re

import numpy as np
from astropy.table import Table

_format_fcn = {'d': int, 'f': float, 's': str}


PADataReturn = np.ndarray | Table  # returns an array if there is no spectral information, otherwise table


def upper(x):
    return x.upper()


def identity(x):
    return x


def dict_from_pattern(pattern, string, key_to_upper=True):
    compiled_pattern = re.compile(re.sub(r"{(\w+?)(:.+?)?}", r"(?P<\1>.+?)", pattern))
    casts = re.findall(r"{(\w+?)(?::.*?(\w))?}", pattern)
    transform = {k: _format_fcn[v] if v in _format_fcn else str for k, v in casts}
    match = re.search(compiled_pattern, string)
    if match is None:
        return {}

    groups = match.groupdict()
    caser = upper if key_to_upper else identity

    casted_match = {caser(k): transform[k](match[k]) for k, v in groups.items()}

    return casted_match
