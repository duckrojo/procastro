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


def trim_to_python(value, maxlims=None):
    result = re.search(r'\[(\d+):(\d+),(\d+):(\d+)\]', value).groups()
    if maxlims is None:
        return int(result[2]), int(result[3]), int(result[0]), int(result[1])

    ret = int(result[2]), int(result[3]), int(result[0]), int(result[1])
    return (ret[0], ret[1] if maxlims[0] > ret[1] else maxlims[0] + ret[0] - 1,
            ret[2], ret[3] if maxlims[1] > ret[3] else maxlims[1] + ret[2] - 1)


def python_to_trim(trim):
    if trim is None:
        return None
    return f'[{str(trim[2])}:{str(trim[3])},{str(trim[0])}:{str(trim[1])}]'


def common_trim_fcn(trim_all):
    trim = [t for t in trim_all if t is not None]
    if not len(trim):
        return None
    result = (np.array(trim) * np.array([1, -1, 1, -1])).max(0)
    return result[0], -result[1], result[2], -result[3]


def extract_common(tdata, trim, common_trim):
    # accommodating to use the same operators both sides of the array
    if trim is None:
        return tdata, False

    delta = trim * np.array([1, -1, 1, -1]) - common_trim * np.array([1, -1, 1, -1])
    if np.all(delta == 0):
        return tdata, False
    else:
        delta = list(delta)
        # if there is no trimming at the end of the array
        delta[1] = None if delta[1] == 0 else delta[1]
        delta[3] = None if delta[3] == 0 else delta[3]

        ret = tdata[-delta[0]:delta[1], -delta[2]:delta[3]]
        return ret, True
