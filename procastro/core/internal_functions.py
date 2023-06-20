import re
import numpy as np


def trim_to_python(value):
    result = re.search(r'\[(\d+):(\d+),(\d+):(\d+)\]', value).groups()
    return int(result[2]), int(result[3]), int(result[0]), int(result[1])


def common_trim_fcn(trim_all):
    trim = [t for t in trim_all if t is not None]
    result = (np.array(trim) * np.array([1, -1, 1, -1])).max(0)
    return result[0], -result[1], result[2], -result[3]


def extract_common(tdata, trim, common_trim):
    # reacommodating to use the same operators both sides of the array
    delta = trim * np.array([1, -1, 1, -1]) - common_trim * np.array([1, -1, 1, -1])
    if np.all(delta == 0):
        return tdata, False
    else:
        delta = list(delta)
        # if there is no trimming at the end of the array
        delta[1] = None if delta[1] == 0 else delta[1]
        delta[3] = None if delta[3] == 0 else delta[3]

        return tdata[-delta[0]:delta[1], -delta[2]:delta[3]], True
