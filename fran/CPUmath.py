__author__ = 'fran'

from functools import wraps as _wraps
import numpy as np


def _check_input(function):
    """ Decorator that checks all images in the input are
        of the same size and type. Else, raises an error.
    """
    @_wraps(function)
    def input_checker(instance, *args, **kwargs):
        init_shape = instance[0].shape
        init_type = type(instance[0])
        for i in instance:
            if i.shape == init_shape and type(i) is init_type:
                continue
            else:
                raise IOError("Files to be combined are not of the same shape or type.")
        return function(instance, *args, **kwargs)
    return input_checker

@_check_input
def mean_combine(*args):
    """ Combines an array of images, using mean. All images on the
    array must be the same size, otherwise an error is raised.
    Returns the combined image.
    :param args: array of .FITS images, astrofiles, or an astroarray
    :return: if args is an array of FITS files, a FITS file. If args
            is an array of astrofiles or an astroarray, an astrofile.
            The header of the returned file is the header of first image.
    """
    if len(args) > 1:  # Files come separately, not in an array. I'm not sure if this should be allowed.
        new_array = np.array(args)
    else:
        new_array = np.vstack(args)

    return np.mean(new_array, axis=0)


@_check_input
def median_combine(*args):
    """ Combines an array of images, using median. All images on the
    array must be the same size, otherwise an error is raised.
    Returns the combined image.
    :param args: array of .FITS images, astrofiles, or an astroarray
    :return: if args is an array of FITS files, a FITS file. If args
            is an array of astrofiles or an astroarray, an astrofile.
            The header of the returned file is the header of first image.
    """
    if len(args) > 1:  # Files come separately, not in an array. I'm not sure if this should be allowed.
        new_array = np.array(args)
    else:
        new_array = np.vstack(args)

    return np.median(new_array, axis=0)