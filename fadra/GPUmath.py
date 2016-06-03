__author__ = 'fran'

import pyopencl as cl
import warnings
import scipy as sp

def _kernel_warning(k):
    message = "Given kernel size is not odd. (kernel_size+1)=" + str(k) + " will be used."
    warnings.warn(message, Warning)

def _check_array(function):
    """ Decorator to check if input is one array only.
        Else, raises IOError.
    """
    @_wraps(function)
    def array_check(instance, *args, **kwargs):
        if len(args) > 1:  # Files come separately, not in an array.
            raise IOError("Files to be combined must come in an array.")
        return function(instance, *args, **kwargs)
    return array_check

def _check_combine_input(function):
    """ Decorator that checks all images in the input are
        of the same size and type. Else, raises an error.
    """
    @_wraps(function)
    def input_checker(instance, *args, **kwargs):
        try:
            init_shape = instance[0].shape
        except AttributeError:
            raise IOError("First file to be combined in CPUmath.%s is not a supported type." % function.__name__)

        init_type = type(instance[0])

        for i in instance:
            try:
                i_shape = i.shape
            except AttributeError:
                raise IOError("File to be combined in CPUmath.%s is not a supported type." % function.__name__)

            if i_shape == init_shape and type(i) is init_type:
                continue
            else:
                if i_shape != init_shape:
                    raise IOError("Files to be combined in CPUmath.%s are not all of the same shape." % function.__name__)
                else:
                    raise IOError("Files to be combined in CPUmath.%s are not all of the same type." % function.__name__)
        return function(instance, *args, **kwargs)
    return input_checker


@_check_array
@_check_combine_input
def mean_combine(imgs, sigmaclip=False):
    """ Combines an array of images, using mean. All images on the
        array must be the same size, otherwise an error is raised.
        Returns the combined image.
    :param args: array of np.ndarrays
    :return: np.ndarray
    """
    mean_comb = sp.mean(imgs, axis=0)
    if sigmaclip:
        return sigma_clipping(mean_comb)
    else:
        return mean_comb


@_check_array
@_check_combine_input
def median_combine(imgs, sigmaclip=False):
    """ Combines an array of images, using median. All images on the
        array must be the same size, otherwise an error is raised.
        Returns the combined image.
    :param args: array of np.ndarrays
    :return: np.ndarray
    """
    from numpy import median
    median_comb = median(imgs, axis=0)
    if sigmaclip:
        return sigma_clipping(median_comb)
    else:
        return median_comb

@_check_array
@_check_combine_input
def add_images(*args):
    """ Sums images on array.
    :param args: array of np.ndarrays
    :return: np.ndarray
    """
    return sp.sum(*args, axis=0)

def sigmask(image, sigma=3, eps=0.05, iterations=1):
    # TODO chequear que usuario entrege valores de iterations y de eps antes de empezar!
    # TODO chequear condiciones de termino...
    # for it in range(iterations):
    mean = sp.mean(image)
    std_dev = sp.std(image)
    min_sig = mean - sigma*std_dev
    max_sig = mean + sigma*std_dev
    mask = [sp.logical_and((image < min_sig), (image > max_sig))]
    new_stdev = sp.std(image)
    tres = (std_dev - new_stdev)/std_dev
    if tres < eps:
        return mask

def sigma_clipping(image, funct, sigma=3, eps=0.05):
    mask = sigmask(image, sigma, eps)
    # positions = np.transpose(np.nonzero(image*mask))
    # values = [small_mean(image, p) for p in positions]
    return [image, mask]


def shift_generator(a, k, m):
    """ Generates all the shifts needed to do the mean and median filters.
    :param a: original array
    :param k: kernel size, but actually it is the border of the kernel. kernel radius? must be even number
    :return: list with all the shifted np arrays
    """

    if k % 2 == 0:
        k2 = k/2
    else:
        k2 = (k + 1)/2

    # TODO try to get rid of this kind of ifs
    if m == 'constant':
        p = sp.pad(a, (k, k), 'constant', constant_values=0)  # Fully padded array
    else:
        p = sp.pad(a, (k, k), 'edge')

    # "Linear" shifts
    top_shift = p[k:, k2:-k2]  # lime green
    bottom_shift = p[:-k, k2:-k2]  # hot pink
    right_shift = p[k2:-k2, :-k]  # blue
    left_shift = p[k2:-k2, k:]  # orange

    if k % 2 == 1:  # Correction for odd k
        col = sp.zeros((top_shift.shape[0], 1))
        row = sp.zeros((1, top_shift.shape[0]))
        top_shift = sp.concatenate((col, top_shift), 1)
        bottom_shift = sp.concatenate((bottom_shift, col), 1)
        right_shift = sp.concatenate((row, right_shift), 0)
        left_shift = sp.concatenate((left_shift, row), 0)

    # Diagonal shifts
    if m == 'constant':
        top_left = sp.pad(a, (0, k), 'constant', constant_values=0)  # pink
    else:
        top_left = sp.pad(a, (0, k), 'edge')
    top_right = p[k:, :-k]  # purple
    bottom_left = p[:-k, k:]  # dark green
    if m == 'constant':
        bottom_right = sp.pad(a, (k, 0), 'constant', constant_values=0)  # sky blue
    else:
        bottom_right = sp.pad(a, (k, 0), 'edge')

    return [top_shift, bottom_shift, right_shift, left_shift, top_left, top_right, bottom_left, bottom_right]


def median_filter(image, kernel_size):
    if kernel_size % 2 == 0:
        k = kernel_size - 1
        kernel_size += 1
        _kernel_warning(kernel_size)
    else:
        k = (kernel_size - 1)/2

    c = shift_generator(image, k, 'edge')

    from numpy import median
    d = median(c, axis=0)
    dif = d.shape[0] - image.shape[0]
    print(d.shape)
    print("dif = " + str(dif))
    if dif % 2 == 0:
        return d[dif/2:-dif/2, dif/2:-dif/2]
    else:
        return d[dif-1:-1, dif-1:-1]


def mean_filter(image, kernel_size):
    if kernel_size % 2 == 0:
        k = kernel_size/2
        kernel_size += 1
        _kernel_warning(kernel_size)
    else:
        k = (kernel_size - 1)/2

    c = shift_generator(image, k, 'constant')
    d = sp.mean(c, axis=0)
    dif = d.shape[0] - image.shape[0]
    print(d.shape)
    print("dif = " + str(dif))
    if dif % 2 == 0:
        return d[dif/2:-dif/2, dif/2:-dif/2]
    else:
        return d[dif-1:-1, dif-1:-1]
