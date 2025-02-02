from typing import Callable

import numpy as np
from numpy.ma import MaskedArray
from scipy import interpolate, stats
from scipy.interpolate import dfitpack
from scipy.optimize import curve_fit
from scipy.stats import linregress

import matplotlib.pyplot as plt


def plot_fitline(x, ydata, ymodel, label='', filename='temp.png'):
    plt.clf()
    plt.plot(x, ydata, label="data")
    plt.plot(x, ymodel, label="model")
    plt.xlabel(label)
    plt.legend()

    plt.savefig(filename)


def initialize_p0_bounds(variables, y):
    """
    Initialize the initial guesses (p0), bounds, and the fitting mask for the variables.

    This function prepares the initial guess values, bounds, and fitting mask for a
    nonlinear fitting process based on the given variables and an array of dependent
    values. The `variables` parameter defines initial values, fixed values, or ranges
    for the fitting parameters, while the dependent `y` values are used for evaluating
    callable initial guesses or bounds.

    Parameters:
    variables : list[Union[int, float, tuple, list, NoneType]]
        A list defining the initial guesses, bounds, or fixed parameters for the
        fitting. For each variable, each element can be:
        - A numeric value (int or float) indicating a fixed initial guess.
        - A tuple (initial_guess, [lower_bound, upper_bound]) specifying a that the initial
          guess should not be fitted. Lower and upper bounds values are ignored if given.
        - A list (initial_guess, lower_bound, upper_bound) to define a range while also
          allowing optimization (variable will be treated as free to optimize). If only
          an initial guess is provided, no bounds are enforced.
        - Any of the above initial guesses can be callable for dynamic dependency
          calculations using `y` for the initial_guesses, or the initial_guesses for
          the bounds.
        - None for no guess initialization for that parameter.

    y : array-like
        Dependent values used to evaluate callable initial guesses or boundary functions
        for each data point.

    Returns:
    tuple
        params:  ndarray(dtype=bool, shape=(n,))
            List with initial guesses as can be used to construct the fitting function, fitted
            (masked) parameters should be overwritten during the fitting procedure.
        mask : ndarray(dtype=bool, shape=(n,))
            A boolean mask indicating which parameters are free for optimization versus
            those that are fixed.
        p0s : ndarray(dtype=float, shape=(len(y), number_of_free_parameters))
            The initial guesses for the free parameters, evaluated for each value in `y`.
        bounds : ndarray(dtype=float), shape=(len(y), 2, number_of_free_parameters)
            A list containing arrays of shape (2, number_of_free_parameters) for each
            value in `y`, representing the lower and upper bounds for the free parameters.
    """
    n = len(variables)
    do_fitting = [True] * n
    bound = [[-np.inf] * n, [np.inf] * n]
    initial_guesses = [0] * n

    for i, var in enumerate(variables):
        if isinstance(var, (tuple, list)):
            initial_guesses[i] = var[0]
            if len(var) == 3:
                bound[0][i] = var[1]
                bound[1][i] = var[2]
            do_fitting[i] = isinstance(var, list)

        elif var is not None:
            initial_guesses[i] = var

    mask = np.array(do_fitting)

    p0s = []
    bounds = []
    for yy in y:
        # aa = initial_guesses[1](yy)
        evaluated_guesses = np.array([pp(yy) if callable(pp) else pp for pp in initial_guesses])
        p0s.append(evaluated_guesses[mask])
        evaluated_bounds = np.array([[bb(pp) if callable(bb) else bb
                                      for pp, bb in zip(evaluated_guesses, b)]
                                     for b in bound])
        bounds.append(evaluated_bounds[:, mask])

    return (np.array([0 if callable(g) else g for g in initial_guesses], dtype=float),
            mask,
            np.array(p0s, dtype=float),
            np.array(bounds))


def _force_increasing(x: np.ndarray,
                      y: np.ndarray,
                      ):
    """
    Ensure an array x is in increasing order along with corresponding y values.

    This function checks if the `x` array is in increasing order. If `x` is
    in decreasing order, it reverses the order of both `x` and `y`. If `x`
    is unsorted (neither strictly increasing nor decreasing), it raises a
    ValueError. This ensures that `x` is monotonically increasing for
    further processing.

    Parameters
    ----------
    x : numpy.ndarray
        The array of x values to be checked and potentially reversed.
    y : numpy.ndarray
        The array of y values to be reversed alongside x if necessary.

    Returns
    -------
    tuple
        A tuple containing the potentially reversed x and y arrays.
        Both arrays will have x in increasing order.

    Raises
    ------
    ValueError
        If the x array is neither in increasing nor decreasing order.
    """
    if not all(x[:-1] <= x[1:]):
        x = x[::-1]
        y = np.flip(y, axis=1)
        if not all(x[:-1] <= x[1:]):
            raise ValueError("X must be sorted either increasing or decreasing order")
    return x, y


def _remove_unused_dimension(func):
    """Prepare plotting parameters"""

    def wrapper(self, x):

        ret = func(self, x)
        if ret.shape[0] == 1:
            return ret[0]

        return ret

    return wrapper


class FcnWavSol:
    def __init__(self, x, y):
        xx = np.asarray(x)
        yy = np.asarray(y)
        n = xx.shape[-1]
        if len(yy.shape) == 1:
            yy = yy.reshape(1, n)
        if len(yy.shape) != 2 or yy.shape[-1] != n:
            raise TypeError(f"'y' array (shape: {yy.shape})can only be 1 or 2D, with the length of the last"
                            " dimension matching the length of 'x'")

        self.x = xx
        self.y = yy

        self._config = None

    def short(self):
        return str(self)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Error. The fitting function {self.__class__} needs to"
                                  f" implement __call__ before being able to use it.")

    def config(self):
        return self._config

    def set_config(self, name):
        self._config = name


class GenNorm(FcnWavSol):
    def __str__(self):
        return "Generalized Normal"

    def short(self):
        return "GenNorm"

    def __init__(self, x, y, c=None, h=None, w=None, b=(2,), o=None, uncertainty=5):
        """
Initialize and give initial guessses
        :param x:
        :param y:
        :param b:
            Initial value for beta
            If passed as the first element of a tuple, then it is not varied
        :param w:
           initial width, if not given then it is assumed to be one fourth of the total x extent.
            If passed as the first element of a tuple, then it is not varied
        """
        super().__init__(x, y)
        x, y = _force_increasing(self.x, self.y)

        c = c or [x[len(x) // 2],
                  lambda v: v-uncertainty, lambda v: v+uncertainty]   # center's default is middle of array
        h = h or [lambda v: max(v[np.argmin(np.abs(c[0] - uncertainty - x)):
                                  np.argmin(np.abs(c[0] + uncertainty - x))]),
                  lambda v: 0.5*v, lambda v: 1.5*v]  # height's default
        w = w or [10, 3, 40]  # width's default
        o = o or [min, 0, lambda v: 2*v]  # baseline default

        params, mask, p0s, bounds = initialize_p0_bounds([c, h, w, b, o], y)  # center, height, width, beta

        def function_ret(xx, *args):
            params[mask] = np.array(args)
            return params[1] * 2.0 * stats.gennorm.pdf((xx - params[0]) / params[2],
                                                 params[3]) + params[4]

        popts, pcovs = tuple(zip(*[curve_fit(function_ret, x, yy,
                                             p0=p0, bounds=tuple(bb),
                                             )
                                   for p0, yy, bb in zip(p0s, y, bounds)]
                                 )
                             )

        parameters = np.array(len(popts) * [params])
        parameters[:, mask] = popts

        self.function_fit = function_ret
        chi = np.array([(yy - function_ret(x, *opt))  # /np.sqrt(yy)
                        for yy, opt in zip(y, popts)])
        self.chi2r = (chi**2).sum(axis=1) / (len(x) - mask.sum())
        if y.shape[0] == 1:
            parameters = parameters[0, :]
            self.chi2r = self.chi2r[0]

#        plot_fitline(x, y[0], function_ret(x, *popts[0]),
#                     label=self.chi2r, filename=f"temp{popts[0][2]:.1g}.png")

        self.baseline = parameters[4]
        self.center = parameters[0]
        self.height = parameters[1]
        self.width = parameters[2]
        self.beta = parameters[3]

        self.popts = popts
        self.x, self.y = x, y

    @_remove_unused_dimension
    def __call__(self, x):

        return np.array([self.function_fit(x, *popt) for popt in self.popts])


class MultiGenNorm(FcnWavSol):
    def __str__(self):
        return "Multi-line Generalized Normal"

    def short(self):
        return "MultiGNorm"

    def __init__(self, x, y, c=None, w=None, b=(2,), precision_pixel=5):
        """
Initialize and give initial guessses
        :param x:
        :param y:
        :param b:
            Initial value for beta
            If passed as the first element of a tuple, then it is not varied
        :param w:
           initial width, if not given then it is assumed to be one fourth of the total x extent.
            If passed as the first element of a tuple, then it is not varied
        """
        super().__init__(x, y)
        x, y = _force_increasing(self.x, self.y)
        if y.shape[0] != 1:
            raise TypeError("MultiGenNorm only enabled for one y-fit at a time")
        y = y[0]

        if not isinstance(c, (list, np.ndarray)):
            raise TypeError("multicenters must be given in list to initialize")

        n_var = len(c)
        c_idx = [np.argmin(np.abs(cc - x)) for cc in c]

        h = [y[ci-precision_pixel: ci+precision_pixel].mean() for ci in c_idx]
        if not isinstance(w, (list, tuple, np.ndarray)):
            w = [w] * n_var
        upper_bound = tuple(list(np.array(c)+precision_pixel) + list(np.array(h)*1.25))
        lower_bound = tuple(list(np.array(c)-precision_pixel) + list(np.array(h)*0.75))

        def function_ret(xx, *args):
            cf = args[: n_var]
            hf = args[n_var: 2*n_var]
            array = np.array([hh * 2.0 * stats.gennorm.pdf((xx - cc) / ww, b)
                             for cc, hh, ww in zip(cf, hf, w)])

            return array.sum(axis=0)

        ret = curve_fit(function_ret, x, y,
                        p0=list(c)+list(h), bounds=(lower_bound, upper_bound),
                        )
        popt, pcov = ret

        self.function_fit = function_ret

        self.centers = popt[:n_var]
        self.heights = popt[n_var: 2*n_var]
        self.popt = popt
        self.x, self.y = x, y

    def __call__(self, x):
        return self.function_fit(x, *self.popt)


class LinearRegression(FcnWavSol):
    def __str__(self):
        return "Linear regression"

    def short(self):
        return "LinReg"

    def __init__(self, x, y):
        super().__init__(x, y)

    @_remove_unused_dimension
    def __call__(self, x):
        ox, oy = _force_increasing(self.x, self.y)
        results = [linregress(ox, yy) for yy in oy]
        ret = [result.slope * x + result.intercept
               for result in results]
        return np.array(ret)


class LinearInterpolation(FcnWavSol):
    def __str__(self):
        return "Linear interpolation"

    def short(self):
        return "LinInt"

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(x, y)

    @_remove_unused_dimension
    def __call__(self, x: np.ndarray):
        xo, yo = _force_increasing(self.x, self.y)
        slope0 = (yo[:, 1] - yo[:, 0]) / (xo[1] - xo[0])
        slope1 = (yo[:, -1] - yo[:, -2]) / (xo[-1] - xo[-2])

        ret = np.zeros((yo.shape[0], len(x)))

        before = x < xo[0]
        ret[:, before] = yo[:, 0] - (xo[0] - x[before]) * slope0
        after = x > xo[-1]
        ret[:, after] = yo[:, -1] - (xo[-1] - x[after]) * slope1
        remainder = ~(before + after)
        ret[:, remainder] = np.array([np.interp(x[remainder], xo, y)
                                      for y in yo
                                      ])

        return ret


class NoneFcn(FcnWavSol):
    def __str__(self):
        return "None"

    def __init__(self, x, y):
        super().__init__(x, y)

    @_remove_unused_dimension
    def __call__(self, x):
        return np.zeros(len(x))


class Identity(FcnWavSol):
    def __str__(self):
        return "Identity"

    def short(self):
        return "Idnt"

    def __init__(self, x, y):
        super().__init__(x, y)

    @_remove_unused_dimension
    def __call__(self, x):
        return x


class Polynomial(FcnWavSol):
    def __str__(self):
        return f"Polynomial degree {self.degree}"

    def short(self):
        return f"Poly{self.degree}d"

    def __init__(self, x, y, d=2):
        super().__init__(x, y)
        self.degree = d

    @_remove_unused_dimension
    def __call__(self, x):
        d = self.degree
        return np.array([np.polyval(np.polyfit(self.x, yy, d), x) for yy in self.y])


class Spline(FcnWavSol):
    def __str__(self):
        return f"Spline smoothing {self.smoothing:.1f}"

    def short(self):
        return f"spl{self.smoothing:.1f}s"

    def __init__(self, x, y, s=None):
        super().__init__(x, y)
        if s is None:
            s = len(x)
        self.smoothing = s
        self.splines = [interpolate.UnivariateSpline(self.x, yy, s=self.smoothing) for yy in self.y]

    @_remove_unused_dimension
    def __call__(self, x):
        ret = np.array([sp(x) for sp, yy in zip(self.splines, self.y)])
        return ret


class OtfSpline(FcnWavSol):
    def __str__(self):
        return f"Spline smoothing {self.smoothing:.1f}"

    def short(self):
        return f"spl{self.smoothing:.1f}s"

    def __init__(self, x, y, s=None):
        super().__init__(x, y)
        if s is None:
            s = x.shape[-1]
        self.smoothing = s

    @_remove_unused_dimension
    def __call__(self, x):
        x_orig = self.x
        y_orig = self.y

        if isinstance(x, MaskedArray):
            mask_out = ~x.mask
            x = x[mask_out]
        else:
            mask_out = x*0 == 0

        if len(x_orig.shape) == 1:
            x_orig = [x_orig] * y_orig.shape[0]

        ret = np.zeros([x_orig.shape[0], len(mask_out)]) + np.nan
        for i, (xx, yy) in enumerate(zip(x_orig, y_orig)):
            mask = ~np.isnan(yy)
            if mask.sum() < 5:
                continue
            ret[i, mask_out] = interpolate.UnivariateSpline(xx[mask], yy[mask], s=self.smoothing)(x)

        return ret


def use_function(name, x, y):
    names = name.split(":")
    fname = names[0]

    keys = list(available.keys())
    options = [fname == av_name[:len(fname)] for av_name in keys]
    if np.array(options).sum() > 1:
        raise ValueError(f"Ambiguous function name {fname}, the following options"
                         f" match: {list(available.keys())[np.array(options)]}")

    function = available[keys[options.index(True)]]

    kwargs = {arg[0]: cast(arg[1:]) for arg, cast in zip(names[1:], function[1:])}

    ret = function[0](x, y, **kwargs)

    ret.set_config(name)

    return ret


available: dict[str, list[FcnWavSol | Callable]] = {'lin_interp': [LinearInterpolation],
                                              'identity': [Identity],
                                              'lin_regression': [LinearRegression],
                                              'polynomial': [Polynomial, int],
                                              'spline': [Spline, float],
                                              'otf_spline': [OtfSpline, float],
                                              'none': [NoneFcn],
                                              'gen_norm': [GenNorm, float, float, float, float],
                                                    }
