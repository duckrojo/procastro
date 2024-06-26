#
#
# Copyright (C) 2014 Patricio Rojo
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of version 2 of the GNU General
# Public License as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
#


__all__ = ['gauss', 'bipol', 'parabolic_x',
           ]

from typing import Optional

import numpy as np
import astropy.units as u


def gauss(grid, sigma, center=None, norm=False, ndim=None):
    """
    Build a gaussian from a dense multi-dimensional meshgrid

    #TODO: allow multivariate with symmetric covariance matrix

    Parameters
    ----------
    grid: array_like
        Deafult sampling
    sigma: float
        Gaussian width
    center:
        Gaussian center
    norm: bool, optional
        True if output should be normalized
    ndim : int, optional
        Dimensions of gaussian

    Returns
    -------
    array_like
    """

    grid = np.array(grid)

    if ndim is None:
        ndim = len(grid.shape)
    if ndim==1:
        grid = [grid.ravel()]

    if isinstance(sigma, (int, float)):
        sigma = [sigma]*ndim

    if center is None:
        center = np.zeros(ndim)

    gaexpo = np.sum([((dgrid-c)/dsigma)**2
                     for dsigma, dgrid, c
                     in zip(sigma, grid, center)
                     ], axis=0)

    gauss = np.exp(-gaexpo/2)

    if norm:
        norm = np.product(sigma) * np.sqrt(2*np.pi)**len(sigma)
        gauss *= norm

    return gauss


def bipol(coef, x, y):
    """
    Polynomial fit for sky subtraction

    Parameters
    ----------
    coef : scipy ndarray
        Sky fit polynomial coefficients
    x : scipy ndarray
        Horizontal coordinates
    y : vertical coordinates

    Returns
    -------
    scipy ndarray
    """
    plane = np.zeros(x.shape)
    deg = np.sqrt(coef.size).astype(int)
    coef = coef.reshape((deg, deg))

    if deg * deg != coef.size:
        print("Malformed coefficient: " + str(coef.size) + "(size) != " + str(deg) + "(dim)^2")

    for i in np.arange(coef.shape[0]):
        for j in np.arange(i + 1):
            plane += coef[i, j] * (x ** j) * (y ** (i - j))

    return plane


def parabolic_x(yy_or_xx: list,
                yy: Optional[list] = None,
                central_idx: int = None,
                vertex: bool = True,
                ) -> float:
    """
    Adapted and modified to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points

    Parameters
    ----------
    yy_or_xx: list
        if `yy` is present, then it is `xx`. Else it is `yy` and x-axis becomes [-1, 0, 1]
    yy: list, array, 3-element if central_idx is None
    central_idx : int, optional
    If specified, then extract the three x&y points around index idx, otherwise they are expected as
     three-element array

    vertex : bool
       If True then return x position of vertex, otherwise return closest zero-crossing to middle point


    Returns
    -------
    float
        Returns index sub-position if xx is specified; otherwise, it returns requested xx value

    """

    if central_idx is not None:
        if not (0 < central_idx < len(yy_or_xx) - 1):
            raise ValueError(f"central_idx has to be within [1, {len(yy_or_xx)-1}] (not {central_idx}), cannot be a border of `yy_or_xx`")
        if yy is None:
            yy = yy_or_xx[central_idx - 1: central_idx + 2]
            xx = [central_idx - 1, central_idx, central_idx + 1]
        else:
            xx = yy_or_xx[central_idx - 1: central_idx + 2]
            yy = yy[central_idx - 1: central_idx + 2]
    else:
        if yy is None:
            yy = yy_or_xx
            xx = [-1, 0, 1]
        else:
            xx = yy_or_xx

    x1, x2, x3 = xx
    y1, y2, y3 = yy

    denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denominator
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denominator

    xv = -b / (2*a)
    # do not return quantity if x is not.
    if not isinstance(xx, u.Quantity) and isinstance(xv, u.Quantity):
        key = lambda x: x.to(u.dimensionless_unscaled).value
    else:
        key = lambda x: x

    if vertex:
        return key(xv)

    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denominator
    delta = np.sqrt(b * b - 4 * a * c)
    xz = xv + delta * np.array([1, -1]) / (2 * a)

    return key(xz[np.argmin(np.abs(xz - x2))])
