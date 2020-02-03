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

from __future__ import print_function, division

__all__ = ['gauss', 'bipol',
           ]

import numpy as np
import inspect
import scipy.signal as sg


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
