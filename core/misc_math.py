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

import scipy as sp
import inspect
import scipy.signal as sg

def gauss(grid, sigma, center=None, norm=False, ndim=None):
    """
    Build a gaussian from a dense multi-dimensional meshgrid

    #todo: allow multivariate with symmetric covariance matrix
    :param grid: deafult sampling
    :param sigma: gaussian width
    :param center:  gaussian center
    :param norm: True if output should be normalized
    :param ndim: dimensions of gaussian
    :return:
"""

    grid = sp.array(grid)

    if ndim is None:
        ndim = len(grid.shape)
    if ndim==1:
        grid = [grid.ravel()]

    if isinstance(sigma, (int,float)):
        sigma = [sigma]*ndim

    if center is None:
        center = sp.zeros(ndim)
    
    gaexpo = sp.sum([((dgrid-c)/dsigma)**2 
                     for dsigma, dgrid, c
                     in zip(sigma, grid, center)
                     ], axis=0)

    gauss = sp.exp(-gaexpo/2)

    if norm:
        norm = sp.product(sigma) * sp.sqrt(2*sp.pi)**len(sigma)
        gauss *= norm

    return gauss


def bipol(coef, x, y):
    """Polynomial fit for sky subtraction

    :param coef: sky fit polynomial coefficients
    :type coef: sp.ndarray
    :param x: horizontal coordinates
    :type x: sp.ndarray
    :param y: vertical coordinates
    :type y: sp.ndarray
    :rtype: sp.ndarray
    """
    plane = sp.zeros(x.shape)
    deg = sp.sqrt(coef.size).astype(int)
    coef = coef.reshape((deg, deg))

    if deg * deg != coef.size:
        print("Malformed coefficient: " + str(coef.size) + "(size) != " + str(deg) + "(dim)^2")

    for i in sp.arange(coef.shape[0]):
        for j in sp.arange(i + 1):
            plane += coef[i, j] * (x ** j) * (y ** (i - j))

    return plane
