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
import scipy as sp
import inspect
import scipy.signal as sg
import pyfits as pf



def gauss(grid, sigma, center=None, norm=False, ndim=None):
    """
    Build a gaussian from a dense multi-dimensional meshgrid

    todo: allow multivariate with symmetric covariance matrix 
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
