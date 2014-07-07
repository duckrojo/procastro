#
#
# Copyright (C) 2013 Patricio Rojo
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

import scipy as sp


class AstroPlot(object):

    def __init__(self):
        pass


    def doratio(self, trg, ref=None, normframes=None):
        """Computes ratio of science and reference

        :param trg: label name of target star
        :type trg: string (key of flx and coo)
        :param ref: list of stars to be used as reference
        :type ref: None or list of strings. If None, then all stars except target
        :param normframes: list of frames to normalize (for example, in case only out of transit want to be considered)
        :type normframes: None or boolean array. If None, then normalize by all frames
        :rtype: None
        """

        from scipy import asarray
        if trg not in self.flx.keys():
            print("Reference star '%s' not in stellar list: %s" %
                  (trg, self.flx.keys()))
            return None
        if ref is None:
            ref = [k for k in self.flx.keys() if k != trg]
        if len(ref) is 0:
            return None

        science = asarray(self.flx[trg])
        reference = science * 0.0
        variance_ref = science * 0.0
        for k in ref:
            reference += asarray(self.flx[k],dtype=float) / asarray(self.flx[k]).mean()
            variance_ref += (asarray(self.err[k],dtype=float) /
                             asarray(self.flx[k]).mean() ) **2
        reference /= len(ref)
        variance_ref /= len(ref)
        self.var=variance_ref
        self.rr = reference

        self.ratio = science / reference
        self.ratio_error = self.ratio * sp.sqrt(variance_ref/(reference**2) +
                                                (self.err[trg]/science)**2)
        if normframes is None:
            normframes = sp.ones(len(science)) == 1
        
        norm = self.ratio[normframes].mean()
        self.ratio /= norm
        self.ratio_error /= norm
        return
