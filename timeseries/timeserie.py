#
#
# Copyright (C) 2016 Francisca Concha, Patricio Rojo
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
from IPython.core.debugger import Tracer

import dataproc as dp
from datetime import datetime
from matplotlib import dates
import matplotlib.pyplot as plt

__author__ = 'fran'


class TimeSeries(object):
    def __repr__(self):
        return "<timeseries object with {channels} channels of {size} " \
               "elements (Extras={extras}).>".format(channels=len(self.channels),
                                                     size=len(self.channels[0]),
                                                     extras=self.extras.keys(), )

    def __init__(self, data, errors=None, labels=None, epoch=None, extras=None,
                 grouping='mean'):

        self.combine = {}
        self.grouping_with(grouping)
        self.channels = [sp.array(d) for d in
                         data]  # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]

        # Default grouping: 1st coordinate is 1 group, all other objects are another group
        self.groups = sp.ones(len(self.channels)) + 1
        self.groups[0] = 1

        self.labels = []
        if labels is not None:
            self.labels = labels

        if extras is None:
            extras = {}
        elif not isinstance(extras, dict):
            raise TypeError("extras must be a dictionary")
        else:
            if False in [isinstance(v, (list,)) for v in extras.values()]:
                raise TypeError("Each element of 'extras' dictionary must be a list")

        self.extras = extras
        self._extra_call = None

        if errors is None:
            errors = []
        self.extras['errors'] = [sp.array(e) for e in errors]
        # [[error_target1_im1, error_target1_im2, ...], [error_target2_im1, error_target2_im2, ...]]

        self.epoch = epoch

    def __call__(self, **kwargs):
        if len(kwargs) != 1:
            raise ValueError("Only one argument in the form 'field=target' must be given to index it")
        k, v = kwargs.items()[0]

        if k in self.extras:
            ret_extra = sp.array(self.extras[k])
        else:
            raise ValueError("Cannot find requested field '{0}'".format(k))

        if isinstance(v, str):
            item = self.labels.index(v)
        else:
            item = v

        if isinstance(item, int):
            if item < 0:
                if k == 'errors':
                    if not (self.combine['name'] == 'mean' or self.combine['name'] == 'median'):
                        raise NotImplementedError("Only know how to compute error of median- or mean-combined groups.")
                    sample = sp.array(self.groups) == abs(item)
                    sigma = ret_extra[sample]
                    variance = (sigma ** 2).sum(0)
                    #Tracer()()
                    return sp.sqrt(variance) / sample.sum()
                else:
                    raise NotImplementedError("Only combined errors have been implemented")
            else:
                return ret_extra[item]
        else:
            raise TypeError("unrecognized channel specification ({}) for "
                            "extra '{}'".format(item, k))

    def __getitem__(self, item):
        """To recover errors or any of the other extras try: ts('error')[0] ts['centers_xy']['Target'], etc"""
        # This is so I can do ts[0]/ts[2] and it works directly with the channels!

        if isinstance(item, str):
            item = self.labels.index(item)
        elif item < 0:
            return self.combine['op'](sp.array(self.channels)[sp.array(self.groups) == abs(item)],
                                      **(self.combine['prm']))

        return self.channels[item]

    def grouping_with(self, op):
        self.combine['name'] = op
        if op == 'mean':
            self.combine['op'] = sp.mean
            self.combine['prm'] = {'axis': 0}
        elif op == 'median':
            self.combine['op'] = sp.median
            self.combine['prm'] = {'axis': 0}
        else:
            raise ValueError("Unrecognized combine operation '{}'".format(op))


    def plot(self, label=None, axes=None, normalize=False):
        """Display the timeseries data: flux (with errors) as function of mjd

        :param label: Specify a single star to plot
        :rtype label: basestring

        :rtype: None (and plot display)
        """

        fig, ax = dp.figaxes(axes)

        if label is None:
            disp = self.labels
        else:
            disp = [label]

        for lab in disp:
            value = self[lab]
            error = self(errors=lab)
            if normalize:
                norm = value.mean()
                value /= norm
                error /= norm
            ax.errorbar(self.epoch, value, yerr=error,
                        marker="o", label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        plt.show()

    def plot_ratio(self, label=None, axes=None):
        fig, ax = dp.figaxes(axes)

        ratio = self[-1] / self[-2]
        ratio_error = ratio * sp.sqrt((self(errors=-1) / self[-1]) ** 2 +
                                      (self(errors=-2) / self[-2]) ** 2)
        ax.errorbar(self.epoch, ratio, yerr=ratio_error,
                    marker="o")

        labs = [', '.join(sp.array(self.labels)[sp.array(self.groups) == grp]) for grp in [1, 2]]
        ax.set_title("Flux Ratio: <{}>/<{}>".format(labs[0], labs[1]))
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux Ratio")

        plt.show()

        # def get_error(self, item):
        #     return self(errors=item)
        #
        # def group1(self):
        #     return sp.array(self.channels[:-2])[sp.array(self.group)]
        #
        # def group2(self):
        #     return sp.array(self.channels[:-2])[sp.array(self.group)==False]
        #
        # def set_group(self, new_group):  # Receives pe [0 1 0 0 1 0] and that is used to define 2 groups
        #     self.group = new_group
        #
        # def errors_group1(self):
        #     return [self(errors=lab) for lab,g in zip(self.labels, self.group) if g]
        #
        # def errors_group2(self):
        #     return [self(errors=lab) for lab,g in zip(self.labels, self.group) if not g]
        #
        # def set_labels(self, ids):
        #     self.ids = ids
        #     for i in range(len(self.ids)):
        #         self.labels[self.ids[i]] = self.channels[i]
        #     return self.labels
        #
        # def set_epoch(self, e):
        #     self.epoch = e
        #
        # def mean(self, group_id):
        #     if group_id > 2:
        #         warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #     elif group_id == 1:
        #         group = self.group1()
        #         g_errors = self.errors_group1()
        #     else:
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #
        #     self.channels[-group_id] = sp.mean(group, axis=0)
        #     err = np.zeros((1, len(g_errors[0])))
        #     for i in range(len(g_errors)):
        #         err += np.divide(g_errors[i]/group[i])**2
        #     self('errors')[-group_id] = np.sqrt(err)
        #
        #     return self.channels[-group_id]
        #
        # def median(self, group_id):
        #     if group_id > 2:
        #         warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #     elif group_id == 1:
        #         group = self.group1()
        #         g_errors = self.errors_group1()
        #     else:
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #
        #     self.channels[-group_id] = sp.median(group, axis=0)
        #     err = np.zeros((1, len(g_errors[0])))
        #     for i in range(len(g_errors)):
        #         err += np.divide(g_errors[i]/group[i])**2
        #     self.errors[-group_id] = np.sqrt(err)
        #
        #     return self.channels[-group_id]
        #
