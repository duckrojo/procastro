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
import matplotlib.pyplot as plt
#from cycler import cycler

class TimeSeries:
    """Stores many alternatives of information to show for a series of channels"""

    def __repr__(self):
        return "<timeseries object with {n_channels} channels (channels) of {size} " \
               "elements. " \
               "Information available: {infos}>".format(n_channels=len(self),
                                                        channels=self._tss[self.default_info].labels,
                                                        size=len(self._tss[self.default_info]),
                                                        infos=self._tss.keys(),
                                   )

    def __len__(self):
        return len(self._tss[self.default_info])

    def __init__(self, data, errors=None, labels=None, epoch=None, extras=None,
                 default_info=None, grouping='mean'):
        """
Receives a timeseries object with, optionally, several different kinds of information
(flux, errors, centroid, ...)

        :param data: dict or scipy.array
        :param errors: dict or scipy.array
        if dict, then it should have the same keys as data
        :param labels:
        :param epoch:
        :param extras:
        :param default_info:
        :param grouping:
        """

        # blank definition
        self.default_info = None

        if isinstance(data, dict):
            if default_info is None:
                self.set_default_info(data.keys()[0])
            self._tss = {k: TimeSeriesSingle(v, errors[k] if (k in errors) else None,
                                             labels=labels, epoch=epoch, group_op=grouping)
                         for k, v in data.items()}
            # todo check whether this old style will be implemented.
            # if False in [isinstance(v, (list, tuple, sp.ndarray)) for v in extras.values()]:
            #     raise TypeError("Each element of 'extras' dictionary must be a list")

        elif isinstance(data, sp.array):
            if default_info is None:
                default_info = "data"
            self._tss["data"] = TimeSeriesSingle(data, errors,
                                                 labels=labels, epoch=epoch, group_op=grouping)
        self.set_default_info(default_info)

    def set_default_info(self, name):
        self.default_info = name

    def plot(self, info=None, **kwargs):
        """execute .plot() on the default TimeSeriesSingle"""
        if info is None:
            info = self.default_info
        self._tss[info].plot(**kwargs)

    def plot_ratio(self, info=None, **kwargs):
        """execute .plot_ratio() on the default TimeSeriesSingle"""
        if info is None:
            info = self.default_info
        self._tss[info].plot_ratio(**kwargs)

    def __call__(self, *args):
        if len(args) != 1:
            raise ValueError("Only one argument in the form 'field=target' must be given to index it")
        info = args[0]

        if info in self._tss:
            return self._tss[info]
        else:
            raise ValueError("Cannot find requested field '{0}'".format(info))

    def _search_channel(self, target):
        return self._tss[self.default_info]._search_channel(target)



class TimeSeriesSingle:
    """Stores a single kind of information"""

    def __repr__(self):
        return "<single timeseries object with {channels} channels of {size} " \
               "elements ({err} error channel).>".format(channels=len(self.channels),
                                                         size=len(self),
                                                         err="it has" if len(self.errors)
                                                         else "has no")

    def __init__(self, data, errors=None, labels=None, epoch=None, extras=None,
                 group_op='mean'):

        self.channels = [sp.array(d) for d in
                         data]  # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]

        self.labels = []
        if labels is not None:
            self.labels = labels

        if errors is None:
            errors = []

        self.errors = errors
        self.epoch = epoch

        self.combine = {}
        self.grouping_with(group_op)
        self.groups = sp.zeros(len(self))  # group #0 is ignored
        # Default group_op: 1st channel is group #1, all other channels are group #2
        self.set_main(0)

    # todo: implement ignore channels
    def set_main(self, target, ignore=None):
        """
Set target channel as group #1, and all other channels as group #2
        :param ignore: Channel, or list of channels, to be placed in group 3
        :param target: Channel to be set to group 1
        :return:
        """
        target = self._search_channel(target)
        self.groups[:] = 2
        self.groups[target] = 1
        return self

    def __len__(self):
        return len(self.channels[0])

    def __getitem__(self, target):
        """To recover errors or any of the other extras try: ts('error')[0] ts['centers_xy']['Target'], etc"""
        # This is so I can do ts[0]/ts[2] and it works directly with the channels!

        target = self._search_channel(target)
        if target < 0:
            return self.combine['op'](sp.array(self.channels)[sp.array(self.groups) == abs(target)],
                                      **(self.combine['prm']))

        return self.channels[target]

    def _search_channel(self, target):

        if isinstance(target, str):
            target = self.labels.index(target)

        return target

    def grp_errors(self, group):
        if not (self.combine['name'] == 'mean' or self.combine['name'] == 'median'):
            raise NotImplementedError("Only know how to compute error of median- or mean-combined groups. "
                                      "Errors of median computed as errors of mean, actually")
        sample = sp.array(self.groups) == abs(group)
        sigma = self.errors[sample]
        variance = (sigma ** 2).sum(0)
        # Tracer()()
        return sp.sqrt(variance) / sample.sum()

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

        :param axes:
        :param normalize:
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
            lab = self._search_channel(lab)
            value = self[lab]
            error = self.errors[lab]
            if normalize:
                norm = value.mean()
                value /= norm
                error /= norm
            ax.errorbar(self.epoch, value, yerr=error,
                        marker="o", label=lab)

        ax.set_title("TimeSeries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        plt.show()

    # todo grouping not functional
    def plot_ratio(self, label=None, axes=None, fmt='x', grouping=None):
        fig, ax = dp.figaxes(axes)

        ratio = self[-1] / self[-2]
        ratio_error = ratio * sp.sqrt((self.grp_errors(1) / self[-1]) ** 2 +\
                                      (self.grp_errors(2) / self[-2]) ** 2)

        if grouping is None:
            grouping = {}
            group_by = self.epoch*0
            group_labels = ['']
        else:
            group_by = list(set(self.extras[grouping['extra']]))
            group_labels = group_by

        colors = grouping.pop('colors', ['c', 'm', 'y', 'k'])
        markers = grouping.pop('markers', ['^', 'v', 'x'])

        epochs, groups, errors = self.make_groups(self.epoch, ratio,
                                                  group_by, ratio_error)

        for x, y, e, c, lab in zip(epochs, groups, errors, colors, group_labels):
            # ax.set_prop_cycle(cycler('color', colors),
            #                   cycler('marker', markers),)
            ax.errorbar(x, y, yerr=e, props={"ls": 'None'}, label=lab)

        labs = [', '.join(sp.array(self.labels)[sp.array(self.groups) == grp]) for grp in [1, 2]]
        ax.set_title("Flux Ratio: <{}>/<{}>".format(labs[0], labs[1]))
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux Ratio")

        plt.show()

    # todo reimplement save_to_file
    # def save_to_file(self, filename):
    #     to_save=sp.zeros([len(self), len(extras)])
    #
    #     sp.savetxt(header="# {}".format(", ".join(self.epoch, self[-1]/self[-2], err, )))

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
