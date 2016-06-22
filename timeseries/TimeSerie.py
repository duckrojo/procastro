#
#
# Copyright (C) 2016 Francisca Concha
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



__author__ = 'fran'

import scipy as sp
import warnings
import numpy as np
import dataproc as dp
from datetime import datetime
from matplotlib import dates

class TimeSeries(object):

    def __init__(self, data, errors, labels=None, epoch=None):
        dc = []
        for d in data:
            dc.append(sp.array(d))  # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]
        self.channels = dc
        de = []
        for e in errors:
            de.append(sp.array(e))
        self.errors = de
        # [[error_target1_im1, error_target1_im2, ...], [error_target2_im1, error_target2_im2, ...]]
        self.group = [1] + [0 for i in range(len(data)-1)]
        # Default grouping: 1st coordinate is 1 group, all other objects are another group
        self.labels = {}
        if labels is not None:
            self.labels = self.set_labels(labels)  # Dictionary for names?
        #else:
        #    self.labels = {}

        self.channels.append([])  # Group 1 operation result; is overwritten every time a new op is defined
        self.errors.append([])

        self.channels.append([])  # Group 2 operation result; is overwritten every time a new op is defined
        self.errors.append([])

        self.epoch = epoch

    def __getitem__(self, item, error=False):
        # This is so I can do ts[0]/ts[2] and it works directly with the channels!
        try:
            if error is False:
                return self.channels[item]
            else:
                return self.errors[item]
        except TypeError:
            if error is False:
                return self.channels[self.ids.index(item)]
            else:
                return self.errors[self.ids.index(item)]

    def get_error(self, item):
        return self.__getitem__(item, error=True)

    def group1(self):
        return [self.channels[i] for i in range(len(self.channels) - 2) if self.group[i]]

    def group2(self):
        return [self.channels[i] for i in range(len(self.channels) - 2) if not self.group[i]]

    def set_group(self, new_group):  # Receives pe [0 1 0 0 1 0] and that is used to define 2 groups
        self.group = new_group

    def errors_group1(self):
        return [self.errors[i] for i in range(len(self.errors) - 2) if self.group[i]]

    def errors_group2(self):
        return [self.errors[i] for i in range(len(self.errors) - 2) if not self.group[i]]

    def set_labels(self, ids):
        self.ids = ids
        for i in range(len(self.ids)):
            self.labels[self.ids[i]] = self.channels[i]
        return self.labels

    def set_epoch(self, e):
        self.epoch = e

    def mean(self, group_id):
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
            g_errors = self.errors_group2()
        elif group_id == 1:
            group = self.group1()
            g_errors = self.errors_group1()
        else:
            group = self.group2()
            g_errors = self.errors_group2()

        self.channels[-group_id] = sp.mean(group, axis=0)
        err = np.zeros((1, len(g_errors[0])))
        for i in range(len(g_errors)):
            err += np.divide(g_errors[i]/group[i])**2
        self.errors[-group_id] = np.sqrt(err)

        return self.channels[-group_id]

    def median(self, group_id):
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
            g_errors = self.errors_group2()
        elif group_id == 1:
            group = self.group1()
            g_errors = self.errors_group1()
        else:
            group = self.group2()
            g_errors = self.errors_group2()

        self.channels[-group_id] = sp.median(group, axis=0)
        err = np.zeros((1, len(g_errors[0])))
        for i in range(len(g_errors)):
            err += np.divide(g_errors[i]/group[i])**2
        self.errors[-group_id] = np.sqrt(err)

        return self.channels[-group_id]

    def plot(self, label=None, axes=None):
        """Display the timeseries data: flux (with errors) as function of mjd

        :param label: Specify a single star to plot
        :rtype label: basestring

        :rtype: None (and plot display)
        """
        import dataproc as dp
        from datetime import datetime
        from matplotlib import dates

        date_epoch = [datetime.strptime(e, "%Y-%m-%dT%H:%M:%S.%f") for e in self.epoch]
        newepoch = [dates.date2num(dts) for dts in date_epoch]
        #newepoch = self.epoch

        fig, ax, epoch = dp.axesfig_xdate(axes, newepoch)

        if label is None:
            disp = self.labels.keys()
        else:
            disp = [label]

        # TODO check yerr
        for lab in disp:
            if self.__getitem__(lab, error=True) is None:
                yerr = None
            else:
                yerr = self.__getitem__(lab, error=True)

            ax.errorbar(epoch,
                        self.labels[lab],
                        yerr=yerr,
                        marker="o",
                        label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        import matplotlib.pyplot as plt
        plt.show()
        #return
