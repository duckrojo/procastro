__author__ = 'fran'

import scipy as sp
import warnings
import numpy as np

class TimeSeries(object):

    def __init__(self, data, errors, ids=None, epoch=None):
        self.channels = data  # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]
        self.errors = errors
        # [[error_target1_im1, error_target1_im2, ...], [error_target2_im1, error_target2_im2, ...]]
        self.group = [1] + [0 for i in range(len(data)-1)]
        # Default grouping: 1st coordinate is 1 group, all other objects are another group
        if ids is not None:
            self.flx = self.set_ids(ids)  # Dictionary for names?
        else:
            self.flx = {}

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

    def group1(self):
        """
        Returns all channels in group 1
        :return: List
        """
        return [self.channels[i] for i in range(len(self.channels) - 2) if self.group[i]]

    def group2(self):
        """ Returns all channels in group 2
        :return: List
        """
        return [self.channels[i] for i in range(len(self.channels) - 2) if not self.group[i]]

    def set_group(self, new_group):
        """ Used to define group of channels within time series, using a mask of 0s and 1s.
        For example [0 1 0 0 1 0] is used to define 2 different groups.
        Only 2 groups per TimeSerie supported.
        :param new_group: Group mask
        """
        self.group = new_group

    def errors_group1(self):
        """ Returns error channels from all the channels in group 1
        :return: List
        """
        return [self.errors[i] for i in range(len(self.errors) - 2) if self.group[i]]

    def errors_group2(self):
        """ Returns error channels from all the channels in group 2
        :return: List
        """
        return [self.errors[i] for i in range(len(self.errors) - 2) if not self.group[i]]

    def set_ids(self, ids):
        """ Sets list of target IDs to TimeSerie in order to plot with target names.
        :param ids: List
        """
        self.ids = ids
        for i in range(len(self.ids)):
            self.flx[self.ids[i]] = self.channels[i]
        return self.flx

    def set_epoch(self, e):
        """ Sets list of observation epochs to TimeSerie in order to plot
        :param e: List
        """
        self.epoch = e

    def mean(self, group_id):
        """
        Calculates the mean of all the channels in the group given by group_id.
        This operation returns the resulting new channel, but also overwrites TimeSerie[-group_id] with the result.
        :param group_id: id of group to operate on. {1, 2}
        :return: []
        """
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
        """
        Calculates the median of all the channels in the group given by group_id
        This operation returns the resulting new channel, but also overwrites TimeSerie[-group_id] with the result.
        :param group_id: id of group to operate on. {1, 2}
        :return: []
        """
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
        print("PLOT!")
        import dataproc as dp
        fig, ax, epoch = dp.axesfig_xdate(axes, self.epoch)

        if label is None:
            disp = self.flx.keys()
        else:
            disp = [label]

        # TODO check yerr
        for lab in disp:
        #    if self.__getitem__(lab, error=True) is None:
        #        yerr = None
        #    else:
        #        yerr = self.__getitem__(lab, error=True)
        print lab

            ax.errorbar(epoch,
                        self.flx[lab],
                        self.errors[lab],
                        marker="o",
                        label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        import matplotlib.pyplot as plt
        plt.show()
        #return
