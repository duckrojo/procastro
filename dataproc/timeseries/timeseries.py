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

import numpy as np
import dataproc as dp

def vspan_plot(vspan, ax):
    if isinstance(vspan, (list, tuple, np.ndarray)):
        if len(vspan) == 2 and isinstance(vspan[0], (int, float)):
            vspan = [{'range': vspan}]
        elif not isinstance(vspan[0], dict):
            raise ValueError("vspan needs to be a dict, a 2-number array, or a list of dicts")
    elif isinstance(vspan, dict):
        vspan = [vspan]
    for vs in vspan:
        ax.axvspan(*vs['range'], facecolor="gray" if 'facecolor' not in vs.keys() else vs['facecolor'],
                   alpha=0.5)


class TimeSeries:
    """
    Stores different data channels using multiple TimeSeriesSingle instances.
    Each channel represents a column on a timeseries table, this object selects
    one channel as a default for all related methods, to display different
    channels the user must set this default to the desired channel.

    Receives a timeseries object with, optionally, several different kinds of
    information (flux, errors, centroid, ...)

    Attributes
    ----------
    _tss: dict
        Name to TSS dictionary for all channels

    Parameters
    ----------
    data : dict or scipy.array
        Contains the data used by each channel for all targets
    errors : dict or scipy.array, then it should have the same keys as data
        Errors for some or all channels for al targets
    labels : str list
        List of names for the targets
    epoch : array_like
        Contains the time axis to be used for all channels
    extras :
        No use as of yet
    default_info :
        Channel used when plotting graphs
    grouping : {'mean', 'median'}
        Mathematical Operation used when grouping the data channels
    """

    def __repr__(self):
        return "<timeseries object with {n_channels} channels of {size} " \
                "elements. " \
                "Information available: " \
                "{infos}>" \
                .format(n_channels=len(self._tss[self.default_info].channels),
                        channels=self._tss[self.default_info].labels,
                        size=len(self._tss[self.default_info]),
                        infos=", ".join(
                            ["{}{}".format(k, "*"
                                              if self._tss[k].has_errors()
                                              else "")
                             for k in self._tss.keys()])
                        )

    def __len__(self):
        return len(self._tss[self.default_info])

    def __getitem__(self, item):
        return self._tss[self.default_info][item]

    def __init__(self, data, errors=None, labels=None, epoch=None,
                 extras=None, default_info=None, grouping='mean'):

        # Blank definition
        self.default_info = None
        if isinstance(data, dict):
            if default_info is None:
                # TODO: Posible error here keys[0] invalid
                self.set_default_info(data.keys()[0])
            self._tss = {k: TimeSeriesSingle(v,
                                             errors[k] if (k in errors)
                                             else None,
                                             labels=labels,
                                             epoch=epoch,
                                             group_op=grouping)
                         for k, v in data.items()}
            # TODO: check whether this old style will be implemented.
            # if False in [isinstance(v, (list, tuple, np.ndarray))
            #              for v in extras.values()]:
            #     raise TypeError("Each element of 'extras' dictionary must "
            #                     "be a list")

        elif isinstance(data, np.array):
            if default_info is None:
                default_info = "data"
            self._tss["data"] = TimeSeriesSingle(data, errors,
                                                 labels=labels, epoch=epoch,
                                                 group_op=grouping)
        self.set_default_info(default_info)

    def set_default_info(self, name):
        self.default_info = name

    def plot(self, info=None, **kwargs):
        """
        Executes .plot() on the default TimeSeriesSingle

        Parameters
        ----------
        info : str, optional
            Data channel to be plotted, default is the current default channel

        See Also
        --------
        TimeSeriesSingle
        """
        if info is None:
            info = self.default_info
        self._tss[info].plot(**kwargs)

    def get_ratio(self, info=None, **kwargs):
        """
        Recovers ratio

        Parameters
        ----------
        info : str, optional
            Data channel, default is the current default channel

        Returns
        -------

        """

        if info is None:
            info = self.default_info
        ratio, ratio_error, sigma, err_m = self._tss[info].get_ratio(**kwargs)
        return ratio, ratio_error, sigma, err_m

    def ignore_with_outliers(self, info=None, min=None, max=None):
        if info is None:
            info = self.default_info
        return self._tss[info].ignore_with_outliers(min, max)

    def plot_ratio(self, info=None, **kwargs):
        """
        Execute .plot_ratio() on the default TimeSeriesSingle

        Parameters
        ----------
        info : str, optional
            Data channel, default is the current default channel

        """
        if info is None:
            info = self.default_info
        self._tss[info].plot_ratio(**kwargs)

    def __call__(self, *args):
        if len(args) != 1:
            raise ValueError("Only one argument in the form 'field=target' "
                             "must be given to index it")
        info = args[0]

        if info in self._tss:
            return self._tss[info]
        else:
            raise ValueError("Cannot find requested field '{0}'".format(info))

    def _search_channel(self, target):
        return self._tss[self.default_info]._search_channel(target)

    def __getitem__(self, item):
        return self._tss[self.default_info]


class TimeSeriesSingle:
    """
    Stores a single kind of information for all targets

    Attributes
    ----------
    channels : array_like
        Data contained in the default channel
    combine : dict
        Contains the function used for grouping
    groups : array_like
        Groups generated after gropuing

    Parameters
    ----------
    data : array_like, optional
        Data contained in this channel
    errors : array_like, optional
    labels : array_like, optional
        Name for each target
    epoch : array_like, optional
    extras :
        Aditional header items
    group_op : str, optional
        Operation used to group the targets

    """

    def __repr__(self):
        return "<single timeseries object with {channels} channels of " \
               "{size} elements ({err} error channel).>" \
               .format(channels=len(self.channels),
                       size=len(self),
                       err="it has" if self.has_errors()
                       else "has no")

    def __init__(self, data, errors=None, labels=None, epoch=None, extras=None,
                 group_op='mean'):

        # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]
        self.channels = [np.array(d) for d in
                         data]

        self.labels = []
        if labels is not None:
            self.labels = labels

        if errors is None:
            errors = []

        self.errors = errors
        self.epoch = epoch

        self.combine = {}
        self.grouping_with(group_op)
        self.groups = np.zeros(self.n_channels())  # group #0 is ignored
        # Default group_op: 1st channel is group #1,
        # all other channels are group #2
        self.set_main(0)

    def has_errors(self):
        return len(self.errors) > 0

    def ignore_with_outliers(self, min=None, max=None):
        ignored = []
        for chan in range(self.n_channels()):
            if self.groups[chan] == 1:
                continue
            if min is not None and np.any(self[chan]<min):
                self.groups[chan] = 3
                ignored.append(chan)
                continue
            if max is not None and np.any(self[chan]>max):
                self.groups[chan] = 3
                ignored.append(chan)
                continue

        return ignored

    def set_main(self, target, ignore=None):
        """
        Set target channel as group #1, and all other channels as group #2

        Parameters
        ----------
        target: str
            Channel which will be set to group 1
        ignore: Channel, or list of channels.
            Channels to be ignored (placed in group 3)
        """
        target = self._search_channel(target)
        self.groups[:] = 2
        if ignore is not None:
            for out in ignore:
                self.groups[out] = 3
        self.groups[target] = 1
        return self

    def __len__(self):
        return len(self.channels[0])

    def n_channels(self):
        return len(self.channels)

    def __getitem__(self, target):
        """
        To recover errors or any of the other extras try:
            ts('error')[0] ts['centers_xy']['Target'], etc
        """
        # This is so I can do ts[0]/ts[2] and it works directly
        # with the channels!

        target = self._search_channel(target)

        # if negative, then group the channels
        if target < 0:
            return self.combine['op'](
                        np.array(self.channels)[np.array(self.groups)
                                                == abs(target)],
                        **(self.combine['prm']))

        return self.channels[target]

    def _search_channel(self, target):

        if isinstance(target, str):
            target = self.labels.index(target)

        return target

    def grp_errors(self, group):
        if not (self.combine['name'] == 'mean'
                or self.combine['name'] == 'median'):
            raise NotImplementedError(
                "Only know how to compute error of median- or mean-combined "
                "groups. Errors of median computed as errors "
                "of mean, actually")
        sample = np.array(self.groups) == abs(group)
        sigma = self.errors[sample]
        variance = (sigma ** 2).sum(0)
        return np.sqrt(variance) / sample.sum()

    def grouping_with(self, op):
        """
        Sets grouping method used to group the channels

        Parameters
        ----------
        op : str
            Operation to be applied, possible operations
            are 'mean' and 'median'
        """
        self.combine['name'] = op
        if op == 'mean':
            self.combine['op'] = np.mean
            self.combine['prm'] = {'axis': 0}
        elif op == 'median':
            self.combine['op'] = np.median
            self.combine['prm'] = {'axis': 0}
        else:
            raise ValueError("Unrecognized combine operation '{}'".format(op))

    def plot(self, label=None, axes=None, normalize=False, save=None,
             clear=True, fmt_time="MJD", title="TimeSeries Data", show=None):
        """
        Display the timeseries data: flux (with errors) as function of mjd

        Parameters
        ----------
        label : str, optional
            Specify a single star to plot
        normalize: bool, optional
            If set, normalizes the data before plotting
        axes : int, plt.Figure, plt.Axes
        save : str, optional
            Filename where the plot will be saved
        clear : bool, optional
             Clear previous axes info
        fmt_time : str, optional
            Specify a format for epoch time like "JD", by default it's "MJD"
        title : str, optional

        Returns
        -------
        None
            Displays plot if 'save' set to None
        """

        if show is None:
            show = save is None
        import matplotlib.pyplot as plt
        fig, ax = dp.figaxes(axes, clear=clear)

        if label is None:
            disp = self.labels
        else:
            disp = [label]

        for lab in disp:
            lab_num = self._search_channel(lab)
            value = self[lab_num]
            error = self.errors[lab_num]
            if normalize:
                norm_0 = self[0].mean()
                norm = value.mean()
                value = value / norm
                error = error / norm
                ratio = round(norm / norm_0, 2)
                lab = lab + " flux ratio " + str(ratio)
            ax.errorbar(self.epoch, value, yerr=error,
                        marker="o", label=lab)

        ax.set_title(title)
        ax.set_xlabel(fmt_time)
        ax.set_ylabel(f"{'Normalized ' if normalize else ''}Flux")

        ax.legend()
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()

    def get_ratio(self, label=None, axes=None, sector=None):
        """
        Calculates the ratio of the stored data

        Parameters
        ----------
        label : str, optional
            Name of the target
        axes : matplotlib.pyplot axes object, optional
        sector : tuple, optional
            Range from self.epoch to be used

        Returns
        -------
        ratio_cut : array_like
            Ratio obtained, if sector is enabled will return the ratio of said
            sector.
        ratio_error_cut : array_like
            Error of the calculated sector.
        sigma : float
            Standard deviation of the data
        errbar_media : float
            Median value of the ratio error
        """
        ratio = self[-1] / self[-2]

        ratio_error = np.absolute(ratio) * np.sqrt((self.grp_errors(1) / self[-1]) ** 2 +
                                                   (self.grp_errors(2) / self[-2]) ** 2)

        x1 = 0
        x2 = len(ratio)
        if isinstance(sector, list):
            x1 = sector[0]
            x2 = sector[1]
        sigma = np.std(ratio[x1:x2])
        errbar_media = np.median(ratio_error[x1:x2])
        ratio_cut = ratio[x1:x2]
        ratio_error_cut = ratio_error[x1:x2]

        return ratio_cut, ratio_error_cut, sigma, errbar_media

    # TODO: grouping not functional
    def JD(self, sector=None):
        """
        Recovers the epoch values from the x-axis

        Parameters
        ----------
        sector : 2 item list, optional
            Range to be retrieved from self.epoch

        Returns
        -------
        array_like
        """
        x1 = 0
        x2 = len(self.epoch)
        if isinstance(sector, list):
            x1 = sector[0]
            x2 = sector[1]
        return self.epoch[x1:x2]

    def plot_ratio(self, label=None, axes=None, fmt='x', title="",
                   grouping=None, sector=None, save=None,
                   clear=True, vspan=None, show=None):
        """
        Plots data ratio

        Parameters
        ----------
        label : str, optional
            Target name
        axes : int, matplotlib.pyplot Axes, optional
        fmt : string, optional
        grouping :
        sector :
        save : str, optional
            Filename of image file where the plot will be saved
        clear : bool, optional
            Clear the axes content if needed
        """

        if show is None:
            show = save is None
        import matplotlib.pyplot as plt
        fig, ax = dp.figaxes(axes, clear=clear)

        ratio, ratio_error, s, erbm = self.get_ratio(sector=sector)

        if label is None:
            label = self.labels[0]
        # if grouping is None:
        #     grouping = {}
        #     group_by = self.epoch*0
        #     group_labels = ['']
        # else:
        #     group_by = list(set(self.extras[grouping['extra']]))
        #     group_labels = group_by
        #
        # colors = grouping.pop('colors', ['c', 'm', 'y', 'k'])
        # markers = grouping.pop('markers', ['^', 'v', 'x'])
        #
        # epochs, groups, errors = self.make_groups(self.epoch, ratio,
        #                                           group_by, ratio_error)
        #
        # for x, y, e, c, lab in zip(epochs, groups, errors, colors,
        #                            group_labels):
        #     # ax.set_prop_cycle(cycler('color', colors),
        #     #                   cycler('marker', markers),)
        #     ax.errorbar(x, y, yerr=e, props={"ls": 'None'}, label=lab)

        # labs = [', '.join(np.array(
        #                   self.labels)[np.array(self.groups) == grp])
        #                                         for grp in [1, 2]]
        # ax.set_title("Flux Ratio: <{}>/<{}>".format(labs[0], labs[1]))

        ax.errorbar(self.JD(sector=sector), ratio, yerr=ratio_error,
                    label=label, fmt='o-')
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Flux Ratio of {}".format(self.labels[0]))
        ax.set_xlabel("JD")
        ax.set_ylabel("Flux Ratio")
        if vspan is not None:
            vspan_plot(vspan, ax)

        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()

    # todo reimplement save_to_file
    # def save_to_file(self, filename):
    #     to_save=np.zeros([len(self), len(extras)])
    #
    #     np.savetxt(header="# {}".format(", ".join(self.epoch,
    #                                     self[-1]/self[-2], err, )))

        # def get_error(self, item):
        #     return self(errors=item)
        #
        # def group1(self):
        #     return np.array(self.channels[:-2])[np.array(self.group)]
        #
        # def group2(self):
        #     return np.array(self.channels[:-2])[np.array(self.group)==False]
        #
        # # Receives pe [0 1 0 0 1 0] and that is used to define 2 groups
        # def set_group(self, new_group):
        #     self.group = new_group
        #
        # def errors_group1(self):
        #     return [self(errors=lab) for lab,g in zip(self.labels,
        #                                               self.group) if g]
        #
        # def errors_group2(self):
        #     return [self(errors=lab) for lab,g in zip(self.labels,
        #                                               self.group) if not g]
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
        #         warnings.warn("group_id must be 1 or 2 only. "
        #                       "Group 2 will be used as default.")
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #     elif group_id == 1:
        #         group = self.group1()
        #         g_errors = self.errors_group1()
        #     else:
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #
        #     self.channels[-group_id] = np.mean(group, axis=0)
        #     err = np.zeros((1, len(g_errors[0])))
        #     for i in range(len(g_errors)):
        #         err += np.divide(g_errors[i]/group[i])**2
        #     self('errors')[-group_id] = np.sqrt(err)
        #
        #     return self.channels[-group_id]
        #
        # def median(self, group_id):
        #     if group_id > 2:
        #         warnings.warn("group_id must be 1 or 2 only. "
        #                       "Group 2 will be used as default.")
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #     elif group_id == 1:
        #         group = self.group1()
        #         g_errors = self.errors_group1()
        #     else:
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #
        #     self.channels[-group_id] = np.median(group, axis=0)
        #     err = np.zeros((1, len(g_errors[0])))
        #     for i in range(len(g_errors)):
        #         err += np.divide(g_errors[i]/group[i])**2
        #     self.errors[-group_id] = np.sqrt(err)
        #
        #     return self.channels[-group_id]
        #
        #     return self.channels[-group_id]
        #
        # def median(self, group_id):
        #     if group_id > 2:
        #         warnings.warn("group_id must be 1 or 2 only."
        #                        Group 2 will be used as default.")
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #     elif group_id == 1:
        #         group = self.group1()
        #         g_errors = self.errors_group1()
        #     else:
        #         group = self.group2()
        #         g_errors = self.errors_group2()
        #
        #     self.channels[-group_id] = np.median(group, axis=0)
        #     err = np.zeros((1, len(g_errors[0])))
        #     for i in range(len(g_errors)):
        #         err += np.divide(g_errors[i]/group[i])**2
        #     self.errors[-group_id] = np.sqrt(err)
        #
        #     return self.channels[-group_id]
        #
