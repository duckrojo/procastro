#
#
# Copyright (C) 2012 Patricio Rojo
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
import warnings

from . import obscalc as ocalc

import numpy as np
import astropy.time as apt
import astropy.coordinates as apc
import astropy.units as u
#import ephem

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.pyplot import cm

__all__ = ['Obsrv']


def _plot_poly(ax, x, up, down, alpha=0.5, facecolor='0.8', edgecolor='k'):
    """
    Plot a band as a semi-transparent polygon

    Parameters
    ----------
    ax : matplotlib.pyplot.axes
    x : list
    up : list of booleans
    down : list of booleans
    alpha : float, optional
    facecolor : string, optional
    edgecolor : string, optional
    """

    vertices = list(zip(x, down*24)) + list(zip(x[::-1], (up*24)[::-1]))
    poly = Polygon(vertices,
                   facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(poly)


def _update_plot(func):
    """
    Decorator used to update the displayed figures after an attribute is
    modified by the user.
    """
    def wrapper(self, *args, **kwargs):

        ret = func(self, *args, **kwargs)

        if 'plot_figure' in self.params:
            fig = self.params['plot_figure']
        else:
            fig = plt.figure(figsize=(10, 5))
            if self.params['interact']:
                self.cid = [fig.canvas.mpl_connect('button_press_event',
                                                   self._onclick),
                            fig.canvas.mpl_connect('key_press_event',
                                                   self._onkey)]
            self.params['plot_figure'] = fig

        fig.clf()
        # subplot(121)
        self.params["plot_ax-airmass"] = \
            ax_airmass = \
            fig.add_axes([0.06, 0.1, 0.35, 0.85])

        # fig.add_subplot(122)
        self.params["plot_ax-elev"] = fig.add_axes([0.55, 0.1, 0.4, 0.83])
        # Delete right y-axis of the transit night if exists since fig.clf()
        # disconnects it.
        #
        if 'plot_ax-elev2' in self.params:
            del self.params['plot_ax-elev2']

        if self.airmass is not None:
            self._plot_airmass(ax_airmass)
            if self.params["show_colorbar"]:
                self._plot_labels(ax_airmass)
        if self.params["show_twilight"]:
            self._plot_twilight(ax_airmass)
        if self.params["show_months"]:
            self._plot_months(ax_airmass)
        if self.params["show_transits"] and self.transits is not None:
            self._plot_transits(ax_airmass)

        ax_airmass.set_ylim(self.ylims)
        ax_airmass.set_xlim(self.xlims)

        fig.show()

        return ret

    return wrapper


# noinspection PyCompatibility,PyUnresolvedReferences
class Obsrv(ocalc.ObsCalc):
    """
    Class used to schedule day/hour of an exoplanet transit

    During initialization this object will generate two images which can
    be interacted with.

    Right Image:
        Displays an exoplanet eclipsing transits across a specific frame,
        where the x-axis shows the day of occurrence while the y-axis
        corresponds to the hour of the event.

        Each transit is shown as a white dot which can be clicked to display
        a plot image specific to that transit located to the left.

        Yellow lines are used to delimit months, the background color is used
        to plot the expected airmass on that date and the grey bands shown
        above and below represent the sun and twilight set and rise.

    Left Image:
        Elevation vs time graph, showing the elevation of the transit in blue
        vs the moon distance in yellow.

    On each click, the object will print the datetime of the transit in
    standard format and JD format.

    Setters are used to change the parameters of the current Obsrv instance
    while the interactive plot is active. Closing the plot and setting data
    afterwards will cause unexpected behaviour.

    Parameters
    ----------
    target : string, optional
        Object to be observed
    show_twilight : bool, optional
    show_months : bool, optional
    show_colorbar : bool, optional
    show_transits : bool, optional
    interact : bool, optional
    savedir : str, optional
        Directory where the plotted figures will be stored
    altitude_limit : int, optional
    central_time : int, optional
        Central time of the y-axis. If outside the [-12,24] range then only
        shows nighttime.

    Attributes
    ----------
    params : dict
        Store each parameter current values
    cid : list
        Stores interactive mode pyplot variables and methods
    airmass : array_like
    xlims, ylims : array_like

    See Also
    --------
    procastro.obsrv.ObsCalc : Parent class which computes transit data

    """

    def __init__(self,
                 target=None, show_twilight=True, show_months=True,
                 show_colorbar=True, show_transits=True,
                 interact=True, savedir='fig/', altitude_limit=30,
                 timespan=2024,
                 **kwargs):

        if not hasattr(self, 'params'):
            self.params = {}

        self.params["interact"] = interact
        self.params["show_twilight"] = show_twilight
        self.params["show_months"] = show_months
        self.params["show_transits"] = show_transits
        self.params["show_colorbar"] = show_colorbar
        self.params["savedir"] = savedir
        self.params["altitude_limit"] = altitude_limit

        super(Obsrv, self).__init__(target=target, timespan=timespan, **kwargs)

    @_update_plot
    def set_target(self, *args, **kwargs):
        super(Obsrv, self).set_target(*args, **kwargs)

    @_update_plot
    def set_vertical(self, *args, **kwargs):
        super(Obsrv, self).set_vertical(*args, **kwargs)

    @_update_plot
    def set_transits(self, *args, **kwargs):
        super(Obsrv, self).set_transits(*args, **kwargs)

    def _plot_airmass(self, ax):
        ams = np.arange(1, 3.01, 0.1)
        self.params["plot_airmass"] = ax.contourf(self.days.jd-self.days[0].jd,
                                                  self.hours,
                                                  np.array(self.airmass),
                                                  levels=ams,
                                                  cmap=cm.jet_r,
                                                  extend="max")

    def _plot_twilight(self, ax):
        """
        Plots sunrise and twilight limits for the transit display (First image)

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
        """
        sx = self.days.jd-self.days[0].jd
        _plot_poly(ax, sx, self.daily["twilight_set"], self.daily["sunset"])
        _plot_poly(ax, sx, self.daily["sunrise"], self.daily["twilight_rise"])

    def _plot_months(self, ax):
        """
        Plot month separator

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
        """

        month_length = np.array([31, 28, 31, 30, 31, 30,
                                 31, 31, 30, 31, 30, 31])
        tm = self.days[0]
        y, m, d, _, _, _ = tm.ymdhms
        cum = -(d - 1)
        m -= 1

        while cum < self.xlims[1]:
            ax.axvline(cum, ls='--', color='yellow')
            cum += month_length[m % 12]
            m += 1

        return self

    def _plot_transits(self, ax):
        """
        Parameters
        ----------
        ax : matplotlib.pyplot.axes
        """
        x = self.transits-self.days[0].jd
        y = self.transit_hours
        half_length = self.transit_info['length']/2

        ax.errorbar(x, y, yerr=half_length, fmt='o', color="w")
        ax.errorbar(x, y-24, yerr=half_length, fmt='o', color="w")

    def _plot_labels(self, ax, title=''):
        """
        Parameters
        ----------
        ax : matplotlib.pyplot.axes
        """
        ax.set_title(title)
#        days = apt.Time((self.days[0] + 2415020), format='jd').strftime('%Y.%m.%d')
        ax.set_xlabel(f"Days from {self.days[0].isot[0:10]}"
                      f" (Site: {self.params['site']})")
        ax.set_ylabel(f'Time (UT). Target: {self.params["target"]}.'
                      f' Offset: {self.transit_info["offset"]:.2f}')
        ax.set_xlim(self.xlims)
        ax.set_ylim(np.array(self.ylims))
        ax_borders = [0.45, 0.20, 0.035,  0.7]
        cax = ax.figure.add_axes(ax_borders)
        ax.figure.colorbar(self.params["plot_airmass"], cax=cax)
        cax.yaxis.set_label_position('left')
        cax.yaxis.set_ticks_position('left')

        ax.text(1.12, 0.04, "Min.", transform=ax.transAxes, ha="center")
        ax.text(1.12, 0.005,
                '{0:.2f}'.format(min(self.airmass.flatten())),
                transform=ax.transAxes,
                ha="center")

        return self

    def _plot_elev_transit(self, day, hour):

        if not hasattr(self, "transits"):
            return

        ax = self.params["plot_ax-elev"]
        close_jd = self.get_closer_transit(day, hour)

        return self._plot_night(close_jd, ax)

    def get_closer_transit(self, cd, ch):
        """
        Obtains the closest transit of the target from the given day and hour

        Parameters
        ----------
        cd, ch : float (JD format)
            Current day and hour

        Returns
        -------
        array_like
        """
        xy_ratio = 5.0/4.0
        dx = self.xlims[1]-self.xlims[0]
        dy = self.ylims[1]-self.ylims[0]

        dist = []
        for htr, dtr in zip(self.transit_hours, self.transits):
            dist.append(np.sqrt(((cd+self.days[0].jd-dtr)*xy_ratio/dx)**2
                        + ((ch-htr)/dy)**2))
            dist.append(np.sqrt(((cd+self.days[0].jd-dtr)*xy_ratio/dx)**2
                        + ((ch-(htr-24))/dy)**2))
        return np.array(list(zip(self.transits,
                                 self.transits))).flatten()[np.argmin(dist)]

    def plot_values(self, x, y, idx,
                    rel_pos=1.0, abs_pos=None,
                    color="black", marker=None):

        ax = self.params["plot_ax-elev"]
        altitude = y[idx].value
        time = x[idx]
        if abs_pos is None:
            pos = rel_pos * altitude
        else:
            pos = abs_pos

        if marker in ['x', '+', 'c', '.']:
            ax.plot(time, altitude, marker, color=color)
        if marker in ['-', '--', ':']:
            ax.plot([time, time], [altitude, pos],
                    ls=marker, color=color)

        time_label = time
        if time_label < 0:
            time_label += 24
        minutes = int(60 * (time_label % 1) + 0.5)
        ax.annotate(f'{altitude:.1f}$^\circ$@ {int(time_label)}:{minutes:02d}UT',
                    (time, pos),
                    ha='center', va='top', color=color)

    def _plot_night(self, jd, ax):

        ax.cla()
        # todo: ax_elev2 should be always initialized
        if 'plot_ax-elev2' in self.params:
            self.params['plot_ax-elev2'].cla()

        loc = self._location
        star_coords = self._target

        with apc.solar_system_ephemeris.set('builtin'):

            n_hours = 200
            previous_24 = apt.Time(jd, format='jd') - np.arange(0, 1, 0.05)
            ref_at_night = previous_24[0]
            previous_midday_idx = np.argmax(apc.get_sun(
                ref_at_night).transform_to(apc.AltAz(obstime=previous_24,
                                                     location=loc)).alt)
            the_24 = previous_24[previous_midday_idx] + np.linspace(0, 1, n_hours)

            the_24_alt = apc.get_sun(
                ref_at_night).transform_to(apc.AltAz(obstime=the_24,
                                                     location=loc)).alt
            twi_set_rise_twi_idx = [np.argmin(np.absolute(the_24_alt[:n_hours // 2])),
                                    np.argmin(np.absolute((the_24_alt+18*u.deg)[:n_hours // 2])),
                                    np.argmin(np.absolute((the_24_alt+18*u.deg)[n_hours // 2:])) + n_hours // 2,
                                    np.argmin(np.absolute(the_24_alt[n_hours // 2:])) + n_hours // 2,
                                    ]

            night_span = the_24[twi_set_rise_twi_idx[3]] - the_24[twi_set_rise_twi_idx[0]]
            delta = 1.1*night_span / (n_hours - 1)
            hours = the_24[twi_set_rise_twi_idx[0]] - 0.05*night_span + delta*np.arange(n_hours)

            moon_coords = apc.get_moon(hours, loc)
            alt_moon = moon_coords.transform_to(apc.AltAz(obstime=hours, location=loc)).alt
            alt_target = star_coords.transform_to(apc.AltAz(obstime=hours, location=loc)).alt

        et_out = np.fix(hours[0].jd)+0.5
        ut_hours = 24*(hours.jd - et_out)

        ax2 = ax.twinx()
        if 'plot_ax-elev2' in self.params:
            self.params['plot_figure'].delaxes(self.params['plot_ax-elev2'])
        self.params['plot_ax-elev2'] = ax2

        ax.plot(ut_hours, alt_target)
        ax.plot(ut_hours, alt_moon, '--')

        setev = np.array([the_24[twi_set_rise_twi_idx[0]].jd] * 2 +
                         [the_24[twi_set_rise_twi_idx[1]].jd] * 2)
        risev = np.array([the_24[twi_set_rise_twi_idx[2]].jd] * 2 +
                         [the_24[twi_set_rise_twi_idx[3]].jd] * 2)
        altitude_limit = self.params['altitude_limit']

        ax.plot((setev - et_out)*24, [0, 90, 90, 0], 'k:')
        ax.plot((risev - et_out)*24, [0, 90, 90, 0], 'k:')
        ax.plot([ut_hours[0], ut_hours[-1]], [altitude_limit]*2, 'k:')
        ax.set_ylim([10, 90])
        ax.set_xlim(ut_hours[[0, -1]])

        tm = apt.Time(jd, format='jd')
        datetime = tm.iso.replace("-", "/")

        phase_info = f"phase {self.transit_info['offset']}: " if self.transit_info['offset'] != 0.0 else ""
        print("{0:s} {1:s} {2:f}".format(phase_info, str(datetime)[:-3], jd))

        ax.set_title(f'{phase_info}{str(datetime[:-4])} JD{jd:.2f}')
        ax.set_ylim(ax.get_ylim())
        sam = np.array([1, 1.5, 2, 3, 4, 5])
        ax2.set_yticks(np.arcsin(1.0/sam)*180.0/np.pi)
        ax2.set_yticklabels(sam)
        self.params['current_transit'] = str(datetime).replace(' ', '_')
        self.params['current_moon_distance'], self.params['current_moon_phase'] = self._moon_distance(tm)
        percent = 0  # todo: ts<np.array(ut_hours)<tr

        ax.set_ylabel(f'Elevation ({percent}% inside twilight and {altitude_limit}${{^\\degree}}$)')
        ax.set_xlabel(f'UT time. Moon distance and phase: '
                      f'{int(self.params["current_moon_distance"].degree)}${{^\\degree}}$ '
                      f'{float(self.params["current_moon_phase"]):.0f}%')

        if hasattr(self, 'transits'):
            length = self.transit_info['length']
            if length is None:
                length = 1
                warnings.warn(f"Length was not found in database, using default {length}")

            enter_transit = ((jd-et_out)*24 - length/2)
            exit_transit = ((jd-et_out)*24 + length/2)
            if enter_transit > exit_transit:
                exit_transit += 24
            facecolor = '0.5'
            if self.params['current_moon_distance'].degree < 30:
                facecolor = 'orange'
            if self.params['current_moon_distance'].degree < 10:
                facecolor = 'red'
            _plot_poly(ax, [enter_transit, exit_transit],
                       [0, 0], [90, 90], facecolor=facecolor)

            idx = np.argmin(np.abs(ut_hours + 1 - enter_transit))
            self.plot_values(ut_hours, alt_target, idx, abs_pos=20, marker='-', color='red')
            idx = np.argmin(np.abs(ut_hours - 1 - exit_transit))
            self.plot_values(ut_hours, alt_target, idx, abs_pos=25, marker='-', color='red')

        mm_idx = np.argmax(alt_target)
        self.plot_values(ut_hours, alt_target, mm_idx, abs_pos=85, marker='+', color='blue')

        ax.figure.canvas.draw()

        return self

    def _onkey(self, event):

        axe = self.params["plot_ax-elev"]
        axa = self.params["plot_ax-airmass"]

        if event.key == 'e' and event.inaxes == axa:  # at position
            self._plot_night(event.xdata + self.days[0].jd, axe)
        # Save file at no transit or transit if it has been set before
        elif event.key == 'P':
            target_no_space = self.params['target'].replace(' ', '_')
            if 'current_transit' in self.params:
                current_transit = self.params['current_transit'].replace('/', '-')[:-3].replace(':', '')
                filename = "{0:s}/{1:s}_{2:s}_{3:s}.png".format(self.params['savedir'],
                                                                target_no_space,
                                                                current_transit,
                                                                self.params['site'])
            else:
                filename = "{0:s}/{1:s}_T{2:s}_{3:s}.png".format(self.params['savedir'],
                                                                 target_no_space,
                                                                 str(self.params['timespan']),
                                                                 self.params['site'])

            print("Saving: {0:s}".format(filename,))
            self.params['plot_figure'].savefig(filename)
        elif event.key == 'p':  # recenter transit and save file
            self._plot_elev_transit(event.xdata, event.ydata)
            target_no_space = self.params['target'].replace(' ', '_')
            site = self.params['site']
            current_transit = self.params['current_transit'].replace('/', '-')[:-3].replace(':', '')
            filename = '{0:s}/{1:s}_{2:s}_{3:s}.png'.format(self.params['savedir'],
                                                            target_no_space,
                                                            current_transit,
                                                            site)
            print("Saving: {0:s}".format(filename,))
            self.params['plot_figure'].savefig(filename)

    def _onclick(self, event):

        if event.inaxes != self.params["plot_ax-airmass"]:
            return

        day, hour = event.xdata, event.ydata
        self._plot_elev_transit(day, hour)
