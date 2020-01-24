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

from __future__ import print_function, division

import scipy as sp
import ephem
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pdb

from . import obscalc as ocalc

__all__ = ['Obsrv']


def _plot_poly(ax, x, up, down, alpha=0.5, facecolor='0.8', edgecolor='k'):
    """ Plot a band  as semi-transparent polygon """
    vertices = list(zip(x, down*24)) + list(zip(x[::-1], (up*24)[::-1]))
    poly = Polygon(vertices,
                   facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(poly)


def _update_plot(func):

    def wrapper(self, *args, **kwargs):

        ret = func(self, *args, **kwargs)

        if 'plot_figure' in self.params:
            fig = self.params['plot_figure']
        else:
            fig = plt.figure(figsize=(10, 5))
            if self.params['interact']:
                self.cid = [fig.canvas.mpl_connect('button_press_event', self._onclick),
                            fig.canvas.mpl_connect('key_press_event', self._onkey)]
            self.params['plot_figure'] = fig

        fig.clf()
        self.params["plot_ax-airmass"] = ax_airmass = fig.add_axes([0.06, 0.1, 0.35, 0.85])  # subplot(121)
        self.params["plot_ax-elev"] = fig.add_axes([0.55, 0.1, 0.4, 0.83])  # fig.add_subplot(122)
        # delete right y-axis of the transit night if exists since fig.clf() disconnects it.
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

        #pdb.set_trace()
        fig.show()

        return ret

    return wrapper


# noinspection PyCompatibility,PyUnresolvedReferences
class Obsrv(ocalc.ObsCalc):

    def __init__(self,
                 target=None, show_twilight=True, show_months=True,
                 show_colorbar=True, show_transits=True,
                 interact=True, savedir='fig/', altitude_limit=30,
                 **kwargs):
        """ Initializes obsrv class.

        :param central_time: Central time of y-axis.  If outside the [-12,24] range, then only shows nighttime
        """
        if not hasattr(self, 'params'):
            self.params = {}

        self.params["interact"] = interact
        self.params["show_twilight"] = show_twilight
        self.params["show_months"] = show_months
        self.params["show_transits"] = show_transits
        self.params["show_colorbar"] = show_colorbar
        self.params["savedir"] = savedir
        self.params["altitude_limit"] = altitude_limit

        super(Obsrv, self).__init__(target=target, **kwargs)

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
        ams = sp.arange(1, 3.01, 0.1)
        self.params["plot_airmass"] = ax.contourf(self.days-self.days[0], self.hours,
                                                  sp.array(self.airmass),
                                                  levels=ams,
                                                  cmap=plt.cm.jet_r)

    def _plot_twilight(self, ax):
        sx = self.days-self.days[0]
        _plot_poly(ax, sx, self.daily["twilight_set"], self.daily["sunset"])
        _plot_poly(ax, sx, self.daily["sunrise"], self.daily["twilight_rise"])

    def _plot_months(self, ax):
        """Plot month separator"""

        month_length = sp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        cum = month_length.copy()
        for i in range(len(month_length))[1:]:
            cum[i:] += month_length[:-i]
        month_cumulative = sp.array(list(zip(cum, cum))).flatten()
        vertical_lims = list(ax.get_ylim())
        vertical_lims = (vertical_lims + vertical_lims[::-1])*6

        y, m, d = ephem.Date(self.days[0]).triple()
        jan1 = self.days[0]-((m-1 != 0)*cum[m-2]+d-1)
        ax.plot(self.days[0]-jan1 + month_cumulative, vertical_lims, 'y--')
        ny = (self.days[-1]-jan1)//365

        for i in range(int(ny)):
            ax.plot(i*365+self.days[0]-jan1+month_cumulative, vertical_lims, 'y--')
        return self

    def _plot_transits(self, ax):
        x = self.transits-self.jd0
        y = self.transit_hours
        half_length = self.transit_info['length']/2

        ax.errorbar(x, y, yerr=half_length, fmt='o', color="w")
        ax.errorbar(x, y-24, yerr=half_length, fmt='o', color="w")

    def _plot_labels(self, ax, title=''):
        ax.set_title(title)
        ax.set_xlabel(f"Days from {ephem.Date(self.days[0]).datetime().strftime('%Y.%m.%d')}"
                      f" (Site: {self.params['site']})")
        ax.set_ylabel(f'Time (UT). Target: {self.params["target"]}.'
                      f' Offset: {self.transit_info["offset"]:.2f}')
        ax.set_xlim(self.xlims)
        ax.set_ylim(sp.array(self.ylims))
        ax_borders = [0.45, 0.20, 0.035,  0.7]
        cax = ax.figure.add_axes(ax_borders)
        ax.figure.colorbar(self.params["plot_airmass"], cax=cax)
        cax.yaxis.set_label_position('left')
        cax.yaxis.set_ticks_position('left')

        ax.text(1.12, 0.04, "Min.", transform=ax.transAxes, ha="center")
        ax.text(1.12, 0.005, '%.2f' % min(self.airmass.flatten()), transform=ax.transAxes, ha="center")

        return self

    def _plot_elev_transit(self, day, hour):

        if not hasattr(self, "transits"):
            return

        ax = self.params["plot_ax-elev"]
        close_jd = self.get_closer_transit(day, hour)

        return self._plot_night(close_jd, ax)

    def get_closer_transit(self, cd, ch):

        xy_ratio = 5.0/4.0
        dx = self.xlims[1]-self.xlims[0]
        dy = self.ylims[1]-self.ylims[0]

        dist = []
        for htr, dtr in zip(self.transit_hours, self.transits):
            dist.append(sp.sqrt(((cd+self.jd0-dtr)*xy_ratio/dx)**2
                        + ((ch-htr)/dy)**2))
            dist.append(sp.sqrt(((cd+self.jd0-dtr)*xy_ratio/dx)**2
                        + ((ch-(htr-24))/dy)**2))
        return sp.array(list(zip(self.transits,
                                 self.transits))).flatten()[sp.argmin(dist)]

    def _plot_night(self, jd, ax):

        ax.cla()
        #todo: ax_elev2 should be always initialized
        if 'plot_ax-elev2' in self.params:
            self.params['plot_ax-elev2'].cla()

        moon = ephem.Moon()
        obs = self._obs
        altitude_limit = self.params['altitude_limit']
        obs.date = jd - self.jd0 + self.days[0]
        midday = obs.previous_transit(self._sun)
        obs.date = midday
        ss = obs.next_setting(self._sun)
        sr = obs.next_rising(self._sun)
        obs.horizon = '-18:00'
        ts = obs.next_setting(self._sun)
        tr = obs.next_rising(self._sun)
        obs.horizon = '0:00'
        hours = sp.arange(ss-0.03, sr+0.03, 0.007)

        moon_altitude = []
        star_altitude = []
        for h in hours:
            obs.date = h
            moon.compute(obs)
            moon_altitude.append(moon.alt*180/sp.pi)
            self.star.compute(obs)
            star_altitude.append(self.star.alt*180/sp.pi)

        et_out = sp.fix(hours[0])+0.5
        ut_hours = (hours - et_out)*24

        ax2 = ax.twinx()
        if 'plot_ax-elev2' in self.params:
            self.params['plot_figure'].delaxes(self.params['plot_ax-elev2'])
        self.params['plot_ax-elev2'] = ax2

        ax.plot(ut_hours, star_altitude)
        ax.plot(ut_hours, moon_altitude, '--')
        setev = sp.array([ss, ss, ts, ts])
        risev = sp.array([sr, sr, tr, tr])
        ax.plot((setev - et_out)*24, [0, 90, 90, 0], 'k:')
        ax.plot((risev - et_out)*24, [0, 90, 90, 0], 'k:')
        ax.plot([ut_hours[0], ut_hours[-1]], [altitude_limit]*2, 'k:')
        ax.set_ylim([10, 90])
        ax.set_xlim([(ss-et_out)*24-0.5, (sr-et_out)*24+0.5])
        datetime = ephem.date(jd-self.jd0+self.days[0])
        phase_info = f"phase {self.transit_info['offset']}: " if self.transit_info['offset'] != 0.0 else ""
        print("%s%s %s" % (phase_info, str(datetime)[:-3], jd))
        ax.set_title('%s%s' % (phase_info, str(datetime)[:-3]))
        ax.set_ylim(ax.get_ylim())
        sam = sp.array([1, 1.5, 2, 3, 4, 5])
        ax2.set_yticks(sp.arcsin(1.0/sam)*180.0/sp.pi)
        ax2.set_yticklabels(sam)
        self.params['current_transit'] = str(datetime).replace(' ', '_')
        self.params['current_moon_distance'], self.params['current_moon_phase'] = self._moon_distance(datetime)
        percent = 0  # todo: ts<sp.array(ut_hours)<tr
        ax.set_ylabel(f'Elevation ({percent}% inside twilight and {altitude_limit}${{^\\degree}}$)')
        ax.set_xlabel(f'UT time. Moon distance and phase: '
                      f'{int(self.params["current_moon_distance"].degree)}${{^\\degree}}$ '
                      f'{float(self.params["current_moon_phase"]):.0f}%')

        if hasattr(self, 'transits'):
            enter_transit = (jd-self.jd0+self.days[0]-et_out)*24 - self.transit_info['length']/2
            exit_transit = (jd-self.jd0+self.days[0]-et_out)*24 + self.transit_info['length']/2
            if enter_transit > exit_transit:
                exit_transit += 24
            facecolor = '0.5'
            if self.params['current_moon_distance'].degree < 30:
                facecolor = 'orange'
            if self.params['current_moon_distance'].degree < 10:
                facecolor = 'red'
            _plot_poly(ax, [enter_transit, exit_transit],
                       [0, 0], [90, 90], facecolor=facecolor)

        ax.figure.canvas.draw()

        return self

    def _onkey(self, event):

        axe = self.params["plot_ax-elev"]
        axa = self.params["plot_ax-airmass"]

        if event.key == 'e' and event.inaxes == axa:  # at position
            self._plot_night(event.xdata + self.jd0, axe)
        elif event.key == 'P':  # save file at no transit or transit if it has been set before
            target_no_space = self.params['target'].replace(' ', '_')
            if 'current_transit' in self.params:
                current_transit = self.params['current_transit'].replace('/', '-')
                filename = '%s/%s_%s_%s.png' % (self.params['savedir'],
                                                target_no_space,
                                                current_transit,
                                                self.params['site'])
            else:
                filename = '%s/%s_T%s_%s.png' % (self.params['savedir'],
                                                 target_no_space,
                                                 self.params['timespan'],
                                                 self.params['site'])
            print("Saving: %s" % (filename,))
            self.params['plot_figure'].savefig(filename)
        elif event.key == 'f':  # recenter transit and save file
            self._plot_elev_transit(event.xdata, event.ydata)
            target_no_space = self.params['target'].replace(' ', '_')
            site = self.params['site']
            current_transit = self.params['current_transit'].replace('/', '-')[:-3].replace(':', '')
            filename = '%s/%s_%s_%s.png' % (self.params['savedir'],
                                            target_no_space,
                                            current_transit,
                                            site)
            print("Saving: %s" % (filename,))
            self.params['plot_figure'].savefig(filename)

    def _onclick(self, event):

        if event.inaxes != self.params["plot_ax-airmass"]:
            return

        day, hour = event.xdata, event.ydata
        self._plot_elev_transit(day, hour)


if __name__ == '__MAIN__':
    import obsrv
    test_target = ['HD83443', 9+(37 + 11.82841/60)/60.0, -(43+(16+19.9354/60)/60.0),
                   2455943.20159650, 2.985, 3]
    test_target = ['WASP-34', 11+(1+36/60.0)/60.0, -(23+(51+38/60)/60),
                   2454647.55358, 4.3176782, 3]
    a = obsrv.Obsrv((test_target[1], test_target[2]),
                          tr_period=test_target[4],
                          tr_epoch=test_target[3],
                          title=test_target[0])


