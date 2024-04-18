#
# Copyright (C) 2023 Patricio Rojo
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
# Contributors: Pablo Cañas
#

import os.path
from pathlib import Path

import astropy.coordinates as apc
import matplotlib.pyplot as plt
import astropy.time as apt
import astropy.units as u
import matplotlib.axes
import pyvo as vo
import pyvo.dal.exceptions

import procastro as pa
import procastro.astro as paa
import matplotlib as mpl
import numpy as np
import pandas as pd
import time
from typing import Union, Optional, Tuple

import procastro.astro.coordinates

TwoTuple = Tuple[float, float]

__all__ = ['query_full_exoplanet_db', 'Nightly']


def query_full_exoplanet_db(force_reload: bool = False,
                            reload_days: float = 7
                            ):
    file = pa.user_confdir("exodb.pickle")
    if not force_reload and os.path.isfile(file):
        days_since = (os.path.getmtime(file) - time.time()) / 3600 / 24
        if days_since < reload_days:
            return pd.read_pickle(file)

    exo_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    try:
        resultset = exo_service.search(
            f"SELECT pl_name,ra,dec,pl_orbper,pl_tranmid,disc_facility,pl_trandur,sy_pmra,sy_pmdec,sy_vmag,sy_gmag "
            f"FROM exo_tap.pscomppars "
            f"WHERE pl_tranmid!=0.0 and pl_orbper!=0.0 ")
    except pyvo.dal.exceptions.DALFormatError:
        if os.path.isfile(file):
            days_since = (time.time() - os.path.getmtime(file)) / 3600 / 24
            print(f"Cannot connect to the Exoplanet Archive Database, "
                  f"using cached version instead ({days_since:.1f} days old)")
            return pd.read_pickle(file)
        else:
            raise ConnectionError("Cannot connect to the Exoplanet Archive dabatase")
    planets_df = resultset.to_table().to_pandas()
    planets_df.to_pickle(file)

    return planets_df


def _choose(dataframe: pd.DataFrame,
            field1: str,
            field2: Union[str, float],
            return_max: bool = True
            ) -> list:
    """Choose the maximum or minimum between two columns or between a column and a scalar"""
    ret = dataframe[field1].copy()
    if isinstance(field2, (int, float)):
        field2 = np.ones(len(ret)) * field2
    else:
        field2 = dataframe[field2]
    two_over_one = dataframe[field1] < field2
    if return_max:
        ret.mask(two_over_one, field2[two_over_one], inplace=True)
    else:
        ret.mask(~two_over_one, field2[~two_over_one], inplace=True)

    return list(ret)


class Nightly:
    def __init__(self,
                 observatory="lasilla",
                 altitude_min=25,

                 # observing constraints
                 night_angle=-12,
                 moon_separation_min=10,
                 vmag_min=10,
                 vmag_max=15,

                 # transiting constraints
                 transit_percent_min=0.999,
                 baseline_min=0.2,
                 baseline_max=1,
                 baseline_both=True,

                 # defaults for transits
                 default_duration=2,

                 # DB paprameter
                 force_reload=False,
                 reload_days=7,
                 ):
        self._sidereal_at_sets = None
        self._moon_coord = None
        self._constraints = {}
        self._observatory = None
        self._observatory_name = None
        self._start_night = None
        self._end_night = None
        self._date = None
        self._civil_midday = None

        self._stage = 0  # processing stage... 0:no filter, 1:pre-ephem filter, 2:post-ephem filter
        self._planets = None
        self._planets_after_pre_filter = None

        # create Dataframe and update missing magnitude
        planets = query_full_exoplanet_db(force_reload=force_reload, reload_days=reload_days)
        no_filter_data = planets['sy_vmag'] == 0
        planets.loc[:, 'sy_vmag'].where(~no_filter_data, other=planets['sy_gmag'][no_filter_data], inplace=True)
        self._planets_all = planets

        self.observatory(observatory)
        self.constraints(altitude_min=altitude_min,
                         night_angle=night_angle,
                         moon_separation_min=moon_separation_min,
                         vmag_min=vmag_min,
                         vmag_max=vmag_max,
                         transit_percent_min=transit_percent_min,
                         baseline_max=baseline_max,
                         baseline_min=baseline_min,
                         baseline_both=baseline_both,
                         default_duration=default_duration,
                         )

    #################################
    #
    # DEFINE PARAMETERS

    def observatory(self,
                    observatory):
        self._observatory_name = observatory
        self._observatory = apc.EarthLocation.of_site(observatory)
        self._planets = self._planets_all
        self._stage = 0

    def constraints(self,
                    # observatory parameters
                    altitude_min=None,
                    # observing constraints
                    night_angle=None, moon_separation_min=None,
                    vmag_min=None, vmag_max=None,
                    # transiting constraints
                    transit_percent_min=None,
                    baseline_min=None, baseline_max=None, baseline_both=None,
                    default_duration=None,
                    ):

        # check if any of the new constraints affect the pre-ephemeris filter
        if self._stage > 0 and (night_angle is not None or
                                vmag_min is not None or
                                vmag_max is not None or
                                default_duration is not None):
            self._stage = 0

        # check if any of the new constraints affect the post-ephemeris filter
        if self._stage > 1 and (altitude_min is not None or
                                moon_separation_min is not None or
                                transit_percent_min is not None or
                                baseline_max is not None or
                                baseline_min is not None or
                                baseline_both is not None):
            self._stage = 1

        self._constraints = {"altitude_min": altitude_min,

                             # observing constraints
                             "night_angle": night_angle,
                             "moon_separation_min": moon_separation_min,
                             "vmag_min": vmag_min,
                             "vmag_max": vmag_max,

                             # transiting constraints
                             "transit_percent_min": transit_percent_min,
                             "baseline_min": baseline_min,
                             "baseline_max": baseline_max,
                             'baseline_both': baseline_both,

                             # transit defaults
                             "default_duration": default_duration,
                             }

    def set_date(self, date, n_points=1400, equinox_db='J2000'):
        if date is None:
            return

        self._date = date
        self._civil_midday = apt.Time(date) + 12 * u.hour - (int(self._observatory.lon.degree / 15)) * u.hour
        the_day = self._civil_midday + np.linspace(0, 24, n_points) * u.hour
        sun_alt = np.array(apc.get_body("sun", the_day).transform_to(apc.AltAz(obstime=the_day, location=self._observatory)
                                                             ).alt.degree)
        above = list(sun_alt > self._constraints['night_angle'])

        start_night_idx = above.index(False)
        self._start_night = the_day[start_night_idx]
        self._end_night = the_day[above.index(True, start_night_idx + 1)]


        self._moon_coord = apc.get_body("moon", self._civil_midday + 12 * u.hour, location=self._observatory)
        times_at_sets = apt.Time([self._start_night, self._end_night])
        self._sidereal_at_sets = times_at_sets.sidereal_time('apparent', self._observatory).hourangle

        # update database values
        planets = self._planets
        closest_transit_n = (((self._civil_midday.jd + 0.5 - planets['pl_tranmid']) /
                              planets['pl_orbper']) + 0.5).astype('int')
        self._planets['closest_transit'] = planets['pl_tranmid'] + closest_transit_n * planets['pl_orbper']
        self._planets['pl_trandur'].mask(planets['pl_trandur'] == 0.0, self._constraints['default_duration'],
                                         inplace=True)
        self._planets['transit_i'] = planets['closest_transit'] - (planets['pl_trandur'] / 48)
        self._planets['transit_f'] = planets['closest_transit'] + (planets['pl_trandur'] / 48)
        self._planets['pl_trandur'].fillna(self._constraints['default_duration'], inplace=True)
        self._planets['sy_pmra'].fillna(0, inplace=True)
        self._planets['sy_pmdec'].fillna(0, inplace=True)

        pm_right_ascension = planets['sy_pmra'].array * u.mas / u.yr
        pm_declination = planets['sy_pmdec'].array * u.mas / u.yr
        coords = apc.SkyCoord(planets['ra'].array * u.degree, planets['dec'].array * u.degree,
                              frame='icrs', pm_ra_cosdec=pm_right_ascension, pm_dec=pm_declination,
                              equinox=equinox_db)
        self._planets['star_coords'] = coords
        self._planets['moon_separation'] = np.array(self._moon_coord.separation(coords))

        self._stage = 0

    ################################
    #
    # COMPUTATIONS

    def _rank_events(self):
        self._planets['rank'] = 10

    def _hour_angle_for_altitude(self, skycoord, altitude):
        if not isinstance(altitude, u.Quantity):
            altitude = (altitude*u.degree).to(u.radian).value

        return procastro.astro.coordinates.hour_angle_for_altitude(skycoord.dec.radian, self._observatory.lat.radian, altitude)

    def _ephemeris(self):
        planets = self._planets.copy()

        skycoords = apc.SkyCoord(list(planets['star_coords']))
        hour_angle_sets = self._hour_angle_for_altitude(skycoords, self._constraints['altitude_min'])
        # following is for not filtering circumpolar stars
        hour_angle_sets[np.isnan(hour_angle_sets)] = 13 * u.hourangle

        delta_ra = skycoords.ra.hourangle - self._sidereal_at_sets[0]
        delta_ra[delta_ra < -12] += 24
        planets['starrise'] = (self._start_night+(delta_ra - hour_angle_sets.value)*u.sday/24).jd
        planets['starset'] = (self._start_night+(delta_ra + hour_angle_sets.value)*u.sday/24).jd

        planets['start_observation'] = planets['transit_i'] - self._constraints['baseline_max']/24
        planets['start_observation'] = _choose(planets, 'start_observation', self._start_night.jd, return_max=True)
        planets['start_observation'] = _choose(planets, 'start_observation', 'starrise', return_max=True)
        planets['end_observation'] = planets['transit_f'] + self._constraints['baseline_max']/24
        planets['end_observation'] = _choose(planets, 'end_observation', self._end_night.jd, return_max=False)
        planets['end_observation'] = _choose(planets, 'end_observation', 'starset', return_max=False)

        self._planets = planets.copy()

    # ###############################
    #
    # FILTERING

    def apply_filters_needed(self):
        if self._stage == 0:
            self._planets = self._planets_all.copy()
            self._filter_pre_ephemeris()
            self._ephemeris()
            self._planets_after_pre_filter = self._planets.copy()
            self._filter_post_ephemeris()

        if self._stage == 1:
            self._planets = self._planets_after_pre_filter.copy()
            self._filter_post_ephemeris()

        self._rank_events()
        self._stage = 2

    def _filter_post_ephemeris(self):
        """"Filters to be applied that depend on stellar ephemeris for epoch"""
        self._transit_percent_altitude_filter()
        self._baseline_filter()
        self._moon_separation_filter()

    def _filter_pre_ephemeris(self):
        """Filters to be applied that do not depend on stellar ephemeris"""
        self._closest_transit_filter()
        self._vmag_filter()
        self._transit_percent_twilight_filter()
        self._max_altitude_filter()
        self._altitude_at_night_filter()

    def _altitude_at_night_filter(self):
        """Filters out those that never reach the minimum altitude during the night"""
        planets = self._planets
        night_length = (self._end_night - self._start_night).to(u.hour).value
        times_at_sets = apt.Time([self._start_night, self._end_night])
        planet_hour_angle = planets['ra'] - self._sidereal_at_sets[0]
        planet_hour_angle[planet_hour_angle < 0] += 24

        # Any star whose max altitude is reached during the night is a possible observation, otherwise they need
        # to be checked whether the min altitude is reached at sunset or sunrise. If so, upgrade them into possible.
        # Any other (reaching max altitude during daytime) is out
        stars_with_posibilites = planet_hour_angle > night_length
        to_be_checked = ~stars_with_posibilites
        frame_with_uncertain = apc.AltAz(obstime=times_at_sets, location=self._observatory)
        star_coords = list(planets[to_be_checked]['star_coords'])
        sunset_altitude_of_uncertain = apc.SkyCoord(star_coords,
                                                    ).transform_to(frame_with_uncertain[0]).alt.degree
        sunrise_altitude_of_uncertain = apc.SkyCoord(star_coords,
                                                     ).transform_to(frame_with_uncertain[1]).alt.degree
        possible = ((sunset_altitude_of_uncertain > self._constraints['altitude_min']) +
                    (sunrise_altitude_of_uncertain > self._constraints['altitude_min']))

        stars_with_posibilites[to_be_checked] = possible

        self._planets = planets[stars_with_posibilites].copy()

    def _max_altitude_filter(self):
        """Filters out stars that never reach the minimum altitude"""
        latitude = self._observatory.lat.degree
        self._planets = self._planets[90 - np.abs(latitude - self._planets['dec']) > self._constraints['altitude_min']]

    def _baseline_filter(self):
        """Filter for enough baseline either in one or both sides"""
        planets = self._planets
        planets['delta_i_baseline'] = planets['transit_i'] - planets['start_observation']
        planets['delta_f_baseline'] = planets['end_observation'] - planets['transit_f']

        self._planets = planets.query(f'delta_i_baseline>@self._constraints["baseline_min"]/24'
                                      f' {"or" if self._constraints["baseline_both"] else "and"} '
                                      f'delta_f_baseline>@self._constraints["baseline_min"]/24'
                                      ).copy()

    def _closest_transit_filter(self):
        delta_transit_jd = apt.Time(self._planets['closest_transit'], format='jd') - self._civil_midday + 12 * u.hour
        self._planets = self._planets[-0.5 <= (delta_transit_jd.to(u.day).value < 0.5)]

    def _moon_separation_filter(self):
        self._planets = self._planets.query('moon_separation > @self._constraints["moon_separation_min"]'
                                            )

    def _vmag_filter(self):
        self._planets = self._planets.query('sy_vmag > @self._constraints["vmag_min"] and '
                                            'sy_vmag < @self._constraints["vmag_max"]'
                                            )

    def _transit_percent_twilight_filter(self):
        """filter for enough observable transit, considering only twilight"""
        planets = self._planets.copy()

        planets.query('transit_i < @self._end_night.jd', inplace=True)
        planets.query('transit_f > @self._start_night.jd', inplace=True)

        planets['transit_i_twi'] = _choose(planets, 'transit_i', self._start_night.jd, return_max=True)
        planets['transit_f_twi'] = _choose(planets, 'transit_f', self._end_night.jd, return_max=False)
        planets['transit_twilight_percent'] = (planets['transit_f_twi'] -
                                               planets['transit_i_twi']) / (planets['pl_trandur'] / 24)

        planets.query('transit_twilight_percent > @self._constraints["transit_percent_min"]',
                      inplace=True)
        self._planets = planets.copy()

    def _transit_percent_altitude_filter(self):
        """filter for enough observable transit, considering altitude and twilight"""
        planets = self._planets.copy()

        planets.query('transit_i < end_observation', inplace=True)
        planets.query('transit_f > start_observation', inplace=True)

        planets['transit_i_obs'] = _choose(planets.copy(), 'transit_i', 'start_observation', return_max=True)
        planets['transit_f_obs'] = _choose(planets.copy(), 'transit_f', 'end_observation', return_max=False)
        planets['transit_observable_ratio'] = (planets['transit_f_obs'] -
                                               planets['transit_i_obs']) / (planets['pl_trandur'] / 24)

        self._planets = planets.query('transit_observable_ratio > @self._constraints["transit_percent_min"]'
                                      ).copy()

    ############################
    #
    # PLOTTING

    def plot(self, date,
             precision: int = 150,
             extend: bool = True,
             altitude_separation: float = 60,
             ax: Union[matplotlib.axes.Axes, matplotlib.figure.Figure, int] = None,
             mark_ra: Optional[TwoTuple] = None,
             colorbar: bool = False,    # todo: enable this once rank is working
             ):
        """
        Plots the altitudes of the encountered exoplanet's stars for the given date with information about the
        observation

        Parameters
        ----------
            date: str
                Date for which to plot
            colorbar
            mark_ra: (float, float)
                RA region to mark in output plot
            ax: matplotlib.Axes, matplotlib.Figure, int
                axes where plot will be produced
            precision : int
                the number of altitudes points to plot between the start and the end of the stars' observation
            extend : bool , optional
                if True is given , then the plot will have only the transit's interval of the observation. If not,
                then the plot will have the complete observations.
            altitude_separation : float
                separation in degrees between vertical altitude curves

        Returns
        -------
        object
        """
        self.set_date(date)
        if not self._date:
            raise ValueError("No date specified for calculation")
        self.apply_filters_needed()
        filtered_planets = self._planets.sort_values('transit_i', axis=0)

        f, ax = pa.figaxes(ax, figsize=(10, 15))
        cum_altitude = 0  # cumulative offset
        cmap = mpl.cm.get_cmap(name='OrRd')
        grade_norm = mpl.colors.Normalize(vmin=0, vmax=10)
        scalar_mappable = mpl.cm.ScalarMappable(norm=grade_norm, cmap=cmap)

        for index, info in filtered_planets.iterrows():
            transit_time = apt.Time(info['starrise'], format='jd', scale='utc') \
                           + np.linspace(0, (info['starset'] - info['starrise']) * 24,
                                         precision) * u.hour
            local_frame = apc.AltAz(obstime=transit_time, location=self._observatory)

            altitudes = (info['star_coords'].transform_to(local_frame).alt.degree
                         - self._constraints['altitude_min'] + cum_altitude)
            jd = transit_time.jd

            transit_i_index = np.argmin(np.abs(jd - info['transit_i']))
            transit_f_index = np.argmin(np.abs(jd - info['transit_f']))
            baseline_i_index = np.argmin(np.abs(jd - (info['transit_i'] - self._constraints['baseline_max']/24)))
            baseline_f_index = np.argmin(np.abs(jd - (info['transit_f'] + self._constraints['baseline_max']/24)))

            if extend:
                ax.fill_between(jd, altitudes, cum_altitude, color='lightblue', alpha=0.5)

            ax.fill_between(jd[transit_i_index:transit_f_index], altitudes[transit_i_index:transit_f_index],
                            cum_altitude, color=cmap(info['rank']/10), alpha=0.8)
            if baseline_i_index < transit_i_index:
                ax.fill_between(jd[baseline_i_index:transit_i_index], altitudes[baseline_i_index:transit_i_index],
                                cum_altitude, color='blue', alpha=0.8)
            if baseline_f_index > transit_f_index:
                ax.fill_between(jd[transit_f_index:baseline_f_index], altitudes[transit_f_index:baseline_f_index],
                                cum_altitude, color='blue', alpha=0.8)

            if info['transit_observable_ratio'] < 0.99999:
                middle_index = (transit_i_index+transit_f_index)//2
                ax.text(jd[middle_index], cum_altitude,
                        f"{100 * info['transit_observable_ratio']:.0f}%",
                        fontsize=9, color='goldenrod', ha="center")
            ax.text(jd[baseline_f_index], cum_altitude,
                    s=f"{transit_time[baseline_f_index].iso[11:16]} - P{info['pl_orbper']:.1f}d",
                    fontsize=9)
            ax.text(jd[baseline_f_index], cum_altitude + altitude_separation//2,
                    s=f"{info['sy_vmag']}, {info['moon_separation']:.0f}$^\circ$",
                    fontsize=9)
            ax.text(jd[baseline_i_index], cum_altitude,
                    s=transit_time[baseline_i_index].iso[11:16],
                    fontsize=9, ha='right')
            ax.text(jd[baseline_i_index], cum_altitude + altitude_separation//2,
                    s=f"$\\delta$ {info['dec']:.1f}",
                    fontsize=9, ha='right')

            cum_altitude += altitude_separation

        if colorbar:
            cax = f.add_axes([0.18, 0.55, 0.06, 0.3])
            f.colorbar(mappable=scalar_mappable, ticks=[0, 5, 10],
                       cax=cax, orientation='vertical', shrink=0.5,
                       )
        ax.set_xlabel(f'Planetary transits at "{self._observatory_name}" for the night after {self._date}', fontsize=13)
        delta_night = self._end_night.jd - self._start_night.jd
        ticks = [self._start_night.jd - 0.15 * delta_night, self._start_night.jd, self._end_night.jd,
                 self._end_night.jd + 0.2 * delta_night]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["", str(self._start_night.value), str(self._end_night.value), ""])
        ax.set_yticks(altitude_separation*np.arange(len(self._planets)))
        ax.set_yticklabels(filtered_planets['pl_name'])
        ax.set_xlim([self._start_night.jd - 0.15 * delta_night,
                     self._end_night.jd + 0.2 * delta_night])
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=5, va="bottom")
        ax.grid(visible=True, axis="y")
        ax2 = ax.twiny()
        lims = apt.Time(ax.get_xlim(), format='jd').sidereal_time("apparent", self._observatory).value
        if lims[1] < lims[0]:
            lims[1] += 24
        ax2.set_xlim(lims)
        ax2.set_xlabel("Local Apparent Sidereal Time")

        if mark_ra is not None:
            ax2.axvspan(mark_ra[0], mark_ra[1], alpha=0.2, color='gray')

    def __getitem__(self, item):
        if item not in self._planets['pl_name']:
            raise ValueError(f"Name '{item} not found in remaining dataset")
        return self._planets.loc[self._planets['pl_name'] == item]


if __name__ == '__main__':
    a = Nightly("lasilla")
    a.plot("2023-11-21",
           mark_ra=(8, 15),  # Highlighting a RA range can help identify TESS targets, among others
           )
