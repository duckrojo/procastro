import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pyvo as vo
import math
from astropy.time import Time
from astropy import coordinates as coord, units as u
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates import get_sun
from astropy.coordinates import get_moon
from matplotlib import cm
from itertools import chain
import time
import warnings
from pandas.plotting import table

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas

__all__ = ['AvailableAt']


def to_hms(jd):
    return Time(jd, format='jd').to_value('iso', 'date_hms')[11:20]


def two_slopes(values, grades, ref_values):
    is_top = (values > ref_values[1]) * 1
    grades = np.array(grades)
    ref_values = np.array(ref_values)

    ref_grade = grades[is_top]
    ref_percent = ref_values[is_top]
    slope = (grades[1 + is_top] - grades[is_top]) / (ref_values[1 + is_top] - ref_values[is_top])

    grade = (values - ref_percent) * slope + ref_grade
    grade *= (grade > 0)
    grade = (grade < grades[2]) * (grade - grades[2]) + grades[2]

    return grade


def query_full_exoplanet_db():
    exo_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    resultset = exo_service.search(
        f"SELECT pl_name,ra,dec,pl_orbper,pl_tranmid,disc_facility,pl_trandur,sy_pmra,sy_pmdec,sy_vmag,sy_gmag "
        f"FROM exo_tap.pscomppars "
        f"WHERE pl_tranmid!=0.0 and sy_pmra is not null and sy_pmdec is not null and pl_orbper!=0.0 ")
    planets_df = resultset.to_table().to_pandas()
    return planets_df


class AvailableAt:
    """
    A class that contains information about the available exoplanets at a given observatory and night.
    ...

    Attributes
    ----------
    dataframe : object
        the dataset with all the confirmed exoplanets and its attributes
    observatory : object
        the location's coordinates of the observatory
    utc_offset : float
        the utc offset of the observatory's location
    min_transit_percent : float
        the minimum transit percentage value for the observation
    min_transit_percent_ix : float
        the extreme minimum transit percentage acceptble for the observation
    night_angle : float
        the local sun altitude that defines the start and end of the night
    min_obs_ex : float
        the star's extreme minimum altitude in degrees that is acceptable for the observation. Must be equal or lower
        than min_obs.
    min_obs : float
        the star's minimum altitude in degrees required for the observation
    max_obs : float
        the star's ideal altitude in degrees required for the observation
    min_baseline_ex : float
        the baseline's extreme minimum value in days acceptable for the transit observation
    min_baseline : float
        the baseline's minimum value in days, required for the transit observation
    max_baseline : float
        the baseline's ideal value in days , required for the transit observation
    moon_separation_min_ex : float
        the extreme minimum separation angle between the moon and the star acceptable for the observation
    moon_separation_min : float
        the minimum separation angle between the moon and the star required for the observation
    moon_separation_max : float
        the ideal separation angle between the moon and the star required for the observation
    vmag_min_ex : float
        the extreme minimum v magnitude value of the stars acceptable for the observation

    Methods
    -------
    run_day(date):
        determines the avalaible exoplanets on the observatory location at a given date.
    update(args):
        determines the avalaible exoplanets on the observatory location at the given date but
        with a change in the initial args.
    plot(precision,extention=False):
        plots the altitudes of the avalaible exoplanets on the observatory location at the given date.
    """

    def __init__(self, observatory, min_transit_percent=0.9, night_angle=-18, min_obs=30,
                 min_obs_ex=23.5,
                 max_obs=40.0, min_baseline_ex=0.01, max_baseline=0.04, min_baseline=0.02, moon_separation_min_ex=10,
                 moon_separation_min=20, moon_separation_max=50, vmag_min_ex=12.5):
        """
        Constructs all the neccesary attributes for the object

        Parameters
        ----------
            observatory : str
                name of the observatory
            min_transit_percent : float
                the minimum transit percentage value for the observation
            night_angle : float
                the local sun altitude that defines the start and end of the night
            min_obs : float
                the star's minimum altitude in degrees required for the observation
            min_obs_ex : float
                the star's extreme minimum altitude in degrees that is acceptable for the observation. Must be equal or
                lower than min_obs.
            max_obs : float
                the star's ideal altitude in degrees required for the observation
            min_baseline_ex : float
                the baseline's extreme minimum value in days acceptable for the transit observation
            max_baseline : float
                the baseline's ideal value in days , required for the transit observation
            min_baseline : float
                the baseline's minimum value in days, required for the transit observation
            moon_separation_min_ex : float
                the extreme minimum separation angle in degrees between the moon and the star acceptable for the
                observation
            moon_separation_min : float
                the minimum separation angle between the moon and the star required for the observation
            moon_separation_max : float
                the ideal separation angle between the moon and the star required for the observation
            vmag_min_ex : float
                the extreme minimum v magnitude value of the stars acceptable for the observation


        """
        self.night_angle = night_angle
        print('empieza el dataframe')
        self.dataframe = query_full_exoplanet_db()
        print('termina el dataframe')
        self.observatory = coord.EarthLocation.of_site(observatory)
        self.utc_offset = (int(self.observatory.lon.degree / 15)) * u.hour
        # dejar todo en self.date_offset
        self.min_transit_percent = min_transit_percent
        self.min_transit_percent_ix = self.min_transit_percent / 2
        self.max_obs = max_obs
        self.min_obs = min_obs  # cambiar nombre a min_star_airmass
        self.min_obs_ix = min_obs_ex  # cambiar nombre a min_star_airmass_ix
        self.min_baseline_ix = min_baseline_ex
        self.min_baseline = min_baseline
        self.max_baseline = max_baseline
        self.moon_separation_min_ix = moon_separation_min_ex
        self.moon_separation_min = moon_separation_min
        self.moon_separation_max = moon_separation_max
        self.vmag_min_ix = vmag_min_ex

        self.selected_planets = None

    def run_day(self, date):
        """
        Determines the avalaible exoplanets at a certain date at the given observatory

        Parameters
        ----------
        date : str
            the observation's date
        Returns
        -------
        None
        """
        self.date = Time(date) + 12 * u.hour
        self.date_offset = self.date - self.utc_offset
        self.start_night = self.delta_midnight_times()[0]
        self.end_night = self.delta_midnight_times()[1]
        self.delta_midnight_ = self.delta_midnight()
        start = time.time()
        self.pre_ephemerides()
        self.ephemerides()
        self.post_ephemerides()
        end = time.time()
        print(end - start)

        return self

    def pre_ephemerides(self):
        selected_planets = self.dataframe.copy()
        selected_planets = self.closest_transit_filter(selected_planets, self.date_offset)
        selected_planets = self.vmag_filter(selected_planets)
        self.selected_planets = self.transit_percent_filter(selected_planets)
        self.pre_ephemerides_planets_to_update = self.selected_planets.copy()

    def ephemerides(self):
        self.stars_altitudes_ = self.stars_altitudes()
        self.start_and_end_observation_assignment()

    def post_ephemerides(self):
        self.selected_planets = self.transit_observation_percent()
        self.baseline_filter_ = self.baseline_filter()
        self.post_ephemerides_planets_to_update = self.baseline_filter_.copy()

    def set_default_parameteres(self, min_transit_percent, night_angle, min_obs, min_obs_ix, max_obs, min_baseline_ix,
                                max_baseline, min_baseline, moon_separation_min_ix, moon_separation_min,
                                moon_separation_max):
        args_index = ['min_transit_percent', 'night_angle', 'min_obs', 'min_obs_ix', 'max_obs', 'min_baseline_ix',
                      'max_baseline', 'min_baseline', 'moon_separation_min_ix', 'moon_separation_min',
                      'moon_separation_max']
        args = pd.Series((min_transit_percent, night_angle, min_obs,
                          min_obs_ix, max_obs, min_baseline_ix, max_baseline, min_baseline, moon_separation_min_ix,
                          moon_separation_min, moon_separation_max), index=args_index)

        default_parameters = list((
            self.min_transit_percent, self.night_angle, self.min_obs, self.min_obs_ix, self.max_obs,
            self.min_baseline_ix, self.max_baseline, self.min_baseline, self.moon_separation_min_ix,
            self.moon_separation_min, self.moon_separation_max))
        args = args.fillna(0)
        args = pd.Series(np.where(args == 0.0, default_parameters, args), index=args_index)

        return args

    def set_parameters(self, args):
        self.night_angle = args[
            'night_angle']  # esta no es necesaria tenerla ya solo se utiliza para determinar el self.date_offset
        self.min_transit_percent = args['min_transit_percent']
        self.min_transit_percent_ix = self.min_transit_percent / 2
        self.max_obs = args['max_obs']
        self.min_obs = args['min_obs']  # cambiar nombre a min_star_airmass
        self.min_obs_ix = args['min_obs_ix']  # cambiar nombre a min_star_airmass_ix
        self.min_baseline_ix = args['min_baseline_ix']
        self.min_baseline = args['min_baseline']
        self.max_baseline = args['max_baseline']
        self.moon_separation_min_ix = args['moon_separation_min_ix']
        self.moon_separation_min = args['moon_separation_min']
        self.moon_separation_max = args['moon_separation_max']
        self.start_night = self.delta_midnight_times()[0]
        self.end_night = self.delta_midnight_times()[1]
        self.delta_midnight_ = self.delta_midnight()

    def update(self, min_transit_percent=None, night_angle=None, min_obs=None, min_obs_ix=None, max_obs=None,
               min_baseline_ix=None, max_baseline=None, min_baseline=None, moon_separation_min_ix=None,
               moon_separation_min=None, moon_separation_max=None):
        """
        Updates the attributes of the constructor and determines the avalaible exoplanets with the new attributes

        Parameters
        ----------
        Same paramateres of the constructor

        Returns
        -------
        None

        """
        args = self.set_default_parameteres(min_transit_percent, night_angle, min_obs, min_obs_ix, max_obs,
                                            min_baseline_ix, max_baseline, min_baseline, moon_separation_min_ix,
                                            moon_separation_min, moon_separation_max)
        if args['min_transit_percent'] < self.min_transit_percent and args['night_angle'] == self.night_angle and args[
            'min_obs_ix'] == self.min_obs_ix:
            self.set_parameters(args)
            prev_transit_percent_filter = self.pre_ephemerides_planets_to_update
            prev_filtered_dataframe = self.post_ephemerides_planets_to_update
            self.pre_ephemerides()
            self.selected_planets = self.selected_planets.drop(prev_transit_percent_filter.index)
            self.ephemerides()
            self.post_ephemerides()
            self.baseline_filter_ = pd.concat([prev_filtered_dataframe, self.baseline_filter_])
            self.baseline_filter_ = self.baseline_filter_.sort_index()
        elif args['min_transit_percent'] > self.min_transit_percent and args['night_angle'] == self.night_angle:
            self.set_parameters(args)
            self.selected_planets = self.pre_ephemerides_planets_to_update[
                self.pre_ephemerides_planets_to_update['transit_percent'] > self.min_transit_percent_ix]
            self.post_ephemerides()
        elif args[
            'night_angle'] > self.night_angle or self.min_obs_ix != args['min_obs_ix']:
            self.set_parameters(args)
            self.pre_ephemerides()
            self.ephemerides()
            self.post_ephemerides()
        else:
            self.set_parameters(args)
            self.baseline_filter_ = self.baseline_filter_[
                self.start_night.jd < self.baseline_filter_['end_observation']]
            self.baseline_filter_ = self.baseline_filter_[
                self.end_night.jd > self.baseline_filter_['start_observation']]
            self.baseline_filter_['start_observation'] = self.baseline_filter_['start_observation'].apply(
                lambda x: x
                if x > self.start_night.jd else self.start_night.jd)
            self.baseline_filter_['end_observation'] = self.baseline_filter_['end_observation'].apply(
                lambda x: x
                if x < self.end_night.jd else self.end_night.jd)
            self.selected_planets = self.baseline_filter_
            self.post_ephemerides()
        return

    def sun_airmass(self, precision=1440):
        midnight = self.date_offset
        delta_midnight_loc = np.linspace(0, 24, precision) * u.hour
        times_july12_to_13 = midnight + delta_midnight_loc
        frame_july12_to_13 = AltAz(obstime=times_july12_to_13, location=self.observatory)
        sunaltazs_july12_to_13 = get_sun(times_july12_to_13).transform_to(frame_july12_to_13)
        return sunaltazs_july12_to_13

    def closest_transit_filter(self, planets_df, date):
        """
        Computes the exoplanet's closest transit to the observation's date and filters all the exoplanets which doesnt
        have a closest transit to the observation's date.

        Parameters:
            planets_df(pd.DataFrame): The dataframe that contains the information of the exoplanets and that will be
            filtered.
            date(Time): the date offset of the observation date in UTC
        Returns:
            filtered_df(pd.DataFrame): The exoplanets' dataframe filtered.
        """
        closest_transit_n = (((date.jd - planets_df['pl_tranmid']) / planets_df['pl_orbper'])
                             + 1.0).astype('int')
        closest_transit_jd = planets_df['pl_tranmid'] + closest_transit_n * planets_df['pl_orbper']
        delta_transit_jd = closest_transit_jd - date.jd
        planets_df['closest_transit'] = closest_transit_jd
        return planets_df[0 <= (delta_transit_jd < 1.0)]

    def vmag_filter(self, planets_df):
        no_filter_data = planets_df['sy_vmag'] == 0
        planets_df.loc[:, 'sy_vmag'].where(~no_filter_data, other=planets_df['sy_gmag'][no_filter_data], inplace=True)
        return planets_df[planets_df['sy_vmag'] > self.vmag_min_ix]

    def star_and_end_night(self, precision):
        sun_alt = np.array(self.sun_airmass(precision).alt.degree)
        index = np.array([])
        for altitude in range(0, len(sun_alt)):
            if sun_alt[altitude] < self.night_angle:
                index = np.append(index, altitude)
            else:
                continue
        ndarray = np.array([index[0], index[len(index) - 1]])
        return ndarray

    def delta_midnight_times(self, precision=1440):  # con este se determinar el self.end_night y el self.start_night
        night_index = self.star_and_end_night(precision)
        delta_midnight_loc = np.linspace(0, 24, precision) * u.hour
        delta_date = self.date_offset + delta_midnight_loc
        start_night = delta_date[int(night_index[0])]
        end_night = delta_date[int(night_index[1])]
        night_limits = np.array([start_night, end_night])
        return night_limits

    def delta_midnight(self, precision=750):  # aca habia un 250
        delta_date = self.end_night - self.start_night
        delta_date = delta_date.to_value('hour')
        delta_midnight = np.linspace(0, delta_date, precision) * u.hour
        delta_midnight = self.start_night + delta_midnight
        return delta_midnight

    def transit_percent_filter(self, planets_df, default_duration=2.5):
        """
        Computes if the minimum extreme transit percent occurs on the observation's night and discards all the exoplanets
        which doesn't have that transit minimum.

        Parameters:
            planets_df(pd.DataFrame): The dataframe that contains the information of the exoplanets and that will be
            filtered.
            default_duration(float): the default transit duration for those exoplanets which doesnt have the information
            about its transit duration
        Returns:
            filtered_df(pd.DataFrame): The exoplanets' dataframe filtered.
        """
        planets_df['pl_trandur'][planets_df['pl_trandur'] == 0.0] = default_duration
        planets_df['transit_i'] = planets_df['closest_transit'] - (planets_df['pl_trandur'] / 48)
        planets_df['transit_f'] = planets_df['closest_transit'] + (planets_df['pl_trandur'] / 48)

        planets_df = planets_df[planets_df['transit_i'] < self.end_night.jd]
        planets_df = planets_df[planets_df['transit_f'] > self.start_night.jd]

        transit_i = planets_df['transit_i']
        transit_i[self.start_night.jd > transit_i] = self.start_night.jd
        transit_f = planets_df['transit_f']
        transit_f[self.end_night.jd < transit_f] = self.end_night.jd

        planets_df['transit_percent'] = (transit_f - transit_i) / (planets_df['pl_trandur'] / 24)
        return planets_df[planets_df.transit_percent > self.min_transit_percent_ix]

    def stars_coordinates(self, planets_df_input):
        right_ascention = Angle(planets_df_input['sy_pmra'], u.degree)
        right_ascention = right_ascention.degree
        right_ascention = right_ascention * u.mas / u.yr
        declination = Angle(planets_df_input['sy_pmdec'], u.degree)
        declination = declination.degree
        declination = declination * u.mas / u.yr
        obstime_in = Time(planets_df_input['pl_tranmid'], format='jd', scale='utc')
        stars_coords_i = SkyCoord(planets_df_input['ra'] * u.degree, planets_df_input['dec'] * u.degree,
                                  distance=1 * u.pc,
                                  frame='icrs', pm_ra_cosdec=right_ascention, pm_dec=declination, obstime=obstime_in)
        obstime_fi = Time(planets_df_input['closest_transit'], format='jd', scale='utc')
        stars_coords_f = stars_coords_i.apply_space_motion(
            new_obstime=obstime_fi)  # esta es la parte que quizas se podría omitir
        if len(planets_df_input.columns) == len(self.selected_planets.columns):
            self.selected_planets['star_coords'] = stars_coords_f
        return stars_coords_f

    def transit_observation_percent(self):
        """
        Computes if the minimum extreme transit percent occurs between the start and end of the observation and discards
        all the exoplanets which doesn't have that transit minimum.

        Parameters:
            self()
        Returns:
            filtered_df(pd.DataFrame): The exoplanets' dataframe filtered.

        """
        planets_df = self.selected_planets
        planets_df = planets_df[planets_df['transit_i'] < planets_df['end_observation']]
        planets_df = planets_df[planets_df['transit_f'] > planets_df['start_observation']]
        planets_df['transit_i_obs'] = planets_df[['transit_i', 'start_observation']].max(axis=1)
        planets_df['transit_f_obs'] = planets_df[['transit_f', 'end_observation']].min(axis=1)
        planets_df['transit_observation_percent'] = (planets_df['transit_f_obs']
                                                     - planets_df['transit_i_obs']) / (planets_df['pl_trandur'] / 24)

        return planets_df[planets_df['transit_observation_percent'] > self.min_transit_percent_ix]

    def baseline_filter(self):
        """
        Computes the baseline of the exoplanets and filters by the duration of the baseline.

        Parameters:
            self()
        Return:
            None
        """
        planets_df = self.selected_planets
        planets_df['delta_i_baseline'] = planets_df['transit_i'] - planets_df['start_observation']
        planets_df['delta_f_baseline'] = planets_df['end_observation'] - planets_df['transit_f']
        planets_df.loc[:, 'delta_i_baseline'].where(planets_df['delta_i_baseline'] > self.min_baseline_ix,
                                                    other=0.0, inplace=True)
        planets_df.loc[:, 'delta_f_baseline'].where(planets_df['delta_f_baseline'] > self.min_baseline_ix,
                                                    other=0.0, inplace=True)
        planets_df = planets_df.query('delta_i_baseline!=0.0 or delta_f_baseline!=0.0')
        return planets_df

    def transit_percent_rank(self):

        planets_df_input = self.baseline_filter_

        transit_observation_percent = np.array(planets_df_input['transit_observation_percent'])

        min_transit_percent = self.min_transit_percent
        min_transit_percent_ix = self.min_transit_percent_ix

        planets_df_input.loc[:, 'transit_percent_grade'] = two_slopes(transit_observation_percent,
                                                                      [0, 5, 10],
                                                                      [min_transit_percent_ix, min_transit_percent, 1])

        return planets_df_input

    def baseline_percent_rank(self):
        planets_df_input = self.baseline_filter_
        min_baseline_ix = self.min_baseline_ix
        delta_baseline = self.min_baseline - self.min_baseline_ix
        planets_df_input['delta_i_baseline_grade'] = np.where(planets_df_input['delta_i_baseline'] >= min_baseline_ix,
                                                              planets_df_input['delta_i_baseline'] - min_baseline_ix, 0)
        planets_df_input['delta_i_baseline_grade'] = np.where(planets_df_input['delta_i_baseline_grade'] != 0,
                                                              5 * planets_df_input['delta_i_baseline_grade'] /
                                                              delta_baseline, 0)
        planets_df_input['delta_i_baseline_grade'] = np.where(planets_df_input['delta_i_baseline_grade'] > 5.0, 5.0,
                                                              planets_df_input['delta_i_baseline_grade'])
        planets_df_input['delta_f_baseline_grade'] = np.where(planets_df_input['delta_f_baseline'] >= min_baseline_ix,
                                                              planets_df_input['delta_f_baseline'] - min_baseline_ix, 0)
        planets_df_input['delta_f_baseline_grade'] = np.where(planets_df_input['delta_f_baseline_grade'] != 0,
                                                              5 * planets_df_input['delta_f_baseline_grade'] /
                                                              delta_baseline, 0)
        planets_df_input['delta_f_baseline_grade'] = np.where(planets_df_input['delta_f_baseline_grade'] > 5.0, 5.0,
                                                              planets_df_input['delta_f_baseline_grade'])
        planets_df_input['baseline_percent_grade'] = planets_df_input['delta_f_baseline_grade'] + planets_df_input[
            'delta_i_baseline_grade']
        self.baseline_filter_ = self.baseline_filter_.drop('delta_f_baseline_grade', axis=1)
        self.baseline_filter_ = self.baseline_filter_.drop('delta_i_baseline_grade', axis=1)
        return planets_df_input

    def altitude_grades(self):
        min_obs_ix = self.min_obs_ix
        min_obs = self.min_obs
        max_obs = self.max_obs

        # transit_altitude = np.array(planets_df_input['closest_transit_altitudes'])
        #
        # planets_df_input.loc[:, 'altitude_grade'] = two_slopes(transit_observation_percent,
        #                                                               [0, 5, 10],
        #                                                               [min_transit_percent_ix, min_transit_percent, 1])

        delta_alt_1 = min_obs - min_obs_ix
        delta_alt_2 = max_obs - min_obs
        closest_transit_altitude_ndarray = np.array([])
        planets_df_input = self.baseline_filter_.copy()
        for i in planets_df_input.index:
            closest_transit = Time(planets_df_input.loc[i, :]['closest_transit'], format='jd', scale='utc')
            local_frame = AltAz(obstime=closest_transit, location=self.observatory)
            from_icrs_to_alt_az = self.selected_planets.loc[i, :]['star_coords'].transform_to(local_frame).alt.degree
            closest_transit_altitude_ndarray = np.append(closest_transit_altitude_ndarray, from_icrs_to_alt_az)
        planets_df_input.loc[:, 'closest_transit_altitudes'] = closest_transit_altitude_ndarray
        planets_df_input.loc[:, 'altitude_grade'] = np.where(planets_df_input['closest_transit_altitudes']
                                                             <= min_obs_ix,
                                                             0, planets_df_input['closest_transit_altitudes'])
        planets_df_input.loc[:, 'altitude_grade'] = np.where((planets_df_input['altitude_grade'] > min_obs_ix) &
                                                             (planets_df_input['altitude_grade'] <= min_obs),
                                                             5 * (planets_df_input[
                                                                      'altitude_grade'] - min_obs_ix) / delta_alt_1,
                                                             planets_df_input['altitude_grade'])
        planets_df_input.loc[:, 'altitude_grade'] = np.where(planets_df_input['altitude_grade'] >= min_obs,
                                                             (5 * (planets_df_input[
                                                                       'altitude_grade'] - min_obs) / delta_alt_2) + 5,
                                                             planets_df_input['altitude_grade'])
        planets_df_input.loc[:, 'altitude_grade'] = np.where(planets_df_input['altitude_grade'] > 10, 10,
                                                             planets_df_input['altitude_grade'])
        self.baseline_filter_ = self.baseline_filter_.assign(
            altitude_grades=planets_df_input.loc[:, 'altitude_grade'].values)
        return planets_df_input

    def moon_separation_rank(self):
        planets_df_input = self.baseline_filter_
        min_separation_ix = self.moon_separation_min_ix
        min_separation = self.moon_separation_min
        max_separation = self.moon_separation_max
        delta_separation_1 = min_separation - min_separation_ix
        delta_separation_2 = max_separation - min_separation
        obs_time = Time(planets_df_input['closest_transit'], format='jd', scale='utc')
        moon_coord = get_moon(obs_time, self.observatory)
        stars_coords = self.stars_coordinates(planets_df_input)
        moon_separation = np.array(moon_coord.separation(stars_coords))
        planets_df_input['moon_separation'] = moon_separation
        planets_df_input['moon_separation_grade'] = np.where(planets_df_input['moon_separation'] < min_separation_ix, 0,
                                                             planets_df_input['moon_separation'])
        planets_df_input['moon_separation_grade'] = np.where(
            (planets_df_input['moon_separation_grade'] <= min_separation) &
            (planets_df_input[
                 'moon_separation_grade'] > min_separation_ix),

            5 * (planets_df_input[
                     'moon_separation_grade'] - min_separation_ix) /
            delta_separation_1,
            planets_df_input['moon_separation_grade'])
        planets_df_input['moon_separation_grade'] = np.where(planets_df_input['moon_separation_grade'] > min_separation,
                                                             (5 * (planets_df_input[
                                                                       'moon_separation_grade'] - min_separation) /
                                                              delta_separation_2) + 5,
                                                             planets_df_input['moon_separation_grade'])
        planets_df_input['moon_separation_grade'] = np.where(planets_df_input['moon_separation_grade'] > 10, 10,
                                                             planets_df_input['moon_separation_grade'])
        return planets_df_input

    def star_max_altitude_utc_time(self, precision=750):  # aca habia un 250
        local_midnight_times = Time(self.delta_midnight(precision), scale='utc', location=self.observatory)
        local_sidereal_times = local_midnight_times.sidereal_time('mean').degree
        self.selected_planets['max_altitude_index'] = self.selected_planets.apply(lambda x:
                                                                                  np.argmin(np.abs(local_sidereal_times
                                                                                                   - x.ra)),
                                                                                  axis=1)
        self.selected_planets['max_altitude_time'] = local_midnight_times[self.selected_planets['max_altitude_index']]

        return

    def altitudes(self, x, y):
        ephemerides_times = Time(
            (self.start_night.value, x.value,
             self.end_night.value), format='iso',
            scale='utc')
        local_frame = AltAz(obstime=ephemerides_times, location=self.observatory)
        from_icrs_to_alt_az = np.array(
            y.transform_to(local_frame).alt.degree)
        return from_icrs_to_alt_az

    def stars_altitudes(self):
        """
        Computes the altitudes of the exoplanets' host stars at the start of the night, end of the night and at the time
        of maximum altitude

        Parameters:
            self()
        Returns:
            stars_altitudes(pd.DataFrame): A dataframe which contains the computed 3 altitudes for each exoplanet's host
            star
        """
        self.stars_coordinates(self.selected_planets)
        start = time.time()
        self.star_max_altitude_utc_time()
        times = self.selected_planets['max_altitude_time'].values
        end = time.time()
        start = time.time()
        ra_array = self.selected_planets['star_coords'].apply(lambda x: x.ra.value).values
        dec_array = self.selected_planets['star_coords'].apply(lambda x: x.dec.value).values
        star_coords = SkyCoord(ra_array * u.degree, dec_array * u.degree, frame='icrs', obstime=times)
        local_frame = AltAz(location=self.observatory)
        start_night_altaz_local_frame = AltAz(obstime=self.start_night, location=self.observatory)
        end_night_altaz_local_frame = AltAz(obstime=self.end_night, location=self.observatory)
        max_altitudes = star_coords.transform_to(local_frame).alt.degree
        start_night_altitudes = star_coords.transform_to(start_night_altaz_local_frame).alt.degree
        end_night_altitudes = star_coords.transform_to(end_night_altaz_local_frame).alt.degree
        altitudes_dict = dict(start_night_altitudes=start_night_altitudes, max_altitudes=max_altitudes,
                              end_night_altitudes=end_night_altitudes)
        altitudes = pd.DataFrame(altitudes_dict)
        altitudes['start_night_altitudes'] = altitudes['start_night_altitudes'].apply(lambda x: np.array([x]))
        altitudes['concatenated_altitudes'] = altitudes.apply(
            lambda x: np.append(x.start_night_altitudes, [x.max_altitudes, x.end_night_altitudes]), axis=1)
        self.selected_planets['altitudes'] = altitudes['concatenated_altitudes'].values
        end = time.time()
        print('termina self.altitudes, tiempo total: ' + str(end - start))
        stars_altitudes = pd.DataFrame(self.selected_planets['altitudes'].explode())
        index_array = '0 1 2 ' * len(stars_altitudes)
        index_array = np.array(index_array.split()).astype('int')[:len(stars_altitudes)]
        stars_altitudes['index'] = index_array
        stars_altitudes = pd.pivot_table(stars_altitudes, columns=stars_altitudes.index, values='altitudes',
                                         index='index')
        return stars_altitudes

    def linear_aproximation_to_min_obs_ix(self, star_coords, iterative_delta_midnight, first_altitude, mid_altitudes,
                                          end_altitude,
                                          obs_type):
        d_t = int(len(iterative_delta_midnight) / 90)
        from_icrs_to_alt_az = np.append(first_altitude,
                                        np.append(mid_altitudes,
                                                  end_altitude)) - self.min_obs_ix
        closest_observation_time_index = np.argmin(np.abs(from_icrs_to_alt_az)) * d_t
        closest_observation_altitude = from_icrs_to_alt_az[int(closest_observation_time_index / d_t)]
        closest_observation_altitude_time = iterative_delta_midnight[
            closest_observation_time_index - int(closest_observation_time_index / len(iterative_delta_midnight))]
        if obs_type == 'growing':
            if closest_observation_altitude < 0:
                closest_observation_altitudes_and_times = dict(
                    altitudes=[closest_observation_altitude,
                               from_icrs_to_alt_az[int((closest_observation_time_index + d_t) / d_t)]],
                    times=[closest_observation_altitude_time,
                           iterative_delta_midnight[closest_observation_time_index + d_t
                                                    - int((closest_observation_time_index + d_t) /
                                                          len(iterative_delta_midnight))]])
            else:
                closest_observation_altitudes_and_times = dict(
                    altitudes=[from_icrs_to_alt_az[int((closest_observation_time_index - d_t) / d_t)],
                               closest_observation_altitude],
                    times=[iterative_delta_midnight[closest_observation_time_index - d_t],
                           closest_observation_altitude_time])
        else:
            if closest_observation_altitude < 0:
                closest_observation_altitudes_and_times = dict(
                    altitudes=[from_icrs_to_alt_az[int((closest_observation_time_index - d_t) / d_t)],
                               closest_observation_altitude],
                    times=[iterative_delta_midnight[closest_observation_time_index - d_t],
                           closest_observation_altitude_time])
            else:
                closest_observation_altitudes_and_times = dict(
                    altitudes=[closest_observation_altitude,
                               from_icrs_to_alt_az[int((closest_observation_time_index + d_t) / d_t)]],
                    times=[closest_observation_altitude_time,
                           iterative_delta_midnight[closest_observation_time_index + d_t - int(
                               (closest_observation_time_index + d_t) / len(iterative_delta_midnight))]])
        for i in range(0, 350):
            delta_altitude = float((closest_observation_altitudes_and_times['altitudes'][0] -
                                    closest_observation_altitudes_and_times['altitudes'][1]))
            delta_time = float((closest_observation_altitudes_and_times['times'][0] -
                                closest_observation_altitudes_and_times['times'][1]).to_value('hour'))
            slope = delta_altitude / delta_time
            min_obs_ix_delta = float(closest_observation_altitudes_and_times['altitudes'][0])
            min_obs_ix_time = (
                    (closest_observation_altitudes_and_times['times'][0].jd * 24) - min_obs_ix_delta / slope)

            min_obs_ix_time = Time(min_obs_ix_time / 24, format='jd', scale='utc')
            min_obs_ix_time = Time(min_obs_ix_time, format='iso')
            local_frame = AltAz(obstime=min_obs_ix_time, location=self.observatory)
            min_obs_ix_altitude = float(star_coords.transform_to(local_frame).alt.degree) - self.min_obs_ix
            if abs(min_obs_ix_altitude) > 0.001:
                closest_observation_altitudes_and_times['times'][1] = closest_observation_altitudes_and_times['times'][
                    0]
                closest_observation_altitudes_and_times['altitudes'][1] = \
                closest_observation_altitudes_and_times['altitudes'][0]
                closest_observation_altitudes_and_times['times'][0] = min_obs_ix_time
                closest_observation_altitudes_and_times['altitudes'][0] = min_obs_ix_altitude
            else:
                break
        return min_obs_ix_time.jd

    def iterative_delta_midnight(self, x):
        if x.iloc[0] > self.min_obs_ix and x[2] > self.min_obs_ix:  # no hay aproximacion lineal
            iterative_delta_midnight_times_1_linear_aproximation = True
        elif x.iloc[0] == x.iloc[1] and x.iloc[
            0] > self.min_obs_ix:  # hay aproximacion lineal con iterative_delta_midnight = self.delta_midnight_
            iterative_delta_midnight_times_1_linear_aproximation = self.delta_midnight_
        elif x.iloc[2] == x.iloc[1] and x.iloc[
            2] > self.min_obs_ix:  # hay aproximacion lineal con iterative_delta_midnight = self.delta_midnight_
            iterative_delta_midnight_times_1_linear_aproximation = self.delta_midnight_
        elif x.iloc[2] != x.iloc[1] \
                and x.iloc[0] != x.iloc[1] \
                and x.iloc[1] > self.min_obs_ix:
            if x.iloc[
                0] >= self.min_obs_ix:  # hay aproximacion lineal con iterative_delta_midnight = self.delta_midnight_[
                # self.selected_planets.loc[x.name, :]['max_altitude_index']:len(
                # self.delta_midnight_)]

                iterative_delta_midnight_times_1_linear_aproximation = self.delta_midnight_[
                                                                       self.selected_planets.loc[x.name, :][
                                                                           'max_altitude_index']:len(
                                                                           self.delta_midnight_)]

            elif x.iloc[
                2] >= self.min_obs_ix:  # hay aproximacion lineal con iternative delta_midnight = self.delta_midnight_[0:self.selected_planets.loc[x.name, :]
                # ['max_altitude_index']]

                iterative_delta_midnight_times_1_linear_aproximation = self.delta_midnight_[
                                                                       0:self.selected_planets.loc[x.name, :][
                                                                           'max_altitude_index']]
            else:  # hay 2 aproximaciones lineales , una desde el inicio hasta el maximo, otra del maximo hasta el final
                iterative_delta_midnight_times1 = \
                    self.delta_midnight_[0:self.selected_planets.loc[x.name, :]['max_altitude_index']]
                iterative_delta_midnight_times2 = self.delta_midnight_[self.selected_planets.loc[x.name, :] \
                                                                           ['max_altitude_index']:len(
                    self.delta_midnight_) - 1]

                x.loc[4] = iterative_delta_midnight_times1, iterative_delta_midnight_times2
                return x.loc[4]
        else:  # no hay aproximacion lineal
            iterative_delta_midnight_times_1_linear_aproximation = float('nan')

        x.loc[4] = iterative_delta_midnight_times_1_linear_aproximation
        return x.loc[4]

    def start_and_end_observation_times(self, x):
        if x.name==3073:
            a = 0
        start_observation = np.array([])
        end_observation = np.array([])
        if x.iloc[0] > self.min_obs_ix and x[2] > self.min_obs_ix:
            start_observation = np.append(start_observation, self.start_night.jd)
            end_observation = np.append(end_observation, self.end_night.jd)
        elif x.iloc[0] == x.iloc[1] and x.iloc[0] > self.min_obs_ix:
            start_observation = np.append(start_observation, self.start_night.jd)
            end_observation = np.append(end_observation, self.linear_aproximation_to_min_obs_ix(
                self.selected_planets.loc[x.name, :]['star_coords'], self.delta_midnight_,
                x.iloc[0], x.iloc[3], x.iloc[2], 'decreasing'))
        elif x.iloc[2] == x.iloc[1] and x.iloc[2] > self.min_obs_ix:
            end_observation = np.append(end_observation, self.end_night.jd)
            start_observation = np.append(start_observation, self.linear_aproximation_to_min_obs_ix(
                self.selected_planets.loc[x.name, :]['star_coords'], self.delta_midnight_,
                x.iloc[0], x.iloc[3],
                x.iloc[2], 'growing'))
        elif x.iloc[2] != x.iloc[1] \
                and x.iloc[0] != x.iloc[1] \
                and x.iloc[1] > self.min_obs_ix:
            if x.iloc[0] >= self.min_obs_ix:
                start_observation = np.append(start_observation, self.start_night.jd)
                end_observation = np.append(end_observation,
                                            self.linear_aproximation_to_min_obs_ix(
                                                self.selected_planets.loc[x.name, :]['star_coords'],
                                                self.delta_midnight_[
                                                self.selected_planets.loc[x.name, :]['max_altitude_index']:len(
                                                    self.delta_midnight_)],
                                                x.iloc[1], x.iloc[3],
                                                x.iloc[2], 'decreasing'))

            elif x.iloc[2] >= self.min_obs_ix:
                end_observation = np.append(end_observation, self.end_night.jd)
                start_observation = np.append(start_observation,
                                              self.linear_aproximation_to_min_obs_ix(
                                                  self.selected_planets.loc[x.name, :]['star_coords'],
                                                  self.delta_midnight_[0:self.selected_planets.loc[x.name, :]
                                                  ['max_altitude_index']], x.iloc[0], x.iloc[3],
                                                  x.iloc[1], 'growing'))
            else:
                start_observation = np.append(start_observation, self.linear_aproximation_to_min_obs_ix(
                    self.selected_planets.loc[x.name, :]['star_coords'],
                    self.delta_midnight_[0:self.selected_planets.loc[x.name, :]['max_altitude_index']],
                    x.iloc[0], x.iloc[3],
                    x.iloc[1], 'growing'))
                end_observation = np.append(end_observation, self.linear_aproximation_to_min_obs_ix(
                    self.selected_planets.loc[x.name, :]['star_coords'], self.delta_midnight_[
                                                                         self.selected_planets.loc[
                                                                         x.name, :][
                                                                             'max_altitude_index']:len(
                                                                             self.delta_midnight_) - 1],
                    x.iloc[1], x.iloc[4], x.iloc[2], 'decreasing'))
        else:
            start_observation = np.append(start_observation, False)
            end_observation = np.append(end_observation, False)

        x.loc[5] = start_observation[0]
        x.loc[6] = end_observation[0]
        return x.loc[5], x.loc[6]

    def start_and_end_observation_assignment(self):
        """
        Computes a given number of altitudes at equispaced times of the observation's night of the exoplanet's. Then it
        calls another function (start_and_end_observation_times) that computes the times at which the minimum extreme
        altitude is found. Then this function assign the start and end observation times to each planet of the dataframe.
        Parameters:
            self()
        Return:
            None
        """
        start = time.time()
        iterative_delta_midnight_times = pd.DataFrame(
            self.stars_altitudes_.apply(lambda x: self.iterative_delta_midnight(x), axis=0))
        iterative_delta_midnight_times.dropna(inplace=True)
        no_linear_aproximation_required_planets = iterative_delta_midnight_times[
            iterative_delta_midnight_times == True].dropna()
        no_linear_aproximation_required_planets_altitudes = self.stars_altitudes_.loc[:,
                                                            list(no_linear_aproximation_required_planets.index)]
        iterative_delta_midnight_times = iterative_delta_midnight_times[iterative_delta_midnight_times != True].dropna()
        self.stars_altitudes_ = self.stars_altitudes_.loc[:, list(iterative_delta_midnight_times.index)]
        self.selected_planets = self.selected_planets.loc[list(iterative_delta_midnight_times.index), :]
        times = pd.DataFrame(iterative_delta_midnight_times.loc[:, 0].values,
                             index=iterative_delta_midnight_times.index)
        times['0_values'] = times.loc[:, 0].apply(lambda x: len(x) if len(x) != 2 else 0)
        iterative_delta_midnight_times_1_linear_aproximation = times[times['0_values'] > 0].copy()
        iterative_delta_midnight_times_1_linear_aproximation.loc[:,
        'iterative_delta_midnight'] = iterative_delta_midnight_times_1_linear_aproximation.loc[
                                      :, 0].apply(lambda x: x[int(len(x) / 90):int(
            (89 / 90) * len(x)) + 1:int(len(x) / 90)] if int(len(x) / 90) > 0 else x)
        iterative_delta_midnight_times_2_linear_aproximation = times[times['0_values'] == 0]
        iterative_delta_midnight_times_2_linear_aproximation = iterative_delta_midnight_times_2_linear_aproximation.explode(
            0).reset_index()
        index = np.array(iterative_delta_midnight_times_2_linear_aproximation.index)
        start_night_to_max_altitude_times = \
        iterative_delta_midnight_times_2_linear_aproximation.drop(index[1:len(index):2])[0]
        end_night_to_max_altitude_times = iterative_delta_midnight_times_2_linear_aproximation.drop(
            index[0:len(index):2])[0]
        times_dict = dict(start_night_to_max_altitude_times=start_night_to_max_altitude_times.values,
                          end_night_to_max_altitude_times=end_night_to_max_altitude_times.values)
        iterative_delta_midnight_times_2_linear_aproximation = pd.DataFrame(times_dict,
                                                                            index=times[times['0_values'] == 0].index)
        iterative_delta_midnight_times_2_linear_aproximation[
            'start_night_to_max_altitude_times'] = iterative_delta_midnight_times_2_linear_aproximation.iloc[:,
                                                   0].apply(
            lambda x: x[int(len(x) / 90):int(
                (89 / 90) * len(x)) + 1:int(len(x) / 90)] if int(len(x) / 90) > 0 else x)
        iterative_delta_midnight_times_2_linear_aproximation[
            'end_night_to_max_altitude_times'] = iterative_delta_midnight_times_2_linear_aproximation.iloc[:, 1].apply(
            lambda x: x[int(len(x) / 90):int(
                (89 / 90) * len(x)) + 1:int(len(x) / 90)] if int(len(x) / 90) > 0 else x)
        number_of_night_points_1 = iterative_delta_midnight_times_1_linear_aproximation.apply(
            lambda x: len(x.iterative_delta_midnight), axis=1)
        number_of_night_points_2 = iterative_delta_midnight_times_2_linear_aproximation.apply(
            lambda x: (len(x.start_night_to_max_altitude_times), len(x.end_night_to_max_altitude_times)), axis=1)
        selected_planets_1_linear_aproximation = self.selected_planets.loc[
                                                 list(iterative_delta_midnight_times_1_linear_aproximation.index), :]
        selected_planets_2_linear_aproximation = self.selected_planets.loc[
                                                 list(iterative_delta_midnight_times_2_linear_aproximation.index), :]
        selected_planets_1_linear_aproximation['number_of_night_points'] = number_of_night_points_1
        selected_planets_2_linear_aproximation['number_of_night_points'] = number_of_night_points_2
        end = time.time()
        print('tiempo total a lo de antes de los repeated_ra y repeated_dec :' + str(end - start))
        start = time.time()
        # selected_planets_1_linear_aproximation['repeats_number'] = times['0_values']
        selected_planets_1_linear_aproximation['repeated_ra'] = selected_planets_1_linear_aproximation.apply(
            lambda x: np.array([x.ra]).repeat(x.number_of_night_points), axis=1)
        # selected_planets_1_linear_aproximation['repeats_number'] = times['0_values']
        selected_planets_1_linear_aproximation['repeated_dec'] = selected_planets_1_linear_aproximation.apply(
            lambda x: np.array([x.dec]).repeat(x.number_of_night_points), axis=1)
        # selected_planets_2_linear_aproximation['repeats_number'] = times['0_values']
        selected_planets_2_linear_aproximation['repeated_ra_1'] = selected_planets_2_linear_aproximation.apply(
            lambda x: np.array([x.ra]).repeat(x.number_of_night_points[0]), axis=1)
        # selected_planets_2_linear_aproximation['repeats_number'] = times['0_values']
        selected_planets_2_linear_aproximation['repeated_dec_1'] = selected_planets_2_linear_aproximation.apply(
            lambda x: np.array([x.dec]).repeat(x.number_of_night_points[0]), axis=1)
        selected_planets_2_linear_aproximation['repeated_ra_2'] = selected_planets_2_linear_aproximation.apply(
            lambda x: np.array([x.ra]).repeat(x.number_of_night_points[1]), axis=1)
        # selected_planets_2_linear_aproximation['repeats_number'] = times['0_values']
        selected_planets_2_linear_aproximation['repeated_dec_2'] = selected_planets_2_linear_aproximation.apply(
            lambda x: np.array([x.dec]).repeat(x.number_of_night_points[1]), axis=1)
        end = time.time()
        print('tiempo total de los repeated_ra y repeated_dec ' + str(end - start))
        start = time.time()
        ra_array = list(chain.from_iterable(selected_planets_1_linear_aproximation['repeated_ra'].values))
        dec_array = list(chain.from_iterable(selected_planets_1_linear_aproximation['repeated_dec'].values))
        times = iterative_delta_midnight_times_1_linear_aproximation['iterative_delta_midnight'].values, \
                iterative_delta_midnight_times_2_linear_aproximation['start_night_to_max_altitude_times'].values, \
                iterative_delta_midnight_times_2_linear_aproximation[
                    'end_night_to_max_altitude_times'].values  # estos son los tiempos que deben ir al crear el SkyCoord
        star_coords_1_linear_aproximation = SkyCoord(ra_array * u.degree,
                                                     dec_array * u.degree, frame='icrs', obstime=times[0])
        ra_array = list(chain.from_iterable(selected_planets_2_linear_aproximation['repeated_ra_1'].values))
        dec_array = list(chain.from_iterable(selected_planets_2_linear_aproximation['repeated_dec_1'].values))
        star_coords_2_linear_aproximation_1 = SkyCoord(ra_array * u.degree,
                                                       dec_array * u.degree, frame='icrs', obstime=times[1])
        ra_array = list(chain.from_iterable(selected_planets_2_linear_aproximation['repeated_ra_2'].values))
        dec_array = list(chain.from_iterable(selected_planets_2_linear_aproximation['repeated_dec_2'].values))
        star_coords_2_linear_aproximation_2 = SkyCoord(ra_array * u.degree,
                                                       dec_array * u.degree, frame='icrs', obstime=times[2])
        altaz_local_frame = AltAz(location=self.observatory)
        number_of_night_points = 0
        start = time.time()
        altitudes_1 = star_coords_1_linear_aproximation.transform_to(altaz_local_frame).alt.degree
        altitudes_2 = star_coords_2_linear_aproximation_1.transform_to(altaz_local_frame).alt.degree
        altitudes_3 = star_coords_2_linear_aproximation_2.transform_to(altaz_local_frame).alt.degree
        end = time.time()
        print('tiempo para sacar las altitudes: ' + str(end - start))
        mid_altitudes_1 = pd.DataFrame()
        count = 0
        end = time.time()
        print('tiempo total antes de los ciclos:' + str(end - start))
        start = time.time()
        for i in selected_planets_1_linear_aproximation['number_of_night_points']:
            array = pd.Series([altitudes_1[count:i + count]])
            mid_altitudes_1 = mid_altitudes_1.append(array, ignore_index=True)
            count += i
        count = 0
        mid_altitudes_2 = pd.DataFrame()
        for i in selected_planets_2_linear_aproximation['number_of_night_points']:
            array = pd.Series([altitudes_2[count:i[0] + count]])
            mid_altitudes_2 = mid_altitudes_2.append(array, ignore_index=True)
            count += i[0]
        count = 0
        mid_altitudes_3 = pd.DataFrame()
        for i in selected_planets_2_linear_aproximation['number_of_night_points']:
            array = pd.Series([altitudes_3[count:i[1] + count]])
            mid_altitudes_3 = mid_altitudes_3.append(array, ignore_index=True)
            count += i[1]
        end = time.time()
        print('tiempo en los ciclos: ' + str(end - start))
        altitudes_1 = self.stars_altitudes_.loc[:, iterative_delta_midnight_times_1_linear_aproximation.index]
        altitudes_2 = self.stars_altitudes_.loc[:, iterative_delta_midnight_times_2_linear_aproximation.index]
        altitudes_3 = self.stars_altitudes_.loc[:, iterative_delta_midnight_times_2_linear_aproximation.index]
        altitudes_1 = altitudes_1.transpose()
        altitudes_1.loc[:, 3] = mid_altitudes_1.values
        altitudes_1 = altitudes_1.transpose()
        altitudes_2 = altitudes_2.transpose()
        altitudes_2.loc[:, 3] = mid_altitudes_2.values
        altitudes_2.loc[:, 4] = mid_altitudes_3.values
        altitudes_2 = altitudes_2.transpose()
        altitudes_3 = altitudes_3.transpose()
        self.altitudes_1 = altitudes_1
        self.altitudes_2 = altitudes_2
        start = time.time()
        observation_times_1 = self.altitudes_1.apply(lambda x: self.start_and_end_observation_times(x), axis=0)
        observation_times_2 = self.altitudes_2.apply(lambda x: self.start_and_end_observation_times(x), axis=0)
        no_linear_aproximation_required_planets_observartion_times = no_linear_aproximation_required_planets_altitudes \
            .apply(lambda x: self.start_and_end_observation_times(x), axis=0)
        observation_times = pd.concat(
            [observation_times_1, observation_times_2, no_linear_aproximation_required_planets_observartion_times],
            axis=1).sort_index(axis=1).dropna()
        end = time.time()
        print('termina start_and_end_observation___, tiempo total: ' + str(end - start))
        self.selected_planets['start_observation'] = observation_times.iloc[0, :]
        self.selected_planets['end_observation'] = observation_times.iloc[1, :]
        return

    def planets_rank(self):
        if 'rank' in self.baseline_filter_:
            return

        self.altitude_grades()
        self.transit_percent_rank()
        self.baseline_percent_rank()
        self.moon_separation_rank()
        self.baseline_filter_['rank'] = self.baseline_filter_.filter(like='_grade').mean(axis=1)
        self.baseline_filter_.sort_values('transit_i', axis=0, inplace=True)

    def plot(self, precision=100, extend=False, altitude_separation=70):
        """
        Plots the altitudes of the encountered exoplanet's stars for the given date with information about the
        observation

         Parameters
         ----------
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
        self.planets_rank()
        cum = 0  # cumulative offset
        filtered_planets = self.baseline_filter_.copy()
        fig, axes = plt.subplots(figsize=(10, 15))
        cmap = cm.get_cmap(name='OrRd')
        grade_norm = mpl.colors.Normalize(vmin=0,vmax=10)
        scalar_mappable = mpl.cm.ScalarMappable(norm=grade_norm,cmap=cmap)
        for index in filtered_planets.index:
            info = filtered_planets.loc[index]
            transit_time = Time(info['start_observation'], format='jd', scale='utc') \
                           + np.linspace(0, (info['end_observation'] - info['start_observation']) * 24,
                                         precision) * u.hour
            local_frame = AltAz(obstime=transit_time, location=self.observatory)

            altitudes_min = np.zeros(precision) + altitude_separation * cum
            altitudes = np.array(info['star_coords'].transform_to(local_frame).alt.degree) \
                        - self.min_obs_ix + altitudes_min
            x_axis = transit_time.jd

            transit_i_index = np.argmin(np.abs(x_axis - info['transit_i']))
            transit_f_index = np.argmin(np.abs(x_axis - info['transit_f']))
            baseline_i_index = np.argmin(np.abs(x_axis - (info['transit_i'] - self.max_baseline)))
            baseline_i_index = baseline_i_index if baseline_i_index > 0 else 0
            baseline_f_index = np.argmin(np.abs(x_axis - (info['transit_f'] + self.max_baseline)))
            baseline_f_index = baseline_f_index if baseline_f_index < precision else precision - 1
            if extend:
                axes.fill_between(x_axis, altitudes, altitudes_min, color='yellow')
            if transit_i_index!=baseline_i_index and transit_f_index!=baseline_f_index:
                i = 1,1
            elif transit_i_index==baseline_i_index and transit_f_index!=baseline_f_index:
                i = 0,1
            elif transit_f_index==baseline_f_index and transit_i_index!=baseline_i_index:
                i = 1,0
            else:
                i = 0,0
            axes.fill_between(x_axis[transit_i_index:transit_f_index],
                              altitudes[transit_i_index:transit_f_index],
                              altitudes_min[transit_i_index:transit_f_index], color=cmap(info['rank'] / 10))
            axes.fill_between(x_axis[baseline_i_index:transit_i_index + i[0]],
                              altitudes[baseline_i_index:transit_i_index + i[0]],
                              altitudes_min[baseline_i_index:transit_i_index + i[0]], color='blue', )
            axes.fill_between(x_axis[transit_f_index - i[1]:baseline_f_index],
                              altitudes[transit_f_index - i[1]:baseline_f_index],
                              altitudes_min[transit_f_index - i[1]:baseline_f_index], color='blue', )

            if extend==False:
                axes.text(x_axis[baseline_f_index], altitudes_min[-1],
                          s=f"{transit_time[baseline_f_index].iso[11:16]} - "
                            f"{100 * info['transit_observation_percent'] if info['transit_observation_percent'] < 1 else 100:.0f}%",
                          fontsize=9)
                axes.text(x_axis[baseline_f_index], altitudes_min[-1] + 25,
                          s=f"{info['pl_name']} ({info['sy_vmag']}, {info['moon_separation']:.0f}$^\circ$)",
                          fontsize=9)
                axes.text(x_axis[baseline_i_index], altitudes_min[-1],
                          s=transit_time[baseline_i_index].iso[11:16],
                          fontsize=9, ha='right')
            if extend==True:
                axes.text(x_axis[len(x_axis)-1], altitudes_min[-1],
                          s=f"{transit_time[len(transit_time)-1].iso[11:16]} - "
                            f"{100 * info['transit_observation_percent'] if info['transit_observation_percent'] < 1 else 100:.0f}%",
                          fontsize=9)
                axes.text(x_axis[len(x_axis)-1], altitudes_min[-1] + 25,
                          s=f"{info['pl_name']} ({info['sy_vmag']}, {info['moon_separation']:.0f}$^\circ$)",
                          fontsize=9)
                axes.text(x_axis[0], altitudes_min[-1],
                          s=transit_time[0].iso[11:16],
                          fontsize=9, ha='right')


            cum += 1

        plt.colorbar(mappable=scalar_mappable,ticks=[0,5,10],location='right',orientation='vertical',shrink=0.5)
        axes.set_xlabel('Planets by Observation Rank', fontsize=20)
        axes.set_xticks([self.start_night.jd, self.end_night.jd, 1.5 * self.end_night.jd - 0.5 * self.start_night.jd])
        axes.set_xticklabels([str(self.start_night.value), str(self.end_night.value), ""])


def star_sidereal_time_to_local_hour(sun_ra, star_ra, midday):
    star_to_sun_distance = star_ra.hour - sun_ra.hour
    if star_to_sun_distance < 0:
        star_to_sun_distance = 24 - math.fabs(star_to_sun_distance)
    star_time_in_utc = midday + star_to_sun_distance * u.hour
    return star_time_in_utc


available_exoplanets = AvailableAt('La Silla Observatory', min_obs_ex=25.0,
                                   night_angle=-12)

available_exoplanets.run_day('2023-05-10')
available_exoplanets.plot()
#a.update(night_angle=-70)