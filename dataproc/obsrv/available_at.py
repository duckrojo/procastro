import matplotlib.pyplot as plt
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

__all__ = ['AvailableAt']


def to_hms(jd):
    return Time(jd, format='jd').to_value('iso', 'date_hms')[11:20]

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
    A class that contains information about the avalaible exoplanets at a given observatory and night.
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
    min_obs_ix : float
        the star's extreme minimum altitude in degrees that is acceptable for the observation. Must be equal or lower
        than min_obs.
    min_obs : float
        the star's minimum altitude in degrees required for the observation
    max_obs : float
        the star's ideal altitude in degrees required for the observation
    min_baseline_ix : float
        the baseline's extreme minimum value in days acceptable for the transit observation
    min_baseline : float
        the baseline's minimum value in days, required for the transit observation
    max_baseline : float
        the baseline's ideal value in days , required for the transit observation
    moon_separation_min_ix : float
        the extreme minimum separation angle between the moon and the star acceptable for the observation
    moon_separation_min : float
        the minimum separation angle between the moon and the star required for the observation
    moon_separation_max : float
        the ideal separation angle between the moon and the star required for the observation
    vmag_min_ix : float
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
                 min_obs_ix=23.5,
                 max_obs=40.0, min_baseline_ix=0.01, max_baseline=0.04, min_baseline=0.02, moon_separation_min_ix=10,
                 moon_separation_min=20, moon_separation_max=50, vmag_min_ix=12.5):
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
            min_obs_ix : float
                the star's extreme minimum altitude in degrees that is acceptable for the observation. Must be equal or
                lower than min_obs.
            max_obs : float
                the star's ideal altitude in degrees required for the observation
            min_baseline_ix : float
                the baseline's extreme minimum value in days acceptable for the transit observation
            max_baseline : float
                the baseline's ideal value in days , required for the transit observation
            min_baseline : float
                the baseline's minimum value in days, required for the transit observation
            moon_separation_min_ix : float
                the extreme minimum separation angle in degrees between the moon and the star acceptable for the
                observation
            moon_separation_min : float
                the minimum separation angle between the moon and the star required for the observation
            moon_separation_max : float
                the ideal separation angle between the moon and the star required for the observation
            vmag_min_ix : float
                the extreme minimum v magnitude value of the stars acceptable for the observation


        """
        self.night_angle = night_angle
        self.dataframe = query_full_exoplanet_db()
        self.observatory = coord.EarthLocation.of_site(observatory)
        self.utc_offset = (int(self.observatory.lon.degree / 15)) * u.hour
        # dejar todo en self.date_offset
        self.min_transit_percent = min_transit_percent
        self.min_transit_percent_ix = self.min_transit_percent / 2
        self.max_obs = max_obs
        self.min_obs = min_obs  # cambiar nombre a min_star_airmass
        self.min_obs_ix = min_obs_ix  # cambiar nombre a min_star_airmass_ix
        self.min_baseline_ix = min_baseline_ix
        self.min_baseline = min_baseline
        self.max_baseline = max_baseline
        self.moon_separation_min_ix = moon_separation_min_ix
        self.moon_separation_min = moon_separation_min
        self.moon_separation_max = moon_separation_max
        self.vmag_min_ix = vmag_min_ix

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
        self.pre_ephemerides()
        self.ephemerides()
        self.post_ephemerides()

        return self

    def pre_ephemerides(self):
        selected_planets = self.dataframe.copy()
        selected_planets = self.closest_transit_filter(selected_planets, self.date_offset)
        selected_planets = self.vmag_filter(selected_planets)
        self.selected_planets = self.transit_percent_filter(selected_planets)

    def ephemerides(self):
        self.stars_altitudes_ = self.stars_altitudes()
        self.start_and_end_observation___()

    def post_ephemerides(self):
        self.selected_planets = self.transit_observation_percent()
        self.baseline_percent_filter = self.baseline_percent()

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

    def sun_airmass(self, precision=1440):
        midnight = self.date_offset
        delta_midnight_loc = np.linspace(0, 24, precision) * u.hour
        times_july12_to_13 = midnight + delta_midnight_loc
        frame_july12_to_13 = AltAz(obstime=times_july12_to_13, location=self.observatory)
        sunaltazs_july12_to_13 = get_sun(times_july12_to_13).transform_to(frame_july12_to_13)
        return sunaltazs_july12_to_13

    def closest_transit_filter(self, planets_df, date):
        closest_transit_n = (((date.jd - planets_df['pl_tranmid']) / planets_df['pl_orbper'])
                             + 1.0).astype('int')
        closest_transit_jd = planets_df['pl_tranmid'] + closest_transit_n * planets_df['pl_orbper']
        delta_transit_jd = closest_transit_jd - date.jd
        planets_df['closest_transit'] = closest_transit_jd
        return planets_df[0 <= (delta_transit_jd < 1.0)]

    def vmag_filter(self, planets_df):
        no_filter_data = planets_df['sy_vmag'] == 0
        planets_df['sy_vmag'][no_filter_data] = planets_df['sy_gmag'][no_filter_data]
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

    def delta_midnight(self, precision=250):  # crea un array de 250 intervalos, del delta_midnight
        delta_date = self.end_night - self.start_night
        delta_date = delta_date.to_value('hour')
        delta_midnight = np.linspace(0, delta_date, precision) * u.hour
        delta_midnight = self.start_night + delta_midnight
        return delta_midnight

    def transit_percent_filter(self, planets_df, default_duration=2.5):
        planets_df['pl_trandur'][planets_df['pl_trandur'] == 0.0] = default_duration
        planets_df['transit_i'] = planets_df['closest_transit'] - (planets_df['pl_trandur'] / 48)
        planets_df['transit_f'] = planets_df['closest_transit'] + (planets_df['pl_trandur'] / 48)

        planets_df = planets_df[planets_df['transit_i'] < self.end_night.jd]
        planets_df = planets_df[planets_df['transit_f'] > self.start_night.jd]

        transit_i = planets_df['transit_i']
        transit_i[self.start_night.jd > transit_i] = self.start_night.jd
        transit_f = planets_df['transit_f']
        transit_f[self.end_night.jd < transit_f] = self.end_night.jd

        planets_df['transit_percent'] = (transit_f - transit_i) / (planets_df['pl_trandur']/24)
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
        planets_df = self.selected_planets
        planets_df = planets_df[planets_df['transit_i'] < planets_df['end_observation']]
        planets_df = planets_df[planets_df['transit_f'] > planets_df['start_observation']]
        planets_df['transit_i_obs'] = planets_df[['transit_i', 'start_observation']].max(axis=1)
        planets_df['transit_f_obs'] = planets_df[['transit_f', 'end_observation']].min(axis=1)
        planets_df['transit_observation_percent'] = (planets_df['transit_f_obs']
                                                     - planets_df['transit_i_obs'])/(planets_df['pl_trandur']/24)

        return planets_df[planets_df['transit_observation_percent'] > self.min_transit_percent_ix]

    def baseline_percent(self):
        planets_df = self.selected_planets
        planets_df['delta_i_baseline'] = planets_df['transit_i'] - planets_df['start_observation']
        planets_df['delta_f_baseline'] = planets_df['end_observation'] - planets_df['transit_f']
        planets_df['delta_i_baseline'] = np.where(self.min_baseline_ix < planets_df['delta_i_baseline'],
                                                  planets_df['delta_i_baseline'], 0.0)
        planets_df['delta_f_baseline'] = np.where(self.min_baseline_ix < planets_df['delta_f_baseline'],
                                                  planets_df['delta_f_baseline'], 0.0)
        planets_df = planets_df.query('delta_i_baseline!=0.0 or delta_f_baseline!=0.0')
        return planets_df

    def transit_percent_rank(self):
        planets_df_input = self.baseline_percent_filter
        min_transit_percent = self.min_transit_percent
        min_transit_percent_ix = self.min_transit_percent_ix
        max_transit_percent = planets_df_input['transit_observation_percent'].max()
        if max_transit_percent > 1.0:
            max_transit_percent = 1.0
        delta_grade_1 = min_transit_percent - min_transit_percent_ix
        delta_grade_2 = max_transit_percent - min_transit_percent
        planets_df_input['transit_percent_grade'] = np.where(
            planets_df_input['transit_observation_percent'] < min_transit_percent_ix,
            0, planets_df_input['transit_observation_percent'])
        planets_df_input['transit_percent_grade'] = np.where(
            (planets_df_input['transit_observation_percent'] >= min_transit_percent_ix) &
            (planets_df_input['transit_observation_percent'] < min_transit_percent),
            5 * (planets_df_input['transit_observation_percent'] - min_transit_percent_ix) / delta_grade_1,
            planets_df_input['transit_observation_percent'])
        planets_df_input['transit_percent_grade'] = np.where(
            (planets_df_input['transit_observation_percent'] >= min_transit_percent),
            (5 * (planets_df_input['transit_observation_percent'] - min_transit_percent) / delta_grade_2) + 5,
            planets_df_input['transit_observation_percent'])
        planets_df_input['transit_percent_grade'] = np.where(planets_df_input['transit_percent_grade'] > 10.0, 10.0,
                                                             planets_df_input['transit_percent_grade'])

        return planets_df_input

    def baseline_percent_rank(self):
        planets_df_input = self.baseline_percent_filter
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
        self.baseline_percent_filter = self.baseline_percent_filter.drop('delta_f_baseline_grade', axis=1)
        self.baseline_percent_filter = self.baseline_percent_filter.drop('delta_i_baseline_grade', axis=1)
        return planets_df_input

    def altitude_grades(self):
        min_obs_ix = self.min_obs_ix
        min_obs = self.min_obs
        max_obs = self.max_obs
        delta_alt_1 = min_obs - min_obs_ix
        delta_alt_2 = max_obs - min_obs
        closest_transit_altitude_ndarray = np.array([])
        planets_df_input = self.baseline_percent_filter
        for i in planets_df_input.index:
            closest_transit = Time(planets_df_input.loc[i, :]['closest_transit'], format='jd', scale='utc')
            local_frame = AltAz(obstime=closest_transit, location=self.observatory)
            from_icrs_to_alt_az = self.selected_planets.loc[i, :]['star_coords'].transform_to(local_frame).alt.degree
            closest_transit_altitude_ndarray = np.append(closest_transit_altitude_ndarray, from_icrs_to_alt_az)
        planets_df_input['closest_transit_altitudes'] = closest_transit_altitude_ndarray
        planets_df_input['altitude_grade'] = np.where(planets_df_input['closest_transit_altitudes'] <= min_obs_ix,
                                                      0, planets_df_input['closest_transit_altitudes'])
        planets_df_input['altitude_grade'] = np.where((planets_df_input['altitude_grade'] > min_obs_ix) &
                                                      (planets_df_input['altitude_grade'] <= min_obs),
                                                      5 * (planets_df_input[
                                                               'altitude_grade'] - min_obs_ix) / delta_alt_1,
                                                      planets_df_input['altitude_grade'])
        planets_df_input['altitude_grade'] = np.where(planets_df_input['altitude_grade'] >= min_obs,
                                                      (5 * (planets_df_input[
                                                                'altitude_grade'] - min_obs) / delta_alt_2) + 5,
                                                      planets_df_input['altitude_grade'])
        planets_df_input['altitude_grade'] = np.where(planets_df_input['altitude_grade'] > 10, 10,
                                                      planets_df_input['altitude_grade'])
        return planets_df_input

    def moon_separation_rank(self):
        planets_df_input = self.baseline_percent_filter
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

    def star_max_altitude_utc_time(self, precision=250):
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
        self.stars_coordinates(self.selected_planets)
        self.star_max_altitude_utc_time()
        self.selected_planets['altitudes'] = self.selected_planets.apply(lambda x: self.altitudes(x.max_altitude_time,
                                                                                                  x.star_coords),
                                                                         axis=1)
        stars_altitudes = pd.DataFrame(self.selected_planets['altitudes'].explode())
        index_array = '0 1 2 ' * len(stars_altitudes)
        index_array = np.array(index_array.split()).astype('int')[:len(stars_altitudes)]
        stars_altitudes['index'] = index_array
        stars_altitudes = pd.pivot_table(stars_altitudes, columns=stars_altitudes.index, values='altitudes',
                                         index='index')
        return stars_altitudes

    def linear_aproximation_to_min_obs_ix(self, star_coords, iterative_delta_midnight, first_altitude, end_altitude,
                                          obs_type):
        d_t = int(len(iterative_delta_midnight) / 5)
        local_frame = AltAz(obstime=iterative_delta_midnight[int(len(iterative_delta_midnight) / 5):int(
            (4 / 5) * len(iterative_delta_midnight)) + 1:d_t], location=self.observatory)
        from_icrs_to_alt_az = np.append(first_altitude,
                                        np.append(np.array(star_coords.transform_to(local_frame).alt.degree),
                                                  end_altitude))
        closest_observation_time_index = np.argmin(np.abs(from_icrs_to_alt_az - self.min_obs_ix)) * d_t
        closest_observation_altitude = from_icrs_to_alt_az[int(closest_observation_time_index / d_t)]
        closest_observation_altitude_time = iterative_delta_midnight[
            closest_observation_time_index - int(closest_observation_time_index / len(iterative_delta_midnight))]
        if obs_type == 'growing':
            if closest_observation_altitude < self.min_obs_ix:
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
            if closest_observation_altitude < self.min_obs_ix:
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
        min_obs_ix_altitude = 0
        count = 0
        while not (self.min_obs_ix - 0.001 < min_obs_ix_altitude < self.min_obs_ix + 0.001):
            delta_altitude = float((closest_observation_altitudes_and_times['altitudes'][0] -
                                    closest_observation_altitudes_and_times['altitudes'][1]))
            delta_time = float((closest_observation_altitudes_and_times['times'][0] -
                                closest_observation_altitudes_and_times['times'][1]).to_value('hour'))
            slope = delta_altitude / delta_time
            min_obs_ix_delta = float(closest_observation_altitudes_and_times['altitudes'][0] - self.min_obs_ix)
            min_obs_ix_time = ((closest_observation_altitudes_and_times['times'][0].jd * 24) - min_obs_ix_delta / slope)
            min_obs_ix_time = Time(min_obs_ix_time / 24, format='jd', scale='utc')
            min_obs_ix_time = Time(min_obs_ix_time, format='iso')
            local_frame = AltAz(obstime=min_obs_ix_time, location=self.observatory)
            min_obs_ix_altitude = float(star_coords.transform_to(local_frame).alt.degree)
            if count > 350:
                min_obs_ix_altitude = self.min_obs_ix
            elif min_obs_ix_altitude < self.min_obs_ix and obs_type == 'growing':
                closest_observation_altitudes_and_times['altitudes'][0] = min_obs_ix_altitude
                closest_observation_altitudes_and_times['times'][0] = min_obs_ix_time
            elif min_obs_ix_altitude > self.min_obs_ix and obs_type == 'growing':
                closest_observation_altitudes_and_times['altitudes'][1] = min_obs_ix_altitude
                closest_observation_altitudes_and_times['times'][1] = min_obs_ix_time
            elif min_obs_ix_altitude < self.min_obs_ix and obs_type == 'decreasing':
                closest_observation_altitudes_and_times['altitudes'][1] = min_obs_ix_altitude
                closest_observation_altitudes_and_times['times'][1] = min_obs_ix_time
            elif min_obs_ix_altitude > self.min_obs_ix and obs_type == 'decreasing':
                closest_observation_altitudes_and_times['altitudes'][0] = min_obs_ix_altitude
                closest_observation_altitudes_and_times['times'][0] = min_obs_ix_time
            count += 1
        return min_obs_ix_time.jd

    def start_and_end_observation__(self, x):
        start_observation = np.array([])
        end_observation = np.array([])
        if x.iloc[0] > self.min_obs_ix and x[2] > self.min_obs_ix:
            start_observation = np.append(start_observation, self.start_night.jd)
            end_observation = np.append(end_observation, self.end_night.jd)
        elif x.iloc[0] == x.iloc[1] and x.iloc[0] > self.min_obs_ix:
            start_observation = np.append(start_observation, self.start_night.jd)
            end_observation = np.append(end_observation, self.linear_aproximation_to_min_obs_ix(
                self.selected_planets.loc[x.name, :]['star_coords'], self.delta_midnight_,
                x.iloc[0], x.iloc[2], 'decreasing'))
        elif x.iloc[2] == x.iloc[1] and x.iloc[2] > self.min_obs_ix:
            end_observation = np.append(end_observation, self.end_night.jd)
            start_observation = np.append(start_observation, self.linear_aproximation_to_min_obs_ix(
                self.selected_planets.loc[x.name, :]['star_coords'], self.delta_midnight_,
                x.iloc[0],
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
                                                x.iloc[1],
                                                x.iloc[2], 'decreasing'))

            elif x.iloc[2] >= self.min_obs_ix:
                end_observation = np.append(end_observation, self.end_night.jd)
                start_observation = np.append(start_observation,
                                              self.linear_aproximation_to_min_obs_ix(
                                                  self.selected_planets.loc[x.name, :]['star_coords'],
                                                  self.delta_midnight_[0:self.selected_planets.loc[x.name, :]
                                                  ['max_altitude_index']], x.iloc[0],
                                                  x.iloc[1], 'growing'))
            else:
                start_observation = np.append(start_observation, self.linear_aproximation_to_min_obs_ix(
                    self.selected_planets.loc[x.name, :]['star_coords'],
                    self.delta_midnight_[0:self.selected_planets.loc[x.name, :]['max_altitude_index']],
                    x.iloc[0],
                    x.iloc[1], 'growing'))
                end_observation = np.append(end_observation, self.linear_aproximation_to_min_obs_ix(
                    self.selected_planets.loc[x.name, :]['star_coords'], self.delta_midnight_[
                                                                         self.selected_planets.loc[
                                                                         x.name, :][
                                                                             'max_altitude_index']:len(
                                                                             self.delta_midnight_) - 1],
                    x.iloc[1], x.iloc[2], 'decreasing'))
        else:
            start_observation = np.append(start_observation, False)
            end_observation = np.append(end_observation, False)

        x.loc[3] = start_observation[0]
        x.loc[4] = end_observation[0]
        return x.loc[3], x.loc[4]

    def start_and_end_observation___(self):
        observation_times = self.stars_altitudes_.apply(lambda x: self.start_and_end_observation__(x), axis=0)
        self.selected_planets['start_observation'] = observation_times.iloc[0, :]
        self.selected_planets['end_observation'] = observation_times.iloc[1, :]
        self.selected_planets = self.selected_planets[self.selected_planets['start_observation'] != False]
        return

    def planets_rank(self):
        if 'rank' in self.baseline_percent_filter:
            return

        self.altitude_grades()
        self.transit_percent_rank()
        self.baseline_percent_rank()
        self.moon_separation_rank()
        self.baseline_percent_filter['rank'] = self.baseline_percent_filter.filter(like='_grade').mean(axis=1)
        self.baseline_percent_filter.sort_values('transit_i', axis=0, inplace=True)

    def plot(self, precision=100, extend=False, altitude_separation=50):
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
        filtered_planets = self.baseline_percent_filter

        fig, axes = plt.subplots(figsize=(10, 15))
        cmap = cm.get_cmap(name='OrRd')
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

            axes.fill_between(x_axis[transit_i_index:transit_f_index],
                              altitudes[transit_i_index:transit_f_index],
                              altitudes_min[transit_i_index:transit_f_index], color=cmap(info['rank'] / 10))
            axes.fill_between(x_axis[baseline_i_index:transit_i_index],
                              altitudes[baseline_i_index:transit_i_index],
                              altitudes_min[baseline_i_index:transit_i_index], color='blue', )
            axes.fill_between(x_axis[transit_f_index:baseline_f_index],
                              altitudes[transit_f_index:baseline_f_index],
                              altitudes_min[transit_f_index:baseline_f_index], color='blue', )

            axes.text(x_axis[baseline_f_index], altitudes_min[-1],
                      s=f"{transit_time[baseline_f_index].iso[11:16]} - "
                        f"{100*info['transit_observation_percent'] if info['transit_observation_percent'] < 1 else 100:.0f}%",
                      fontsize=9)
            axes.text(x_axis[baseline_f_index], altitudes_min[-1]+25,
                      s=f"{info['pl_name']} ({info['sy_vmag']}, {info['moon_separation']:.0f}$^\circ$)",
                      fontsize=9)
            axes.text(x_axis[baseline_i_index], altitudes_min[-1],
                      s=transit_time[baseline_i_index].iso[11:16],
                      fontsize=9, ha='right')

            cum += 1

        axes.set_xlabel('Planets by Observation Rank', fontsize=20)
        axes.set_xticks([self.start_night.jd, self.end_night.jd, 1.5*self.end_night.jd-0.5*self.start_night.jd])
        axes.set_xticklabels([str(self.start_night.value), str(self.end_night.value), ""])



def star_sidereal_time_to_local_hour(sun_ra, star_ra, midday):
    star_to_sun_distance = star_ra.hour - sun_ra.hour
    if star_to_sun_distance < 0:
        star_to_sun_distance = 24 - math.fabs(star_to_sun_distance)
    star_time_in_utc = midday + star_to_sun_distance * u.hour
    return star_time_in_utc
