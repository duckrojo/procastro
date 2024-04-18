
#
# Copyright (C) 2014 Patricio Rojo
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

import astropy.coordinates as apc
import astropy.time as apt
import astropy.units as u
import procastro.astro as paa
import numpy as np
import os
import pyvo as vo

import procastro.astro.coordinates
import procastro.astro.exoplanet

exo_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")


def _update_airmass(func):
    def wrapper_airmass(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if self._target is not None:
            self._get_airmass()

        return ret

    return wrapper_airmass


def _update_transits(func):
    def wrapper_transits(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if self.transit_info is not None:
            self.set_transits()

        return ret

    return wrapper_transits


# noinspection PyCompatibility
class ObsCalc(object):
    """
    Object used to compute exoplanet transits

    Parameters
    ----------
    timespan : int, optional
    target : str, optional
    site : string, optional
    equinox : string, optional
    central_time: int, optional
        Central time of y-axis.  If outside the [-12,24] range,
        then only shows nighttime

    Attributes
    ----------
    params : dict
        General parameter values.
        Available keys: 'target', 'current transit', 'equinox', 'lat_lon',
        'site', 'to_local_midday', 'timespan'
    daily : dict
        Sunset, sunrise and twilight information
    jd0 : float
        Initial transit timespan value in JD
    star : SkyCoord instance
        Host star
    transit_info : dict
        Data relevant to the current transit being observed
        Current keys include: length, epoch, period and offset
    airmass : array_like
    transits : array_like
        Available transits for the current target
    transit_hours : array_like
        Hours in which the exoplanet transit is detectable
    days : array_like
        Days in which the current transit is visible
    hours : array_like
    xlims, ylims : array_like
        Dimensions of grid used to create the main figures

    """
    def __init__(self, timespan=2017, target=None,
                 site='paranal',
                 equinox="J2000",
                 phase_offset=0,
                 **kwargs):

        if not hasattr(self, 'params'):
            self.params = {}
        self.params["equinox"] = equinox
        self.daily = {}

        # The following are attributes that will be initialized elsewhere
        self._target = None
        self._location = None
        self.star = None
        self.transit_info = None
        self.airmass = None
        self.transits = None
        self.xlims = []
        self.ylims = []
        self.days = apt.Time.now()
        self.hours = np.array([])
        self.transit_hours = []

        self.set_site(site)
        self.set_timespan(timespan, **kwargs)

        if target is not None:
            self.set_target(target, phase_offset=phase_offset, **kwargs)

    def set_site(self, site):
        """
        Define name from list aailable from EarthLocation

        Parameters
        ----------
        site: string ,optional
            Identifies the observatory as a name, or (lat, lon) tuple,
        """

        self._location = apc.EarthLocation.of_site(site)

        # Parameters for user-friendly values
        self.params["site"] = site
        # fraction of day to move from UT to local midday
        self.params["to_local_midday"] = 0.5 - self._location.lon/(360.0*u.deg)

        print(f"Selected (lat, lon): {self._location.lat}, {self._location.lon}")

        return self

    def _moon_distance(self, date):
        """
        Computes moon distance

        Parameters
        ----------
        date : str (YYYY/MM/DD HH:MM:SS.mm)

        Returns
        -------
        int, float :
            Distance and moon phase represented as percentage
        """
        with apc.solar_system_ephemeris.set('builtin'):
            moon = apc.get_body("moon", date, location=self._location)
            sun = apc.get_sun(date)

        separation = moon.separation(self._target)
        sun_moon_dist = np.sqrt(sun.distance**2 + moon.distance**2 -
                                2*sun.distance*moon.distance*np.cos(sun.separation(moon)))
        cos_phase_angle = (moon.distance**2 + sun_moon_dist**2 -
                           sun.distance**2)/moon.distance/sun_moon_dist/2
        illumination = 100*(1+cos_phase_angle)/2

        return separation, illumination

    @_update_airmass
    @_update_transits
    def set_timespan(self, timespan, samples=120,
                     central_time=25,
                     **kwargs):
        """
        Set time span

        Parameters
        ----------
        timespan : str, int
        samples : int, optional
        central_time : int
        """
        if timespan is None:
            if 'timespan' in self.params:
                timespan = self.params["timespan"]
            else:
                # TODO: default to next year!
                raise ValueError("Timespan needs to be specified")
        else:
            self.params["timespan"] = timespan

        if isinstance(timespan, int):   # Year
            # times always at midnight (UT)
            t0 = apt.Time('{0:d}-01-01'.format(timespan,))
            t1 = apt.Time('{0:d}-01-01'.format(timespan+1,))

        elif isinstance(timespan, str):  # Start Year - End Year
            years = timespan.split('-')
            if len(years) != 2:
                raise NotImplementedError(
                    "Requested timespan ({0:s}) is not valid. Only a string "
                    "in the format <FROMYEAR-TOYEAR> is accepted (only one "
                    "dash separating integers)".format(years[0]))
            elif int(years[0]) > int(years[1]):
                raise ValueError(
                    "Starting year must be lower than the end year")

            t0 = apt.Time('{0:d}-1-1'.format(int(years[0]),))
            t1 = apt.Time('{0:d}-1-1'.format(int(years[1])+1,))

        else:
            raise NotImplementedError(
                "Requested timespan ({0:s}) not implemented "
                "yet. Currently supported: * single "
                "integer (year)".format(timespan,))

        dt = (t1 - t0)/(samples-1)
        dt = int(dt.to(u.day).value + 0.5)
        self.days = t0 + dt*u.day*np.arange(samples)
        self.xlims = [0, (t1 - t0).to(u.day).value]

        self._get_sun_set_rise(**kwargs)
        self.set_vertical(central_time)

        return self

    def _get_sun_set_rise(self, nhours=40):
        """
        Compute sunsets and sunrises

        """

        hours = self.params["to_local_midday"] + np.linspace(-1, 0, nhours)
        the_24_365 = apt.Time((self.days.jd + hours[:, np.newaxis]).value, format='jd')

        half = len(hours)//2
        with apc.solar_system_ephemeris.set('builtin'):
            suns = apc.get_sun(the_24_365).transform_to(apc.AltAz(obstime=the_24_365,
                                                                  location=self._location))

        first_half = np.flip(np.array(suns.alt[:half, :]).transpose(), axis=1)
        first_hours = np.flip(hours[:half])
        sunset = np.array([np.interp(0, alts, first_hours) for alts in first_half])
        twilight_set = np.array([np.interp(-18, alts, first_hours) for alts in first_half])

        second_half = np.array(suns.alt[half:, :]).transpose()
        second_hours = hours[half:]
        sunrise = np.array([np.interp(0, alts, second_hours) for alts in second_half])
        twilight_rise = np.array([np.interp(-18, alts, second_hours) for alts in second_half])

        self.daily["sunset"] = np.array(sunset)
        self.daily["twilight_set"] = np.array(twilight_set)
        self.daily["twilight_rise"] = np.array(twilight_rise)
        self.daily["sunrise"] = np.array(sunrise)

        return self

    @_update_airmass
    def set_vertical(self, central_time, hour_step=0.2, **kwargs):
        """
        Sets values for the Y-axis

        Parameters
        ----------
        central_time : int
        hour_step : float, optional
            Time interval between ibservations

        """
        if central_time > 24 or central_time < -12:
            ylims = [min(self.daily["sunset"])*24-0.5,
                     max(self.daily["sunrise"])*24+0.5
                     ]
            self.hours = np.arange(ylims[0], ylims[1], hour_step)*u.hour
            self.ylims = ylims
        else:
            raise NotImplementedError(
                "Centering at times different than middle of night is"
                " not supported yet")

        return self

    @_update_airmass
    @_update_transits
    def set_target(self, target, magnitude=10, star_name='',
                   transit_epoch=None, transit_period=None, transit_length=1,
                   phase_offset=0.0,
                   **kwargs):
        """
        Set star and site into pyephem

        Parameters
        ----------
        target:
            Either RA and Dec in hours and degrees, or target name
            to be queried
        magnitude:
        star_name: string, optional
            Name of host star
        transit_length: float, optional
        transit_period: float, optional
        transit_epoch: float, optional
        phase_offset: float, optional
            Set to 0.5 to show occultations instead of transits
        """
        self.params['target'] = target
        if 'current_transit' in self.params:
            del self.params['current_transit']

        paths = [os.path.dirname(__file__)+'/coo.txt',
                 os.path.expanduser("~")+'/.coostars'
                 ]
        self._target = procastro.astro.coordinates.find_target(target,
                                                               coo_files=paths,
                                                               equinox=self.params["equinox"])

        print("Star at RA/DEC: {0:s}/{1:s}"
              .format(self._target.ra.to_string(sep=':'),
                      self._target.dec.to_string(sep=':')))

        transit_epoch, transit_period, transit_length = \
            procastro.astro.exoplanet.get_transit_ephemeris(target)
        print(f"Found in file: {transit_epoch}+E*{transit_period} +- {transit_length}")

        if transit_epoch is None or transit_period is None:
            print("Attempting to query transit information")

            query = f"SELECT pl_name,pl_tranmid,pl_orbper,pl_trandur FROM exo_tap.pscomppars " \
                    f"WHERE lower(pl_name) like '%{target}%' "
            resultset = exo_service.search(query)
            try:
                req_cols = [resultset['pl_orbper'].data[0], resultset['pl_tranmid'].data[0]]
            except IndexError:
                raise IndexError(f"Planet {target} not found in exoplanet database")
            trandur = resultset['pl_trandur'].data[0]
            if trandur is None:
                req_cols.append(1)
                warnings.warn("Using default 1hr length for transit duration", UserWarning)
            else:
                req_cols.append(trandur)

            transit_period, transit_epoch, transit_length = req_cols

            print("  Found ephemeris: {0:f} + E*{1:f} (length: {2:f})"
                  .format(transit_epoch, transit_period, transit_length))

        if transit_period != 0:
            transit_epoch += phase_offset * transit_period
            self.transit_info = {'length': transit_length,
                                 'epoch': transit_epoch,
                                 'period': transit_period,
                                 'offset': phase_offset}
        else:
            self.set_transits(tr_period=0)

        return self

    def set_transits(self,
                     tr_period=None, tr_epoch=None,
                     **kwargs):
        """
        Calculate the transits. It assumes that the decorator has already
        checked transit_info existence

        Parameters
        ----------
        tr_period : float, optional
            Transit period
        tr_epoch : float, optional
            Transit epoch
        """
        if tr_period == 0:
            if hasattr(self, 'transits'):
                del self.transits
                del self.transit_hours

            return self

        if tr_period is None:
            tr_period = self.transit_info['period']
        if tr_epoch is None:
            tr_epoch = self.transit_info['epoch']

        jd0 = self.days[0].jd
        jd1 = self.days[-1].jd
        n_transits = int((jd1-jd0)/tr_period)+2
        tr1 = tr_epoch+tr_period*int((jd0-tr_epoch)/tr_period+0.9)
        self.transits = tr1+np.arange(n_transits)*tr_period
        self.transit_hours = (self.transits-(np.fix(self.transits-0.5)+0.5))*24

        if jd0 < tr_epoch:
            warnings.warn("WARNING: Reference transit epoch is in the future."
                          "Are you certain that you are using JD?", UserWarning)

        return self

    def _get_airmass(self):
        """
        Get airmass

        Parameters
        ----------
        """
        # obs = self._obs
        hours = self.hours
        days = self.days

        times = days + hours[:, np.newaxis]

        self.airmass = self._target.transform_to(apc.AltAz(obstime=times,
                                                           location=self._location)).secz
        self.airmass[self.airmass < 0] = 100

        return self
