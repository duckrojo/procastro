
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

from __future__ import print_function, division

import ephem
import numpy as np
import astropy.coordinates as apc
import astropy.units as u
import os
import dataproc.astro as dpa
from urllib.error import URLError
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import pdb


def _update_airmass(func):
    def wrapper_airmass(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if self.star is not None:
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

    Attributes
    ----------
    _obs : ephem.Observer instance
    _sun : ephem.Sun instance
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
        Current keys include:
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

    Parameters
    ----------
    timespan : int, optional
    target : str, optional
    site : string, optional
    equinox : string, optional
    central_time: int, optional
        Central time of y-axis.  If outside the [-12,24] range, then
        only shows nighttime

    """
    def __init__(self, timespan=2017, target=None,
                 site='paranal',
                 equinox="J2000",
                 **kwargs):

        self._obs = ephem.Observer()
        self._sun = ephem.Sun()
        if not hasattr(self, 'params'):
            self.params = {}
        self.params["equinox"] = equinox
        self.daily = {}

        # The following are attributes that will be initialized elsewhere
        self.jd0 = 0
        self.star = None
        self.transit_info = None
        self.airmass = None
        self.transits = None
        self.days = []
        self.xlims = []
        self.ylims = []
        self.hours = []
        self.transit_hours = []

        self.set_site(site, **kwargs)
        self.set_timespan(timespan, **kwargs)

        if target is not None:
            self.set_target(target, **kwargs)

    def set_site(self, site,
                 site_filename=None, **kwargs):
        """
        Checks whether name's coordinate are known from a list or whether it
        is a (lat, lon) tuple

        Parameters
        ----------
        site_filename: string, optional
            Filename storing site coordinates
        site: string ,optional
            Identifies the observatory as a name, or (lat, lon) tuple,
        """

        obs = self._obs

        if site_filename is None:
            site_filename = os.path.dirname(__file__) + '/observatories.coo'

        coordinates = {}
        for line in open(site_filename, 'r').readlines():
            field = line.split(',')
            coordinates[field[0]] = (eval(field[1]), eval(field[2]))

        if site is None:
            return coordinates.keys()

        if isinstance(site, str):
            try:
                lat_lon = coordinates[site]
            except KeyError:
                raise KeyError("site keyword is not defined. Valid values "
                               "are " + ', '.join(coordinates) + '.')
        elif isinstance(site, tuple) and len(site) == 2:
            lat_lon = site
        else:
            raise TypeError("object can only be a 2-component tuple or a string")

        # Parameters for user-friendly values
        self.params["lat_lon"] = lat_lon
        self.params["site"] = site
        # fraction of day to move from UT to local midday
        self.params["to_local_midday"] = 0.5-lat_lon[1]/360.0

        # parameters in pyephem
        obs.lon = str(lat_lon[1])
        obs.lat = str(lat_lon[0])

        obs.elevation = 0
        epoch = str(self.params["equinox"])
        if epoch[0] == 'J':
            epoch = epoch[1:]
        obs.epoch = epoch

        print("Selected (lat, lon): %s" % (lat_lon,))

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
        obs = self._obs
        obs.date = date
        self.star.compute(obs)
        moon = ephem.Moon()
        moon.compute(obs)
        st = apc.SkyCoord(ra=self.star.ra*u.radian, dec=self.star.dec*u.radian,
                          frame='icrs')
        mn = apc.SkyCoord(ra=moon.ra*u.radian, dec=moon.dec*u.radian,
                          frame='icrs')
        dist = st.separation(mn)
        return dist, moon.phase
        # return np.cos(self.star.dec)*np.cos(moon.dec)*np.cos(self.star.ra-moon.ra) \
        #     + np.sin(self.star.dec)*np.sin(self.star.dec)

    @_update_airmass
    @_update_transits
    def set_timespan(self, timespan, samples=60,
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
            ed0 = ephem.Date('%i/1/1' % (timespan,))
            ed1 = ephem.Date('%i/1/1' % (timespan+1,))

        elif isinstance(timespan, str): # Start Year - End Year
            years = timespan.split('-')
            if len(years) != 2:
                raise NotImplementedError("Requested timespan (%s) is not valid. Only a string "
                                          "in the format <FROMYEAR-TOYEAR> is accepted (only one "
                                          "dash separating integers)")
            elif int(years[0]) > int(years[1]):
                raise ValueError("Starting year must be lower than the end year")
                
            ed0 = ephem.Date('%i/1/1' % (int(years[0]),))
            ed1 = ephem.Date('%i/1/1' % (int(years[1])+1,))

        else:
            raise NotImplementedError("""Requested timespan (%s) not implemented yet. Currently supported:
            * single integer (year)
            """ % (timespan,))

        ed = np.arange(ed0, ed1, int((ed1 - ed0) / samples))
        xlims = [ed[0] - ed0, ed[-1] - ed0]

        self.jd0 = ephem.julian_date(ephem.Date(ed[0]))
        self.days = ed
        self.xlims = xlims

        self._get_sun_set_rise(**kwargs)
        self.set_vertical(central_time)

        return self

    def _get_sun_set_rise(self, **kwargs):
        """
        Compute sunsets and sunrises

        """
        sunset = []
        sunrise = []
        twilight_set = []
        twilight_rise = []
        obs = self._obs
        for day in self.days:
            # sunrise/set calculated from local midday
            obs.date = day + self.params["to_local_midday"]
            sunset.append(obs.next_setting(self._sun)-day)
            sunrise.append(obs.next_rising(self._sun)-day)
            obs.horizon = '-18:00'
            twilight_set.append(obs.next_setting(self._sun)-day)
            twilight_rise.append(obs.next_rising(self._sun)-day)
            obs.horizon = '0:00'
        self.daily["sunrise"] = np.array(sunrise)
        self.daily["twilight_rise"] = np.array(twilight_rise)
        self.daily["sunset"] = np.array(sunset)
        self.daily["twilight_set"] = np.array(twilight_set)

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
        if central_time>24 or central_time<-12:
            ylims = [min(self.daily["sunset"])*24-0.5, max(self.daily["sunrise"])*24+0.5]
            self.hours = np.arange(ylims[0], ylims[1], hour_step)
            self.ylims = ylims
        else:
            raise NotImplementedError("centering at times different than middle of night is"
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
            Either RA and Dec in hours and degrees, or target name to be queried
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

        ra_dec = dpa.read_coordinates(target,
                                     coo_files=[os.path.dirname(__file__)+'/coo.txt',
                                                os.path.expanduser("~")+'/.coostars'],
                                     equinox=self.params["equinox"])

        print("Star at RA/DEC: %s/%s" %(ra_dec.ra.to_string(sep=':'),
                                        ra_dec.dec.to_string(sep=':')))
        epoch = float(ra_dec.equinox.value[1:])
        self.star = ephem.readdb("%s,f,%s,%s,%.2f, %f" %
                                 (star_name,
                                  ra_dec.ra.to_string(sep=':', unit=u.hour),
                                  ra_dec.dec.to_string(sep=':'),
                                  magnitude, epoch))

        transit_epoch, transit_period, transit_length = dpa.get_transit_ephemeris(target, os.path.dirname(__file__))

        if transit_epoch is None or transit_period is None:
            print("Attempting to query transit information")
            try:
                planet = NasaExoplanetArchive.query_planet(target, all_columns=True) ### Included all_columns tags
            except URLError as mesg:
                raise NotImplementedError(f"NASA has not yet fixed the SSL validation error ({mesg}). Info has to be isput "
                       "manually in ~/.transits")
            
            # As of astroquery 0.3.10, normal columns are u.Quantity while extra
            # columns are floats. 
            # NOTE: If astroquery is upgraded check this block for any API changes.
            req_cols = ['pl_orbper', 'pl_tranmid', 'pl_trandur']
            for i in range(len(req_cols)):
                col = req_cols[i]
                # Missing information can be returned as None or as a numpy
                # Masked array
                if planet[col] is None or isinstance(planet[col], np.ma.MaskedArray):
                    raise ValueError("Requested data {} is not available on the \
                                       NASA archives for target {}".format(col, target))
                elif isinstance(planet[col],  u.Quantity):
                    req_cols[i] = planet[col].value
                else:
                    req_cols[i] = planet[col]
            
            transit_period, transit_epoch, transit_length = req_cols
            transit_length *= 24
            
            print("  Found ephemeris: %f + E*%f (length: %f)" % (transit_epoch, transit_period, transit_length))

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

        jd0 = self.jd0
        jd1 = ephem.julian_date(ephem.Date(self.days[-1]))
        n_transits = int((jd1-jd0)/tr_period)+2
        tr1 = tr_epoch+tr_period*int((jd0-tr_epoch)/tr_period+0.9)
        self.transits = tr1+np.arange(n_transits)*tr_period
        self.transit_hours = (self.transits-(np.fix(self.transits-0.5)+0.5))*24

        if jd0 < tr_epoch:
            print("WARNING: Reference transit epoch is in the future. Are you certain that you are using JD?")

        return self

    def _get_airmass(self, max_airmass=3.0, **kwargs):
        """
        Get airmass

        Parameters
        ----------
        max_airmass : float, optional
        """
        obs = self._obs
        hours = self.hours

        airmass = []
        for d in self.days:
            alts = []
            for h in hours:
                obs.date = d+h/24.0
                self.star.compute(obs)
                alts.append(self.star.alt)

            cosz = np.sin(np.array(alts))
            cosz[cosz < (1.0 / max_airmass)] = 1.0 / max_airmass
            airmass.append(1.0/cosz)

        self.airmass = np.array(airmass).transpose()

        return self
