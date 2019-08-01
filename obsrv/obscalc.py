
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
import scipy as sp
import astropy.coordinates as apc
import astropy.units as u
import os
import dataproc as dp
from urllib.error import URLError
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive


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

    def __init__(self, timespan=2017, target=None,
                 site='paranal',
                 equinox="J2000",
                 **kwargs):
        """ =Initializes obsrv class.

        :param central_time: Central time of y-axis.  If outside the [-12,24] range, then only shows nighttime
        """

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
        """Checks whether name's coordinate are known from a list or whether it is a (lat, lon) tuple

        :param site_filename: Filename storing site coordinates
        :param site: identifies the observatory as a name, or (lat, lon) tuple,
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
        # return sp.cos(self.star.dec)*sp.cos(moon.dec)*sp.cos(self.star.ra-moon.ra) \
        #     + sp.sin(self.star.dec)*sp.sin(self.star.dec)

    @_update_airmass
    @_update_transits
    def set_timespan(self, timespan, samples=60,
                     central_time=25,
                     **kwargs):
        """Set time span"""
        if timespan is None:
            if 'timespan' in self.params:
                timespan = self.params["timespan"]
            else:
                # TODO: default to next year!
                raise ValueError("Timespan needs to be specified")
        else:
            self.params["timespan"] = timespan
        # pdb.set_trace()
        if isinstance(timespan, int):   # Year
            # times always at midnight (UT)
            ed0 = ephem.Date('%i/1/1' % (timespan,))
            ed1 = ephem.Date('%i/1/1' % (timespan+1,))

        elif isinstance(timespan, str):
            years = timespan.split('-')
            if len(years) != 2:
                raise NotImplementedError("Requested timespan (%s) is not valid. Only a string "
                                          "in the format <FROMYEAR-TOYEAR> is accepted (only one "
                                          "dash separating integers)")
            ed0 = ephem.Date('%i/1/1' % (int(years[0]),))
            ed1 = ephem.Date('%i/1/1' % (int(years[1])+1,))

        else:
            raise NotImplementedError("""Requested timespan (%s) not implemented yet. Currently supported:
            * single integer (year)
            """ % (timespan,))

        ed = sp.arange(ed0, ed1, int((ed1 - ed0) / samples))
        xlims = [ed[0] - ed0, ed[-1] - ed0]

        self.jd0 = ephem.julian_date(ephem.Date(ed[0]))
        self.days = ed
        self.xlims = xlims

        self._get_sun_set_rise(**kwargs)
        self.set_vertical(central_time)

        return self

    def _get_sun_set_rise(self, **kwargs):
        """Compute sunsets and sunrises"""
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
        self.daily["sunrise"] = sp.array(sunrise)
        self.daily["twilight_rise"] = sp.array(twilight_rise)
        self.daily["sunset"] = sp.array(sunset)
        self.daily["twilight_set"] = sp.array(twilight_set)

        return self

    @_update_airmass
    def set_vertical(self, central_time, hour_step=0.2, **kwargs):
        if central_time>24 or central_time<-12:
            ylims = [min(self.daily["sunset"])*24-0.5, max(self.daily["sunrise"])*24+0.5]
            self.hours = sp.arange(ylims[0], ylims[1], hour_step)
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
        """Set star and site into pyephem

        :param phase_offset: Set to 0.5 to show occultations instead of transits
        :param transit_length:
        :param transit_period:
        :param transit_epoch:
        :param magnitude:
        :param star_name:
        :param target: either RA and Dec in hours and degrees, or target name to be queried
        """

        self.params['target'] = target
        if 'current_transit' in self.params:
            del self.params['current_transit']

        ra_dec = dp.read_coordinates(target,
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

        transit_filename_locations = [os.path.expanduser("~")+'/.transits',
                                      os.path.dirname(__file__)+'/transits.txt',
                                      ]

        for transit_filename in transit_filename_locations:
            try:
                open_file = open(transit_filename)

                override = []
                for line in open_file.readlines():
                    if line[0] == '#' or len(line) < 3:
                        continue
                    data = line[:-1].split()
                    planet = data.pop(0)
                    if dp.accept_object_name(planet, target):
                        for d in data:
                            if d[0].lower() == 'p':
                                override.append('period')
                                transit_period = float(eval(d[1:]))
                            elif d[0].lower() == 'e':
                                override.append("epoch")
                                transit_epoch = float(eval(d[1:]))
                            elif d[0].lower() == 'l':
                                override.append("length")
                                transit_length = float(eval(d[1:]))
                            elif d[0].lower() == 'c':
                                override.append("comment")
                            else:
                                raise ValueError("data field not understood, it must "
                                                 "start with L, P, C, or E:\n%s" % (line,))
                        print("Overriding for '%s' from file '%s':\n %s" % (planet,
                                                                            transit_filename,
                                                                            ', '.join(override),))

                if len(override):
                    break

            except IOError:
                pass

        if transit_epoch is None or transit_period is None:
            print("Attempting to query transit information")
            try:
                planet = NasaExoplanetArchive.query_planet(target)
            except URLError as mesg:
                raise NotImplementedError(f"NASA has not yet fixed the SSL validation error ({mesg}). Info has to be input "
                       "manually in ~/.transits")
            transit_period = planet['pl_orbper']
            transit_epoch = planet['pl_tranmid']
            transit_length = planet['pl_trandur'] * 24
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
        """ Calculate the transits.  It assumes that the decorator has already checked transit_info existence """
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
        self.transits = tr1+sp.arange(n_transits)*tr_period
        self.transit_hours = (self.transits-(sp.fix(self.transits-0.5)+0.5))*24

        if jd0 < tr_epoch:
            print("WARNING: Reference transit epoch is in the future. Are you certain that you are using JD?")

        return self

    def _get_airmass(self, max_airmass=3.0, **kwargs):
        """Get airmass"""
        obs = self._obs
        hours = self.hours

        airmass = []
        for d in self.days:
            alts = []
            for h in hours:
                obs.date = d+h/24.0
                self.star.compute(obs)
                alts.append(self.star.alt)

            cosz = sp.sin(sp.array(alts))
            cosz[cosz < (1.0 / max_airmass)] = 1.0 / max_airmass
            airmass.append(1.0/cosz)

        self.airmass = sp.array(airmass).transpose()

        return self
