# Copyright (C) 2016 Andrea Egas
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

from __future__ import print_function, division

import ephem as ep
import scipy as sp
import numpy as np
import astropy as ap
import astropy.coordinates as apc
import astropy.units as u
import os
import xml.etree.ElementTree as ET, urllib, gzip, io
import dataproc as dp


class observation_night(object):
    # Inicialization
    def __init__(self, date, site='paranal', equinox=2000, **kwargs):
        self.date = str(date)  # Date of night observation
        self._obs = ep.Observer()  # Observer created for pyephem
        self._sun = ep.Sun()

        self.params = {}  # Dictionary that will contain parameters of night observation
        self.params["equinox"] = equinox
        self.day = {}
        self.store = {}
        self.star_cat = {}

        self.set_location(site, date)

        print(kwargs)

    # Observer place
    def set_location(self, site, date, observatories_file=None, **kwargs):

        obs = self._obs

        if observatories_file is None:
           # observatories_file = os.path.dirname(os.path.abspath(__file__)) + '/observatories.coo'  # Observatories file
            observatories_file= '/home/amia/Desktop/Practica2/dataproc/obsrv/observatories.coo'

        coordinates = {}  # Dictionary that will contain elements whose format is "Observatory_name: (lat, long)"
        for line in open(observatories_file, 'r').readlines():  # Each line of file contains name, lat and long
            field = line.split(',')
            coordinates[field[0]] = (eval(field[1]), eval(field[2]))

        if site == None: # If "site" is empty, it returns the name of each observatory
            return coordinates.keys()

        if isinstance(site, str):   # If the value of "site" is a string, then it stores the coordinates of this
            try:                    # observatory in "latlon" variable, except if keyword is not defined. In this last
                latlon = coordinates[site]  # case, it returns an error message with the valid names of each
            except KeyError:                # observatory
                raise KeyError("Site keyword is not defined. Valid values are " + ', '.join(coordinates) + '.')

        elif isinstance(site, tuple) and len(site) == 2:
            latlon = site   # If the value of "site" is a 2-component tuple, then it stores this value in "latlon"
                            # variable, except if "site" is different from the already defined
        else:
            raise TypeError("Object can only be a 2-component tuple or a string.")

        # Parameters for user-friendly values
        self.params["site"] = site
        self.params["latlon"] = latlon

        # Fraction of day to move from UT to local midday
        self.params["local_time"] = 0.5 - latlon[1] / 360.0

        # Parameters in pyephem
        obs.lat = str(latlon[0])  # Assigns the latitude to object "obs" (from Observer class defined in Pyephem)
        obs.lon = str(latlon[1])  # Similary, assigns the longitude to object "obs"

        obs.elevation = 0   # Assigns the elevation to object "obs"
        obs.epochs = str(self.params["equinox"]) # Assings the epoch to object "obs"

        print("Selected (lat, lon): (%s, %s)" %(latlon[0],latlon[1]))

        self.set_date(date)

        return self

    # Observation date format
    def set_date(self, date, **kwargs):  # Date format: yyyy/mm/dd
        lenght_date = len(date)
        # Format conditions
        if 7 < lenght_date < 11:    # 8 and 10 are the minimum and maximum quantity of characters that could have
            separator = date.find('/')  # the date format. ".find('/')" indicates the index of the first position
                                        # where slash appears in the date format
            # Separation conditions
            if separator != 4:
                raise TypeError("Correct date format: yyyy/mm/dd")
            elif separator == 4:
                day_observation = ep.Date(date)
                day_observation_after = ep.Date(day_observation + 1)

            # Another way of separation (correction dates in aaaa-mm-dd or another format)
            elif date[4].isdigit() == False:
                # Allowed format
                date_components = date.split(date[4])  # Different formats (for example: yyyy-mm-dd)
                corrected_date = "/".join(date_components)  # Correction to the format yyyy/mm/dd
                day_observation = ep.Date(corrected_date)
                day_observation_after = ep.Date(day_observation + 1)
            else:
                raise TypeError("Correct date format: yyyy/mm/dd")

            date_jd = ep.julian_date(date)  # Converts date to julian date and saves it in "date_jd"

            # Local midnight UT
            midnight = ep.Date(day_observation_after + 0.5 + self.params["local_time"])
            midnight_jd = ep.julian_date(midnight)
            # Local midday UT
            midday = ep.Date(day_observation_after + self.params["local_time"])
            midday_jd = ep.julian_date(midday)

        else:
            raise TypeError("Date format: yyyy/mm/dd")

        self.day["Observation date"] = date
        self.day["Julian date"] = date_jd
        self.day["Midday"] = midday
        self.day["Midnight"] = midnight

        print("Midday %s and midnight %s" % (midday, midnight))
        print("Observation date %s" % (date))

        self.set_sunset_rise(date, **kwargs)
        self.set_target(**kwargs)

        return self

    # Twilights: sunsets and sunrises
    def set_sunset_rise(self, date, **kwargs):
        obs = self._obs

        obs.date = self.day["Midnight"]
        self.day["Sunset"] = obs.previous_setting(self._sun)
        self.day["Sunrise"] = obs.next_rising(self._sun)

        print("Sunrise %s and sunset %s at %s" % (self.day["Sunrise"], self.day["Sunset"], self.day["Observation date"]))

        return self

    # Catalogue
    def set_target(self, magn=10, starname='', epoch=None, period=None, length=1, **kwargs):
        #catalogue = os.path.dirname(os.path.abspath(__file__)) + '/test_input_catalogue.dat'
        catalogue= '/home/amia/Desktop/Practica2/dataproc/obsrv/test_input_catalogue.dat'
  
        target = []
        period_transit = []
        epoch_transit = []
        length_transit = []

        radec = []
        star_information = []

        for lines in open(catalogue, 'r').readlines(): # Each line contains planet, ra, dec
            text = lines.split('\r')
            for line in text:
                data = line.split('\t')
                planet = data[0]
                target.append(planet)
                ra = data[1]
                dec = data[2]
                tuple_radec = (data[1], data[2])

                equinox = self.params["equinox"]
                # Print("Star at RA,DEC: (%s,%s)" %(ra,dec))
                self.star = ep.readdb("%s,f,%s,%s,%.2f,%f" % (starname, ra, dec, magn, equinox))

                for d in data[3:]:
                    if d[0].lower() == 'p':
                        period = float(eval(d[1:]))
                    elif d[0].lower() == 'e':
                        epoch = float(eval(d[1:]))
                    elif d[0].lower() == 'l':
                        length = float(eval(d[1:]))
                        # elif d[0].lower()=='c':
                        # override.append("comment")
                    else:
                        raise ValueError("Params must start with L, P, C, \or E:\n%s" % (data,))

                period_transit.append(period)
                epoch_transit.append(epoch)
                length_transit.append(length)

                radec.append(tuple_radec)
                star_information.append(self.star)

                # print ("Transit information of %s is: period: %s, epoch: %s, length: %s, equinox: %s" % (planet,period, epoch, length,equinox))

        self.store['target'] = sp.array(target)
        self.store['period'] = sp.array(period_transit)
        self.store['epoch'] = sp.array(epoch_transit)
        self.store['length'] = sp.array(length_transit)
        self.star_cat['RA,DEC'] = sp.array(radec)
        self.star_cat['Star info'] = sp.array(star_information)

        print(self.store['target'])
        print(self.store['period'])
        print(self.store['epoch'])
        print(self.store['length'])
        print(self.star_cat['RA,DEC'])
        print(self.star_cat['Star info'])

        self.set_curve()
        self.set_transits()

        return self

    # Find closest transits
    def set_transits(self, **kwargs):
        transit = []

        for target, epoch, period, length in zip(self.store['target'], self.store['epoch'], self.store['period'],
                                                 self.store['length']):
            day_julian = ep.julian_date(ep.Date(self.day["Midnight"]))
            transit_at_midnight = ((day_julian - epoch) / period)  # de donde sale esta formula
            phase = transit_at_midnight % 1  # una cifra significaiva, que es phase
            closest_transit = period * int(transit_at_midnight + 0.5) + epoch
            # closest_transit = period*int(transit_at_midnight)+epoch + (phase>0.5)
            transit.append(ep.Date(closest_transit - 2415020))  # Por que resto ese numero
            print(target, ep.Date(closest_transit - 2415020))

        self.store['transit'] = sp.array(transit)
        print(self.store['transit'])

        self.set_restr_twil()

        return self

    # Curve
    def set_curve(self, hr_sep=0.2, **kwargs):  # hr_sep=0.2 y luego hr_sep=0
        obs = self._obs
        night = [ep.Date(self.day["Sunset"] - 0.5 * ep.hour), ep.Date(self.day["Sunrise"] + 0.5 * ep.hour)]

        lar = int((ep.Date(night[1]) - ep.Date(night[0])) * 24 / hr_sep)
        hours = []
        r = 0
        hr_sep = 0
        while r < lar:
            val = ep.Date(night[0] + hr_sep * ep.hour)
            hours.append(val)
            # print(val) -> fecha hora
            hr_sep = hr_sep + 0.2
            r = r + 1

        hours.append(night[1])
        self.hours = sp.array(hours)
        print(self.hours)

        dn = self.day["Midnight"]
        alts = []

        elevat = []

        for star in self.star_cat['Star info']:
            for hour in self.hours:
                obs.date = hour
                self.star.compute(obs)
                alt_val = self.star.alt
                elev = (hour, alt_val)
                # print(elev) -> (ep.date, alt)
                alts.append(elev)

            elevat.append(alts)
            alts = []

        self.star_cat['Elev'] = sp.array(elevat)
        print(self.star_cat['Elev'])

        return self

    # Restricciones para los transitos
    def set_restr_twil(self, bline=40, k=60, **kwargs):
        tr_beg_ar = []
        tr_end_ar = []

        for target, transit, length in zip(self.store['target'], self.store['transit'], self.store['length']):
            tr_beg = ep.Date(transit - length / 2 * ep.hour)
            tr_end = ep.Date(transit + length / 2 * ep.hour)
            print(target, ep.Date(tr_beg), ep.Date(transit), ep.Date(tr_end))
            tr_beg_ar.append(tr_beg)
            tr_end_ar.append(tr_end)

        self.store['Transit beg'] = sp.array(tr_beg_ar)
        self.store['Transit end'] = sp.array(tr_end_ar)

        # Twilight
        twil_beg = ep.Date(self.day["Sunset"] + 50 * ep.minute)
        twil_end = ep.Date(self.day["Sunrise"] - 50 * ep.minute)
        print(ep.Date(twil_beg), ep.Date(twil_end))

        tr_begnew_ar = []
        tr_endnew_ar = []

        bl_prenew_ar = []
        bl_posnew_ar = []

        tr_obs_ar = []
        tr_per_ar = []

        for target, tr_beg_real, tr_end_real, length in zip(self.store['target'], self.store['Transit beg'],
                                                            self.store['Transit end'], self.store['length']):
            bl_pre = ep.Date(tr_beg_real - bline * ep.minute)
            bl_pos = ep.Date(tr_end_real + bline * ep.minute)

            ed0 = ep.Date(self.day["Sunset"])
            ed1 = ep.Date(self.day["Sunset"] + 30 * ep.minute)

            ed2 = ep.Date(twil_beg + bline * ep.minute)
            ed3 = ep.Date(twil_end - bline * ep.minute)

            ed4 = ep.Date(self.day["Sunrise"] - 30 * ep.minute)
            ed5 = ep.Date(self.day["Sunrise"])

            if ed0 < ep.Date(tr_beg_real) < ed1 or ed4 < ep.Date(tr_end_real) < ed5:
                tr_beg_new = 0
                tr_end_new = 0
                tr_obs = 0
                tr_per = tr_obs / length * 100
                bl_pre_new = 0
                bl_pos_new = 0

            elif ed1 < ep.Date(tr_beg_real) < twil_beg:
                tr_beg_new = twil_beg
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs / length * 100
                bl_pre_new = 0
                bl_pos_new = 40

            elif twil_beg <= ep.Date(tr_beg_real) < ed2:
                tr_beg_new = tr_beg_real
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs / length * 100
                aux = ep.Date(tr_beg_new - twil_beg)
                (yy, mm, dd, hh, mn, sg) = aux.tuple()
                bl_pre_new = mn + sg / 60
                bl_pos_new = 40

            elif ed2 <= ep.Date(tr_beg_real) and ep.Date(tr_end_real) <= ed3:
                tr_beg_new = tr_beg_real
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs / length * 100
                bl_pre_new = 40
                bl_pos_new = 40

            elif ed3 < ep.Date(tr_end_real) < twil_end:
                tr_beg_new = tr_beg_real
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs / length * 100
                bl_pre_new = 40
                aux = ep.Date(twil_end - tr_end_real)
                (yy, mm, dd, hh, mn, sg) = aux.tuple()
                bl_pos_new = mn + sg / 60

            elif twil_end <= ep.Date(tr_end_real) < ed4:
                tr_beg_new = tr_beg_real
                tr_end_new = twil_end
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs / length * 100
                bl_pre_new = 40
                bl_pos_new = 0

            else:
                tr_beg_new = 0
                tr_end_new = 0
                tr_obs = 0
                tr_per = tr_obs / length * 100
                bl_pre_new = 0
                bl_pos_new = 0

            tr_begnew_ar.append(tr_beg_new)
            tr_endnew_ar.append(tr_end_new)

            bl_prenew_ar.append(bl_pre_new)
            bl_posnew_ar.append(bl_pos_new)

            tr_obs_ar.append(tr_obs)
            tr_per_ar.append(tr_per)

            print(target, tr_beg_new, tr_end_new, bl_pre_new, bl_pos_new)

        self.tr_begnew_twil = sp.array(tr_begnew_ar)
        self.tr_endnew_twil = sp.array(tr_begnew_ar)
        self.tr_obs_twil = sp.array(tr_obs_ar)
        self.tr_per_twil = sp.array(tr_per_ar)
        self.bl_prenew_twil = sp.array(bl_prenew_ar)
        self.bl_posnew_twil = sp.array(bl_posnew_ar)

        self.set_restr_elev()

        return self

    def set_restr_elev(self, minElev=30, bline=40, **kwargs):
        tr_begnew_ar2 = []
        tr_endnew_ar2 = []

        bl_prenew_ar2 = []
        bl_posnew_ar2 = []

        tr_obs_ar2 = []
        tr_per_ar2 = []

        # Limite de elevacion
        for line, tr_begtwil, tr_endtwil, length in zip(self.star_cat['Elev'], self.tr_begnew_twil, self.tr_endnew_twil,
                                                        self.store['length']):
            for elem in line:
                if elem[1] <= minElev and ep.Date(tr_begtwil) < elem[0]:  # No entiendo por que hace esta comparacion
                    tr_beg_new2 = elem[0]  # Desde donde partian los indices en python
                    tr_end_new2 = tr_endtwil
                    tr_obs2 = tr_end_new2 - tr_beg_new2
                    tr_per2 = tr_obs2 / length * 100
                    bl_pre_new2 = 0
                    bl_pos_new2 = ep.Date(tr_end_new2 + bline * ep.minute)

                elif elem[1] <= minElev and elem[0] < ep.Date(tr_endtwil):
                    tr_beg_new2 = tr_begtwil
                    tr_end_new2 = elem[0]
                    tr_obs2 = tr_end_new2 - tr_beg_new2
                    tr_per2 = tr_obs2 / length
                    bl_pre_new2 = ep.Date(tr_beg_new2 - bline * ep.minute)
                    bl_pos_new2 = 0

                elif elem[1] > minElev and elem[0] > ep.Date(tr_begtwil - bline * ep.minute):
                    tr_beg_new2 = tr_begtwil
                    tr_end_new2 = tr_endtwil
                    tr_obs2 = tr_end_new2 - tr_beg_new2
                    tr_per2 = tr_obs2 / length * 100
                    bl_pre_new2 = ep.Date(tr_beg_new2 - bline * ep.minute)
                    bl_pos_new2 = ep.Date(tr_end_new2 + bline * ep.minute)

            tr_begnew_ar2.append(tr_beg_new2)
            tr_endnew_ar2.append(tr_end_new2)

            bl_prenew_ar2.append(bl_pre_new2)
            bl_posnew_ar2.append(bl_pos_new2)

            tr_obs_ar2.append(tr_obs2)
            tr_per_ar2.append(tr_per2)

        self.store['Obs transit'] = sp.array(tr_obs_ar2)
        self.store['Per transit'] = sp.array(tr_per_ar2)
        self.store['Baseline pre'] = sp.array(bl_prenew_ar2)
        self.store['Baseline post'] = sp.array(bl_posnew_ar2)

        return self


prueba = observation_night('2016/03/20')
