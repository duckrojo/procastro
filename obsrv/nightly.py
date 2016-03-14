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

class obsnight(object):
    # Inicializacion
    def __init__(self,date, site='paranal', equinox=2000, **kwargs):
        self.date = str(date)
        self._obs = ep.Observer()
        self._sun = ep.Sun()

        self.params = {}
        self.params["equinox"] = equinox
        self.day = {}
        self.store = {}
        self.star_cat = {}

        self.set_loc(site, date)

        print (kwargs)

    # Observer place
    def set_loc(self, site, date, obsfile=None, **kwargs):

        obs=self._obs

        if obsfile is None:
            obsfile = os.path.dirname(os.path.abspath(__file__))+'/observatories.coo'

        coordinates = {}
        for line in  open(obsfile, 'r').readlines():
            field=line.split(',')
            coordinates[field[0]] = (eval(field[1]), eval(field[2]))

        if site==None:
            return coordinates.keys()

        if isinstance(site, str):
            try:
                latlon = coordinates[site]
            except KeyError:
                raise KeyError("site keyword is not defined. Valid values are " + \
            ', '.join(coordinates) + '.')
        elif isinstance(site, tuple) and len(site)==2:
            latlon = site

        else:
            raise TypeError("object can only be a 2-component tuple or a string")

        # Parameters for user-friendly values
        self.params["site"] = site
        self.params["latlon"] = latlon

        # fraction of day to move from UT to local midday
        self.params["local_time"] = 0.5 - latlon[1]/360.0

        # parameters in pyephem
        obs.lat = str(latlon[0])  
        obs.lon = str(latlon[1])

        obs.elevation = 0
        obs.epoch = str(self.params["equinox"])

        print ("Selected (lat, lon): %s" % (latlon,))
        self.set_date(date)

        return self

    # Fecha de observacion forma                                       
    def set_date(self, date, **kwargs):
        lar = len(date)
        # Condiciones de formato (para meses o dias menores a 10 se puede anteponer 0)
        if 7 < lar < 11:
            ind = date.find('/')
            # Condiciones de separacion                              
            if ind != 4:
                raise TypeError("Date format: yyyy/mm/dd %s %s")
            elif ind==4:
                day_obs = ep.Date(date)
                day_obs_plus = ep.Date(day_obs + 1)
            # Otra forma de separacion                                                       
            elif date[4].isdigit()==False:
                # Formato permitido                                                          
                sep = date.split(date[4])
                date2 = "/".join(sep)
                day_obs = ep.Date(date2)
                day_obs_plus = ep.Date(day_obs + 1)
            else:
                raise TypeError("Date format: yyyy/mm/dd")
            
            day_jd = ep.julian_date(date)
            # medianoche local UT
            midnight = ep.Date(day_obs_plus + 0.5 + self.params["local_time"])
            midnight_jd = ep.julian_date(midnight)
            # mediodia local UT                                                        
            midday = ep.Date(day_obs_plus + self.params["local_time"])
            midday_jd=ep.julian_date(midday)
            
        else:
            raise TypeError("Date format: yyyy/mm/dd")

        self.day["Obs date"] = date
        self.day["julian date"] = day_jd
        self.day["Midday"] = midday
        self.day["Midnight"] = midnight
        
        print ("Mediodia %s y Medianoche %s" % (midday,midnight))
        print ("Fecha de observacion %s" % (date,))

        self.set_sunsetrise(date,**kwargs)
        self.set_target(**kwargs)
        
        return self

    # Twilights: sunsets and sunrises                                                        
    def set_sunsetrise(self, date, **kwargs):
        obs = self._obs

        obs.date = self.day["Midnight"]
        self.day["Sunset"] = obs.previous_setting(self._sun)
        self.day["Sunrise"] = obs.next_rising(self._sun)

        print ("Sunrise %s and sunset %s at %s" % (self.day["Sunrise"],self.day["Sunset"],self.day["Obs date"]))

        return self

    # Del catalogo
    def set_target(self, magn=10, starname='', epoch=None, period=None, length=1, **kwargs):
        catalogue = os.path.dirname(os.path.abspath(__file__))+'/test_input_catalogue.dat'

        target = []
        period_tr = []
        epoch_tr = []
        length_tr = []

        radec = []
        star_info = []
        
        for lines in open(catalogue, 'r').readlines():
            texto = lines.split('\r')
            for line in texto:
                data = line.split('\t')
                planet = data[0]
                target.append(planet)
                radec_ra = data[1]
                radec_dec = data[2]
                radec_tup = (data[1],data[2])
                
                equinox = self.params["equinox"]
                #print("Star at RA,DEC: (%s,%s)" %(radec_ra,radec_dec))
                self.star = ep.readdb("%s,f,%s,%s,%.2f,%f" % (starname,radec_ra,radec_dec,magn,equinox))
            
                for d in data[3:]:
                    if d[0].lower()=='p':
                        period = float(eval(d[1:]))
                    elif d[0].lower()=='e':
                        epoch = float(eval(d[1:]))
                    elif d[0].lower()=='l':
                        length = float(eval(d[1:]))
                    #elif d[0].lower()=='c':
                        #override.append("comment")
                    else:
                        raise ValueError("Params must start with L, P, C, \or E:\n%s" % (data,))

                period_tr.append(period)
                epoch_tr.append(epoch)
                length_tr.append(length)

                radec.append(radec_tup)
                star_info.append(self.star)

                #print ("Transit information of %s is: period: %s, epoch: %s, length: %s, equinox: %s" % (planet,period, epoch, length,equinox))
                   
        self.store['target'] = sp.array(target)
        self.store['period'] = sp.array(period_tr)
        self.store['epoch'] = sp.array(epoch_tr)
        self.store['length'] = sp.array(length_tr)
        self.star_cat['RA,DEC'] = sp.array(radec)
        self.star_cat['Star info'] = sp.array(star_info)


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

        for target,epoch,period,length in zip(self.store['target'],self.store['epoch'],self.store['period'],self.store['length']):
            jd = ep.julian_date(ep.Date(self.day["Midnight"]))            
            tr_at_midn = ((jd - epoch)/period)
            phase = tr_at_midn%1
            closest_tr = period*int(tr_at_midn+0.5)+epoch
            #closest_tr = period*int(tr_at_midn)+epoch + (phase>0.5)
            transit.append(ep.Date(closest_tr - 2415020))
            print(target,ep.Date(closest_tr - 2415020))

        self.store['transit'] = sp.array(transit)
        print(self.store['transit'])

        self.set_restr_twil()

        return self

    # Curva
    def set_curve(self,hr_sep=0.2,**kwargs):
        obs = self._obs
        night = [ep.Date(self.day["Sunset"] - 0.5*ep.hour), ep.Date(self.day["Sunrise"] + 0.5*ep.hour)]

        lar = int((ep.Date(night[1]) - ep.Date(night[0]))*24/hr_sep)
        hours = []
        r = 0
        hr_sep = 0
        while r < lar:
            val = ep.Date(night[0] + hr_sep*ep.hour)
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
                elev = (hour,alt_val)
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

        for target,transit,length in zip(self.store['target'],self.store['transit'],self.store['length']):
            tr_beg = ep.Date(transit - length/2*ep.hour)
            tr_end = ep.Date(transit + length/2*ep.hour)
            print(target,ep.Date(tr_beg),ep.Date(transit),ep.Date(tr_end))
            tr_beg_ar.append(tr_beg)
            tr_end_ar.append(tr_end)
    
        self.store['Transit beg'] = sp.array(tr_beg_ar)
        self.store['Transit end'] = sp.array(tr_end_ar)

        # Twilight                                                                   
        twil_beg  = ep.Date(self.day["Sunset"] + 50*ep.minute)
        twil_end  = ep.Date(self.day["Sunrise"] - 50*ep.minute)
        print(ep.Date(twil_beg),ep.Date(twil_end))

        tr_begnew_ar = []
        tr_endnew_ar = []

        bl_prenew_ar = []
        bl_posnew_ar = []

        tr_obs_ar = []
        tr_per_ar = []

        for target, tr_beg_real, tr_end_real, length in zip(self.store['target'],self.store['Transit beg'],self.store['Transit end'],self.store['length']):
            bl_pre = ep.Date(tr_beg_real - bline*ep.minute)
            bl_pos = ep.Date(tr_end_real + bline*ep.minute)

            ed0 = ep.Date(self.day["Sunset"])
            ed1 = ep.Date(self.day["Sunset"] + 30*ep.minute)

            ed2 = ep.Date(twil_beg + bline*ep.minute)
            ed3 = ep.Date(twil_end - bline*ep.minute)

            ed4 = ep.Date(self.day["Sunrise"] - 30*ep.minute)
            ed5 = ep.Date(self.day["Sunrise"])

            if ed0 < ep.Date(tr_beg_real) < ed1 or ed4 < ep.Date(tr_end_real) < ed5:
                tr_beg_new = 0
                tr_end_new = 0
                tr_obs = 0
                tr_per = tr_obs/length*100
                bl_pre_new = 0
                bl_pos_new = 0

            elif ed1 < ep.Date(tr_beg_real) < twil_beg:
                tr_beg_new = twil_beg
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs/length*100
                bl_pre_new = 0
                bl_pos_new = 40

            elif twil_beg <= ep.Date(tr_beg_real) < ed2:
                tr_beg_new = tr_beg_real
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs/length*100
                aux = ep.Date(tr_beg_new - twil_beg)
                (yy,mm,dd,hh,mn,sg) = aux.tuple()
                bl_pre_new = mn + sg/60
                bl_pos_new = 40

            elif ed2 <= ep.Date(tr_beg_real) and ep.Date(tr_end_real) <= ed3:
                tr_beg_new = tr_beg_real
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs/length*100
                bl_pre_new = 40
                bl_pos_new = 40

            elif ed3 < ep.Date(tr_end_real) < twil_end:
                tr_beg_new = tr_beg_real
                tr_end_new = tr_end_real
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs/length*100
                bl_pre_new = 40
                aux = ep.Date(twil_end - tr_end_real)
                (yy,mm,dd,hh,mn,sg) = aux.tuple()
                bl_pos_new = mn + sg/60

            elif twil_end <= ep.Date(tr_end_real) < ed4:
                tr_beg_new = tr_beg_real
                tr_end_new = twil_end
                tr_obs = tr_end_new - tr_beg_new
                tr_per = tr_obs/length*100
                bl_pre_new = 40
                bl_pos_new = 0

            else:
                tr_beg_new = 0
                tr_end_new = 0
                tr_obs = 0
                tr_per = tr_obs/length*100
                bl_pre_new = 0
                bl_pos_new = 0

            tr_begnew_ar.append(tr_beg_new)
            tr_endnew_ar.append(tr_end_new)

            bl_prenew_ar.append(bl_pre_new)
            bl_posnew_ar.append(bl_pos_new)

            tr_obs_ar.append(tr_obs)
            tr_per_ar.append(tr_per)

            print(target,tr_beg_new,tr_end_new,bl_pre_new,bl_pos_new)

        self.tr_begnew_twil = sp.array(tr_begnew_ar)
        self.tr_endnew_twil = sp.array(tr_begnew_ar)
        self.tr_obs_twil = sp.array(tr_obs_ar)
        self.tr_per_twil = sp.array(tr_per_ar)
        self.bl_prenew_twil = sp.array(bl_prenew_ar)
        self.bl_posnew_twil = sp.array(bl_posnew_ar)

        self.set_restr_elev()
        
        return self

    def set_restr_elev(self,minElev=30, bline=40,**kwargs):
        tr_begnew_ar2 = []
        tr_endnew_ar2 = []

        bl_prenew_ar2 = []
        bl_posnew_ar2 = []

        tr_obs_ar2 = []
        tr_per_ar2 = []

        # Limite de elevacion
        for line, tr_begtwil, tr_endtwil, length in zip(self.star_cat['Elev'],self.tr_begnew_twil, self.tr_endnew_twil,self.store['length']):
            for elem in line:
                if elem[1] <= minElev and ep.Date(tr_begtwil) < elem[0]:
                    tr_beg_new2 = elem[0]
                    tr_end_new2 = tr_endtwil
                    tr_obs2 = tr_end_new2 - tr_beg_new2
                    tr_per2 = tr_obs2/length*100
                    bl_pre_new2 = 0
                    bl_pos_new2 = ep.Date(tr_end_new2 + bline*ep.minute)

                elif elem[1] <= minElev and elem[0] < ep.Date(tr_endtwil):
                    tr_beg_new2 = tr_begtwil
                    tr_end_new2 = elem[0]
                    tr_obs2 = tr_end_new2 - tr_beg_new2
                    tr_per2 = tr_obs2/length
                    bl_pre_new2 = ep.Date(tr_beg_new2 - bline*ep.minute)
                    bl_pos_new2 = 0

                elif elem[1] > minElev and elem[0] > ep.Date(tr_begtwil - bline*ep.minute):
                    tr_beg_new2 = tr_begtwil
                    tr_end_new2 = tr_endtwil
                    tr_obs2 = tr_end_new2 - tr_beg_new2
                    tr_per2 = tr_obs2/length*100
                    bl_pre_new2 = ep.Date(tr_beg_new2 - bline*ep.minute)
                    bl_pos_new2 = ep.Date(tr_end_new2 + bline*ep.minute)

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


prueba = obsnight('2016/03/20')
