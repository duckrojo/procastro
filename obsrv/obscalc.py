#
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
import astropy as ap
import astropy.coordinates as apc
import astropy.units as u
import os
import xml.etree.ElementTree as ET, urllib, gzip, io
import dataproc as dp


def _update_airmass(func):
  def wrapper(self, *args, **kwargs):
    ret = func(self, *args, **kwargs)
    if hasattr(self,'star'):
      self._get_airmass()

    return ret

  return wrapper

def _update_transits(func):
  def wrapper(self, *args, **kwargs):
    ret = func(self, *args, **kwargs)
    if hasattr(self,'transit_info'):
      self.set_transits()

    return ret

  return wrapper



class ObsCalc(object):

  def __init__(self, timespan=2015, target=None,
               site='paranal', 
               only_night=True, equinox=2000,
               **kwargs):
    """ =Initializes obsrv class.

    :param central_time: Central time of y-axis.  If outside the [-12,24] range, then only shows nighttime
    """

    self._obs = ephem.Observer()
    self._sun  = ephem.Sun()
    if not hasattr(self, 'params'):
      self.params = {}
    self.params["equinox"] = equinox
    self.daily = {}

    self.set_site(site, timespan, **kwargs)

    print (kwargs)

    if target is not None:
      self.set_target(target, **kwargs)


  def set_site(self, site, timespan=None,
                   obsfilename=None, **kwargs):
    """Checks whether name's coordinate are known from a list or whether it is a (lat, lon) tuple  

    :param site: identifies the observatory as a name, or (lat, lon) tuple, 
"""

    obs = self._obs

    if obsfilename is None:
      obsfilename = os.path.dirname(__file__)+'/observatories.coo'

    coordinates = {}
    for line in  open(obsfilename, 'r').readlines():
      field = line.split(',')
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


    ## Parameters for user-friendly values
    self.params["latlon"] = latlon
    self.params["site"] = site
    #fraction of day to move from UT to local midday
    self.params["to_local_midday"] = 0.5-latlon[1]/360.0

    ##parameters in pyephem
    obs.lon = str(latlon[1])
    obs.lat = str(latlon[0])

    obs.elevation = 0
    obs.epoch = str(self.params["equinox"])

    print ("Selected (lat, lon): %s" % (latlon,))
    self.set_timespan(timespan,  **kwargs)

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
        #TODO: default to next year!
        raise ValueError("Timespan needs to be specified")
    else:
      self.params["timespan"] = timespan

    if isinstance(timespan, int):   #Year
    #times always at midnight (UT)
      ed0 = ephem.Date('%i/1/1' % (timespan,)) 
      ed1 = ephem.Date('%i/1/1' % (timespan+1,))
      ed = sp.arange(ed0,ed1,int((ed1-ed0)/samples))
      xlims = [ed[0]-ed0,ed[-1]-ed0]
    else:
      raise NotImplementedError( """Requested timespan (%s) not implemented yet. Currently supported:
                 * single integer (year)
""" % (timespan,))
    
    self.jd0 = ephem.julian_date(ephem.Date(ed[0]))
    self.days = ed
    self.xlims = xlims

    self._get_sunsetrise(**kwargs)
    self.set_vertical(central_time)

    return self


  def _get_sunsetrise(self, **kwargs):
    """Compute sunsets and sunrises"""
    sunset=[]
    sunrise=[]
    twiset=[]
    twirise=[]
    obs = self._obs
    for day in self.days:
    #sunrise/set calculated from local midday
      obs.date = day + self.params["to_local_midday"]
      sunset.append(obs.next_setting(self._sun)-day)
      sunrise.append(obs.next_rising(self._sun)-day)
      obs.horizon = '-18:00'
      twiset.append(obs.next_setting(self._sun)-day)
      twirise.append(obs.next_rising(self._sun)-day)
      obs.horizon = '0:00'
    self.daily["sunrise"] = sp.array(sunrise)
    self.daily["twirise"] = sp.array(twirise)
    self.daily["sunset"]  = sp.array(sunset)
    self.daily["twiset"]  = sp.array(twiset)

    return self



  @_update_airmass
  def set_vertical(self, central_time, hour_step=0.2, **kwargs):
    if central_time>24 or central_time<-12:
      ylims = [min(self.daily["sunset"])*24-0.5,max(self.daily["sunrise"])*24+0.5]
      self.hours = sp.arange(ylims[0],ylims[1],hour_step)
      self.ylims = ylims
    else:
      raise NotImplementedError("centering at times different than middle of night is not supported yet")

    return self


  @_update_airmass
  @_update_transits
  def set_target(self, target, magn=10, starname='', 
                 tr_epoch=None, tr_period=None, tr_length=1,
                 home_transit = True, 
                 **kwargs):
    """Set star and site into pyephem

    :param target: either RA and Dec in hours and degrees, or target name to be queried
    """

    self.params['target'] = target
    if 'current_transit' in self.params:
      del self.params['current_transit']

    radec = dp.read_coordinates(target, 
                                coo_files = [os.path.dirname(__file__)+'/coo.txt',
                                             os.path.expanduser("~")+ '/.coostars'],
                                equinox=self.params["equinox"])


    print("Star at RA/DEC: %s/%s" %(radec.ra.to_string(sep=':'),
                                    radec.dec.to_string(sep=':')))
    self.star = ephem.readdb("%s,f,%s,%s,%.2f, %f" % 
                             (starname, 
                              radec.ra.to_string(sep=':', unit=u.hour), 
                              radec.dec.to_string(sep=':'), 
                              magn, radec.equinox))

    transit_filename_locations = [os.path.expanduser("~")+'/.transits',
                                  os.path.dirname(__file__)+'/transits.txt',]

    for transit_filename in transit_filename_locations:
      try:
        open_file = open(transit_filename)

        override = []
        for line in open_file.readlines():


          if line[0]=='#' or len(line)<3:
            continue
          data = line[:-1].split()
          planet = data.pop(0).replace('_', ' ')

          if planet.lower() == target.lower():
            for d in data:
              if d[0].lower()=='p':
                override.append('period')
                tr_period = float(eval(d[1:]))
              elif d[0].lower()=='e':
                override.append("epoch")
                tr_epoch  = float(eval(d[1:]))
              elif d[0].lower()=='l':
                override.append("length")
                tr_length = float(eval(d[1:]))
              elif d[0].lower()=='c':
                override.append("comment")
              else:
                raise ValueError("data field not understood, it must start with L, P, C, or E:\n%s" % (line,))
            print ("Overriding for '%s' from file '%s':\n %s" % (planet, 
                                                                 transit_filename,
                                                                 ', '.join(override),))

        if len(override):
          break

      except IOError:
        pass

    if tr_epoch is None or tr_period is None:
      print("Attempting to query transit information")
      url = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"
      oec = ET.parse(gzip.GzipFile(fileobj=io.BytesIO(urllib.urlopen(url).read())))
      query = oec.find(".//planet[name='%s']" % (target,))
      if query is None:
        print ("  not found")
        tr_period = tr_epoch = 0
      else:
        if tr_period is None:
          tr_period = float(query.findtext('period'))
        if tr_epoch is None:
          tr_epoch  = float(query.findtext('transittime'))
      print ("  Found ephemeris: %f + E*%f" % (tr_epoch, tr_period))

    if tr_period != 0: 
      self.transit_info = {'length': tr_length,
                           'epoch' : tr_epoch,
                           'period': tr_period}
    else:
      self.set_transits(tr_period=0)

    return self



  def set_transits(self,
                   tr_period=None, tr_epoch=None, tr_length=None,
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
      tr_epoch  = self.transit_info['epoch']
    if tr_length is None:
      tr_length = self.transit_info['length']

    jd0 = self.jd0
    jd1 = ephem.julian_date(ephem.Date(self.days[-1]))
    ntransits = int((jd1-jd0)/tr_period)+2
    tr1 = tr_epoch+tr_period*int((jd0-tr_epoch)/tr_period+0.9)
    self.transits = tr1+sp.arange(ntransits)*tr_period
    self.transit_hours = (self.transits-(sp.fix(self.transits-0.5)+0.5))*24

    if jd0 < tr_epoch:
      print("WARNING: Reference transit epoch is in the future. Are you certain that you are using JD?")


    return self



  def _get_airmass(self, maxairmass=3.0, **kwargs):
    """Get airmass"""
    obs   = self._obs
    hours = self.hours

    airmass = []
    for d in self.days:
      alts=[]
      for h in hours:
        obs.date = d+h/24.0
        self.star.compute(obs)
        alts.append(self.star.alt)

      cosz = sp.sin(sp.array(alts))
      cosz[cosz<(1.0/maxairmass)] = 1.0/maxairmass
      airmass.append(1.0/cosz)

    self.airmass = sp.array(airmass).transpose()

    return self



 
