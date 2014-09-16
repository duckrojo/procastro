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


import scipy as sp
import ephem
import pylab as pl

import  obsrv.obsplot as oplot
plot = reload(oplot)

__obsrv_version = '3.02'


def NonNoneArgs(func):
  """Function decorator that checks whether any of the arguments is None.
  Returns None if so, otherwise allows the original function"""

  def tocheck(*args,**kwargs):
    if None in args[1:]:
      return None
    else:
      return func(*args, **kwargs)

  return tocheck

def splitcoo(val, name):
  """Split coordinate to XX:XX:XX format"""
  if isinstance(val, str):
    vals = val.split(':')
    if len(vals)<3:
      raise ValueError, " %s string format must be XX:XX:XX and not: %s" \
          % (name, val)
  elif isinstance(val, (float,int)):
    valh = int(val)
    valm = int((val-valh)*60)
    vals = ((val-valh)*60-valm)*60
  else:
    raise ValueError, "Unrecognized format for %s: %s" % (name, val)
  
  return "%i:%i:%.2f" % (valh,valm,vals)



class obsrv(oplot.obsplot):


  def obsll(self, observer, obsfilename='./prg/python/obsrv/observatories.coo', **kwargs):
    """Checks whether name's coordinate are known from a list or whether it is a (lat, lon) tuple  """

    coordinates = {}
    for line in  open(obsfilename, 'r').readlines():
      field = line.split(',')
      coordinates[field[0]] = (eval(field[1]), eval(field[2]))

    if observer==None:
      return coordinates.keys()
    if isinstance(observer, str):
      try:
        latlon = coordinates[observer]
      except KeyError: 
        raise KeyError, "observer keyword is not defined. Valid values are " + \
            ', '.join(coordinates) + '.'
    elif isinstance(observer, tuple) and len(observer)==2: 
      latlon = observer
    else:
      raise TypeError, "object can only be a 2-component tuple or a string"
    self.latlon = latlon

    print "Latitude: ",latlon

    return self



  def setstarsite(self, radec, observer='paranal', epoch_coo=2000, magn=10, starname='', **kwargs):
    """Set star and site variables"""
    self.obsll(observer)

    obs = ephem.Observer()
    obs.lon = str(self.latlon[1])
    obs.lat = str(self.latlon[0])
    obs.elevation = 0
    obs.epoch = str(epoch_coo)
  #fraction of day to move from UT to local midday
    self.tolocalmidday = 0.5-self.latlon[1]/360.0
    self.obs = obs

    dec = splitcoo(radec[1], 'Dec')
    ra = splitcoo(radec[0], 'RA')
    self.star = ephem.readdb("%s,f,%s,%s,%.2f, %f" % 
                             (starname, ra, dec, magn, epoch_coo))
    self.sun = ephem.Sun()
    self.moon = ephem.Moon()

    return self


  def settimespan(self, timespan, daystep=5, **kwargs):
    """Set time span"""
    if isinstance(timespan, int):   #Year
    #times always at midnight (UT)
      ed0 = ephem.Date('%i/1/1' % (timespan,)) 
      ed1 = ephem.Date('%i/1/1' % (timespan+1,))
      ed = sp.arange(ed0,ed1,daystep)
      xlims = [ed[0]-ed0,ed[-1]-ed0]
    else:
      raise ValueError, "timespan must be a single integer (year), ..."

    self.ed = ed
    self.xlims = xlims

    return self


  def sunsetrise(self, **kwargs):
    """Compute sunsets and sunrises"""
    sunset=[]
    sunrise=[]
    twiset=[]
    twirise=[]
    obs = self.obs
    for day in self.ed:
    #sunrise/set calculated from local midday
      obs.date = day + self.tolocalmidday
      sunset.append(obs.next_setting(self.sun)-day)
      sunrise.append(obs.next_rising(self.sun)-day)
      obs.horizon = '-18:00'
      twiset.append(obs.next_setting(self.sun)-day)
      twirise.append(obs.next_rising(self.sun)-day)
      obs.horizon = '0:00'
    self.sunrise= sp.array(sunrise)
    self.twirise= sp.array(twirise)
    self.sunset= sp.array(sunset)
    self.twiset= sp.array(twiset)

    return self



  def setysamp(self, hourstep=0.2, **kwargs):
    ylims = [min(self.sunset)*24-0.5,max(self.sunrise)*24+0.5]
    self.hours = sp.arange(ylims[0],ylims[1],hourstep)
    self.ylims = ylims



  def getairmass(self, maxairmass=3.0, **kwargs):
    """Get airmass"""
    obs = self.obs
    hours=self.hours

    airmass = []
    for d in self.ed:
      cosz=[]
      for h in hours:
        obs.date = d+h/24.0
        self.star.compute(obs)
        cosz.append(sp.sin(self.star.alt))
      cosz=sp.array(cosz)
      cosz[cosz<(1.0/maxairmass)] = 1.0/maxairmass
      airmass.append(1.0/cosz)

    self.am = sp.array(airmass).transpose()

    return self


  def moondist(self, date):
    obs = self.obs
    obs.date = date
    self.star.compute(obs)
    self.moon.compute(obs)
    return sp.cos(self.star.dec)*sp.cos(self.moon.dec)*sp.cos(self.star.ra-self.moon.ra) \
        + sp.sin(self.star.dec)*sp.sin(self.star.dec)

        

  def getmoon(self, maxairmass=3.0, **kwargs):
    """Get moon distance"""
    obs = self.obs
    hours=self.hours

    cosdistance = []
    for d in self.ed:
      cosdist=[]
      for h in hours:
        cosdist.append(self.moondist(d+h/24.0))
      cosdistance.append(cosdist)

    self.cmn = sp.array(cosdistance).transpose()

    return self



  def gettransits(self, tr_epoch, tr_period, **kwargs):
    """ Calculate the transits """
    jd0 = ephem.julian_date(ephem.Date(self.ed[0]))
    jd1 = ephem.julian_date(ephem.Date(self.ed[-1]))
    ntransits = int((jd1-jd0)/tr_period)+2
    tr1 = tr_epoch+tr_period*int((jd0-tr_epoch)/tr_period+0.9)
    self.transits = tr1+sp.arange(ntransits)*tr_period
    self.htran = (self.transits-(sp.fix(self.transits-0.5)+0.5))*24

    if jd0 < tr_epoch:
      print "WARNING: Reference transit epoch is in the future. Are you certain that you are using JD?"

    self.jd0 = jd0

    return self



  def __init__(self,radec, timespan=2013, name='paranal', 
               tr_hlen=2, **kwargs):

    self.tr_hlen = tr_hlen
#    self.obsll(name, **kwargs)
    self.setstarsite(radec, observer=name, **kwargs)
    self.settimespan(timespan)
    self.sunsetrise(**kwargs)
    self.setysamp(**kwargs)
    self.getairmass(**kwargs)
    self.getmoon(**kwargs)

    self.get_plot(kwargs['tr_epoch'],kwargs['tr_period'],
                  interact=kwargs['interact'])


##TD: Change django so that it uses get_plot(epoch, per, django=True) instead of get_plot_figure

  @NonNoneArgs
  def overlay_transit(self, tr_epoch, tr_period, **kwargs):
    print tr_epoch, tr_period
    self.gettransits(tr_epoch, tr_period, **kwargs)
    self.plottransits(self.transits-self.jd0, self.htran, self.tr_hlen)

  def get_plot(self, tr_epoch, tr_period, 
               interact=False, django=False, **kwargs):
    pl.clf()
    if django:
      interact=False

    self.plot(interact, **kwargs)
    self.overlay_transit(tr_epoch, tr_period, **kwargs)
    if django:
      return plot_fig

    pl.show()



  def get_transit_figure(self, cd, ch, tr_epoch, tr_period, **kwargs):

    self.gettransits(tr_epoch, tr_period, **kwargs)
    night_fig = self.onclick_coord(cd,ch,return_figure=True)
    return night_fig


  def get_transit_timestamp(self, cd, ch, tr_epoch, tr_period, **kwargs):

    self.gettransits(tr_epoch, tr_period, **kwargs)
    timestamp = self.get_timestamp(cd,ch)
    return timestamp


#def obsrv(radec, observer='paranal', tr_period=None, tr_epoch=None, tr_hlen=0, ut_offset=0, 
#          timespan=2013, save=None, title=None, magn=10, epoch_coo=2000,
#          showmonths=True, daystep=5, hourstep=0.2):
  """
    Plots the observability window of an astronomical object from a
    given observatory.

    Parameters:
    -----------
    observer: Tuple or Scalar
         Either a tuple indicating observer's location (lat, long) or the
         keyword for a known location (None lists them) 
    radec:    Tuple
         The target's RA and DEC coordinates in a tuple (hours and degrees,
         respectively). Either a float or a 'hh:mm:ss' string
    t_period: Scalar
              Orbital period of the planet (days).
    t_epoch:  Scalar
              Time of transit (or eclipse) epoch (JD-2.450.000).
    t_hlen:   Scalar
              Half-length of the transit duration (hours).
    univtime: Boolean
              If True plot time in universal time. Else plot in hours
              from local midnight.
    jd0: Scalar
         JD-2.450.000 of the first day to be plotted. Default: first
         day of Placefile.
    ndays: Scalar
           Number of days to plot. Default: from jd0 to last day in placefile.
    placegmt: Scalar
              Time correction to GMT in hours. Default is -4 (Chile).
    save: String
          If != None, save the plot to file with name given by save.
    magn: float
          Stellar magnitude


"""

















if __name__ == '__MAIN__':
    import obsrv
    target=['HD83443',9+(37 + 11.82841/60)/60.0, -(43+ (16+ 19.9354/60)/60.0),
            2455943.20159650, 2.985, 3]
    target=['WASP-34', 11+(1+36/60.0)/60.0, -(23+( 51+38/60)/60), 
            2454647.55358, 4.3176782,3]
    a=reload(obsrv).obsrv((target[1],target[2]),
                          tr_period=target[4],
                          tr_epoch=target[3],title=target[0])
