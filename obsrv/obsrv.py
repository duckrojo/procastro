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


from __future__ import print_function, division
import scipy as sp
import ephem
import matplotlib.pyplot as plt
import astropy as ap
from matplotlib.patches import Polygon

from . import obscalc as ocalc


def _plotpoly(ax, x, up, down, alpha=0.5, facecolor='0.8', edgecolor='k'):
    """ Plot a band  as semi-transparent polygon """
    verts = list(zip(x, down*24)) + list(zip(x[::-1], (up*24)[::-1]))
    poly = Polygon(verts, 
                   facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(poly)

# def NonNoneArgs(func):
#   """Function decorator that checks whether any of the arguments is None.
#   Returns None if so, otherwise allows the original function"""

#   def tocheck(*args,**kwargs):
#     if None in args[1:]:
#       return None
#     else:
#       return func(*args, **kwargs)

#   return tocheck




def _update_plot(func):

  def wrapper(self, *args, **kwargs):

    ret = func(self, *args, **kwargs)

    if 'plot_figure' in self.params:
      fig = self.params['plot_figure']
    else:
      fig = plt.figure(figsize=(10,5))
      if self.params['interact']:
        self.cid = [fig.canvas.mpl_connect('button_press_event', self._onclick), 
                    fig.canvas.mpl_connect('key_press_event', self._onkey)]
      self.params['plot_figure'] = fig

    fig.clf()
#    cax = ax.figure.add_axes(cbrect)
    self.params["plot_ax-airm"] = ax_airm = fig.add_axes([0.06,0.1,0.35,0.85])#subplot(121)
    self.params["plot_ax-elev"] = ax_elev = fig.add_axes([0.55,0.1,0.4,0.83])#fig.add_subplot(122)
    #delete right y-axis of the transit night if exists since fig.clf() disconnects it.
    if 'plot_ax-elev2' in self.params:
        del self.params['plot_ax-elev2']

    if hasattr(self,'airmass'):
      self._plot_airmass(ax_airm)
      if self.params["show_colorbar"]:
        self._plot_labels(ax_airm)
    if self.params["show_twilight"]:
      self._plot_twilight(ax_airm)
    if self.params["show_months"]:
      self._plot_months(ax_airm)
    if self.params["show_transits"] and hasattr(self, 'transits'):
      self._plot_transits(ax_airm)

    ax_airm.set_ylim(self.ylims)
    ax_airm.set_xlim(self.xlims)

    fig.show()

    return ret

  return wrapper




class Obsrv(ocalc.ObsCalc):

  def __init__(self, 
               target=None, show_twilight=True, show_months=True,
               show_colorbar=True, show_transits=True,
               interact=True, savedir='fig/',
               **kwargs):
    """ Initializes obsrv class.

    :param central_time: Central time of y-axis.  If outside the [-12,24] range, then only shows nighttime
    """
    if not hasattr(self, 'params'):
      self.params = {}

    self.params["interact"] = interact
    self.params["show_twilight"] = show_twilight
    self.params["show_months"]   = show_months
    self.params["show_transits"] = show_transits
    self.params["show_colorbar"] = show_colorbar
    self.params["savedir"] = savedir

    super(Obsrv, self).__init__(target=target, **kwargs)


  # @_update_plot
  # def set_observer(self, *args, **kwargs):
  #   super(Obsrv, self).set_observer(*args, **kwargs)

  @_update_plot
  def set_target(self, *args, **kwargs):
    super(Obsrv, self).set_target(*args, **kwargs)

  @_update_plot
  def set_vertical(self, *args, **kwargs):
    super(Obsrv, self).set_vertical(*args, **kwargs)

  @_update_plot
  def set_transits(self, *args, **kwargs):
    super(Obsrv, self).set_transits(*args, **kwargs)

  # @_update_plot
  # def set_timespan(self, *args, **kwargs):
  #   super(Obsrv, self).set_timespan(*args, **kwargs)


  def _plot_airmass(self, ax):

    ams = sp.arange(1, 3.01, 0.1)
    self.params["plot_airmass"] = ax.contourf(self.days-self.days[0], self.hours,
                                              sp.array(self.airmass),
                                              levels=ams,
                                              cmap=plt.cm.jet_r)


  def _plot_twilight(self, ax):

    sx = self.days-self.days[0]
    _plotpoly(ax, sx, self.daily["twiset"],  self.daily["sunset"])
    _plotpoly(ax, sx, self.daily["sunrise"], self.daily["twirise"])


  def _plot_months(self, ax):

    """Plot month separator"""

    mnlen = sp.array([31,28,31,30,31,30,31,31,30,31,30,31])
    cum = mnlen.copy()
    for i in range(len(mnlen))[1:]:
      cum[i:] += mnlen[:-(i)]
    mcum = sp.array(list(zip(cum,cum))).flatten()
    vlims = list(ax.get_ylim())
    vlims = (vlims + vlims[::-1])*6
    # vlims = [-10,20,20,-10]*6

    y,m,d = ephem.Date(self.days[0]).triple()
    jan1 = self.days[0]-((m-1!=0)*cum[m-2]+d-1)
    ax.plot(self.days[0]-jan1 + mcum, vlims, 'y--')
    ny=(self.days[-1]-jan1)//365
    for i in range(int(ny)):
      ax.plot(i*365 + self.days[0]-jan1 + mcum, vlims, 'y--')

    return self


  def _plot_transits(self, ax, **kwargs):
    x = self.transits-self.jd0
    y = self.transit_hours
    hlen = self.transit_info['length']/2

    ax.errorbar(x, y, yerr=hlen, fmt='o', color="w")
    ax.errorbar(x, y-24, yerr=hlen, fmt='o', color="w")


  def _plot_labels(self, ax, title='', **kwargs):

    ax.set_title(title)
    ax.set_xlabel(r"Days from %s (Site: %s)" % ( ephem.Date(self.days[0]).datetime().strftime('%Y.%m.%d'), self.params['site']))
    ax.set_ylabel(r'Time (UT). Target: %s' % (self.params['target'],))
    ax.set_xlim(self.xlims)
    ax.set_ylim(sp.array(self.ylims))
    cbrect = [0.45, 0.20, 0.035,  0.7]
    cax = ax.figure.add_axes(cbrect)
    ax.figure.colorbar(self.params["plot_airmass"], cax=cax)
    cax.yaxis.set_label_position('left')
    cax.yaxis.set_ticks_position('left')
#    cax.set_ylabel("Airmass")

    ax.text(1.12, 0.04, "Min.", transform = ax.transAxes, ha="center")
    ax.text(1.12, 0.005, '%.2f' % min(self.airmass.flatten()), transform = ax.transAxes, ha="center")

    return self


  def _plot_elev_transit(self, day, hour):

    if not hasattr(self, "transits"):
      return

    ax = self.params["plot_ax-elev"]
    closejd = self.get_closer_transit(day, hour)

    return self._plot_night(closejd, ax)


  def get_closer_transit(self,cd,ch):

    xovery = 5.0/4.0
    dx = self.xlims[1]-self.xlims[0]
    dy = self.ylims[1]-self.ylims[0]

    dist = []
    for htr, dtr in zip (self.transit_hours, self.transits):
        dist.append(sp.sqrt(((cd+self.jd0-dtr)*xovery/dx)**2
                          + ((ch-htr)/dy)**2))
        dist.append(sp.sqrt(((cd+self.jd0-dtr)*xovery/dx)**2
                          + ((ch-(htr-24))/dy)**2))
    return sp.array(list(zip(self.transits,self.transits))).flatten()[sp.argmin(dist)]


   
  def _plot_night(self, jd, ax, rect=None):

    if rect is None:
      rect = [0.1,0.1,0.8,0.8]
    ax.cla()
    #todo: ax_elev2 should be always initialized
    if 'plot_ax-elev2' in self.params:
        self.params['plot_ax-elev2'].cla()

    moon = ephem.Moon()
    obs  = self._obs
    obs.date = jd - self.jd0 + self.days[0]
    midday = obs.previous_transit(self._sun)
    obs.date = midday
    ss = obs.next_setting(self._sun)
    sr = obs.next_rising(self._sun)
    obs.horizon = '-18:00'
    ts = obs.next_setting(self._sun)
    tr = obs.next_rising(self._sun)
    obs.horizon = '0:00'
    hours = sp.arange(ss-0.03, sr+0.03, 0.007)

    moonalt=[]
    staralt=[]
    for h in hours:
      obs.date=h
      moon.compute(obs)
      moonalt.append(moon.alt*180/sp.pi)
      self.star.compute(obs)
      staralt.append(self.star.alt*180/sp.pi)

    etout=sp.fix(hours[0])+0.5
    uthours = (hours - etout)*24

    ax2 = ax.twinx()
    if 'plot_ax-elev2' in self.params:
        self.params['plot_figure'].delaxes(self.params['plot_ax-elev2'])
    self.params['plot_ax-elev2'] = ax2

    ax.plot(uthours, staralt)
    ax.plot(uthours, moonalt, '--')
    setev=sp.array([ss,ss,ts,ts])
    risev=sp.array([sr,sr,tr,tr])
    ax.plot((setev - etout)*24, [0,90,90,0], 'k:')
    ax.plot((risev - etout)*24, [0,90,90,0], 'k:')
    ax.set_ylim([10,90])
    ax.set_xlim([(ss-etout)*24-0.5,(sr-etout)*24+0.5])
    datetime = ephem.date(jd-self.jd0+self.days[0])
    print("%s %s" % (str(datetime)[:-3], jd))
    ax.set_title('%s' % str(datetime)[:-3])
    ax.set_ylim(ax.get_ylim())
    sam = sp.array([1,1.5,2,3,4,5])
    ax2.set_yticks(sp.arcsin(1.0/sam)*180.0/sp.pi)
    ax2.set_yticklabels(sam)
    self.params['current_transit'] = str(datetime).replace(' ', '_')
    self.params['current_moon_distance'],self.params['current_moon_phase'] = self._moon_distance(datetime)
    ax.set_ylabel('Elevation')
    ax.set_xlabel('UT time. Moon distance and phase: %s${^\degree}$ %.0f%%' %
                  (int(self.params["current_moon_distance"].degree),
                   float(self.params["current_moon_phase"])))

    if hasattr(self, 'transits'):
      intr = (jd-self.jd0+self.days[0]-etout)*24 - self.transit_info['length']/2
      outr = (jd-self.jd0+self.days[0]-etout)*24 + self.transit_info['length']/2
      if intr>outr:
        outr+=24
      facecolor = '0.5'
      if self.params['current_moon_distance'].degree<30:
          facecolor='orange'
      if self.params['current_moon_distance'].degree<10:
          facecolor='red'
      _plotpoly(ax, [intr,outr],
               [0,0], [90,90], facecolor=facecolor)

    ax.figure.canvas.draw()

    return self


  def _onkey(self, event):

    axe = self.params["plot_ax-elev"]
    axa = self.params["plot_ax-airm"]

    if event.key=='e' and event.inaxes == axa:  #at position
      self._plot_night(event.xdata + self.jd0, axe)
    elif event.key=='P':  #save file at no transit or transit if it has been set before
      target_unspace = self.params['target'].replace(' ', '_')
      if 'current_transit' in self.params:
          current_transit = self.params['current_transit'].replace('/', '-')
          filename = '%s/%s_%s_%s.png' %(self.params['savedir'],
                                         target_unspace,
                                         current_transit,
                                         self.params['site'])
      else:
          filename = '%s/%s_T%s_%s.png' %(self.params['savedir'],
                                          target_unspace,
                                          self.params['timespan'],
                                          self.params['site'])
      print ("Saving: %s" % (filename,))
      self.params['plot_figure'].savefig(filename)
    elif event.key=='f':  #recenter transit and save file
      self._plot_elev_transit(event.xdata, event.ydata)
      target_unspace = self.params['target'].replace(' ', '_')
      site = self.params['site']
      current_transit = self.params['current_transit'].replace('/', '-')[:-3].replace(':','')
      filename = '%s/%s_%s_%s.png' %(self.params['savedir'],
                                     target_unspace,
                                     current_transit,
                                     site)
      print ("Saving: %s" % (filename,))
      self.params['plot_figure'].savefig(filename)


  def _onclick(self, event):

    ax = self.params["plot_ax-elev"]
    if event.inaxes != self.params["plot_ax-airm"]:
      return

    day, hour = event.xdata, event.ydata
    self._plot_elev_transit(day, hour)







if __name__ == '__MAIN__':
    import obsrv
    target=['HD83443',9+(37 + 11.82841/60)/60.0, -(43+ (16+ 19.9354/60)/60.0),
            2455943.20159650, 2.985, 3]
    target=['WASP-34', 11+(1+36/60.0)/60.0, -(23+( 51+38/60)/60), 
            2454647.55358, 4.3176782,3]
    a=reload(obsrv).obsrv((target[1],target[2]),
                          tr_period=target[4],
                          tr_epoch=target[3],title=target[0])


