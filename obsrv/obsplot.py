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

import matplotlib.pyplot as pl
from matplotlib.patches import Polygon
import scipy as sp
import ephem

######################
# Plotting


def plotpoly(ax, x, up, down, alpha=0.5, facecolor='0.8', edgecolor='k'):
    """ Plot a band  as semi-transparent polygon """
    verts = zip(x, down*24) + zip(x[::-1], (up*24)[::-1])
    poly = Polygon(verts, 
                   facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(poly)


class obsplot():

    def get_closejd(self,cd,ch):

        xovery = 5.0/4.0
        dx = self.xlims[1]-self.xlims[0]
        dy = self.ylims[1]-self.ylims[0]

        dist = []
        for htr, dtr in zip (self.htran, self.transits):
            dist.append(sp.sqrt(((cd+self.jd0-dtr)*xovery/dx)**2
                                + ((ch-htr)/dy)**2))
            dist.append(sp.sqrt(((cd+self.jd0-dtr)*xovery/dx)**2
                                + ((ch-(htr-24))/dy)**2))
        return sp.array(zip(self.transits,self.transits)).flatten()[sp.argmin(dist)]

        
    def onclick_coord(self,cd,ch,return_figure=False):

        closejd = self.get_closejd(cd,ch)
        self.figon = pl.figure(1, (8,6))
        self.plotnight(closejd, self.figon)
        if return_figure:
           return self.figon
   
    def onclick(self, event):

        cd, ch = event.xdata, event.ydata
        self.onclick_coord(cd,ch)

    def get_timestamp(self,cd,ch):

        transit_jd = self.get_closejd(cd,ch)
        timestamp  = '%s' % ephem.date(transit_jd-self.jd0+self.ed[0])

        idx = list(self.transits).index(transit_jd)
        xcoord = float(ephem.date(transit_jd-self.jd0+self.ed[0]))-float(self.ed[0])
        ycoord = self.htran[idx]
        if ycoord > 12.0:
           ycoord -= 24.0

        return timestamp+","+str(xcoord)+","+str(ycoord)

    def plotnight(self, transit_jd, fig, rect=None):

        if rect is None:
            rect = [0.1,0.1,0.8,0.8]
            fig.clf()

        obs=self.obs
        obs.date = transit_jd - self.jd0 + self.ed[0]
        midday = obs.previous_transit(self.sun)
        obs.date = midday
        ss = obs.next_setting(self.sun)
        sr = obs.next_rising(self.sun)
        obs.horizon = '-18:00'
        ts = obs.next_setting(self.sun)
        tr = obs.next_rising(self.sun)
        obs.horizon = '0:00'
        hours = sp.arange(ss-0.03, sr+0.03, 0.007)

        moonalt=[]
        staralt=[]
        for h in hours:
            obs.date=h
            self.moon.compute(obs)
            self.star.compute(obs)
            moonalt.append(self.moon.alt*180/sp.pi)
            staralt.append(self.star.alt*180/sp.pi)

        etout=sp.fix(hours[0])+0.5
        uthours = (hours - etout)*24

        ax1 = fig.add_axes(rect)
        ax2 = ax1.twinx()

        ax1.plot(uthours, staralt)
        ax1.plot(uthours, moonalt, '--')
        setev=sp.array([ss,ss,ts,ts])
        risev=sp.array([sr,sr,tr,tr])
        ax1.plot((setev - etout)*24, [0,90,90,0], 'k:')
        ax1.plot((risev - etout)*24, [0,90,90,0], 'k:')
        ax1.set_ylim([10,90])
        ax1.set_xlim([(ss-etout)*24-0.5,(sr-etout)*24+0.5])
        datetime = ephem.date(transit_jd-self.jd0+self.ed[0])
        print datetime
        ax1.set_title('Transit: %s' % datetime)
        ax2.set_ylim(ax1.get_ylim())
        sam = sp.array([1,1.5,2,3,4,5])
        ax2.set_yticklabels(sam)
        ax2.set_yticks(sp.arcsin(1.0/sam)*180.0/sp.pi)


        intr = (transit_jd-self.jd0+self.ed[0]-etout)*24-self.tr_hlen
        outr = (transit_jd-self.jd0+self.ed[0]-etout)*24+self.tr_hlen
        if intr>outr:
            outr+=24
        plotpoly(ax1, [intr,outr],
                 [0,0], [90,90], facecolor='0.5')

        fig.canvas.draw()

        
    def plot(self, interact=False, **kwargs):

        fig = pl.figure(3, (8,6))
        
        if interact:
            self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        rect = [0.125, 0.10, 0.7,  0.8]
        self.ax = pl.axes(rect)
        pl.text(1.12, 0.04, "Min. airmass", transform = self.ax.transAxes, ha="center")
        pl.text(1.12, 0.005, '%.2f' % min(self.am.flatten()), transform = self.ax.transAxes, ha="center")

        self.plotam()
        #self.plotmoon()
        self.plotmonths()
        self.plotlabels(**kwargs)
        return fig

    def plotam(self):

        ams = sp.arange(1, 3.01, 0.1)
        self.amp = pl.contourf(self.ed-self.ed[0], self.hours, sp.array(self.am), levels=ams,
                               cmap=pl.cm.jet_r)
        sx = self.ed - self.ed[0]
        plotpoly(self.ax, sx, self.twiset,  self.sunset)
        plotpoly(self.ax, sx, self.sunrise, self.twirise)

    def plotmoon(self):

        cmns = map(sp.cos, sp.array([5,10,20])*sp.pi/180.0)
        self.mnp = pl.contourf(self.ed-self.ed[0], self.hours, sp.array(self.cmn), levels=cmns, 
                               cmap=pl.cm.YlGn, alpha=0.5)

    def plottransits(self, x, y, hlen):

        self.ax.errorbar(x, y, yerr=hlen, fmt='o', color="w")
        self.ax.errorbar(x, y-24, yerr=hlen, fmt='o', color="w")
   
    def plotmonths(self):
        """Plot month separator"""

        mnlen = sp.array([31,28,31,30,31,30,31,31,30,31,30,31])
        cum = mnlen.copy()
        for i in range(len(mnlen))[1:]:
            cum[i:] += mnlen[:-(i)]
        mcum = sp.array(zip(cum,cum)).flatten()
        vlims = [-10,20,20,-10]*6

        y,m,d = ephem.Date(self.ed[0]).triple()
        jan1 = self.ed[0]-((m-1!=0)*cum[m-2]+d-1)
        self.ax.plot(self.ed[0]-jan1 + mcum, vlims, 'y--')
        ny=(self.ed[-1]-jan1)/365
        for i in range(int(ny)):
            self.ax.plot(i*365 + self.ed[0]-jan1 + mcum, vlims, 'y--')

        return self

    def plotlabels(self, title='', **kwargs):

        ax = self.ax
        pl.title(title)
        pl.xlabel(r"Days from %s" % ( ephem.Date(self.ed[0]).datetime().strftime('%Y.%m.%d'), ))
        pl.ylabel(r'Time (UT)')
        pl.xlim(self.xlims)
        pl.ylim(sp.array(self.ylims))
        cbrect = [0.85, 0.20, 0.035,  0.7]
        cax = pl.axes(cbrect)
        pl.colorbar(self.amp, cax=cax)
        cax.set_ylabel("Airmass")


