class obsplot():

        

    def get_timestamp(self,cd,ch):

        transit_jd = self.get_closejd(cd,ch)
        timestamp  = '%s' % ephem.date(transit_jd-self.jd0+self.ed[0])

        idx = list(self.transits).index(transit_jd)
        xcoord = float(ephem.date(transit_jd-self.jd0+self.ed[0]))-float(self.ed[0])
        ycoord = self.htran[idx]
        if ycoord > 12.0:
           ycoord -= 24.0

        return timestamp+","+str(xcoord)+","+str(ycoord)


        
    def plot(self, interact=False, **kwargs):

        fig = pl.figure(3, (8,6))
        

        rect = [0.125, 0.10, 0.7,  0.8]
        self.ax = pl.axes(rect)

        self.plotam()
        #self.plotmoon()
        self.plotmonths()
        self.plotlabels(**kwargs)
        return fig



    def plotmoon(self):

        cmns = map(sp.cos, sp.array([5,10,20])*sp.pi/180.0)
        self.mnp = pl.contourf(self.ed-self.ed[0], self.hours, sp.array(self.cmn), levels=cmns, 
                               cmap=pl.cm.YlGn, alpha=0.5)

   





###########################################################################
#
#
##########################################








  def classic_transit(self):

    self.tr_hlen = tr_hlen

#    self.obsll(Observer)
#    self.settimespan(timespan)
#    self.sunsetrise(**kwargs)
#    self.setysamp(**kwargs)

#    self.setstarsite(radec, **kwargs)
#    self.getairmass(**kwargs)

    self.sun = ephem.Sun()
    self.moon = ephem.Moon()
    self.getmoon(**kwargs)

    self.get_plot(kwargs['tr_epoch'],kwargs['tr_period'],
                  interact=kwargs['interact'])






  def getmoon(self, maxairmass=3.0, **kwargs):
    """Get moon distance"""
    obs = self.obs
    hours=self.hours

    cosdistance = []
    for d in self.days:
      cosdist=[]
      for h in hours:
        cosdist.append(self.moondist(d+h/24.0))
      cosdistance.append(cosdist)

    self.cmn = sp.array(cosdistance).transpose()

    return self






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


#def obsrv(radec, Observer='paranal', tr_period=None, tr_epoch=None, tr_hlen=0, ut_offset=0,
#          timespan=2013, save=None, title=None, magn=10, epoch_coo=2000,
#          showmonths=True, daystep=5, hourstep=0.2):
  """
    Plots the observability window of an astronomical object from a
    given observatory.

    Parameters:
    -----------
    Observer: Tuple or Scalar
         Either a tuple indicating Observer's location (lat, long) or the
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












# def splitcoo(val, name):
#   """Split coordinate to XX:XX:XX format"""
#   if isinstance(val, str):
#     vals = val.split(':')
#     if len(vals)<3:
#       raise ValueError, " %s string format must be XX:XX:XX and not: %s" \
#           % (name, val)
#   elif isinstance(val, (float,int)):
#     valh = int(val)
#     valm = int((val-valh)*60)
#     vals = ((val-valh)*60-valm)*60
#   else:
#     raise ValueError, "Unrecognized format for %s: %s" % (name, val)
  
#   return "%i:%i:%.2f" % (valh,valm,vals)

