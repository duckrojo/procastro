#
#
# Copyright (C) 2013 Patricio Rojo
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


from __future__ import print_function
import misc_process as mp

from io import PrintDebug
#set verbose output
PrintDebug.setv(2)


def _refreshgraph(f):
    from functools import wraps
    @wraps(f)
    def ret(*args, **kwargs):
        toreturn=f(*args, **kwargs)

        # Following is to circumvent an identified Python-language issue: when using pythonized decorator  (@_refreshgraph) then the function is received as having its first element as 'self'. However, when using brute-force decoration (object.function = _refreshgraph(object.function), in function __init__), the function is received as a method and self is thus in f.im_self
        if hasattr(f,'im_self'):
            self=f.im_self
        else:
            self=args[0]

        self.draw(**kwargs)
        return toreturn
    return ret


class axexamine2d(mp.process2d):
    """Graphically examine a ndd array of two dimensions.
    If a pyplot.axes is specified, then use that otherwise make a
    new figure. kwargs are passed to plt.figure and then to plt.add_subplot.
    :param interact: whether mouse and keyboard events are processed """


    def __init__(self, data, axes,
                 datapointers=None, linked=None,
                 showtitle=True,
                 mode='cmap', **kwargs):
        """
Initializes examine2d by creating figure and then calling the initialization of process2d
        :param linked: List of all plots that need to be update together
        :param datapointers: a 2-element tuple indicating the arrays that need to be used as orig and data, respectively. Both need to be of type astropy.nddata.NDdata
        :param mode: Mode to display the data it should be in self._drawmodes dictionary
        :param data: 2d Data to be analyzed
        :param axes: Where to plot
        :param interact: Whether user input is allowed
        :param kwargs: passed to all pyplot functions
        """
        import matplotlib.pyplot as plt
        import astropy.nddata as nd
        import inspect

        if axes is None:
            raise ValueError("Axes need to be specified for axexamine!. Use examine if you just want to plot one frame")
        self.fig = axes.figure
        self.prop = {}
        self.axes = axes
        #TODO: test what happens with axis ratio
        #ax.set_aspect('equal')
        self._defmode=mode
        if mode=='cmap':
            self.prop['xlabel'] = 'column'
            self.prop['ylabel'] = 'row'
        elif mode=='rows':
            self.prop['xlabel'] = 'row'
            self.prop['ylabel'] = ''
        elif mode=='cols':
            self.prop['xlabel'] = 'column'
            self.prop['ylabel'] = ''


        #All the plots that need to get updated together (after this)
        if isinstance(linked,(list,tuple)):
            self.linked=linked
        else:
            self.linked=[]

        #which labels to show
        self.showlabels=['xlabel','ylabel']
        if showtitle:
            self.showlabels.append('title')

        #to plot reference Y,X
        self.refyx=[data.shape[0]/2,data.shape[1]/2]

        #brute-force decorate inherited algorithms
        for name, object in inspect.getmembers(self, inspect.ismethod):
            if name[0] !='_' and name in mp.process2d.__dict__ and name not in ["reset"]:
                setattr(self,name, _refreshgraph(object))

        #following includes a call to reset
        mp.process2d.__init__(self,data, datapointers=datapointers)



    def reset(self, forcedata=None):
        """Drop all modifications and redraw original image
        :param forcedata:  Force data pointer to specified value rather than copy from .orig
        """

        mp.process2d.reset(self, forcedata=forcedata)
        self.toshow=self.data
        self.draw()


    def clone(self, axes, linkrefxy=False, **kwargs):
        """Creates a new plot with the same data
        :param axes:
        :param linkrefxy:
        :param kwargs:
        """
        import matplotlib.axes as ax
        if not isinstance(axes,ax.Axes):
            raise ValueError("A valid axes need to be specified to create a linked examine2d instance")

        new = examine2d(None,datapointers=[self.orig,self.data], linked=self.linked+[self], axes=axes, **kwargs)
        for l in self.linked:
            l.linked+=[new]
        self.linked+=[new]

        if linkrefxy:
            new.refyx=self.refyx

        return new


    def draw(self, mode=None, golink=True, **kwargs):
        """
Draw a new plot using the standard mode or as specified by the user, which then becomes the standard mode. Optionally do the same for all linked objects in self.linked
        :param golink: Whether to update the linked plots
        :param mode: Mode to be used, it must be one of _drawmodes
        :raise: If mode is not recognized in self._drawmodes
        """
        if mode is None or not  isinstance(mode, basestring):
            mode = self._defmode
        if not hasattr(self, "draw_"+mode):
            raise("Mode '%s' is not available. Choose from: %s" % (mode,', '.join([m[5:] for m in dir(self) if m[:5]=="draw_"])))
        getattr(self,"draw_"+mode)(**kwargs)
        #defmode hast to be set after draw_*, so that it can be used to check whether the type has changed (except for the default type cmap
        #todo: allow cmap to profit from checking _defmode to see if it was run.
        self._defmode=mode

        if golink:
            for l in self.linked:
                l.draw(golink=False, **kwargs)


    def draw_surface(self):
        #todo: plot 3d
        print ("todo: plot 3D")


    def draw_cmap(self, **kwargs):
        """Update canvas with data
        """

        self.cla()
        self.imdata = self.axes.imshow(self.toshow, origin='lower')

        #print labels
        if 'title' in self.showlabels:
            if self.toshow is self.orig:
                self.axes.set_title('Original image')
            else:
                self.axes.set_title("Processed (%s)" % ';'.join(["%s%s" % (ac[:3],','.join(map(str,prm))) for ac,prm in self.actionlog]))
        else:
            self.axes.set_title('')
        if 'xlabel' in self.showlabels:
            self.axes.set_xlabel(self.prop['xlabel'])
        if 'ylabel' in self.showlabels:
            self.axes.set_ylabel(self.prop['ylabel'])

        self.changeprops(**kwargs)


    def draw_rows(self, add=False):
        """Draw at a fixed column
        :param add:  If True, then add new plot. Otherwise, just update the data of the last cut
"""
        import scipy as sp

        rows = self.toshow[self.refy,:]
        xaxis = sp.arange(self.toshow.shape[1])

        return self._draw1d(xaxis, rows, add,'rows')


    def draw_cols(self, add=False):
        """Draw at a fixed row
        :param add:  If True, then add new plot. Otherwise, just update the data of the last cut
"""
        import scipy as sp

        cols = self.toshow[:,self.refy]
        xaxis = sp.arange(self.toshow.shape[0])

        return self._draw1d(xaxis, cols, add,'cols')


    def _draw1d(self,x,y,add,name):
        """Only to be called by draw_rows() or draw_cols()
        :param add:  If True, then add new plot. Otherwise, just update the data of the last cut
        :param name: type of plot, either 'cols' or 'rows'
        :param x: X-axis
        :param y: Y-axis
"""
        if self._defmode!=name:  #new instance
            add=True
            self.cla()

        if add:
            self.imdata.extend(self.axes.plot(x, y))
        else:
            self.imdata[-1].set_data(x, y)


    @_refreshgraph
    def showorig(self):
        """Select original image to plot"""

        self.toshow=self.orig

    @_refreshgraph
    def showproc(self):
        """Select processed image to plot"""

        self.toshow=self.data


    @_refreshgraph
    def updaterefx(self, x):
        """change x-reference"""
        PrintDebug("Setting X-ref: %i" %x)
        self.refyx[1]=x
        return self


    @_refreshgraph
    def updaterefy(self, y):
        """change y-reference"""
        PrintDebug("Setting Y-ref: %i" % y)
        self.refyx[0]=yq
        return self


    def cla(self):
        """Clear axis"""
        self.axes.cla()


    def _updateX(self, xout):
        """Updates X-out to new lims and visibility status"""
        import misc_arr as ma
        import matplotlib.pyplot as plt
        x1, x2, y1, y2 = ma.expandlims(self.axes.get_xlim(),
                                       self.axes.get_ylim(), -0.1)
        if hasattr(self,'_xlines'):
            self._xlines[0].set_data([x1, x2], [y1, y2])
            self._xlines[1].set_data([x1, x2], [y1, y2])
            self._xlines[2].set_data([x1, x2], [y2, y1])
            self._xlines[3].set_data([x1, x2], [y2, y1])
        else:
            self._xlines=self.axes.plot([x1, x2], [y1, y2], 'k',
                                        [x1, x2], [y2, y1], 'k', linewidth=5)
            self._xlines.extend(self.axes.plot([x1, x2], [y1, y2], 'w',
                                               [x1, x2], [y2, y1], 'w', linewidth=2))

        for l in self._xlines:
            plt.setp(l, visible=xout)


    def changeprops(self, pl_resetprops=False, **kwargs):
        """
        The following pl_* kwargs are accepted to override defaults
        pl_xout: whether to X-out the image
        pl_contrast: use the indicated method to compute the contrast. The format is a list, where the first element is the method and the remainder are the parameters.
        pl_cmap: colormap
        pl_xl: Limits the X-axis
        pl_yl: Limits the Y-axis

 """
        import misc_arr as ma
        import matplotlib.pyplot as plt

        if (not hasattr(self,'props')) or pl_resetprops:
            self.props={'xl':None,  #auto xlim
                        'yl':None,  #auto ylim
                        'contrast': ('zscale',), #Zscale contrast
                        'cmap': None, #default cmap
                        'xout': False, #not X-out
                        }

        for k in kwargs.keys():
            if k[:3]=='pl_' and k[3:] in self.props:
                self.props[3:]=kwargs[k]

        #avail_contrast contains the available contrast computing
        #methods, as first parameter, they should receive data then at
        #least as many arguments as specified in the second element of
        #the tuple.
        avail_contrast = {'zscale': (ma.zscale, 0),
                          'force': (lambda dt, mn, mx: (mn, mx), 2),
                         }
        scmeth = self.props['contrast'][0]
        scprms = self.props['contrast'][1:]
        if (scmeth not in avail_contrast):
            raise ValueError("Specified contrast computation method (%s) is not valid. Has to be one of: %s" %
                             (scmeth, ', '.join(avail_contrast.keys())))
        if len(scprms) < avail_contrast[scmeth][1]:
            raise ValueError("Not enough arguments supplied to scmeth (%i needed): %s" %
                             (avail_contrast[scmeth][1], ', '.join([str(p) for p in scprms])))
        mn, mx = avail_contrast[scmeth][0](self.toshow, *scprms)
        #todo change contrast in 3d image
        self.imdata.set_clim(vmin=mn, vmax=mx)

        #set the limits
        xl = self.props['xl'] is None and self.axes.get_xlim() or self.props['xl']
        yl = self.props['yl'] is None and self.axes.get_ylim() or self.props['yl']
        #todo: check that it works in 3d
        self.axes.set_xlim(xl)
        self.axes.set_ylim(yl)

        #draw Xout
        self._updateX(self.props['xout'])

        #set colormap
        if self.props['cmap'] is not None:
            self.imdata.set_cmap(self.props['cmap'])

        self.fig.canvas.draw()





class kmevents(object):
    """Provides capture of keyboard and mouse events"""


    def __init__(self):
        raise NotImplementedError('class kmevents is not designed to be called on its own, only inherited')


    def _ihelp(self):
        """
Prints help for all keyboard and mouse capture functions as defined
in the self.action dictionary
    :rtype : None
    :param self: kmevents class
    :param event: Captured event
    """
        #the following will only work while the first letter of the mode is the enconding letter (first) in self.keyaction
        for mode,encod in [["keystrokes", "k"],["mouse buttons","m"]]:
            if self.ev_cid[encod]:
                print("Available %s:" % (mode,))
                for k,v in sorted(self.action.items()):
                    if k[0]==encod:
                        stroke = k[1:]
                        fname = v[1]
                        opcs = v[2]
                        oplist = len(opcs)>0 and "\n   accepts (%s)" % (', '.join(map(lambda x:x[0],opcs))) or ""
                        print(" %s: %s%s" % (stroke, fname, oplist))
        return None


    def interactive(self, iface, updatelists=None):
        """
Starts interactive capture on specified interface
        :param iface: it can be 'both', 'mouse', or 'keyboard' (or just its first initial)
        :param updatelists: is a 3-element tuple, that should indicate (as a list of strings (function names) the functions that require updatex, updatey, and updatexy, respectively.  

        """
        import matplotlib.pyplot as plt

        if updatelists is None:
            updatelists=[]

        if not hasattr(self, 'fig'):
            raise ValueError("Keyboard events capture must be called after initializing .fig attribute with a matplotlib.pyplot.figure")
        if not isinstance(iface, basestring):
            raise ValueError("interactive() argument must be 'mouse', 'keyboard', or 'both'")

        self.action = {}

        #set keyboard capture and initialize help function
        if not hasattr(self,"ev_cid"):
            self.ev_cid={}


        if iface[0]=='b' or iface[0]=='k':
            if "k" in self.ev_cid and self.ev_cid["k"]:
                raise ValueError("Keyboard capture still running, disconnect first")

            self.ev_cid["k"] = self.fig.canvas.mpl_connect('key_press_event',self._keyaction)
            self.addkeyaction('?','_ihelp', "Help", [])
        #set mouse capture
        if iface[0]=='b' or iface[0]=='m':
            if "m" in self.ev_cid and self.ev_cid["m"]:
                raise ValueError("Mouse capture still running, disconnect first")
            self.ev_cid["m"] = self.fig.canvas.mpl_connect('button_press_event', self._mouseaction)


    def uninteract(self):
        """
Disable interactive capture of mouse and/or keyboard

        """
        for encod in ["m","k"]:
            if encod in self.ev_cid and self.ev_cid[encod]:
                self.fig.canvas.mpl_disconnect(self.ev_cid[encod])
                del self.ev_cid[encod]


    def _runevent(self, event, key):
        """
Runs a mouse or keyboard event, it must be called from capturing functions: _keyaction or _mouseaction
        :param event:
        :param key:
        :return: :raise:
        """

        fcn,doc,argsd,onlymodes=self.action[key]

        runax=None
        for a in self.axex:
            if event.inaxes == a.axes:
                if len(onlymodes) and a._defmode not in onlymodes:
                    break
                runax=a
        if runax is None:
            return None

        #find arguments either X,Y or from user input
        args=[]
        for a in argsd:
            if isinstance(a,(list,tuple)):
                a,t=a
            elif isinstance(a,basestring):
                t=float
            else:
                raise ValueError("Each of the list elements of the arguments (field 2 of the action dictionary) can be either a 'description' or a ('description', type) tuple")
            if a.lower() == 'x':
                val=t(event.xdata)
                xl=event.inaxes.get_xlim()
                if not (xl[0]<val<xl[1]):
                    return None
                args.append(val)
            elif a.lower() == 'y':
                val=t(event.ydata)
                yl=event.inaxes.get_xlim()
                if not (yl[0]<val<yl[1]):
                    return None
                args.append(val)
            else:
                args.append(t(input(' %s? '%(a,))))

        #Executes the function requested by the hotkey, either on the axes object (priority) or the figure object
        if not hasattr(runax,fcn):
            if not hasattr(self,fcn):
                raise ValueError("Hotkey '%s' provides '%s', which was not enabled on the figure nor axes" %(key, fcn))
            dummy=getattr(self,fcn)(*args)
        else:
            print ("Running '%s'... " % (fcn,), end='')
            dummy=getattr(runax,fcn)(*args)
            print ("done")

        return self


    def _keyaction(self, event):
        """
        Execute whenever a keyboard pressed is pressed
        :param event:
        """
        if "k"+event.key in self.action.keys():
            return self._runevent(event,'k'+event.key)


    def _mouseaction(self, event):
        """
        Execute whenever mouse is clicked
        :param event:
        """
        if 'm'+str(event.button) in self.action.keys():
            return self._runevent(event, 'm'+event.key)


    def _addaction(self, key,fcn,doc,argsd, onlymodes):
        """
Adds action for either keyboard or mouse, it must be called from addkeyaction or addmouseaction, respectively
        :param key:
        :param fcn:
        :param doc:
        :param argsd:
        :raise:
        """
        from warnings import warn

        if key in self.action:
            warn("Overwritting function for key %s (%s) with function that '%s'"
                 % (key,self.action[key][1],doc))
        if argsd is None:
            argsd=[]
        if onlymodes is None:
            onlymodes=[]
        if not isinstance(argsd,(list,tuple)):
            raise ValueError("args argument to addkeyaction must be a list ([] or None for no parameters")
        self.action[key] = (fcn, doc, argsd, onlymodes)


    def addkeyaction(self, key, fcn, doc="No info", argsd=None, onlymodes=None):
        """Adds a function capture to a keyboard event
        :param key: Key pressed
        :param fcn: Function to be called
        :param doc: Function description
        :param argsd: Description of each argument to 'fcn'
        """
        return self._addaction('k'+key, fcn, doc, argsd, onlymodes)


    def addbuttonaction(self, button, fcn, doc="No info", argsd=None, onlymodes=None):
        """Adds a function capture to a mouse event"""

        return self._addaction('m'+button, fcn, doc, argsd, onlymodes)






class examine2d(kmevents):

    def __init__(self, data, interact=True, fig=None,
                 mode='cmap', multi=False):

        _useraction={'m':('medianfilter','Perform a median filter of the image',[['filter radius',int]]),
                     'd':('difference','Subtract processed from original image',[]),
                     'f':('ratio','Divide processed from original image',[]),
                     'r':('reset','Reset image to its original look',[]),
                     'o':('showorig','Show original image',[]),
                     'p':('showproc','Show processed image',[]),
                     'q':('quit','Close figure', []),
                     'x':('updaterefx','Update X reference',['x'],['cmap']),
                     'y':('updaterefy','Update Y reference',['y'],['cmap']),
                     #'keystroke':{'function (can be in axes (priority) or figure)',"description",[['var',var_type], ...], ['only in modes', ...]}
                     }
        update = {'x': ,
                  }

        import matplotlib.pyplot as plt
        if multi:
            raise NotImplementedError("multi objects not implemented yet")

        if fig is None:
            fig=plt.figure()
        self.fig = fig

        #start interaction
        self.interactset=interact
        if interact:
            self.interactive('both')
            for k,v in _useraction.items():
                self.addkeyaction(k,*v)

        ax = fig.add_subplot(111)
        self.axex = [axexamine2d(data, ax, mode=mode)]

        self.show()

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    def quit(self):
        """Closes the window"""
        import matplotlib.pyplot as plt
        plt.close(self.fig)


    def __getattr__(self, item):
        if hasattr(self.axex[0],item):
            return getattr(self.axex[0],item)
        else:
            raise AttributeError("Attribute %s is not found in the first axexamine" %(item,))


#################################
##
######################################

def _plotaccross(self, event):
    my, mx = self.arrays[0].shape
    if event.inaxes is not None and \
            isinstance(self.axs, (list, tuple)) and \
            hasattr(self, "_acrax"):
        if mx > event.xdata >= 0 and mx > event.ydata >= 0:
            self.plotpix([event.ydata, event.xdata],
                         ax=self._acrax,
                         ytitle='x,y=%i,%i' % (event.xdata, event.ydata))


def _ckaxes(f):
    """Decorator to check axes keyword and create a subplot(111) if not setored in self.axs. If later is list, then use the last one"""
    from functools import wraps

    @wraps(f)
    @_ckfigure
    def ckax(self, *args, **kwargs):
        if 'ax' not in kwargs:
            if hasattr(self, 'axs'):
                if isinstance(self.axs, (list, tuple)):
                    kwargs['ax'] = self.axs[-1]
                else:
                    kwargs['ax'] = self.axs
            else:
                kwargs['ax'] = self.axs = self.fig.add_subplot(111)

        return f(self, *args, **kwargs)

    return ckax


def _ckfigure(f):
    """Decorator to create figure if it doesn't exist yet"""
    from functools import wraps
    import matplotlib.pyplot as plt

    @wraps(f)
    def isfig(self, *args, **kwargs):
        if not hasattr(self, 'fig') or not len(plt.get_fignums()):
            if hasattr(self, 'axs'):
                del self.axs
            self.newfigure()
        return f(self, *args, **kwargs)

    return isfig


def _claxes(f):
    """Decorator to clear axes"""
    from functools import wraps

    @wraps(f)
    @_ckaxes
    def clax(self, *args, **kwargs):
        kwargs['ax'].cla()
        return f(self, *args, **kwargs)

    return clax


def _clfigure(f):
    """Decorator to clear figure"""
    from functools import wraps

    @_ckfigure
    @wraps(f)
    def clfig(self, *args, **kwargs):
        self.fig.clf()
        return f(self, *args, **kwargs)

    return clfig





class examine2dn():
    """Graphically examines a cube dataset, which is a 3-dimensional np.array() stored in self.arrays.  

self.okframes: is used to filter out frames.
self.sigmas: contains accompanying error estimates.
self.headers: is a list of dictionaries containing header values.

self.fig: active figure
self.axs: could be a single ax or a list of axes if multiplot
"""


    def newfigure(self):
        """Creates new figure"""
        import matplotlib.pyplot as plt

        self.fig = plt.figure()
        self.fig.hold(True)


    def __init__(self, arrays, okframes=None, sigmas=None, headers=None):
        import scipy as sp

        if not isinstance(arrays, sp.array) or len(arrays.shape) != 3:
            raise ValueError("Initialization arrays must be a 3 dimensional np.array")
        self.arrays = arrays
        if okframes is None:
            okframes = sp.zeros(len(arrays)) == 0
        self.okframes = okframes
        if sigmas is None:
            sigmas = sp.zeros(len(arrays))
        self.sigmas = sigmas
        if headers is None:
            headers = [{}] * len(arrays)
        self.headers = headers

    @_ckaxes
    def plotpix(self, coords, x=None, ax=None, hidebad=False, ytitle=''):
        """coords: Array that has to have the same size as dimensions in the data. Plot one pixel at specified coordinates. Optionally vs an ordered x-axis, which can be a string (for a header value) or list. This function is ok for n-dimensional data."""

        import scipy as sp
        import matplotlib.pyplot as plt

        if hidebad:
            arr = self.arrays[self.okframes]
            hd = [h for h, o in zip(self.headers, self.okframes) if o]
        else:
            arr = self.arrays
            hd = self.headers

        #if only refreshing
        if coords is None:
            if hasattr(self, '_lastpixcoo'):
                coords = self._lastpixcoo
            else:
                return

        if isinstance(x, basestring):
            x = sp.array([h[x] for h in hd])
        elif x is None:
            x = sp.arange(len(hd))
        elif not isinstance(x, (list, tuple)):
            raise TypeError("X-axis can only be a string (for header field) or a list/tuple")

        if (not isinstance(coords, list)) or (len(coords) != len(arr.shape) - 1):
            raise ValueError("coords must have the same size (%i) as dimensions in data (%i). %i, %i" % (
                len(coords), len(arr.shape) - 1, not hasattr(coords, '__iter__'),
                (len(coords) != len(arr.shape) - 1)))

        a = arr.T
        for c in coords[::-1]:
            a = a[c]

        if not ax.has_data or len(ax.lines) < 2:
            ax.plot(x, a, 'bx')
            if not hidebad:
                ax.plot(x[self.okframes == False], a[self.okframes == False], 'Dr')
            self.axpp = ax
        else: #Update good/bad pixels
            ax.lines[0].set_data(x, a)
            ax.lines[1].set_data(x[self.okframes == False],
                                 a[self.okframes == False])
        if ytitle:
            ax.set_ylabel(ytitle)

        self._lastpixcoo = coords
        self.fig.canvas.draw()
        return ax


    @_clfigure
    def showeach(self, title=None, headerfld=None, interact=False,
                 markf=False, accross=False, keep=False):
        import scipy as sp
        import matplotlib.pyplot as plt

        if interact:
            markf = accross = True

        nf = len(self.arrays)
        lx = int(sp.sqrt(nf))
        ly = int(nf / lx + 0.999)
        if accross:
            ly += 1

        ax0 = None
        axs = []
        for i in range(nf):
            if ax0 is None:
                ax = ax0 = self.fig.add_subplot(ly, lx, i + 1)
            else:
                ax = self.fig.add_subplot(ly, lx, i + 1, sharex=ax0, sharey=ax0)
            if i < (ly - 1) * lx:
                plt.setp(ax.get_xticklabels(), visible=False)
            if (i) % lx != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            _drawax(ax, self.arrays[i], self.okframes[i])
            axs.append(ax)

        if accross:
            #To store axes to draw accross  plot
            self._acrax = self.fig.add_subplot(ly, 1, ly)
            self.ikeyboard()

        if title is not None:
            axs[0].set_title(title)

        if isinstance(headerfld, (tuple, list)):
            return [[h[hf] for hf in headerfld] for h in self.headers]

        self.axs = axs
        if markf:
            self.imarkframes()

        if accross or markf:
            self.show()
            if not keep:
                self.dis_mark()
                self.dis_ikeyboard()
        return axs

    def show(self):
        """Show the active figure"""
        import matplotlib.pyplot as plt

        plt.show()

    def imarkframes(self, keep=False):
        self.cid_mf = self.fig.canvas.mpl_connect('button_press_event', self._togglebad)

    def ikeyboard(self, keep=False):
        self.cid_ac = self.fig.canvas.mpl_connect('key_press_event', self._keyboardaction)

    def dis_ikeyboard(self):
        self.fig.canvas.mpl_disconnect(self.cid_ac)
        self.cid_ac = 0

    def dis_mark(self):
        self.fig.canvas.mpl_disconnect(self.cid_mf)
        self.cid_mf = 0

#    keyaction = {'a': (_plotaccross, "Plot profiles accross frames"),
#                 '?': (_kbdhelp, "Help"),
#    }


    def _keyboardaction(self, event):
        import matplotlib.pyplot as plt

        #        print ('key=%s, button=%%, x=%d, y=%d, xdata=%f, ydata=%f, axes=%s'%(
        #            event.key, event.x, event.y, event.xdata, event.ydata, event.inaxes))

        if event.key in self.keyaction.keys():
            self.keyaction[event.key][0](self, event)

        self.show()

    def _togglebad(self, event):
        import matplotlib.pyplot as plt

        if event.inaxes is None or not isinstance(self.axs, (list, tuple)):
            return

        idx = self.axs.index(event.inaxes)
        self.okframes[idx] = self.okframes[idx] == False
        _drawax(self.axs[idx], self.arrays[idx], self.okframes[idx])
        if self.cid_ac:
            self.plotpix(None, ax=self._acrax)
        self.fig.canvas.draw()

        # if not event.inaxes.has_data:
        #     if event.inaxes is not None and isinstance(self.axs,(list,tuple)):
        #         idx=self.axs.index(event.inaxes)
        #         self.okframes[idx]=self.okframes[idx]==False
        #         tit=self.axs[idx].get_title()
        #         xv=plt.getp(self.axs[idx].get_xticklabels()[0],'visible')
        #         yv=plt.getp(self.axs[idx].get_yticklabels()[0],'visible')
        #         self.axs[idx].cla()
        #         _drawax(self.axs[idx], self.arrays[idx], self.okframes[idx])
        #         plt.setp(self.axs[idx].get_xticklabels(),visible=xv)
        #         plt.setp(self.axs[idx].get_yticklabels(),visible=yv)
        #         self.axs[idx].set_title(tit)
        #


def _drawax(axes, data, ok):
    import misc_arr as ma
    import matplotlib.pyplot as plt

    mn, mx = ma.zscale(data)
    if not axes.has_data(): #then draw, otherwise just update the X mark
        axes.imshow(data, vmin=mn, vmax=mx, origin='lower')
        xl = axes.get_xlim()
        yl = axes.get_ylim()
        dx = xl[1] - xl[0]
        dy = yl[1] - yl[0]
        axes.plot([xl[0] + 0.1 * dx, xl[1] - 0.1 * dx],
                  [yl[0] + 0.1 * dy, yl[1] - 0.1 * dy], 'k', linewidth=5)
        axes.plot([xl[0] + 0.1 * dx, xl[1] - 0.1 * dx],
                  [yl[0] + 0.1 * dy, yl[1] - 0.1 * dy], 'w', linewidth=2)
        axes.plot([xl[1] - 0.1 * dx, xl[0] + 0.1 * dx],
                  [yl[0] + 0.1 * dx, yl[1] - 0.1 * dx], 'k', linewidth=5)
        axes.plot([xl[1] - 0.1 * dx, xl[0] + 0.1 * dx],
                  [yl[0] + 0.1 * dx, yl[1] - 0.1 * dx], 'w', linewidth=2)
        axes.set_xlim(xl)
        axes.set_ylim(yl)

    for l in axes.lines:
        plt.setp(l, visible=not ok)


