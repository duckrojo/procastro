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


from io import PrintDebug

def _actionlog(f):
    from functools import wraps
    @wraps(f)
    def tolog(self, *args, **kwargs):
        ret= f(self, *args, **kwargs)
        self.actionlog.append((f.__name__,args))
        return ret
    return tolog


class process2d():
    """Provides some manipulation to an NDData array, including the capacity to reset to original condition
"""
    
    def __init__(self, data, attrtoclone=None, header=None, **kwargs):
        """Initialize array and backup. 
        :param data: it can be an array, a filename, or an object to clone. If the latter, attrtoclone is madatory
        :param attrtoclone: list of attributes to clone. If specified, data must be object to clone from.
    """
        import astropy.nddata as nd
        from dataproc import astrofile

        if not hasattr(self, 'props'):
            self.props={}

        #If clone make sure both attrtoclone is specified and data is the appropriate object class
        if isinstance(data, basestring): #filename is given
            data, header2=astrofile(data).reader(datahead=True)
            if header is None:
                header=header2

        #do the clonning
        if attrtoclone is not None and not isinstance(data, process2d):
            raise ValueError("data must be a process2d instance whenever attrtoclone is specified and viceversa")
        if isinstance(data, process2d):
            if 'orig' not in attrtoclone:
                attrtoclone.append('orig')
            self.props['clonedattr']=[]
            self.props['clonedfrom']=data
            for a in attrtoclone:
                setattr(self, a, getattr(data,a))
                self.props['clonedattr'].append(a)
            if 'data' not in self.attrtoclone:
                self.reset()
        #initiailize regularly
        else:
            self.orig = nd.NDData(data)
            self.reset()

        if header is not None:
            self.orig.meta=dict(header)


    def reset(self):
        """Resets data array to its original condition"""
        from copy import deepcopy
        self.actionlog = []
        self.data = deepcopy(self.orig)


    @_actionlog
    def meanfilter(self, radius):
        import scipy.signal as sg
        import scipy as sp

        a = sg.medfilt2d(sp.array(self.data.data,dtype=float), int(2*radius+1))
        self.data.data = a
        return self


    @_actionlog
    def medianfilter(self, radius):
        import scipy.signal as sg
        import scipy as sp

        a = sg.medfilt2d(sp.array(self.data.data,dtype=float), int(2*radius+1))
        self.data.data = a
        return self


    @_actionlog
    def difference(self):

        self.data.data=self.orig.data-self.data.data

        return self


    @_actionlog
    def ratio(self):

        self.data.data=self.orig.data/self.data.data

        return self


    def test(self, *args, **kwargs):
        print ("In test")
        print (self)
        print (args)
        print (kwargs)



