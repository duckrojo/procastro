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


class process2d():
    """Provides some manipulation to an NDData array, including the capacity to reset to original condition"""
    
    def __init__(self, data, datapointers=None, header=None, **kwargs):
        """Initialize array and backup.  """
        import astropy.nddata as nd
        from dataproc import AstroFile

        if isinstance(data, basestring): #filename is given
            data, header2=AstroFile(data).reader(datahead=True)
            if header is None:
                header=header2
        if (datapointers is not None):
            if (isinstance(datapointers,(list,tuple)) and
                len(datapointers)==2 and
                map(lambda x: isinstance(x, nd.NDData), datapointers).all()):
                self.orig=datapointers[0]
                self.reset(forcedata=datapointers[1])
            else:
                raise ValueError("datapointers need to be a 2-element list or tuple")

        else:
            self.orig = nd.NDData(data)
            if header is not None:
                self.orig.meta=dict(header)
            self.reset()


    def reset(self, forcedata=None):
        """Resets data array to its original condition"""
        from copy import deepcopy
        if forcedata:
            self.data=forcedata
        else:
            self.data = deepcopy(self.orig)


    def medianfilter(self, radius):
        import scipy.signal as sg
        import scipy as sp

        a = sg.medfilt2d(sp.array(self.data.data,dtype=float), int(2*radius+1))
        self.data.data = a
        return self


    def difference(self,radius):

        self.data.data=self.orig.data-self.data.data

        return self


    def ratio(self,radius):

        self.data.data=self.orig.data/self.data.data

        return self


    def test(self, *args, **kwargs):
        print ("In test")
        print (self)
        print (args)
        print (kwargs)