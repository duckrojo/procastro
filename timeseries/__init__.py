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


import scipy as sp
import astropy as ap
import pyfits as pf



#Defines subarray, size 2*rad+1
def subarray(arr,y,x,rad):
    return arr[(y-rad):(y+rad+1),(x-rad):(x+rad+1)]

def centraldistances(data,c):
    dy,dx = data.shape
    y,x = sp.mgrid[0:dy,0:dx]
    return sp.sqrt((y-c[0])*(y-c[0])+(x-c[1])*(x-c[1]))


def bipol(coef,x,y):
	plane = sp.zeros(x.shape)
	deg = sp.sqrt(coef.size).astype(int)
	coef = coef.reshape((deg,deg))
	if (deg*deg != coef.size):
		print "Malformed coefficient: " + str(coef.size) + \
		    "(size) != " + str(deg) + "(dim)^2"
	for i in sp.arange(coef.shape[0]):
		for j in sp.arange(i+1):
			plane += coef[i,j]*(x**j)*(y**(i-j))
	
	return plane


def apphot(data,cs,sap,sky,deg=3,show=False,reset=True):
    """Do aperture photometry on data array,
    using cs central coordinates, aperture sap, and sky sky
    fit the sky with polinomial degree of degree deg"""
    import scipy.optimize as op

    d = centraldistances(data,cs)
    dy,dx = data.shape
    idx = (d>sky[0])*(d<sky[1])
    y,x = sp.mgrid[-cs[0]:dy-cs[0],-cs[1]:dx-cs[1]]
    errfunc = lambda coef,x,y,z: (bipol(coef,x,y)-z).flatten()
    coef0 = sp.zeros((deg,deg))
    coef0[0,0] = data[idx].mean()
    fit,cov,info,mesg,success = op.leastsq(errfunc, coef0.flatten(),args=(x[idx],y[idx],data[idx]),full_output=1)
    res = data-bipol(fit,x,y) #minus sky
    res = res*(res>0)
    phot = res[d<sap].sum()
    return phot


def centroid(arr):
    """Find centroid of small array"""
    import scipy as sp
    med=sp.median(arr)
    arr=arr-med
    arr=arr*(arr>0)
	
    iy,ix=sp.mgrid[0:len(arr),0:len(arr)]

    cy = sp.sum(iy*arr)/sp.sum(arr)
    cx = sp.sum(ix*arr)/sp.sum(arr)

    return cy, cx



class photometry(object):
    def __init__(self, astrodir,
                 coordsxy=None,
                 labels=None,
                 stamprad=20,
                 maxskip=6):
        import matplotlib.pyplot as plt
        import warnings

        try:
            if labels is None:
                labels=[]
            nstars = len(coordsxy)
            labels = list(labels) + sp.arange(len(labels),
                                              nstars).astype(str).tolist()
            targets = {lab: [[coo[0],coo[1]],]
                       for  coo,lab in zip(coordsxy,labels)}
        except:
            raise ValueError("Coordinates of target stars need to be specified as a list of 2 elements, not: %s" % (str(coordsxy)))

        flx={lab:[] for lab in labels}

        for filename in astrodir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data,head = filename.reader(datahead=True)
            for lab,cooxy in targets.items():
                cx,cy= cooxy[-1]
                sarr=subarray(data, cy, cx, stamprad)
                scy,scx = centroid(sarr)

                #plt.imshow(sarr)
                #plt.show()

                skip=sp.sqrt((stamprad-scy)**2 + (stamprad-scx)**2)
                if skip > maxskip:
                    print("Jump of %f pixels has occurred on frame %s for star %s" %
                          (skip,filename,lab))

                flx[lab].append( apphot(sarr,[scy,scx],5,[10,13]))
                cooxy.append([cx+scx-stamprad, cy+scy-stamprad])


        self.flx = flx
        self.cooxy = targets

    def doratio(self, trg='0', ref=None, normframes=None):
        """Computes ratio of science and reference
        :param trg: label name of target star
        :type trg: string (key of flx and coo)
        :param ref: list of stars to be used as reference
        :type ref: None or list of strings. If None, then all stars except target
        :param normframes: list of frames to normalize (for example, in case only out of transit want to be considered)
        :type normframes: None or boolean array. If None, then normalize by all frames
        """
        from scipy import asarray
        if trg not in self.flx.keys():
            print("Reference star '%s' not in stellar list: %s" % (trg, self.flx.keys()))
            return None
        if ref is None:
            ref = [k for k in self.flx.keys() if k != trg]

        science = asarray(self.flx[trg])
        reference = science*0.0
        for k in ref:
            reference += asarray(self.flx[k])/asarray(self.flx[k]).mean()
        reference /= len(ref)

        self.ratio = science/reference
        if normframes is None:
            normframes = sp.ones(len(science))==1

        self.ratio /= self.ratio[normframes].mean()


    def showdrift(self):
        import matplotlib.pyplot as plt
        import scipy as sp
        lines=[]
        leg=[]
        for k in self.flx.keys():
            cooxy=self.cooxy[k]
            xd, yd  = sp.array(cooxy).transpose()
            l=plt.plot(xd-xd[0],yd-yd[0], 
                       label='%8s. X,Y: %-7.1f,%-7.1f' % (k,xd[0],yd[0]))
        plt.legend(bbox_to_anchor=(0.,1.02,1.,.302),loc=3,
                   ncol=1,mode="expand",borderaxespad=0.,
                   prop={'size':6})





if __name__ == '__main__':
    import astropy as ap
    import warnings
    import dataproc as dp
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        files=dp.astrodir('ptfo1/PTFO*').sort('irafname')

    coo=[[513,1172],[553,1220],[573,636]]

    phot = photometry(files,coo)


