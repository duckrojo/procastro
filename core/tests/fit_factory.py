import astropy.io.fits as pf
import numpy as np
import sys


def create_merge_example(x, y, hdus, path):
    """
    Creates a fit file which will store random data inside a number of hdus
    
    Parameters:
    -----------
    x, y: int
        Size of the example
    hdus : int
        Number of hdus to fill
        
    """
    ##Create primary hdu
    data = np.random.rand(x,y)
    primer = pf.PrimaryHDU(data = data, header = pf.Header())
    hdul = pf.HDUList([primer])
    for i in range(hdus):
        data = np.random.rand(x,y)
        hdu = pf.ImageHDU(data = data, header = pf.Header())
        hdul.append(hdu)
        
    hdul.writeto(path)
    
def create_random_fit(xy, path, header=pf.Header()):
    """
    Generates normal fit with random data
    """
    data = np.random.rand(xy[0],xy[1])
    primer = pf.PrimaryHDU(data = data, header = header)
    hdul = pf.HDUList([primer])
    hdul.writeto(path)