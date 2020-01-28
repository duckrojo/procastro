import astropy.io.fits as pf
import numpy as np
import sys
from astropy.modeling.functional_models import Gaussian2D

def create_targeted_fit(path, size_xy, target_xy, header=pf.Header()):
    """
    Creates a fit file which contains a simulated source on the specified 
    coordinates
    TODO: Allow multiple sources
    TODO: Allow different intensities
    
    Parameters:
    -----------
    path : string
        Path to save image
    size_xy : (int, int)
        Fit dimensions
    target_xy : (int, int)
        Coordinates of the source
    """
    model = Gaussian2D(1,target_xy[0],target_xy[1], 5, 5)
    y,x = np.mgrid[0:size_xy[0], 0:size_xy[1]]
    data = model(x,y)
    save_fit(data, header, path)
    
def create_bias(path, size_xy, header=pf.Header()):
    data = np.random.random(size_xy) / 30
    save_fit(data, header, path)
    
def save_fit(data, header , name):
    """
    Convenience function. Creates a fit based on the provided data
    
    Parameters:
    -----------
    data : numpy array
    header : astropy Header Object
    name : string
    """
    hdu = pf.PrimaryHDU(data = data, header = header)
    hdul = pf.HDUList([hdu])
    hdul.writeto(name)
    
    