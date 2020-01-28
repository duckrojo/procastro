from dataproc import AstroDir, AstroFile
from dataproc.timeseries import Photometry
from numpy.testing import assert_equal
from .fit_factory import create_targeted_fit, create_bias
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import numpy as np
import copy 
import pytest
import os
import pdb

class TestPhotometry(object):
    
    @pytest.fixture(autouse=True)
    def setup_class(self, tmpdir):
        self.size = 50
        with tmpdir.as_cwd():
            self.path = os.getcwd()
            os.mkdir("sci_files")
            os.mkdir("extra")
            for i in range(5):
                create_targeted_fit(os.path.join(".","sci_files","file"+str(i)+".fit") , (self.size, self.size), (self.size/2,self.size/2), pf.Header({'JD': 2457487.62537041+i}))
                create_targeted_fit(os.path.join(".","extra","extra"+str(i)+".fit") , (self.size, self.size), (self.size/2,self.size/2), pf.Header({'JD': 2457487.62537041+i}))
            self.data = AstroDir(os.path.join(self.path, "sci_files"))
            ##### TODO Find way to simulate calibration files
            #create_bias(os.path.join(".","mbias.fits"), (self.size, self.size))
            #bias = AstroFile(os.path.join(".","mbias.fits"))
            #f = ff.create_flat()
            #self.data.add_bias(bias)
            #self.data.add_flat(f)  
            self.phot = Photometry(self.data, [[self.size/2, self.size/2]], aperture = 7.5, sky = [11,20], brightest=0, gain = 1.8, ron = 4.7)
    
    def test_append(self):
        self.phot.append(os.path.join(self.path,"extra"))
        assert(len(self.phot.indexing) == 10)
        
        for i in range(9,4,-1):
            self.phot.remove_from(i)
        
        assert(len(self.phot.indexing) == 5)
    def test_cpu_phot(self): 
        time = self.phot.cpu_phot()
        
    def test_photometry(self):
        #Calls cpu_phot, checks if parameters are sound
        broken = copy.deepcopy(self.phot)
        
        broken.stamp_rad = 1.0
        with pytest.raises(ValueError):     #Stamp radius < Aperture
            broken.photometry(aperture = 7.5)
            
        
        broken.stamp_rad = 10.0
        with pytest.raises(ValueError):     #Stamp radius < Sky
            broken.photometry()
            
        broken.stamp_radius = 25
        broken.aperture = [1.0]
        with pytest.raises(ValueError):     #Aperture > Sky
            broken.photometry()
            
        broken.aperture = None
        with pytest.raises(ValueError):     #Aperture is None
            broken.photometry()
        
        broken.aperture = [7.5]
        broken.sky = None
        with pytest.raises(ValueError):     #Aperture is None
            broken.photometry()
            
        del broken
            
    ##### 
    #   TODO: Find a way to recover numeric data from this methods.
    #   The following tests only check if the plots work properly
     
    def test_plot_radial_profile(self):
        self.phot.plot_radialprofile()
        
    def test_show_stamps(self):
        self.phot.showstamp(last = 3, ncol=3)
        
    def test_imshowz(self):
        fig = plt.figure()
        self.phot.imshowz(axes = fig, ccd_lims_xy = [0, self.size, 0, self.size])
        
    def test_plot_drift(self):
        self.phot.plot_drift()