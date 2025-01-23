from procastro import AstroDir
from ..core.astrofile.astrofile import AstroFile
from procastro.timeseries import Photometry
from numpy.testing import assert_equal
from .test_utils import create_targeted_fit, create_bias
from unittest.mock import patch
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
                create_targeted_fit(os.path.join(".", "sci_files", "file"+str(i)+".fit"), (self.size, self.size), (self.size/2,self.size/2), pf.Header({'JD': 2457487.62537041+i}))
                create_targeted_fit(os.path.join(".", "extra", "extra"+str(i)+".fit"), (self.size, self.size), (self.size/2,self.size/2), pf.Header({'JD': 2457487.62537041+i}))
            self.data = AstroDir(os.path.join(self.path, "sci_files"))
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
        #NOTE: Copying Photometry objects using deepcopy will raise
        #      an error for python 3.6. Initializing a new object 
        #      will fix the issue.
        broken = Photometry(self.data, [[self.size/2, self.size/2]], 
                            aperture = 7.5, sky = [11,20], 
                            brightest=0, gain = 1.8, ron = 4.7)

        broken.stamp_rad = 1.0
        with pytest.raises(ValueError):     #Stamp radius < Aperture
            broken.photometry(aperture = 7.5)

        broken.stamp_rad = 10.0
        with pytest.raises(ValueError):     #Stamp radius < Sky
            broken.photometry()

        broken.stamp_radius = 25
        broken._apertures = [1.0]
        with pytest.raises(ValueError):     #Aperture > Sky
            broken.photometry()

        broken._apertures = None
        with pytest.raises(ValueError):     #Aperture is None
            broken.photometry()

        broken._apertures = [7.5]
        broken.sky = None
        with pytest.raises(ValueError):     #Aperture is None
            broken.photometry()

        del broken

    #####
    # TODO: Find a way to recover numeric data from this methods.
    #       The following tests only check if the plots work properly
    #       
    # NOTE: If you want to visually test if the graphs are correctly
    #       displayed, comment these lines:
    #       * @patch("matplotlib.pyplot.show")
    #       * mock_show.return_value = None
    #       And remove the mock_show parameter from the signature (leave self)
    
    @patch("matplotlib.pyplot.show")
    def test_plot_radial_profile(self, mock_show):
        mock_show.return_value = None
        self.phot.plot_radialprofile()
    
    @patch("matplotlib.pyplot.show")
    def test_show_stamps(self, mock_show):
        mock_show.return_value = None
        self.phot.showstamp(last = 3, ncol=3)
    
    @patch("matplotlib.pyplot.show")
    def test_imshowz(self, mock_show):
        mock_show.return_value = None
        fig = plt.figure()
        self.phot.imshowz(axes = fig, ccd_lims_xy = [0, self.size, 0, self.size])
        
    @patch("matplotlib.pyplot.show")
    def test_plot_drift(self, mock_show):
        mock_show.return_value = None
        self.phot.plot_drift()
    
    def teardown_class(self):
        plt.close('all')    # Finish by cleaning all remaining plots
