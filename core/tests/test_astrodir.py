import pytest
from ..astrodir import AstroDir
from ..astrofile import AstroFile
from numpy.testing import assert_equal
import numpy as np

class TestAstroDir(object):
    #TODO: Craft fit files during execution to reduce space, 
    #      when testing pixel_xy the tested pixel should have a different value 
    #      per file
    def setup_class(self):
        self.dataset = AstroDir(".\\test_dir\\astrodir\\")
        
    def test_get_datacube(self):
        #TODO Add parameters for testing
        cube = self.dataset.get_datacube()
        for i in range(len(cube)):
            assert_equal(cube[None][i], self.dataset[i].reader())

    def test_pixel_xy(self):    
        arr = self.dataset.pixel_xy(10,10)
        expected = np.asarray([13775, 13351, 13749])
        assert_equal(arr, expected)
    
    # TODO: Include group by and check unique parametes
    def test_stats(self):
        mean = AstroFile(".\\test_dir\\results\\mean.fits").reader()
        std = AstroFile(".\\test_dir\\results\\std.fits").reader()
        median = AstroFile(".\\test_dir\\results\\median.fits").reader()
        
        assert_equal(self.dataset.mean(), mean)
        assert_equal(self.dataset.std(), std)
        assert_equal(self.dataset.median(), median)
        
    # TODO: Include group by and check unique parametes
    def test_lin_interp(self):
        data = AstroDir(".\\test_dir\\linterp")
        linterp = AstroFile(".\\test_dir\\results\\linterp.fits")
        
        assert_equal(data.lin_interp(target = 3), linterp)
    
    def test_filter_chained(self):
        assert len(self.dataset.filter(naxis1=20).filter(naxis2=20)) == 3
        assert len(self.dataset.filter(naxis1=20).filter(naxis2=30)) == 0
        
    def teardown_class(self):
        del self.dataset
    