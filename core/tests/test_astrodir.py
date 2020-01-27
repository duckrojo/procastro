import pytest
from ..astrodir import AstroDir
from ..astrofile import AstroFile
from numpy.testing import assert_equal, assert_almost_equal
from .fit_factory import create_random_fit
import astropy.io.fits as pf
import numpy as np
import os

class TestAstroDir(object):

    @pytest.fixture(autouse=True)
    def setup_class(self, tmpdir):
        np.random.seed(42)
        with tmpdir.as_cwd():
            self.path = os.getcwd()
            os.mkdir("sci_files")
            for i in range(5):
                header = pf.Header({'JD': 2457487.62537037+i, 
                                    'EXPTIME': 20+i*10,
                                    })
                create_random_fit((8,8),os.path.join(self.path,"sci_files","file"+str(i)+".fit"),
                                  header = header)
                                  
            self.dataset = AstroDir(os.path.join(self.path,"sci_files"))
        
    def test_get_datacube(self):
        #TODO Add parameters for testing
        cube = self.dataset.get_datacube()
        for i in range(len(cube)):
            assert_equal(cube[None][i], self.dataset[i].reader())

    def test_pixel_xy(self):    
        arr = self.dataset.pixel_xy(4,4)
        expected = np.array([0.30461377, 0.03142919, 0.09028977, 0.89204656, 0.82260056])
        assert_almost_equal(arr, expected)
    
    def test_stats(self):
        res_path = os.path.dirname(__file__)
        mean = np.loadtxt(os.path.join(res_path,'results','mean.txt'))
        std = np.loadtxt(os.path.join(res_path,'results','std.txt'))
        median = np.loadtxt(os.path.join(res_path,'results','median.txt'))
        linterp = np.loadtxt(os.path.join(res_path,'results','linterp.txt'))
        
        assert_equal(self.dataset.mean(check_unique=['NAXIS1']), mean)
        assert_equal(self.dataset.std(check_unique=['NAXIS1']), std)
        assert_equal(self.dataset.median(check_unique=['NAXIS1']), median)
        assert_equal(self.dataset.lin_interp(45), linterp)
        
    def test_filter_chained(self):
        assert len(self.dataset.filter(naxis1=8).filter(naxis2=8)) == 5
        assert len(self.dataset.filter(naxis1=8).filter(naxis2=20)) == 0
        