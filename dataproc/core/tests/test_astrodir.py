import pytest
from ..astrodir import AstroDir
from ..astrofile import AstroFile
from numpy.testing import assert_equal, assert_almost_equal
from .test_utils import create_random_fit, create_empty_fit
import astropy.io.fits as pf
import numpy as np
import os


class TestAstroDir(object):
    @pytest.fixture(autouse=True)
    def setup_class(self, tmpdir):
        np.random.seed(42)
        self.raw_data = []
        with tmpdir.as_cwd():
            self.path = os.getcwd()
            os.mkdir("sci_files")
            for i in range(5):
                # Standard Header, values change slightly between them
                header = pf.Header({'JD': 2457487.62537037+i,
                                    'EXPTIME': 20+i*10,
                                    })
                # We generate fits with random data and store the generated
                # values to compare them in future tests
                path = os.path.join(self.path, "sci_files", "file"+str(i)+".fit")
                gen_data = create_random_fit((8, 8), path, header=header, show=True)
                                  
                self.raw_data.append(gen_data)
            
            # We also create a directory with corrupt files for testing 
            # how the instance is supposed to handle them.
            os.mkdir("botched")
            path = os.path.join(self.path,"botched")
            create_random_fit((8,8), os.path.join(path, "a.fits"), header=header, show = False)
            create_empty_fit(name = os.path.join(path, "b.fits"))
            create_random_fit((8,8), os.path.join(path, "c.fits"))
            
            # Generate main instance
            self.dataset = AstroDir(os.path.join(self.path, "sci_files"))

    def test_file_warnings(self):
        # Generate AstroDir with corrupt data and catch warning
        with pytest.warns(UserWarning):
            path = os.path.join(self.path, "botched")
            cdir = AstroDir(path)
        
        assert len(cdir) == 2   # Check that the corrupt data was actually skipped 
    
    def test_get_datacube(self):
        #TODO Add parameters for testing
        cube = self.dataset.get_datacube()
        for i in range(len(cube)):
            assert_equal(self.raw_data[i], self.dataset[i].reader())

    def test_pixel_xy(self):
        px, py = 4, 4   # Pixel coordinates to be tested
        arr = self.dataset.pixel_xy(px, py)
        
        # Retrieve pixels from the original data
        expected = []
        for frame in self.raw_data:  
            expected.append(frame[py][px])
        assert_almost_equal(arr, np.asarray(expected))

    def test_stats(self):
        # Results were precomputed using the seed 42 and then saved as txt
        # using np.savetxt. If failure ocurrs always check that the seed
        # is the one mentioned.
        res_path = os.path.dirname(__file__)
        mean = np.loadtxt(os.path.join(res_path, 'data', 'mean.txt'))
        std = np.loadtxt(os.path.join(res_path, 'data', 'std.txt'))
        median = np.loadtxt(os.path.join(res_path, 'data', 'median.txt'))
        linterp = np.loadtxt(os.path.join(res_path, 'data', 'linterp.txt'))

        assert_equal(self.dataset.mean(check_unique=['NAXIS1']), mean)
        assert_equal(self.dataset.std(check_unique=['NAXIS1']), std)
        assert_equal(self.dataset.median(check_unique=['NAXIS1']), median)
        assert_equal(self.dataset.lin_interp(45), linterp)

    def test_filter_chained(self):
        # Using filter in this fashion should be equivalent to use AND operations
        assert len(self.dataset.filter(naxis1=8).filter(naxis2=8)) == 5
        assert len(self.dataset.filter(naxis1=8).filter(naxis2=20)) == 0
