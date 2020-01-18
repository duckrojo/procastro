from ..astrofile import AstroFile, AstroCalib, merger
from numpy.testing import assert_equal
import astropy.io.fits as pf
import numpy as np
import fit_factory as ff
import pytest
import os
import pdb
class TestAstroCalib(object):
    
    def test_reduce(self):
        calib = AstroCalib()
        bias = AstroFile(".\\test_dir\\astrofile\\bias.fits") 
        calib.add_bias(bias)
        assert_equal(calib.mbias[-1], AstroFile(".\\test_dir\\astrofile\\bias.fits").reader())
        
        flat = AstroFile(".\\test_dir\\astrofile\\flat.fits")
        calib.add_flat(flat)
        assert_equal(calib.mflat[''], AstroFile(".\\test_dir\\astrofile\\flat.fits").reader())

        #TODO: Test different settings
        data = AstroFile(".\\test_dir\\astrofile\\file.fits").reader()
        
        result = calib.reduce(data)
        assert_equal(result, AstroFile(".\\test_dir\\results\\reduced.fits").reader())
        

class TestAstroFile(object):
    
    def setup_method(self):
        self.file = AstroFile(".\\test_dir\\astrofile\\file.fits")
        
    @pytest.mark.parametrize(("current, edition"), 
                            [("2016-04-09T03:02:30.817", ("date", "01/14/2020")),
                             (None, ("test", "test")), 
                             (True, ("simple", None))])
    def test_header_accessors(self, current, edition):
        #TODO: Include cases where headers have multiple keys with same name, include
        #      case described on docstring
        val = self.file.getheaderval(edition[0])
        assert val == current 

        dict = {edition[0] : edition[1]}
        self.file.setheader(**dict)
        val = self.file.getheaderval(edition[0])
        assert val == edition[1]
        
    def test_reader(self): 
        #Raw data
        f = pf.open(".\\test_dir\\astrofile\\file.fits")
        expected = f[0].data
        assert_equal(self.file.reader(), expected)
        f.close()
        
        #Corrupt files return None
        corrupt = AstroFile(".\\test_dir\\astrofile\\corrupt.fits")
        assert corrupt.reader() == None
        
    
    def test_load(self):
        #Load to empty file
        empty = AstroFile()
        filename = ".\\test_dir\\astrofile\\file.fits"
        data = pf.getheader(filename)
        empty.load(filename, data)
        assert empty.readheader() == self.file.readheader()
        with pytest.raises(ValueError):
            empty.load(filename, data)
    
    @pytest.mark.skip(reason="Pending method refactoring")
    def test_writer(self):
        #No header given
        blank = AstroFile("blank.fits")
        data = self.file.reader()
        blank.writer(data)
        assert_equal(blank.reader(), data)
    
    def test_jd_from_ut(self):
        expected = 2457487.626745567
        
        #UT time is located in one header value
        self.file.jd_from_ut(target="test", source="date")  
        val = self.file.getheaderval("test")
        assert val == expected
        
        #UT time is split between two keys
        self.file.jd_from_ut(target ="test2", source = ["ut-date", "ut-time"])
        split_val = self.file.getheaderval("test2")
        assert split_val == expected
    
    
    def test_checktype(self):
        fake = AstroFile("fake.fits")
        assert self.file.checktype(exists=True) == 'fits'
        assert fake.checktype(exists=True) == None
        assert fake.checktype(exists=False) == 'fits'
        assert AstroFile().checktype(exists=True) == None
    
    @pytest.mark.parametrize(("kwargs, result"), ([({"SIMPLE" : True}, True),
                                                    ({"JD": 2457487.62537037}, True),
                                                    ({"TEST": 1}, False),
                                                    ({"NAXIS1": 20, "TEST": 5}, True),
                                                    ({"NAXIS2_LT": 10}, False),
                                                    ({"NAXIS2_GT": 10}, True),
                                                    ({"UT__DATE_MATCH": "2016-04-09"}, True),
                                                    ({"NAXIS_EQUAL": 2}, True),
                                                    ({"DATE_EQUAL_ICASE": "2016-04-09T03:02:30.817"}, True),
                                                    ({"DATE_BEGIN": "2016"}, True), 
                                                    ({"DATE_END": "817"}, True)]))
    def test_filter(self, kwargs, result):
        assert self.file.filter(**kwargs) == result
        
    @pytest.mark.parametrize(("stats, output"), ([("min",[12718]),
                                                   ("max",[14024]),
                                                   ("mean3sclip",[-2.10239348370971]),
                                                   ("std",[274.9602134400539]),
                                                   ("median",[13130.0])]))
    def test_stats(self, stats, output):
        assert self.file.stats(stats) == output
    
    def test_stats_extra(self):
        assert self.file.stats("min", extra_headers = ["simple", "naxis"]) == [12718, True, 2] 
    
    def test_merge(self, tmpdir):
        #Creates test image inside a temporary directory
        with tmpdir.as_cwd():
            os.mkdir("data")
            src = ".\\data\\merge_example.fits"
            
            ff.create_merge_example(2048, 568, 4, src)
            
            #Creates a symbolic link to the file
            os.symlink(".\\data\\merge_example.fits", ".\\merge_example.fits")
            
            #Generates expected result
            target = pf.open(src)
            prev = len(target)
            expected = target[1].data
            
            for i in range(2, len(target)):
                expected = np.concatenate((expected, target[i].data), axis=1)
            target.close()
            
            #Merge ImageHDU's of image, saves data on current folder
            src = "merge_example.fits"
            merger(src, prev)
            
            #Compare result with expected data
            mod = pf.open(src)
            assert len(mod) == prev+1
            result = mod[-1].data
            assert_equal(result, expected)
            mod.close()
        
        
    def teardown_method(self):
        del self.file
        