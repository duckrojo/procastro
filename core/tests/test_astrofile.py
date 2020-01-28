from ..astrofile import AstroFile, AstroCalib
from numpy.testing import assert_equal, assert_almost_equal
import astropy.io.fits as pf
import numpy as np
from .fit_factory import create_merge_example, create_random_fit, create_empty_fit
import pytest
import os
import pdb

class TestAstroCalib(object):
    
    @pytest.fixture(autouse=True)
    def setup_class(self,tmpdir):
        np.random.seed(61)
        with tmpdir.as_cwd():
            self.path = os.getcwd()
            # Creates target fit
            create_random_fit((8,8),
                                os.path.join(self.path, "target.fits"),
                                min_val = 500,
                                max_val = 2000)
            
            # Creates bias
            create_random_fit((8,8),
                                os.path.join(self.path, "bias.fits"),
                                header = pf.Header({'EXPTIME' : 20})
                                )
            
            # Creates filter
            create_random_fit((8,8),
                                os.path.join(self.path, "flat.fits"),
                                header = pf.Header({'FILTER' : 'I'})
                                )
                                
    def test_reduce(self, tmpdir):
        # Appends calibration files to AstroCalib                    
        calib = AstroCalib()
        
        bias = AstroFile(os.path.join(self.path,"bias.fits"))
        calib.add_bias(bias)
        assert_equal(calib.mbias[-1], AstroFile(os.path.join(self.path,"bias.fits")).reader())
        
        flat = AstroFile(os.path.join(self.path,"flat.fits"))
        calib.add_flat(flat)
        assert_equal(calib.mflat[''], AstroFile(os.path.join(self.path,"flat.fits")).reader())
            
        # Reduces data
        #TODO: Test different settings
        data = AstroFile(os.path.join(self.path, "target.fits"))
        
        res_path = os.path.dirname(__file__)
        expected = np.loadtxt(os.path.join(res_path, 'results', 'reduced.txt'))
        
        result = calib.reduce(data)
        assert_equal(result, expected)
        

class TestAstroFile(object):

    @pytest.fixture(autouse=True)
    def setup_class(self, tmpdir):
        np.random.seed(12)
        with tmpdir.as_cwd():
            self.path = os.getcwd()
            header = pf.Header({'JD' : 2457487.62537037,
                                'UT-DATE' : '2016-04-09',
                                'UT-TIME' : '03:02:30.817',
                                'DATE' : '2016-04-09T03:02:30.817'
                                })
            create_random_fit((20,20),
                                os.path.join(self.path, 
                                "file.fits"), 
                                header = header,
                                min_val = 50,
                                max_val = 100)
            create_empty_fit(os.path.join(self.path, "empty.fits"))
            self.file = AstroFile(os.path.join(self.path, "file.fits"))
       
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
        
        #Recover former values
        if edition[0] != 'test':
            dict = {edition[0] : current}
            self.file.setheader(**dict)
        
    def test_reader(self): 
        #Raw data
        f = pf.open(os.path.join(self.path,"file.fits"))
        expected = f[0].data
        assert_equal(self.file.reader(), expected)
        f.close()
        
        #Corrupt files return None
        corrupt = AstroFile(os.path.join(self.path,"empty.fits"))
        assert corrupt.reader() == None
        
    
    def test_load(self):
        #Load to empty file
        empty = AstroFile()
        filename = os.path.join(self.path,"file.fits")
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
    
    @pytest.mark.parametrize(("stats, output"), ([("min",[50]),
                                                   ("max",[99]),
                                                   ("mean3sclip",[6.252776074688882*10**-15]),
                                                   ("std",[14.434359831665553]),
                                                   ("median",[75.0])]))
    def test_stats(self, stats, output):
        assert_almost_equal(self.file.stats(stats), np.array(output))
    
    def test_stats_extra(self):
        assert self.file.stats("min", extra_headers = ["simple", "naxis"]) == [50, True, 2] 
    
    def test_merge(self):
        #Creates test image inside a temporary directory
        
        os.mkdir(os.path.join(self.path, "data"))
        src = os.path.join(self.path, "data", "merge_example.fits")
        
        create_merge_example(2048, 568, 4, src)
        
        #Creates a symbolic link to the file
        os.symlink(src, os.path.join(self.path, "merge_example.fits"))
        
        #Generates expected result
        target = pf.open(src)
        prev = len(target)
        expected = target[1].data
        
        for i in range(2, len(target)):
            expected = np.concatenate((expected, target[i].data), axis=1)
        target.close()
        
        #Create AstroFile pointing to file
        src = os.path.join(self.path, "merge_example.fits")
        af = AstroFile(src)
        
        #Merge ImageHDU's of image, saves data on current folder
        af.merger()
        
        #Compare result with expected data
        mod = pf.open(src)
        assert len(mod) == prev+1
        result = mod[-1].data
        assert_equal(result, expected)
        mod.close()
        
        #Merger wont execute if a file already has a composite image
        af.merger()
        again = pf.open(src)
        assert len(again) == prev+1
        again.close()
        