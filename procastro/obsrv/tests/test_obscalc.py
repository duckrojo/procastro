from ..obscalc import ObsCalc
from numpy.testing import assert_equal, assert_almost_equal
import astropy.io.fits as pf
import numpy as np
import pytest
import os

class TestObsCalc(object):
    ### TODO: Check if decorators are updating related parameters for each setter
    def setup_method(self):
        self.obs = ObsCalc()

    def test_set_site(self):
        filename = os.path.join(os.path.dirname(__file__), "observatories.coo")
        # Recieves String
        self.obs.set_site('paranal', site_filename=filename)
        assert self.obs.params["lat_lon"] == (-24.6272, 289.5958)

        # Recieves tuple
        self.obs.set_site((-33.8688, 151.2093), site_filename=filename)
        assert self.obs.params["site"] == (-33.8688, 151.2093)

        # Unknown observatory
        with pytest.raises(KeyError):
            self.obs.set_site('test', site_filename=filename)

    def test_set_timespan(self):
        ed1 = np.asarray([42734.5, 42740.5, 42746.5, 42752.5, 42758.5, 42764.5, 42770.5,
                        42776.5, 42782.5, 42788.5, 42794.5, 42800.5, 42806.5, 42812.5,
                        42818.5, 42824.5, 42830.5, 42836.5, 42842.5, 42848.5, 42854.5,
                        42860.5, 42866.5, 42872.5, 42878.5, 42884.5, 42890.5, 42896.5,
                        42902.5, 42908.5, 42914.5, 42920.5, 42926.5, 42932.5, 42938.5,
                        42944.5, 42950.5, 42956.5, 42962.5, 42968.5, 42974.5, 42980.5,
                        42986.5, 42992.5, 42998.5, 43004.5, 43010.5, 43016.5, 43022.5,
                        43028.5, 43034.5, 43040.5, 43046.5, 43052.5, 43058.5, 43064.5,
                        43070.5, 43076.5, 43082.5, 43088.5, 43094.5])
        ed2 = np.asarray([40177.5, 40213.5, 40249.5, 40285.5, 40321.5, 40357.5, 40393.5,
                          40429.5, 40465.5, 40501.5, 40537.5, 40573.5, 40609.5, 40645.5,
                          40681.5, 40717.5, 40753.5, 40789.5, 40825.5, 40861.5, 40897.5,
                          40933.5, 40969.5, 41005.5, 41041.5, 41077.5, 41113.5, 41149.5,
                          41185.5, 41221.5, 41257.5, 41293.5, 41329.5, 41365.5, 41401.5,
                          41437.5, 41473.5, 41509.5, 41545.5, 41581.5, 41617.5, 41653.5,
                          41689.5, 41725.5, 41761.5, 41797.5, 41833.5, 41869.5, 41905.5,
                          41941.5, 41977.5, 42013.5, 42049.5, 42085.5, 42121.5, 42157.5,
                          42193.5, 42229.5, 42265.5, 42301.5, 42337.5])
        # Recieves int
        self.obs.set_timespan(2017)
        assert_equal(self.obs.days, ed1)
        assert_equal(self.obs.xlims, np.asarray([0.0, 360.0]))
        assert self.obs.jd0 == 2457754.5

        # Recieves string
        self.obs.set_timespan("2010-2015")
        assert_equal(self.obs.days, ed2)
        assert_equal(self.obs.xlims, [0.0, 2160.0])
        assert self.obs.jd0 == 2455197.5

        # Reversed timespan should raise an error
        with pytest.raises(ValueError):
            self.obs.set_timespan("2015-2010")

    @pytest.mark.parametrize(('target', 'expected'),
                            [("WASP-8 b", {'length': 24*0.11536,  # All data is available
                                           'epoch': 2454679.33486,
                                           'period': 8.158719,
                                           'offset': 0.0}),
                             ("CoRoT-6 b", {'length': 4.08,    #Period is missing
                                              'epoch': 2454595.6144,
                                              'period': 8.886593,
                                              'offset': 0.0}),
                            ("Kepler-150 d", {'length': 3.763200,   #Transit is not known
                                             'epoch': 2454999.795880,
                                             'period': 12.56093,
                                             'offset': 0.0})])
    def test_set_target(self, target, expected):
        self.obs.set_target(target)
        assert self.obs.transit_info == expected

    def test_set_target_error(self):
        with pytest.raises(ValueError):
            self.obs.set_target("WASP-3 b")   # NASA returns masked array on trandur
    
    #TODO: Add more parametrized inputs
    def test_set_transit(self):
        exp_tr=np.array([2457750, 2457800, 2457850, 2457900, 2457950, 
                                2458000, 2458050, 2458100, 2458150])
        
        exp_hours= np.array([12., 12., 12., 12., 12., 12., 12., 12., 12.])
        
        self.obs.set_target("WASP-8 b")
        self.obs.set_transits(tr_period = 50, tr_epoch = 100)
        
        assert_equal(self.obs.transits, exp_tr)
        assert_equal(self.obs.transit_hours, exp_hours)
        
    def teardown_method(self):
        del self.obs
