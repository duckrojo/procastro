from dataproc.timeseries.deprecated.timeseries_alsodeprecated import TimeSeries
from numpy.testing import assert_equal, assert_almost_equal
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np
import pytest

class TestTimeSeries(object):
    # Both TimeSeries and TimeSeriesSingle are tested here
    def setup_method(self):
        # Generates random data
        np.random.seed(89)
        length = 20
        channels = ['a', 'b']
        n_targets = 3

        data = {'a': np.random.randint(200, 350, size=(n_targets, length)),
                'b': np.random.randint(180, 500, size=(n_targets, length))}
        errors = {'a': np.random.randint(20, 30, size=(n_targets, length))}   # Not all channels have errors
        epoch = np.linspace(2455525.6390277776, 2455525.6410011575, num=length)
        labels = ['1', '2', '3']
        self.TS = TimeSeries(data, errors=errors, default_info='a', epoch=epoch, labels=labels)
    
    # NOTE: Plot tests only check if the method is functional. Unable to obtain
    #       data from them
    @patch("matplotlib.pyplot.show")
    def test_plot(self, mock_show):
        mock_show.return_value = None
        self.TS.plot()
        
    @patch("matplotlib.pyplot.show")
    def test_plot_ratio(self, mock_show):
        mock_show.return_value = None
        self.TS.plot_ratio()

    def test_get_ratio(self):
        exp_ratio = np.array([1.29236499, 0.94650206, 1.35510204, 1.22862823, 1.00452489,
                            0.77388535, 1.13584906, 1.40880503, 1.04638219, 0.67462687,
                            0.93165468, 0.97029703, 0.97959184, 0.95483871, 1.        ,
                            1.09060403, 1.06719368, 0.85762144, 0.90793651, 1.06180666])
        exp_error = np.array([0.13356566, 0.12287716, 0.13194479, 0.11262309, 0.1454639 ,
                            0.09488672, 0.10803414, 0.14137032, 0.10415247, 0.08362799,
                            0.08897242, 0.10248718, 0.11884123, 0.10988774, 0.11209878,
                            0.1000929 , 0.11079955, 0.08668668, 0.09344216, 0.08545837])
        exp_sigma = 0.17903495668001645
        exp_errbar = 0.10896093822800271
        
        rc, rec, s, ebm = self.TS.get_ratio()
        
        assert_almost_equal(exp_ratio, rc)
        assert_almost_equal(exp_error, rec)
        assert_almost_equal(exp_sigma, s)
        assert_almost_equal(exp_errbar, ebm)
        
    def test_set_default_info(self):
        # Changes default
        old = self.TS[0].channels
        self.TS.set_default_info('b')
        new = self.TS[0].channels
        
        # Checks that the channels are actually different
        with pytest.raises(AssertionError):
            assert_equal(old, new)
            
    def test_JD(self):
        epoch = self.TS[0].JD(sector=[5,10])
        exp = np.array([2455525.63954709, 2455525.63965095, 2455525.63975481,
                        2455525.63985867, 2455525.63996254])
        assert_almost_equal(epoch, exp)

    def teardown_method(self):
        plt.close('all')
