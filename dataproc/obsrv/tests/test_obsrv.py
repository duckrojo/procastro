from ..obsrv import Obsrv
from numpy.testing import assert_equal
from unittest.mock import patch
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import numpy as np
import pytest
import os
import pdb

class TestObsrv(object):
    
    @patch("matplotlib.pyplot.show")
    def setup_method(self, method, mock_show):
        mock_show.return_value = None
        self.obs = Obsrv("Kepler-150 d")

    # TODO: Verify if test are not   time dependant
    def test_get_closer_transit(self):
        result = self.obs.get_closer_transit(190, 89)
        assert result == 2457926.4925700002

    def test_update_plot(self):
        # _update_plot is called every time a setter is used, setters are 
        # already tested in test_obscalc so what remains here is to test if 
        # the decorator itself wont cause errors.

        self.obs.set_target("CoRoT-6 b")
        self.obs.set_timespan("2014-2016")
        self.obs.set_vertical(30)

    def teardown_method(self):
        del self.obs
        plt.close('all')
