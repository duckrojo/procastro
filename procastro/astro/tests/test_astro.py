import pytest
from ..exoplanet import get_transit_ephemeris
from ..coordinates import find_target
import os


@pytest.mark.filterwarnings('ignore::UserWarning')  # Will catch Simbad warning
def test_read_coordinates():
    # Paths to test:
    # A: SkyCoords obtains RA and DEC correctly
    sky = find_target("1:12:43.2 +1:12:43")

    # B: SkyCoord fails | File contains RA and DEC associated to target name
    file = os.path.join(os.path.dirname(__file__), "transit_test.txt")
    sky = find_target("CoRoT_18_b", coo_files=file)

    assert sky.ra.value == 98.17240708333331
    assert sky.dec.value == -0.03159027777777778

    # C: SkyCoord and File fails | Simbad query finds target string
    sky = find_target("42 Dra b", coo_files=file)
    assert sky.ra.value == 276.49640958333333
    assert sky.dec.value == 65.56347305555555
    # D: All fail
    with pytest.raises(ValueError):         # Catches expected error
        sky = find_target("test", coo_files=file)


def test_get_transit_ephemeris():
    # File format testing
    # A: Correct order, all columns
    dir = os.path.dirname(__file__)
    epoch, period, length = get_transit_ephemeris("TEST CASE A", dir=dir)
    assert epoch == 2454679.33486
    assert period == 8.158719
    assert length == 24*0.11536

    # B: Correct order, missing columns
    epoch, period, length = get_transit_ephemeris("TEST CASE B", dir=dir)
    assert epoch == 2452826.62852
    assert period is None
    assert length == 184.2/60

    # C: Disorganized correct syntax
    epoch, period, length = get_transit_ephemeris("TEST CASE C", dir=dir)
    assert epoch == 2455983.70472
    assert period == 3.3366487
    assert length == 96.912/60

    # D: Unknown field
    with pytest.raises(ValueError):
        get_transit_ephemeris("TEST CASE D", dir=dir)
