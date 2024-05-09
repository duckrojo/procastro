import numpy as np
from scipy.spatial.transform import Rotation
import astropy.units as u


def _quantity_to_degree(angle):
    """
Converts to quantity if number, defaulting to degrees if True, otherwise radian
    Parameters
    ----------
    angle
    degrees
    """

    if isinstance(angle, list):
        angle = np.array(angle)

    if isinstance(angle, u.Quantity):
        return angle.to(u.deg).value

    return angle


def _angle_to_degree(angle, degrees=True) -> u.Quantity:
    """
Converts to quantity if number, defaulting to degrees if True, otherwise radian
    Parameters
    ----------
    angle
    degrees
    """

    if isinstance(angle, list):
        angle = np.array(angle)

    if isinstance(angle, (float, int, np.ndarray)):
        return angle * (u.deg if degrees else u.rad)

    return angle


def unit_vector(lon, lat, degrees=True):
    """
Returns a unit vector with latitude, longitude = (0,0) towards X

    Parameters
    ----------
    lat
    lon
    degrees

    Returns
    -------

    """
    lat = _angle_to_degree(lat, degrees=degrees)
    lon = _angle_to_degree(lon, degrees=degrees)

    ret = np.array([np.cos(lat)*np.cos(lon),
                    np.cos(lat)*np.sin(lon),
                    np.sin(lat)*(lon.value*0+1)])

    if not lat.isscalar or not lon.isscalar:
        return ret.transpose()
    else:
        return ret


def current_x_axis_to(lon, lat, z_pole_angle=0) -> Rotation:
    """
    Rotates vector such that x-axis in the old geodesic coordinates rotates to lon, lat the new coordinate.

    Parameters
    ----------
    vector:
    3D vector to rotate

    lat:
    latitude of the old geodesic coordinate.

    lon:
    longitude of the old geodesic coordinate.

    z_pole_angle:
    angle from current Z-axis where the projected Z-pole would be found.

    See Also
    --------
    new_x_axis_at
     inverse rotation
    """
    lat = _quantity_to_degree(lat)
    lon = _quantity_to_degree(lon)
    z_pole_angle = _quantity_to_degree(z_pole_angle)

    try:
        return Rotation.from_euler('xyz', [z_pole_angle, -lat, lon], degrees=True)
    except ValueError:
        raise ValueError(f"Invalid rotation angles: {lon}, {lat}, {z_pole_angle}")


def new_x_axis_at(lon, lat, z_pole_angle=0):
    """
    Rotates vector such that lat, lon in the old geodesic coordinates becomes the x-axis in the new coordinate.

    Parameters
    ----------
    vector:
    3D vector to rotate

    lat:
    latitude of the old geodesic coordinate.

    lon:
    longitude of the old geodesic coordinate.

    z_pole_angle:
    angle from new Z-axis where the old Z-pole is found.

    See Also
    --------
    current_x_axis_at
     inverse rotation

    """
    lat = _quantity_to_degree(lat)
    lon = _quantity_to_degree(lon)
    z_pole_angle = _quantity_to_degree(z_pole_angle)

    try:
        return Rotation.from_euler('zyx', [-lon, +lat, -z_pole_angle], degrees=True)
    except ValueError:
        raise ValueError(f"Invalid rotation angles: {lon}, {lat}, {z_pole_angle}")


