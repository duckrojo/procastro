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

    if isinstance(angle, u.Quantity):
        return angle.to(u.deg).value

    return angle


def _angle_to_degree(angle, degrees=True):
    """
Converts to quantity if number, defaulting to degrees if True, otherwise radian
    Parameters
    ----------
    angle
    degrees
    """

    if isinstance(angle, (float, int)):
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

    return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


def rotate_xaxis_to(vector, lon, lat, z_pole_angle=0):
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

    """
    lat = _quantity_to_degree(lat)
    lon = _quantity_to_degree(lon)
    z_pole_angle = _quantity_to_degree(z_pole_angle)

    try:
        return Rotation.from_euler('zyx', [-lon, +lat, -z_pole_angle], degrees=True).apply(vector)
    except ValueError:
        raise ValueError(f"Invalid rotation angles: {lon}, {lat}, {z_pole_angle}")


