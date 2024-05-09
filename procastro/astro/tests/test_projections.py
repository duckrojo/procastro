import numpy as np
from ..projection import new_x_axis_at, unit_vector


def _close_enough(*scalars_or_vectors, tolerance=1e-10, verbose=True):
    prev = scalars_or_vectors[0]

    for vector in scalars_or_vectors[1:]:
        absolute = np.abs(prev - vector)
        if isinstance(absolute, np.ndarray):
            absolute = absolute.sum()
        if absolute > tolerance:
            if verbose:
                print(f"{prev} and {vector} are not close enough")
            return False

        prev = vector
        if verbose:
            print(f"{prev} and {vector} are close enough")

    return True


def test_rotate_xaxis_to():
    vector = unit_vector(0, 0, degrees=True)

    random_lat = np.random.randint(0, 90)
    random_lon = np.random.randint(-180, 180)
    assert _close_enough(new_x_axis_at(unit_vector(random_lon, random_lat), random_lon, random_lat),
                         np.array([1, 0, 0])
                         )

    only_y = new_x_axis_at(vector, -90, 0, 0)
    assert _close_enough(only_y, np.array([0, 1, 0]))

    only_z = new_x_axis_at(vector, 0, -90, 0)
    assert _close_enough(only_z, np.array([0, 0, 1]))

    y_equal_z = new_x_axis_at(vector, -90, 0, -45)
    assert _close_enough(y_equal_z[1], y_equal_z[2])

    y_equal_minus_z = new_x_axis_at(vector, -90, 0, 45)
    assert _close_enough(y_equal_minus_z[1], - y_equal_minus_z[2])
    assert y_equal_minus_z[2] < 0

    x_equal_minus_z = new_x_axis_at(vector, 0, 45, 0)
    assert _close_enough(x_equal_minus_z[0], - x_equal_minus_z[2])
    assert x_equal_minus_z[2] < 0

    x_equal_y_equal_z = new_x_axis_at(vector, -45, -45)
    assert _close_enough(x_equal_y_equal_z[0], x_equal_y_equal_z[2],
                         np.sqrt(x_equal_y_equal_z[1]**2/2))
