import numpy as np

from .test_projections import _close_enough


def test__close_enough():
    assert _close_enough(np.array([1e-16, 2, 3]), np.array([0, 2, 3]),
                         tolerance=1e-15, verbose=True)
    assert not _close_enough(np.array([1e-16, 2, 3]), np.array([0, 2, 3.1]),
                             tolerance=1e-15, verbose=True)
    assert _close_enough(np.array([1e-16, 2, 3]), np.array([0, 2, 3]), np.array([2e-16, 2, 3]),
                         tolerance=1e-15, verbose=True)
