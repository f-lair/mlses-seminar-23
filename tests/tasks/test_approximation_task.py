import numpy as np
import pytest
from numpy.testing import assert_equal
from src.tasks import ApproximationTask


def test_create_windows():
    X = np.arange(60).reshape((5, 12))
    y = -np.arange(20).reshape((5, 4))
    indices = np.array([0, 1, 2])
    window_size = 2

    with pytest.raises(ValueError):
        windows = ApproximationTask.create_windows(
            None, X, y, indices, window_size  # type: ignore
        )

    window_size = 3
    X_windows, y_windows, _, _, _, success = ApproximationTask.create_windows(
        None, X, y, indices, window_size  # type: ignore
    )
    X_windows_expected = np.array([list(range(0, 36)), list(range(12, 48))])
    y_windows_expected = np.array([list(range(-4, -8, -1)), list(range(-8, -12, -1))])
    assert success
    assert_equal(X_windows, X_windows_expected)
    assert_equal(y_windows, y_windows_expected)

    indices = np.array([2, 3, 4])
    X_windows, y_windows, _, _, _, success = ApproximationTask.create_windows(
        None, X, y, indices, window_size  # type: ignore
    )
    X_windows_expected = np.array([list(range(12, 48)), list(range(24, 60))])
    y_windows_expected = np.array([list(range(-8, -12, -1)), list(range(-12, -16, -1))])
    assert success
    assert_equal(X_windows, X_windows_expected)
    assert_equal(y_windows, y_windows_expected)

    window_size = 11
    X_windows, y_windows, _, _, _, success = ApproximationTask.create_windows(
        None, X, y, indices, window_size  # type: ignore
    )
    assert not success
