import numpy as np
import pytest
from numpy.testing import assert_equal
from src.tasks.forecast_task import ForecastTask


def test_create_windows():
    X = np.arange(60).reshape((5, 12))
    y = -np.arange(20).reshape((5, 4))
    indices = np.array([0, 1, 2, 3])
    window_size = 2
    horizon_size = 1

    X_windows, y_windows, _, _, _, success = ForecastTask.create_windows(
        None, X, y, indices, window_size, horizon_size  # type: ignore
    )
    X_windows_expected = np.array(
        [
            list(range(0, 24)) + list(range(0, -8, -1)),
            list(range(12, 36)) + list(range(-4, -12, -1)),
        ]
    )
    y_windows_expected = np.array(
        [
            list(range(24, 36)) + list(range(-8, -12, -1)),
            list(range(36, 48)) + list(range(-12, -16, -1)),
        ]
    )
    assert success
    assert_equal(X_windows, X_windows_expected)
    assert_equal(y_windows, y_windows_expected)

    indices = np.array([2, 3, 4])
    horizon_size = 2
    X_windows, y_windows, _, _, _, success = ForecastTask.create_windows(
        None, X, y, indices, window_size, horizon_size  # type: ignore
    )
    y_windows_expected = np.array(
        [
            list(range(24, 48)) + list(range(-8, -16, -1)),
            list(range(36, 60)) + list(range(-12, -20, -1)),
        ]
    )
    assert success
    assert_equal(X_windows, X_windows_expected)
    assert_equal(y_windows, y_windows_expected)

    window_size = 10
    horizon_size = 1
    X_windows, y_windows, _, _, _, success = ForecastTask.create_windows(
        None, X, y, indices, window_size, horizon_size  # type: ignore
    )
    assert not success

    window_size = 1
    horizon_size = 10
    X_windows, y_windows, _, _, _, success = ForecastTask.create_windows(
        None, X, y, indices, window_size, horizon_size  # type: ignore
    )
    assert not success
