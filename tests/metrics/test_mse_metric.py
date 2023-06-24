import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from src.metrics import MeanSquaredError


def test_mse_update():
    mse_metric = MeanSquaredError()
    rmse_metric = MeanSquaredError(squared=False)

    targets = np.array([[-1.0, 0.0, 1.0]])
    predictions = np.array([-1.0, 0.0, 1.0])

    mse_metric.update(predictions, targets)
    rmse_metric.update(predictions, targets)
    assert_allclose(mse_metric.sse, np.array(0.0))
    assert_equal(mse_metric.n, np.array(3))
    assert_allclose(rmse_metric.sse, np.array(0.0))
    assert_equal(rmse_metric.n, np.array(3))

    predictions = np.array([1.0, 0.0, -1.0])
    mse_metric.update(predictions, targets)
    rmse_metric.update(predictions, targets)
    assert_allclose(mse_metric.sse, np.array(8.0))
    assert_equal(mse_metric.n, np.array(6))
    assert_allclose(rmse_metric.sse, np.array(8.0))
    assert_equal(rmse_metric.n, np.array(6))

    mse_metric.reset()
    assert_allclose(mse_metric.sse, np.array(0.0))
    assert_equal(mse_metric.n, np.array(0))


def test_mse_compute():
    mse_metric = MeanSquaredError()
    rmse_metric = MeanSquaredError(squared=False)

    targets = np.array([[-1.0, 0.0, 1.0]])
    predictions = np.array([-1.0, 0.0, 1.0])

    mse = mse_metric(predictions, targets)
    rmse = rmse_metric(predictions, targets)
    assert_allclose(mse, np.array(0.0))
    assert_allclose(rmse, np.array(0.0))

    predictions = np.array([1.0, 0.0, -1.0])
    mse = mse_metric(predictions, targets)
    rmse = rmse_metric(predictions, targets)
    assert_allclose(mse, np.array(4.0 / 3))
    assert_allclose(rmse, np.array(2.0 / 3**0.5))

    mse_metric.reset()
    rmse_metric.reset()
    mse = mse_metric.compute()
    rmse = rmse_metric.compute()
    assert_allclose(mse, np.array(np.nan))
    assert_allclose(rmse, np.array(np.nan))
