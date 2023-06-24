import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from src.metrics import MeanAbsoluteError


def test_mae_update():
    mae_metric = MeanAbsoluteError()

    targets = np.array([[-1.0, 0.0, 1.0]])
    predictions = np.array([-1.0, 0.0, 1.0])

    mae_metric.update(predictions, targets)
    assert_allclose(mae_metric.sae, np.array(0.0))
    assert_equal(mae_metric.n, np.array(3))

    predictions = np.array([1.0, 0.0, -1.0])
    mae_metric.update(predictions, targets)
    assert_allclose(mae_metric.sae, np.array(4.0))
    assert_equal(mae_metric.n, np.array(6))

    mae_metric.reset()
    assert_allclose(mae_metric.sae, np.array(0.0))
    assert_equal(mae_metric.n, np.array(0))


def test_mae_compute():
    mae_metric = MeanAbsoluteError()

    targets = np.array([[-1.0, 0.0, 1.0]])
    predictions = np.array([-1.0, 0.0, 1.0])

    mae = mae_metric(predictions, targets)
    assert_allclose(mae, np.array(0.0))

    predictions = np.array([1.0, 0.0, -1.0])
    mae = mae_metric(predictions, targets)
    assert_allclose(mae, np.array(2.0 / 3))

    mae_metric.reset()
    mae = mae_metric.compute()
    assert_allclose(mae, np.array(np.nan))
