import numpy as np
import pytest
from numpy.testing import assert_equal
from src import utils


def test_get_day_in_week_one_hot():
    dates = np.array([20230101.0, 20230210.0, 20231018.0, 20231230.0])

    days_in_week_one_hot = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]
    )

    assert_equal(utils.get_day_in_week_one_hot(dates), days_in_week_one_hot)
