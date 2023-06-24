import numpy as np
from metrics.base_metric import BaseMetric


class MeanSquaredError(BaseMetric):
    """(Root) Mean Squared Error (RMSE / MSE) metric with batch support"""

    def __init__(self, squared=True, **kwargs) -> None:
        """
        Initializes metric.

        Args:
            squared (bool, optional): Whether result is squared (MSE). Defaults to True.
        """

        super().__init__(**kwargs)

        self.squared = squared
        self.add_state('sse', np.array(0.0))  # sum of squared errors
        self.add_state('n', np.array(0))  # number of elements

    def update(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> None:
        """
        Updates metric internal state.

        Args:
            predictions (np.ndarray): Predictions.
            targets (np.ndarray): Targets.
        """

        delta = predictions - targets
        self.sse += np.sum(delta * delta)
        self.n += targets.size

    def compute(self) -> np.ndarray:
        """
        Computes metric result.

        Returns:
            np.ndarray: RMSE / MSE.
        """

        if self.squared:
            return self.sse / self.n
        else:
            return np.sqrt(self.sse / self.n)
