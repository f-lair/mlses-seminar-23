import numpy as np
from metrics.base_metric import BaseMetric


class MeanAbsoluteError(BaseMetric):
    """Mean Absolute Error (MAE) metric with batch support"""

    def __init__(self, **kwargs) -> None:
        """
        Initializes metric.
        """

        super().__init__(**kwargs)

        self.add_state('sae', np.array(0.0))  # sum of absolute errors
        self.add_state('n', np.array(0))  # number of elements

    def update(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> None:
        """
        Updates metric internal state.

        Args:
            predictions (np.ndarray): Predictions.
            targets (np.ndarray): Targets.
        """

        delta = predictions - targets
        self.sae += np.sum(np.abs(delta))
        self.n += targets.size

    def compute(self) -> np.ndarray:
        """
        Computes metric result.

        Returns:
            np.ndarray: MAE.
        """

        return self.sae / self.n
