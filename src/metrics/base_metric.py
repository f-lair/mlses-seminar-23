from typing import Any

import numpy as np


class BaseMetric:
    """Abstract base class for metrics with batch support."""

    def __init__(self, **kwargs) -> None:
        self.default_states = {}

    def update(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> None:
        """
        Updates metric internal state.

        Args:
            predictions (np.ndarray): Predictions.
            targets (np.ndarray): Targets.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete metric.
        """

        raise NotImplementedError

    def compute(self) -> np.ndarray:
        """
        Computes metric result.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete metric.

        Returns:
            np.ndarray: Metric result.
        """

        raise NotImplementedError

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
        """
        Updates metric internal state and returns new result.

        Args:
            predictions (np.ndarray): Predictions.
            targets (np.ndarray): Targets.

        Returns:
            np.ndarray: Metric result.
        """

        self.update(predictions, targets, **kwargs)
        return self.compute()

    def add_state(self, name: str, default: np.ndarray) -> None:
        self.default_states[name] = default.copy()
        self.__dict__[name] = default.copy()

    def reset(self) -> None:
        for name in self.default_states:
            self.__dict__[name] = self.default_states[name]
