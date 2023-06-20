from typing import Dict, Tuple

import numpy as np

from utils import TimerContext


class BaseTask:
    """Abstract base class for a task."""

    def __init__(
        self,
        source_data: Dict[str, np.ndarray],
        target_data: Dict[str, np.ndarray],
        test: bool,
        verbose: bool,
    ):
        """
        Initializes task.

        Args:
            source_data (Dict[str, np.ndarray]): Source data (circuit -> ndarray).
            target_data (Dict[str, np.ndarray]): Target data (circuit -> ndarray).
            test (bool): Whether test mode is active.
            verbose (bool): Whether verbose output is active.
        """

        self.test_ = test
        self.verbose = verbose
        self.task_name = self.get_task_name()

        # Preprocessing
        with TimerContext() as tc:
            self.X, self.y = self.preprocess(source_data, target_data)
        if verbose:
            self.print("Data preprocessed.", tc.elapsed_time())

    @staticmethod
    def get_task_name() -> str:
        """
        Returns task name.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete task.

        Returns:
            str: Task name.
        """

        raise NotImplementedError

    def preprocess(
        self, source_data: Dict[str, np.ndarray], target_data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses source and target data dictionaries, yielding two ndarrays suited for supervised model fitting.

        Args:
            source_data (Dict[str, np.ndarray]): Source data (circuit -> ndarray).
            target_data (Dict[str, np.ndarray]): Target data (circuit -> ndarray).

        Raises:
            NotImplementedError: Needs to be implemented for a concrete task.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Source data, target data (preprocessed).
        """

        raise NotImplementedError

    def fit(self, **kwargs) -> None:
        """
        Performs model fitting process.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete task.
        """

        raise NotImplementedError

    def test(self, **kwargs) -> None:
        """
        Performs model testing process.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete task.
        """

        raise NotImplementedError

    def run(self, **kwargs) -> None:
        """
        Runs task by either fitting a model or testing a fitted model.
        """

        if self.test_:
            self.test(**kwargs)
        else:
            self.fit(**kwargs)

    def print(self, message: str, elapsed_time: float | None = None) -> None:
        """
        Extended printing method, which incorporates the task name and (optionally) elapsed time.

        Args:
            message (str): Actual message to be printed.
            elapsed_time (float | None, optional): Elapsed time. Defaults to None.
        """

        if elapsed_time is not None:
            print(
                f"{self.task_name.capitalize()} Task: {message} Elapsed time: {elapsed_time:.3f}s."
            )
        else:
            print(f"{self.task_name.capitalize()} Task: {message}")
