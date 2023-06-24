import math
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import xgboost


class DataIterator(xgboost.DataIter):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        window_size: int,
        create_windows: Callable[
            [np.ndarray, np.ndarray, np.ndarray, int],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool],
        ],
    ) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.window_size = window_size
        self.create_windows = create_windows

        self.num_iter = 0
        self.max_iter = int(math.ceil(len(self.X) / self.batch_size))

        super().__init__()

    def next(self, input_data: Callable):
        """
        Advance the iterator by 1 step and pass the data to XGBoost.
        This function is called by XGBoost during the construction of ``DMatrix``.
        """

        if self.num_iter == self.max_iter:
            return 0  # return 0 to let XGBoost know this is the end of iteration

        indices = np.arange(
            self.num_iter * self.batch_size,
            min((self.num_iter + 1) * self.batch_size, len(self.X)),
            dtype=int,
        )
        X_windows, y_windows, _, _, _, success = self.create_windows(
            self.X, self.y, indices, self.window_size
        )

        if success:
            input_data(data=X_windows, label=y_windows)

        self.num_iter += 1

        # print(f"{self.num_iter} / {self.max_iter}")

        return 1  # return 1 to let XGBoost know we haven't seen all the data yet

    def reset(self):
        """Reset the iterator to its beginning."""

        self.num_iter = 0
