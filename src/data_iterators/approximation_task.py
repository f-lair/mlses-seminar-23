import math
from pathlib import Path
from typing import Callable

import numpy as np
from utils import CIRCUITS, read_data

import xgboost


class ApproximationTaskIterator(xgboost.DataIter):
    def __init__(self, data_dir: str, test: bool, batch_size: int):
        self.batch_size = batch_size
        self.num_iter = 0

        self.source_data, self.target_data = read_data(data_dir, test)
        self.max_iter = int(math.ceil(len(self.source_data) / self.batch_size))

        self.preprocess()

        # XGBoost will generate some cache files under current directory with the prefix "cache"
        super().__init__(cache_prefix=str(Path('.').joinpath('cache')))

    def next(self, input_data: Callable):
        """
        Advance the iterator by 1 step and pass the data to XGBoost.
        This function is called by XGBoost during the construction of ``DMatrix``.
        """

        if self.num_iter == self.max_iter:
            return 0  # return 0 to let XGBoost know this is the end of iteration

        X, y = load_svmlight_file(self._file_paths[self._it])
        input_data(X, y)

        self.num_iter += 1

        return 1  # return 1 to let XGBoost know we haven't seen all the files yet

    def reset(self):
        """Reset the iterator to its beginning."""

        self.num_iter = 0
