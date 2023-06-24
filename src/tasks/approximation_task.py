import math
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from tasks.base_task import BaseTask
from tqdm import trange
from utils import CIRCUITS, VF_to_Qth, compute_metrics, create_metrics


class ApproximationTask(BaseTask):
    """Class for the function approximation task."""

    def get_target_size(self) -> int:
        """
        Returns number of target features.

        Returns:
            int: Number of target features.
        """

        return len(CIRCUITS)  # VF per circuit

    @staticmethod
    def get_task_name() -> str:
        """
        Returns task name.

        Returns:
            str: Task name.
        """

        return 'approximation'

    def create_windows(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Creates windows for sliding window approach.
        For the approximation task, the window is bidirectional.
        It is centered at the target data point to be predicted.
        N: Batch dimension.
        B: Batch size (=number of windows).
        D1: Source feature dimension.
        D2: Target feature dimension.
        W1: Flattened source window dimension.
        W2: Flattened target window dimension.

        Args:
            X (np.ndarray): Source data [N, D1].
            y (np.ndarray): Target data [N, D2].
            indices (np.ndarray): Indices of targets to be used [B].
            window_size (int): Size of sliding window.

        Returns:
            Tuple[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]: Windows for
                source and target data ([B, W1], [B, W2]); T1, T2, T3 features for ground-truth
                Qth computation [B, 4]; success flag (windows non-empty).
        """

        # only odd window sizes permitted, as window is centered in data points
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd for the approximation task!")

        N = len(X)  # number of data points
        B = len(indices)  # batch size = number of windows
        r = window_size // 2  # half window width

        ## compute windows
        # for X, windows are centered in data points corresponding to the targets to be predicted
        # i.e., for y, windows are simply index-selected targets
        # too small windows are discarded
        valid_mask = np.logical_and(indices >= r, indices < N - r)

        if not np.any(valid_mask):
            empty = np.array([])
            return empty, empty, empty, empty, empty, False

        X_windows = np.stack(
            [
                np.reshape(X[slice(indices[idx] - r, indices[idx] + r + 1)], (-1,))
                for idx in range(B)
                if valid_mask[idx]
            ],
            axis=0,
        )  # B*[2r+1, D1] -> [B, (2r+1)*D1]

        y_windows = y[indices[valid_mask]]

        X_valid = X[indices[valid_mask]]
        T_indices = np.array([0, 3, 6, 9])  # cf. preprocess()
        T1 = X_valid[:, T_indices]
        T2 = X_valid[:, T_indices + 1]
        T3 = X_valid[:, T_indices + 2]

        return X_windows, y_windows, T1, T2, T3, True

    def test_predict(
        self,
        model: xgb.Booster,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int,
        window_size: int,
        model_dir: str,
        model_name: str,
        save_first_n_predictions: int,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Computes predictions and metric results for test data.
        N: Batch dimension.
        D1: Source feature dimension.
        D2: Target feature dimension.

        Args:
            model (xgb.Booster): XGBoost model.
            X_test (np.ndarray): Test source data [N, D1].
            y_test (np.ndarray): Test target data [N, D2].
            batch_size (int): Data loading batch size.
            window_size (int): Size of sliding window.
            model_dir (str): Path to model files.
            model_name (str): Name of the model.
            save_first_n_predictions (int): Saves first n predictions of test data on disk.

        Returns:
            Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]: RMSE
                (VF), MAE (VF), RMSE (Qth), MAE (Qth) (keys circuit_names, 'total' per dict).
        """

        max_iter = int(math.ceil(len(X_test) / batch_size))

        # set up metrics
        rmse_vf_metrics, mae_vf_metrics, rmse_qth_metrics, mae_qth_metrics = create_metrics()

        # iterate over batches
        for idx in trange(max_iter, disable=not self.verbose):
            # compute predictions for batch
            indices = np.arange(
                idx * batch_size, min((idx + 1) * batch_size, len(X_test)), dtype=int
            )
            X_windows, y_windows, T1, T2, T3, success = self.create_windows(
                X_test, y_test, indices, window_size
            )

            if not success:
                continue

            # predict only VF
            y_pred = model.inplace_predict(X_windows, iteration_range=(0, model.best_iteration))
            vf_target = y_windows
            vf_pred = y_pred

            # COMPUTE QTH USING PREDICTIONS
            qth_target = VF_to_Qth(vf_target, T1, T2, T3)
            qth_pred = VF_to_Qth(vf_pred, T1, T2, T3)

            # save predictions and targets on disk
            if save_first_n_predictions > 0:
                self.save_predictions(
                    vf_target[:save_first_n_predictions],
                    vf_pred[:save_first_n_predictions],
                    qth_target[:save_first_n_predictions],
                    qth_pred[:save_first_n_predictions],
                    model_dir,
                    model_name,
                )

            # update metrics for batch
            for circuit_id, circuit_name in CIRCUITS.items():
                rmse_vf_metrics[circuit_name].update(
                    y_pred[:, circuit_id], y_windows[:, circuit_id]
                )
                mae_vf_metrics[circuit_name].update(
                    y_pred[:, circuit_id], y_windows[:, circuit_id]
                )
                rmse_qth_metrics[circuit_name].update(
                    qth_pred[:, circuit_id], qth_target[:, circuit_id]
                )
                mae_qth_metrics[circuit_name].update(
                    qth_pred[:, circuit_id], qth_target[:, circuit_id]
                )
            rmse_vf_metrics['total'].update(y_pred, y_windows)
            mae_vf_metrics['total'].update(y_pred, y_windows)
            rmse_qth_metrics['total'].update(qth_pred, qth_target)
            mae_qth_metrics['total'].update(qth_pred, qth_target)

        # compute metrics
        rmse_vf, mae_vf, rmse_qth, mae_qth = compute_metrics(
            rmse_vf_metrics, mae_vf_metrics, rmse_qth_metrics, mae_qth_metrics
        )

        return rmse_vf, mae_vf, rmse_qth, mae_qth
