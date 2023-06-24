import math
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from tasks.base_task import BaseTask
from tqdm import tqdm, trange
from utils import (
    CIRCUITS,
    VF_to_Qth,
    compute_metrics,
    create_metrics,
    get_hour_horizon_size,
)


class ForecastTask(BaseTask):
    """Class for the forecast tasks."""

    def create_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        window_size: int,
        horizon_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Creates windows for sliding window approach.
        For the forecast tasks, the source window is unidirectional into the past.
        It ends prior to the first target data point to be predicted.
        N: Batch dimension.
        B: Batch size (=number of windows).
        W1: Flattened source window dimension.
        W2: Flattened target window dimension.

        Args:
            X (np.ndarray): Source data [N, D1].
            y (np.ndarray): Target data [N, D2].
            indices (np.ndarray): Indices of targets to be used [B].
            window_size (int): Size of sliding window.
            horizon_size (int): Size of the horizon to be predicted.

        Returns:
            Tuple[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]: Windows for
                source and target data ([B, W1], [B, W2]); T1, T2, T3 features for ground-truth Qth
                computation [B]; success flag (windows non-empty).
        """

        N = len(X)  # number of data points
        B = len(indices)  # batch size = number of windows

        ## compute windows
        # for X, windows end prior to data points corresponding to the targets to be predicted and
        # include past targets
        # for y, windows start at data points/targets to be predicted and have size 'horizon_size'
        # too small windows are discarded
        valid_mask = np.logical_and(indices >= window_size, indices <= N - horizon_size)

        if not np.any(valid_mask):
            empty = np.array([])
            return empty, empty, empty, empty, empty, False

        X_windows = np.stack(
            [
                np.concatenate(
                    [
                        np.reshape(X[slice(indices[idx] - window_size, indices[idx])], (-1,)),
                        np.reshape(y[slice(indices[idx] - window_size, indices[idx])], (-1,)),
                    ]
                )
                for idx in range(B)
                if valid_mask[idx]
            ],
            axis=0,
        )  # B*([ws, D1], [ws, D2]) -> [B, ws*(D1+D2)]

        y_windows = np.stack(
            [
                np.concatenate(
                    [
                        np.reshape(X[slice(indices[idx], indices[idx] + horizon_size)], (-1,)),
                        np.reshape(y[slice(indices[idx], indices[idx] + horizon_size)], (-1,)),
                    ]
                )
                for idx in range(B)
                if valid_mask[idx]
            ],
            axis=0,
        )  # B*([hs, D1], [hs, D2]) -> [B, hs*(D1+D2)]

        T_indices = np.array([0, 3, 6, 9])  # cf. preprocess()
        T_indices = (
            T_indices[None, :] + np.arange(0, horizon_size * X.shape[1], X.shape[1])[:, None]
        ).reshape(
            (-1)
        )  # broadcast w.r.t. window dimension
        T1 = y_windows[:, T_indices]
        T2 = y_windows[:, T_indices + 1]
        T3 = y_windows[:, T_indices + 2]

        return X_windows, y_windows, T1, T2, T3, True


class ForecastStepTask(ForecastTask):
    """Class for the timestep forecast task."""

    def get_target_size(self) -> int:
        """
        Returns number of target features.

        Returns:
            int: Number of target features.
        """

        return 4 * len(CIRCUITS)  # T1, T2, T3, VF per circuit; width-1-horizon

    @staticmethod
    def get_task_name() -> str:
        """
        Returns task name.

        Returns:
            str: Task name.
        """

        return 'forecast_step'

    def create_windows(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Creates windows for sliding window approach.
        For the timestep forecast task, the window is unidirectional into the past.
        It ends prior to the target data point to be predicted.
        N: Batch dimension.
        B: Batch size (=number of windows).
        W1: Flattened source window dimension.
        W2: Flattened target window dimension.

        Args:
            X (np.ndarray): Source data [N, D1].
            y (np.ndarray): Target data [N, D2].
            indices (np.ndarray): Indices of targets to be used [B].
            window_size (int): Size of sliding window.

        Returns:
            Tuple[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]: Windows for
                source and target data ([B, W1], [B, W2]); T1, T2, T3 features for ground-truth Qth
                computation [B]; success flag (windows non-empty).
        """

        # call super method with width-1-horizon
        return super().create_windows(X, y, indices, window_size, horizon_size=1)

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

            # predict T1, T2, T3, VF
            y_pred = model.inplace_predict(X_windows, iteration_range=(0, model.best_iteration))

            # cf. create_windows() for shapes
            vf_target = y_windows[:, X_test.shape[1] :]
            vf_pred = y_pred[:, X_test.shape[1] :]

            # COMPUTE QTH USING PREDICTIONS
            qth_target = VF_to_Qth(vf_target, T1, T2, T3)
            # T1 - T3 have to be taken from predictions as well
            T_indices = np.array([0, 3, 6, 9])  # cf. preprocess()
            T1 = y_pred[:, T_indices]
            T2 = y_pred[:, T_indices + 1]
            T3 = y_pred[:, T_indices + 2]
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


class ForecastHourTask(ForecastTask):
    """Class for the hour forecast task."""

    def __init__(
        self,
        source_data: Dict[str, np.ndarray],
        target_data: Dict[str, np.ndarray],
        test: bool,
        time_features: bool,
        horizon_partition_size: int,
        verbose: bool,
    ) -> None:
        """
        Initializes hour forecast task.

        Args:
            source_data (Dict[str, np.ndarray]): Source data (circuit -> ndarray).
            target_data (Dict[str, np.ndarray]): Target data (circuit -> ndarray).
            test (bool): Whether test mode is active.
            time_features (bool): Whether additional time features (day in week) are used.
            horizon_partition_size (int): Size of partitions the horizon for the hour
                forecast task is divided into.
            verbose (bool): Whether verbose output is active.
        """

        if get_hour_horizon_size() % horizon_partition_size != 0:
            raise ValueError(
                f"Horizon partition size musst be a divider of {get_hour_horizon_size()}!"
            )

        self.horizon_partition_size = horizon_partition_size
        self.num_horizon_partitions = int(get_hour_horizon_size() // horizon_partition_size)

        super().__init__(source_data, target_data, test, time_features, verbose)

    def get_target_size(self) -> int:
        """
        Returns number of target features.

        Returns:
            int: Number of target features.
        """

        return (
            self.horizon_partition_size * 4 * len(CIRCUITS)
        )  # T1, T2, T3, VF per circuit; variable-width-horizon

    @staticmethod
    def get_task_name() -> str:
        """
        Returns task name.

        Returns:
            str: Task name.
        """

        return 'forecast_hour'

    def create_windows(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Creates windows for sliding window approach.
        For the hour forecast task, the window is unidirectional into the past.
        It ends prior to the first target data point to be predicted.
        N: Batch dimension.
        B: Batch size (=number of windows).
        W1: Flattened source window dimension.
        W2: Flattened target window dimension.

        Args:
            X (np.ndarray): Source data [N, D1].
            y (np.ndarray): Target data [N, D2].
            indices (np.ndarray): Indices of targets to be used [B].
            window_size (int): Size of sliding window.

        Returns:
            Tuple[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]: Windows for
                source and target data ([B, W1], [B, W2]); T1, T2, T3 features for ground-truth Qth
                computation [B]; success flag (windows non-empty).
        """

        # call super method with variable-width-horizon
        return super().create_windows(
            X, y, indices, window_size, horizon_size=self.horizon_partition_size
        )

    def fit(
        self,
        val_ratio: float,
        random_split: bool,
        num_rounds: int,
        num_early_stopping_rounds: int,
        learning_rate: float,
        max_depth: int,
        min_split_loss: float,
        batch_size: int,
        window_size: int,
        model_dir: str,
        model_name: str,
        rng_seed: int,
        **kwargs,
    ) -> None:
        """
        Performs model fitting process:
        Fits an XGBoost regression model in a supervised fashion.

        Args:
            val_ratio (float): Fractional size of the validation dataset, compared to the size of
                the original training dataset.
            random_split (bool): Whether train and validation datasets are split randomly.
            num_rounds (int): Max number of rounds (=trees) in the XGB model.
            num_early_stopping_rounds (int): Number of rounds without improvement after which
                fitting is stopped.
            learning_rate (float): Learning rate in XGBoost model fitting.
            max_depth (float): Max tree depth in the XGBoost model.
            min_split_loss (float): Minimum loss reduction needed for further leaf-predictions of
                the XGBoost model.
            batch_size (int): Data loading batch size.
            window_size (int): Size of sliding window.
            model_dir (str): Path to model files.
            model_name (str): Name of the model.
            rng_seed (int): Random number generator seed.
        """

        # size of horizon partitions needs to be greater or equal to window size to use previous
        # predictions for next predictions
        if self.horizon_partition_size < window_size:
            raise ValueError("Size of horizon paritions must not be smaller than window size!")

        # use teacher forcing for fitting (not previous predictions, but actual ground-truth
        # targets are used to predict the next horizon partition)

        return super().fit(
            val_ratio,
            random_split,
            num_rounds,
            num_early_stopping_rounds,
            learning_rate,
            max_depth,
            min_split_loss,
            batch_size,
            window_size,
            model_dir,
            model_name,
            rng_seed,
            **kwargs,
        )

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

        y_pred = None
        with tqdm(total=max_iter * self.num_horizon_partitions, disable=not self.verbose) as pbar:
            # iterate over batches
            for idx1 in range(max_iter):
                # create buffer to save predictions over entire horizon
                if save_first_n_predictions > 0:
                    vf_target_buffer = np.zeros(
                        (save_first_n_predictions, get_hour_horizon_size() * y_test.shape[1])
                    )
                    vf_pred_buffer = np.zeros_like(vf_target_buffer)
                    qth_target_buffer = np.zeros_like(vf_target_buffer)
                    qth_pred_buffer = np.zeros_like(vf_target_buffer)

                # iterate over horizon partitions
                for idx2 in range(self.num_horizon_partitions):
                    # compute predictions for batch
                    indices = np.arange(
                        idx1 * batch_size + idx2 * self.horizon_partition_size,
                        min(
                            (idx1 + 1) * batch_size + idx2 * self.horizon_partition_size,
                            len(X_test),
                        ),
                        dtype=int,
                    )
                    X_windows, y_windows, T1, T2, T3, success = self.create_windows(
                        X_test, y_test, indices, window_size
                    )

                    if not success:
                        continue

                    # DO NOT INCLUDE TEST TARGET DATA IN SOURCE DATA: REPLACE BY PREVIOUS
                    # PREDICTIONS
                    if y_pred is not None:
                        X_windows[-y_pred.shape[0] :, window_size * X_test.shape[1] :] = y_pred[
                            -X_windows.shape[0] :, window_size * X_test.shape[1] :
                        ]

                    # predict T1, T2, T3, VF
                    y_pred = model.inplace_predict(
                        X_windows, iteration_range=(0, model.best_iteration)
                    )

                    # cf. create_windows() for shapes
                    vf_target = y_windows[:, self.horizon_partition_size * X_test.shape[1] :]
                    vf_pred = y_pred[:, self.horizon_partition_size * X_test.shape[1] :]

                    # COMPUTE QTH USING PREDICTIONS
                    qth_target = VF_to_Qth(vf_target, T1, T2, T3)
                    # T1 - T3 have to be taken from predictions as well
                    T_indices = np.array([0, 3, 6, 9])  # cf. preprocess()
                    T_indices = (
                        T_indices[None, :]
                        + np.arange(
                            0, self.horizon_partition_size * X_test.shape[1], X_test.shape[1]
                        )[:, None]
                    ).reshape(
                        (-1)
                    )  # broadcast w.r.t. window dimension
                    T1 = y_pred[:, T_indices]
                    T2 = y_pred[:, T_indices + 1]
                    T3 = y_pred[:, T_indices + 2]
                    qth_pred = VF_to_Qth(vf_pred, T1, T2, T3)

                    # fill buffers to save predictions and targets on disk afterwards
                    if save_first_n_predictions > 0:
                        horizon_partition_numel = self.horizon_partition_size * y_test.shape[1]
                        vf_target_buffer[
                            : vf_target.shape[0],
                            idx2 * horizon_partition_numel : (idx2 + 1) * horizon_partition_numel,
                        ] = vf_target[:save_first_n_predictions, :]
                        vf_pred_buffer[
                            : vf_pred.shape[0],
                            idx2 * horizon_partition_numel : (idx2 + 1) * horizon_partition_numel,
                        ] = vf_pred[:save_first_n_predictions, :]
                        qth_target_buffer[
                            : qth_target.shape[0],
                            idx2 * horizon_partition_numel : (idx2 + 1) * horizon_partition_numel,
                        ] = qth_target[:save_first_n_predictions, :]
                        qth_pred_buffer[
                            : qth_pred.shape[0],
                            idx2 * horizon_partition_numel : (idx2 + 1) * horizon_partition_numel,
                        ] = qth_pred[:save_first_n_predictions, :]

                    # update metrics for batch
                    for circuit_id, circuit_name in CIRCUITS.items():
                        rmse_vf_metrics[circuit_name].update(
                            vf_pred[:, circuit_id], vf_target[:, circuit_id]
                        )
                        mae_vf_metrics[circuit_name].update(
                            vf_pred[:, circuit_id], vf_target[:, circuit_id]
                        )
                        rmse_qth_metrics[circuit_name].update(
                            qth_pred[:, circuit_id], qth_target[:, circuit_id]
                        )
                        mae_qth_metrics[circuit_name].update(
                            qth_pred[:, circuit_id], qth_target[:, circuit_id]
                        )
                    rmse_vf_metrics['total'].update(vf_pred, vf_target)
                    mae_vf_metrics['total'].update(vf_pred, vf_target)
                    rmse_qth_metrics['total'].update(qth_pred, qth_target)
                    mae_qth_metrics['total'].update(qth_pred, qth_target)

                    pbar.update(1)

                # save predictions and targets on disk
                if save_first_n_predictions > 0:
                    self.save_predictions(
                        vf_target_buffer,
                        vf_pred_buffer,
                        qth_target_buffer,
                        qth_pred_buffer,
                        model_dir,
                        model_name,
                    )

        # compute metrics
        rmse_vf, mae_vf, rmse_qth, mae_qth = compute_metrics(
            rmse_vf_metrics, mae_vf_metrics, rmse_qth_metrics, mae_qth_metrics
        )

        return rmse_vf, mae_vf, rmse_qth, mae_qth
