import math
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from data_iterator import DataIterator
from utils import (
    CIRCUITS,
    TimerContext,
    get_day_in_week_one_hot,
    get_model_filepath,
    write_logs,
)


class BaseTask:
    """Abstract base class for a task."""

    def __init__(
        self,
        source_data: Dict[str, np.ndarray],
        target_data: Dict[str, np.ndarray],
        test: bool,
        time_features: bool,
        verbose: bool,
    ):
        """
        Initializes task.

        Args:
            source_data (Dict[str, np.ndarray]): Source data (circuit -> ndarray).
            target_data (Dict[str, np.ndarray]): Target data (circuit -> ndarray).
            test (bool): Whether test mode is active.
            time_features (bool): Whether additional time features (day in week) are used.
            verbose (bool): Whether verbose output is active.
        """

        self._test = test
        self.time_features = time_features
        self.verbose = verbose
        self.task_name = self.get_task_name()

        # Preprocessing
        with TimerContext() as tc:
            self.X, self.y = self.preprocess(source_data, target_data)
        if verbose:
            self.print("Data preprocessed.", tc.elapsed_time())

    def get_target_size(self) -> int:
        """
        Returns number of target features.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete task.

        Returns:
            int: Number of target features.
        """

        raise NotImplementedError

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

    def create_windows(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Creates windows for sliding window approach.
        N: Batch dimension.
        B: Batch size (=number of windows).
        W1: Flattened source window dimension.
        W2: Flattened target window dimension.

        Args:
            X (np.ndarray): Source data [N, D1].
            y (np.ndarray): Target data [N, D2].
            indices (np.ndarray): Indices of targets to be used [B].
            window_size (int): Size of sliding window.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete task.

        Returns:
            Tuple[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]: Windows for
                source and target data ([B, W1], [B, W2]); T1, T2, T3 features for ground-truth
                Qth computation [B]; success flag (windows non-empty).
        """

        raise NotImplementedError

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

        Raises:
            NotImplementedError: Needs to be implemented for a concrete task.

        Returns:
            Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]: RMSE
                (VF), MAE (VF), RMSE (Qth), MAE (Qth) (keys circuit_names, 'total' per dict).
        """

        raise NotImplementedError

    def preprocess(
        self, source_data: Dict[str, np.ndarray], target_data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses source and target data dictionaries, yielding two ndarrays.
        First dimension of returned arrays has equal size, due to averaging over the source data
        time line.
        Returned arrays are structured as follows:
        - Shapes are [N, D1] for source_data and [N, D2] for target_data
            - N: Batch dimension (incorporating old batch and time dimensions)
            - D1: Source feature dimension (12 / 18)
                - 0, 3, 6, 9: T1 of solar, boiler, water, heating (respectively)
                - 1, 4, 7, 10: T2 of solar, boiler, water, heating (respectively)
                - 2, 5, 8, 11: T3 of solar, boiler, water, heating (respectively)
                - 12 - 17: One-hot encoding of time feature 'day in week (0-6)' (optional)
            - D2: Target feature dimension (4)
                - 0, 1, 2, 3: VF of solar, boiler, water, heating (respectively)

        Args:
            source_data (Dict[str, np.ndarray]): Source data (circuit -> ndarray).
            target_data (Dict[str, np.ndarray]): Target data (circuit -> ndarray).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Source data, target data (preprocessed).
        """

        source_data_list = []
        target_data_list = []

        for circuit in CIRCUITS.values():
            # make source shape equal to target shape by averaging over data points
            temp_shape = target_data[circuit].shape + (50,)
            source_data[circuit] = np.average(
                np.reshape(source_data[circuit], temp_shape), axis=-1
            )
            # target_data[circuit] = np.repeat(target_data[circuit], repeats=50, axis=1)

            # flatten over first two dimensions
            source_data[circuit] = np.reshape(source_data[circuit], (-1, 4))
            target_data[circuit] = np.reshape(target_data[circuit], (-1, 4))

            # add T1, T2, T3 to source data list
            source_data_list.append(source_data[circuit][:, 1])  # T1
            source_data_list.append(source_data[circuit][:, 2])  # T2
            source_data_list.append(target_data[circuit][:, 2])  # T3

            # add VF to target data list
            target_data_list.append(target_data[circuit][:, 1])  # VF

        if self.time_features:
            # get one-hot-encoding of 'day in week'
            day_in_week_one_hot = get_day_in_week_one_hot(
                source_data[next(iter(CIRCUITS.values()))][:, 0]
            )

        # stack along feature dimension
        source_data = np.stack(source_data_list, axis=1)  # type: ignore
        target_data = np.stack(target_data_list, axis=1)  # type: ignore

        if self.time_features:
            # add 'day in week' to source data
            source_data = np.concatenate(
                [source_data, day_in_week_one_hot], axis=1  # type: ignore
            )

        return source_data, target_data  # type: ignore

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

        X_train = self.X
        y_train = self.y

        # split into training, validation datasets

        (X_train, y_train), (X_val, y_val) = self.create_data_split(
            X_train, y_train, val_ratio, random_split, rng_seed
        )

        # free RAM
        del self.X
        del self.y

        # set up data iterators
        with TimerContext() as tc:
            train_iterator = DataIterator(
                X_train, y_train, batch_size, window_size, self.create_windows
            )
            train_data = xgb.DMatrix(train_iterator)
            val_iterator = DataIterator(X_val, y_val, batch_size, window_size, self.create_windows)
            val_data = xgb.DMatrix(val_iterator)
        if self.verbose:
            self.print("Data batched.", tc.elapsed_time())

        # fit model
        if self.verbose:
            print("[Round]")
        logs = {}
        with TimerContext() as tc:
            model = xgb.train(
                {
                    'tree_method': 'hist',
                    'num_target': self.get_target_size(),
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'min_split_loss': min_split_loss,
                    'seed': rng_seed,
                },
                dtrain=train_data,
                num_boost_round=num_rounds,
                early_stopping_rounds=num_early_stopping_rounds,
                evals=[(train_data, 'training'), (val_data, 'validation')],
                evals_result=logs,
                verbose_eval=self.verbose,
            )

        if self.verbose:
            self.print("Model fitted.", tc.elapsed_time())

        # save fitted model
        save_path = get_model_filepath(model_dir, self.get_task_name(), model_name, '.json')
        with TimerContext() as tc:
            model.save_model(save_path)
        if self.verbose:
            self.print("Model saved.", tc.elapsed_time())

        # Save logs
        with TimerContext() as tc:
            write_logs(logs, model_dir, self.get_task_name(), model_name)
        if self.verbose:
            self.print("Logs written.", tc.elapsed_time())

    def test(
        self,
        batch_size: int,
        window_size: int,
        model_dir: str,
        model_name: str,
        save_first_n_predictions: int,
        **kwargs,
    ) -> None:
        """
        Performs model testing process:
        Loads fitted XGB regression model and predicts outputs for unseen test data.

        Args:
            batch_size (int): Data loading batch size.
            window_size (int): Size of sliding window.
            model_dir (str): Path to model files.
            model_name (str): Name of the model.
            save_first_n_predictions (int): Saves first n predictions of test data on disk.
        """

        X_test = self.X
        y_test = self.y

        # load model
        load_path = get_model_filepath(model_dir, self.get_task_name(), model_name, '.json')
        model = xgb.Booster()
        with TimerContext() as tc:
            model.load_model(load_path)
        if self.verbose:
            self.print("Model loaded.", tc.elapsed_time())

        # predict on test data
        with TimerContext() as tc:
            rmse_vf, mae_vf, rmse_qth, mae_qth = self.test_predict(
                model,
                X_test,
                y_test,
                batch_size,
                window_size,
                model_dir,
                model_name,
                save_first_n_predictions,
            )
        if self.verbose:
            self.print("Predictions computed.", tc.elapsed_time())

        # print results
        for circuit_name in CIRCUITS.values():
            self.print(f"RMSE (VF_{circuit_name}) = {rmse_vf[circuit_name]:.5f}")
            self.print(f"MAE (VF_{circuit_name}) = {mae_vf[circuit_name]:.5f}")
            self.print(f"RMSE (Qth_{circuit_name}) = {rmse_qth[circuit_name]:.5f}")
            self.print(f"MAE (Qth_{circuit_name}) = {mae_qth[circuit_name]:.5f}")
        self.print(f"RMSE (VF_total) = {rmse_vf['total']:.5f}")
        self.print(f"MAE (VF_total) = {mae_vf['total']:.5f}")
        self.print(f"RMSE (Qth_total) = {rmse_qth['total']:.5f}")
        self.print(f"MAE (Qth_total) = {mae_qth['total']:.5f}")

    def run(self, **kwargs) -> None:
        """
        Runs task by either fitting a model or testing a fitted model.
        """

        if self._test:
            self.test(**kwargs)
        else:
            self.fit(**kwargs)

    def create_data_split(
        self, X: np.ndarray, y: np.ndarray, val_ratio: float, random_split: bool, rng_seed: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Creates training and validation splits.

        Args:
            X (np.ndarray): Source data (preprocessed).
            y (np.ndarray): Target data (preprocessed).
            val_ratio (float): Fractional size of the validation dataset, compared to the size of
                the original training dataset.
            random_split (bool): Whether train and validation datasets are split randomly.
            rng_seed (int): Random number generator seed.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: (Source train
                data, target train data), (Source validation data, target validation data).
        """

        val_idx = int((1.0 - val_ratio) * len(X))  # start index for validation split

        if random_split:
            # define random number generator
            rng = np.random.default_rng(rng_seed)

            # permute dataset indices
            indices = rng.permutation(len(X))
        else:
            indices = np.arange(len(X))

        # create training split
        X_train = X[indices[:val_idx]]
        y_train = y[indices[:val_idx]]

        # create validation split
        X_val = X[indices[val_idx:]]
        y_val = y[indices[val_idx:]]

        return (X_train, y_train), (X_val, y_val)

    @classmethod
    def save_predictions(
        cls,
        vf_target: np.ndarray,
        vf_pred: np.ndarray,
        qth_target: np.ndarray,
        qth_pred: np.ndarray,
        model_dir: str,
        model_name: str,
    ) -> None:
        np.save(
            get_model_filepath(model_dir, cls.get_task_name(), model_name, '_vf_target.npy'),
            vf_target,
        )
        np.save(
            get_model_filepath(model_dir, cls.get_task_name(), model_name, '_vf_pred.npy'),
            vf_pred,
        )
        np.save(
            get_model_filepath(model_dir, cls.get_task_name(), model_name, '_qth_target.npy'),
            qth_target,
        )
        np.save(
            get_model_filepath(model_dir, cls.get_task_name(), model_name, '_qth_pred.npy'),
            qth_pred,
        )

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
