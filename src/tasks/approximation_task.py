from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tasks.base_task import BaseTask
from utils import CIRCUITS, TimerContext, get_model_filepath, write_logs


class ApproximationTask(BaseTask):
    """Class for the function approximation task."""

    @staticmethod
    def get_task_name() -> str:
        """
        Returns task name.

        Returns:
            str: Task name.
        """

        return 'approximation'

    def preprocess(
        self, source_data: Dict[str, np.ndarray], target_data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses source and target data dictionaries, yielding two ndarrays suited for supervised model fitting.
        Returned arrays are structured as follows:
        - Shapes are [B, D1] for source_data and [B, D2] for target_data
            - B: Batch dimension (incorporating old batch and time dimensions)
            - D1: Source feature dimension (12)
                - 0, 3, 6, 9: T1 of solar, boiler, water, heating (respectively)
                - 1, 4, 7, 10: T2 of solar, boiler, water, heating (respectively)
                - 2, 5, 8, 11: T3 of solar, boiler, water, heating (respectively)

        Args:
            source_data (Dict[str, np.ndarray]): Source data (circuit -> ndarray).
            target_data (Dict[str, np.ndarray]): Target data (circuit -> ndarray).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Source data, target data (preprocessed).
        """

        source_data_list = []
        target_data_list = []

        for circuit in CIRCUITS.values():
            # make target shape equal to source shape by repeating data points
            target_data[circuit] = np.repeat(target_data[circuit], repeats=50, axis=1)
            assert source_data[circuit].shape == target_data[circuit].shape

            # flatten over first two dimensions
            source_data[circuit] = np.reshape(source_data[circuit], (-1, 4))
            target_data[circuit] = np.reshape(target_data[circuit], (-1, 4))

            # add T1, T2, T3 to source data list
            source_data_list.append(source_data[circuit][:, 1])  # T1
            source_data_list.append(source_data[circuit][:, 2])  # T2
            source_data_list.append(target_data[circuit][:, 2])  # T3

            # add VF to target data list
            target_data_list.append(target_data[circuit][:, 1])  # VF

        # stack along feature dimension
        return np.stack(source_data_list, axis=-1), np.stack(target_data_list, axis=-1)

    def fit(
        self,
        val_ratio: float,
        num_rounds: int,
        num_early_stopping_rounds: int,
        model_dir: str,
        rng_seed: int,
        **kwargs,
    ) -> None:
        """
        Performs model fitting process:
        Fits an XGBoost regression model in a supervised fashion.

        Args:
            val_ratio (float): Fractional size of the validation dataset, compared to the size of the original training dataset.
            num_rounds (int): Max number of rounds (=trees) in the XGB model.
            num_early_stopping_rounds (int): Number of rounds without improvement after which fitting is stopped.
            model_dir (str): Path to model files.
            rng_seed (int): Random number generator seed.
        """

        X_train = self.X
        y_train = self.y

        # split into training, validation datasets
        with TimerContext() as tc:
            (X_train, y_train), (X_val, y_val) = self.create_data_split(
                X_train, y_train, val_ratio, rng_seed
            )
        if self.verbose:
            self.print("Data split.", tc.elapsed_time())

        # free RAM
        del self.X
        del self.y

        # fit model
        if self.verbose:
            print("[Iter]  Train Loss (RMSE)              Validation Loss (RMSE)")
        model = xgb.XGBRegressor(
            n_estimators=num_rounds,
            early_stopping_rounds=num_early_stopping_rounds,
            random_state=rng_seed,
        )
        with TimerContext() as tc:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=self.verbose,
            )
        if self.verbose:
            self.print("Model fitted.", tc.elapsed_time())

        # save fitted model
        save_path = get_model_filepath(model_dir, self.get_task_name())
        with TimerContext() as tc:
            model.save_model(save_path)
        if self.verbose:
            self.print("Model saved.", tc.elapsed_time())

        # Save logs
        logs = model.evals_result()
        with TimerContext() as tc:
            write_logs(logs, model_dir, self.get_task_name())
        if self.verbose:
            self.print("Logs written.", tc.elapsed_time())

    def test(
        self,
        model_dir: str,
        **kwargs,
    ) -> None:
        """
        Performs model testing process:
        Loads fitted XGB regression model and predicts outputs for unseen test data.

        Args:
            model_dir (str): Path to model files.
        """

        X_test = self.X
        y_test = self.y

        # load model
        load_path = get_model_filepath(model_dir, self.get_task_name())
        model = xgb.XGBRegressor()
        with TimerContext() as tc:
            model.load_model(load_path)
        if self.verbose:
            self.print("Model loaded.", tc.elapsed_time())

        # predict on test data
        with TimerContext() as tc:
            y_pred = model.predict(X_test)
        if self.verbose:
            self.print("Predictions computed.", tc.elapsed_time())

        # compute metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)

        # print results
        self.print(f"RMSE = {rmse:.5f}")
        self.print(f"MAE = {mae:.5f}")

    def create_data_split(
        self, X: np.ndarray, y: np.ndarray, val_ratio: float, rng_seed: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Creates training and validation splits.

        Args:
            X (np.ndarray): Source data (preprocessed).
            y (np.ndarray): Target data (preprocessed).
            val_ratio (float): Fractional size of the validation dataset, compared to the size of the original training dataset.
            rng_seed (int): Random number generator seed.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: (Source train data, target train data), (Source validation data, target validation data).
        """

        # define random number generator
        rng = np.random.default_rng(rng_seed)

        # permute dataset indices
        indices = rng.permutation(
            X.shape[0],
        )
        val_idx = int(val_ratio * X.shape[0])  # end index in 'indices' for validation split

        # create validation split
        X_val = X[indices[:val_idx]]
        y_val = y[indices[:val_idx]]

        # create training split
        X_train = X[indices[val_idx:]]
        y_train = y[indices[val_idx:]]

        return (X_train, y_train), (X_val, y_val)
