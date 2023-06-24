import csv
from pathlib import Path
from timeit import default_timer
from typing import Dict, List, Tuple

import numpy as np
from metrics import MeanAbsoluteError, MeanSquaredError
from metrics.base_metric import BaseMetric

CIRCUITS = {0: 'solar', 1: 'water', 2: 'boiler', 3: 'heating'}  # map: Circuit_ID -> Circuit_Name
TIMESTEP = 5  # 5s timesteps
# NOTE: VHC_solar is unknown, but preliminarily set to 4.2 as well
VHC = np.array([4.2, 4.2, 4.2, 4.2])  # Volumetric heat capacity (solar, water, boiler, heating)


class Timer:
    """Time measurements."""

    def __init__(self) -> None:
        """
        Initialization of timer.
        """

        self._start = None
        self._end = None

    def start(self) -> None:
        """
        Starts new time measurement.
        """

        self._end = None
        self._start = default_timer()

    def end(self) -> None:
        """
        Finishes time measurement.
        """

        self._end = default_timer()

    def elapsed_time(self) -> float:
        """
        Computes elapsed time.

        Returns:
            float: Elapsed time in seconds.
        """

        if self._start is not None and self._end is not None:
            return self._end - self._start
        else:
            return 0.0


class TimerContext:
    """Context handler for time measurements."""

    def __init__(self) -> None:
        """
        Initializes timer context.
        """

        self.timer = Timer()

    def __enter__(self) -> Timer:
        """
        Enters timer context and starts time measurement.

        Returns:
            Timer: Timer instance.
        """

        self.timer.start()
        return self.timer

    def __exit__(self, *args) -> None:
        """
        Leaves timer context and finishes time measurement.
        """

        self.timer.end()


def check_paths(
    data_dir: str, model_dir: str, task_name: str, model_name: str, test: bool
) -> None:
    """
    Checks whether relevant directories and file paths exist.

    Args:
        data_dir (str): Path to data files.
        model_dir (str): Path to model files.
        task_name (str): Name of the task to be solved.
        model_name (str): Name of the model.
        test (bool): Whether test mode is active.

    Raises:
        ValueError: Data directory does not exist.
        ValueError: Data file does not exist.
        ValueError: Model file does not exist.
    """

    # check whether data path exists
    if not Path(data_dir).exists():
        raise ValueError(f"Data path '{data_dir}' does not exist!")

    # check whether data files exist
    data_filenames = get_read_map(test, {}, {}).keys()
    for circuit in CIRCUITS.values():
        for data_filename in data_filenames:
            data_filepath = Path(data_dir).joinpath(circuit, data_filename)
            if not data_filepath.exists():
                raise ValueError(f"Data file '{str(data_filepath)}' does not exist!")

    # create model dir, if not existing
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # if test, check whether model file exists
    model_filepath = Path(get_model_filepath(model_dir, task_name, model_name, '.json'))
    if test and not model_filepath.exists():
        raise ValueError(f"Model file '{str(model_filepath)}' does not exist! Fit a model first.")


def read_data(data_dir: str, test: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Reads data from disk.

    Args:
        data_dir (str): Path to data files.
        test (bool): Whether test mode is active.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Source data, target data (mapping via
            circuit, respectively).
    """

    data_source = {}
    data_target = {}
    read_map = get_read_map(test, data_source, data_target)

    for circuit in CIRCUITS.values():
        for filename, data_dict in read_map.items():
            filepath = Path(data_dir).joinpath(circuit, filename)
            data_dict[circuit] = np.load(str(filepath))

    return data_source, data_target


def write_logs(
    logs: Dict[str, Dict[str, List[float]]], model_dir: str, task_name: str, model_name: str
) -> None:
    """
    Writes fitting process logs to disk.

    Args:
        logs (Dict[str, Dict[str, List[float]]]): Fitting process logs.
        model_dir (str): Path to model files.
        task_name (str): Name of the task to be solved.
        model_name (str): Name of the model.
    """

    log_path = get_model_filepath(model_dir, task_name, model_name, '.csv')

    logs_reordered = {
        f'{mode}-{metric}': results
        for mode, temp_dict in logs.items()
        for metric, results in temp_dict.items()
    }

    with open(log_path, 'w', newline='') as file_handler:
        csv_writer = csv.writer(file_handler)
        csv_writer.writerow(logs_reordered.keys())
        csv_writer.writerows(zip(*logs_reordered.values()))


def get_read_map(test: bool, data_source: Dict, data_target: Dict) -> Dict[str, Dict]:
    """
    Returns map (data filename -> data dictionary), which can be used to read-in the correct files.

    Args:
        test (bool): Whether test mode is active.
        data_source (Dict): Dictionary for source data.
        data_target (Dict): Dictionary for target data.

    Returns:
        Dict[str, Dict]: Map: Data filename -> data dictionary
    """

    if test:
        read_map = {
            'source_test.npy': data_source,
            'target_test.npy': data_target,
        }
    else:
        read_map = {'source_training.npy': data_source, 'target_training.npy': data_target}

    return read_map


def get_model_filepath(model_dir: str, task_name: str, model_name: str, suffix: str) -> str:
    """
    Returns path to model or log file.

    Args:
        model_dir (str): Path to model files.
        task_name (str): Name of the task to be solved.
        model_name (str): Name of the model.
        suffix (str): Filename suffix, including file extension.

    Returns:
        str: Path to model or log file.
    """

    return str(Path(model_dir).joinpath(get_model_filename(task_name, model_name, suffix)))


def get_model_filename(task_name: str, model_name: str, suffix: str) -> str:
    """
    Returns filename of model or log file.

    Args:
        task_name (str): Name of the task to be solved.
        model_name (str): Name of the model.
        suffix (str): Filename suffix, including file extension.

    Returns:
        str: Filename of model or log file.
    """

    return f'{task_name}_{model_name}{suffix}'


def get_hour_horizon_size() -> int:
    """
    Returns horizon size for hour forecasting task.

    Returns:
        int: Horizon size.
    """

    # 1h = 3600s
    return int(3600 // TIMESTEP)


def get_day_in_week_one_hot(dates: np.ndarray) -> np.ndarray:
    """
    Computes one-hot-encoding of the time feature 'day in week (0-6)', given the date.
    0: Saturday, 1: Sunday, ..., 6: Friday.
    Uses vectorized implementation of Zeller's congruence.
    cf. https://en.wikipedia.org/wiki/Zeller%27s_congruence
    N: Number of dates.

    Args:
        dates (np.ndarray): Dates in format 'YYYYMMDD' [N].

    Returns:
        np.ndarray: Days in week (one-hot-encoded) [N, 6].
    """

    # extract day, month, and year
    dates = dates.astype(int)
    y, m = np.divmod(dates, 10000)
    m, q = np.divmod(m, 100)

    # Jan needs to be encoded as 13, Feb as 14, and the year has to be deduced by 1 for such dates
    mask = np.logical_or(m == 1, m == 2)
    m[mask] += 12
    y[mask] -= 1

    J, K = np.divmod(y, 100)

    # compute day in week
    days_in_week = (
        q
        + (13 * (m + 1) / 5.0).astype(int)
        + K
        + (K / 4.0).astype(int)
        + (J / 4.0).astype(int)
        - 2 * J
    ) % 7

    # create one-hot-encoding
    N = len(days_in_week)
    one_hot_enc = np.zeros((N, 7))
    one_hot_enc[np.arange(N), days_in_week] = 1

    # cut off first column, as it is redundant
    return one_hot_enc[:, 1:]


def VF_to_Qth(VF: np.ndarray, T1: np.ndarray, T2: np.ndarray, T3: np.ndarray) -> np.ndarray:
    """
    Computes heat transfer Qth from volume flow rate VF and temperatures T1 to T3.
    N: Batch dimension.
    WS: Window size.

    Args:
        VF (np.ndarray): Volume flow rate VF: [l] / [h]; [N, 4*WS].
        T1 (np.ndarray): Temperature 1: [K]; [N, 4*WS].
        T2 (np.ndarray): Temperature 2: [K]; [N, 4*WS].
        T3 (np.ndarray): Temperature 3: [K]; [N, 4*WS].

    Returns:
        np.ndarray: Heat transfer Qth [kWh]; [N, 4*WS].
    """

    # Volume flow rate VF: [l] / [h] -> 1e-3 [m]^3 / 3600 [s] -> [m]^3 / (3.6e6 [s])
    # Temperatures T1 / T2 / T3: [K]
    # Volumetric heat capacity VHC: [MJ] / ([m]^3 * [K]) -> 1e6 [J] / ([m]^3 * [K])

    # Temperature difference: [K]
    delta_T = np.abs(T3 - 0.5 * (T1 + T2))

    VHC_shaped = np.tile(VHC, int(VF.shape[1] // 4))[None, :]  # broadcast w.r.t. window dimension
    # Heat flow rate Qpunkt: ([m]^3 / (3.6e6 [s])) * [K] * (1e6 [J] / ([m]^3 * [K]))
    # -> [J] / (3.6 [s])

    # print(VF.shape)
    # print(delta_T.shape)
    # print(VHC_shaped.shape)

    Qpunkt = VF * delta_T * VHC_shaped

    # Heat transfer Qth: ([J] / (3.6 [s])) * [s] -> [J] / 3.6 -> 3.6e6 [J] / (3.6 * 3.6e6)
    # -> [kWh] / (3.6 * 3.6e6)
    Qth = Qpunkt * TIMESTEP

    corrective_factor = 1 / (3.6 * 3.6e6)

    return corrective_factor * Qth


def create_metrics() -> (
    Tuple[
        Dict[str, BaseMetric], Dict[str, BaseMetric], Dict[str, BaseMetric], Dict[str, BaseMetric]
    ]
):
    """
    Creates metrics for predictions on test data.

    Returns:
        Tuple[Dict[str, BaseMetric], Dict[str, BaseMetric], Dict[str, BaseMetric],
            Dict[str, BaseMetric]]: RMSE (VF), MAE (VF), RMSE (Qth), MAE (Qth)
            (keys circuit_names, 'total' per dict).
    """

    rmse_vf_metrics = {
        circuit_name: MeanSquaredError(squared=False) for circuit_name in CIRCUITS.values()
    }
    mae_vf_metrics = {circuit_name: MeanAbsoluteError() for circuit_name in CIRCUITS.values()}
    rmse_qth_metrics = {
        circuit_name: MeanSquaredError(squared=False) for circuit_name in CIRCUITS.values()
    }
    mae_qth_metrics = {circuit_name: MeanAbsoluteError() for circuit_name in CIRCUITS.values()}

    rmse_vf_metrics['total'] = MeanSquaredError(squared=False)
    mae_vf_metrics['total'] = MeanAbsoluteError()
    rmse_qth_metrics['total'] = MeanSquaredError(squared=False)
    mae_qth_metrics['total'] = MeanAbsoluteError()

    return rmse_vf_metrics, mae_vf_metrics, rmse_qth_metrics, mae_qth_metrics  # type: ignore


def compute_metrics(
    rmse_vf_metrics: Dict[str, BaseMetric],
    mae_vf_metrics: Dict[str, BaseMetric],
    rmse_qth_metrics: Dict[str, BaseMetric],
    mae_qth_metrics: Dict[str, BaseMetric],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Computes metric results for predictions on test data.

    Args:
        rmse_vf_metrics (Dict[str, BaseMetric]): RMSE (VF; keys circuit_names, 'total' per dict).
        mae_vf_metrics (Dict[str, BaseMetric]): MAE (VF; keys circuit_names, 'total' per dict).
        rmse_qth_metrics (Dict[str, BaseMetric]): RMSE (Qth; keys circuit_names, 'total' per dict).
        mae_qth_metrics (Dict[str, BaseMetric]): MAE (Qth; keys circuit_names, 'total' per dict).

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]: RMSE
                (VF), MAE (VF), RMSE (Qth), MAE (Qth) (keys circuit_names, 'total' per dict).
    """

    rmse_vf = {
        circuit_name: rmse_vf_metrics[circuit_name].compute().item()
        for circuit_name in CIRCUITS.values()
    }
    mae_vf = {
        circuit_name: mae_vf_metrics[circuit_name].compute().item()
        for circuit_name in CIRCUITS.values()
    }
    rmse_qth = {
        circuit_name: rmse_qth_metrics[circuit_name].compute().item()
        for circuit_name in CIRCUITS.values()
    }
    mae_qth = {
        circuit_name: mae_qth_metrics[circuit_name].compute().item()
        for circuit_name in CIRCUITS.values()
    }
    rmse_vf['total'] = rmse_vf_metrics['total'].compute().item()
    mae_vf['total'] = mae_vf_metrics['total'].compute().item()
    rmse_qth['total'] = rmse_qth_metrics['total'].compute().item()
    mae_qth['total'] = mae_qth_metrics['total'].compute().item()

    return rmse_vf, mae_vf, rmse_qth, mae_qth
