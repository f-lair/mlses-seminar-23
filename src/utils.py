import csv
from pathlib import Path
from timeit import default_timer
from typing import Dict, List, Tuple

import numpy as np

CIRCUITS = {0: 'solar', 1: 'water', 2: 'boiler', 3: 'heating'}  # map: Circuit_ID -> Circuit_Name


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


def check_paths(data_dir: str, model_dir: str, task_name: str, test: bool) -> None:
    """
    Checks whether relevant directories and file paths exist.

    Args:
        data_dir (str): Path to data files.
        model_dir (str): Path to model files.
        task_name (str): Name of the task to be solved.
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
    model_filepath = Path(model_dir).joinpath(get_model_filepath(model_dir, task_name))
    if test and not model_filepath.exists():
        raise ValueError(f"Model file '{str(model_filepath)}' does not exist! Fit a model first.")


def read_data(data_dir: str, test: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Reads data from disk.

    Args:
        data_dir (str): Path to data files.
        test (bool): Whether test mode is active.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Source data, target data (mapping via circuit, respectively).
    """

    data_source = {}
    data_target = {}
    read_map = get_read_map(test, data_source, data_target)

    for circuit in CIRCUITS.values():
        for filename, data_dict in read_map.items():
            filepath = Path(data_dir).joinpath(circuit, filename)
            data_dict[circuit] = np.load(str(filepath))

    return data_source, data_target


def write_logs(logs: Dict[str, Dict[str, List[float]]], model_dir: str, task_name: str) -> None:
    """
    Writes fitting process logs to disk.

    Args:
        logs (Dict[str, Dict[str, List[float]]]): Fitting process logs.
        model_dir (str): Path to model files.
        task_name (str): Name of the task to be solved.
    """

    log_path = get_model_filepath(model_dir, task_name, log=True)

    logs['training'] = logs['validation_0']
    logs['validation'] = logs['validation_1']
    del logs['validation_0']
    del logs['validation_1']

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


def get_model_filepath(model_dir: str, task_name: str, log: bool = False) -> str:
    """
    Returns path to model or log file.

    Args:
        model_dir (str): Path to model files.
        task_name (str): Name of the task to be solved.
        log (bool, optional): Whether path to log file should be returned. Defaults to False.

    Returns:
        str: Path to model or log file.
    """

    return str(Path(model_dir).joinpath(get_model_filename(task_name, log)))


def get_model_filename(task_name: str, log: bool = False) -> str:
    """
    Returns filename of model or log file.

    Args:
        task_name (str): Name of the task to be solved.
        log (bool, optional): Whether filename of log file should be returned. Defaults to False.

    Returns:
        str: Filename of model or log file.
    """

    if log:
        return f'{task_name}_log.csv'
    else:
        return f'{task_name}_model.json'
