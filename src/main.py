from argparse import ArgumentParser

from tasks import ApproximationTask, ForecastHourTask, ForecastStepTask
from utils import check_paths, get_hour_horizon_size, read_data


def main() -> None:
    # set up CLI
    parser = ArgumentParser(
        description="XGBoost time series forecasting/approximation model for solar thermal systems"
    )
    parser.add_argument('--data-dir', type=str, default='../data/', help="Path to data files.")
    parser.add_argument('--model-dir', type=str, default='../models/', help="Path to model files.")
    parser.add_argument(
        '--model-name',
        type=str,
        default='model',
        help="Model name used as suffix in model/log file names '<task>_<model_name>.json'/ \
            '<task>_<model_name>.csv'",
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help="Task to solve ('forecast_step', 'forecast_hour', 'approximation').",
    )
    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help="Perform inference on test dataset.",
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help="Fractional size of the validation dataset, compared to the size of the original \
            training dataset.",
    )
    parser.add_argument(
        '--random-split',
        default=False,
        action='store_true',
        help="Splits train and validation datasets randomly (otherwise, deterministic cut).",
    )
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=50,
        help="Max number of rounds (=trees) in the XGB model.",
    )
    parser.add_argument(
        '--num-early-stopping-rounds',
        type=int,
        default=10,
        help="Number of rounds without improvement after which fitting is stopped.",
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.3,
        help="Learning rate in XGBoost model fitting.",
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=3,
        help="Max tree depth in the XGBoost model.",
    )
    parser.add_argument(
        '--min-split-loss',
        type=float,
        default=0.1,
        help="Minimum loss reduction needed for further leaf-predictions of the XGBoost model.",
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=1,
        help="Size of sliding window. Must be odd for approximation task.",
    )
    parser.add_argument(
        '--time-features',
        default=False,
        action='store_true',
        help="Uses additional time features (day in week).",
    )
    parser.add_argument(
        '--horizon-partition-size',
        type=int,
        default=5,
        help=f"Size of partitions the horizon for the hour forecast task is divided \
            into. Must be a factor of {get_hour_horizon_size()}.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=262144,  # 2**20
        help="Batch size used in data loading. Has no effect on results, only affects runtime and \
            memory load.",
    )
    parser.add_argument(
        '--save-first-n-predictions',
        type=int,
        default=0,
        help="Saves first n predictions of test data on disk. Must be smaller or equal to batch \
            size.",
    )
    parser.add_argument('--rng-seed', type=int, default=7, help="Random number generator seed.")
    parser.add_argument(
        '--verbose',
        default=False,
        action='store_true',
        help="Activates verbose console output.",
    )
    args = parser.parse_args()

    # check data paths
    check_paths(args.data_dir, args.model_dir, args.task, args.model_name, args.test)

    # read data from disk
    source_data, target_data = read_data(args.data_dir, args.test)

    # set up task
    if args.task == ApproximationTask.get_task_name():
        task = ApproximationTask(
            source_data=source_data,
            target_data=target_data,
            test=args.test,
            time_features=args.time_features,
            verbose=args.verbose,
        )
    elif args.task == ForecastStepTask.get_task_name():
        task = ForecastStepTask(
            source_data=source_data,
            target_data=target_data,
            test=args.test,
            time_features=args.time_features,
            verbose=args.verbose,
        )
    elif args.task == ForecastHourTask.get_task_name():
        task = ForecastHourTask(
            source_data=source_data,
            target_data=target_data,
            test=args.test,
            time_features=args.time_features,
            horizon_partition_size=args.horizon_partition_size,
            verbose=args.verbose,
        )
    else:
        raise ValueError(
            f"Unknown task: {args.task}. Choose one of ('{ForecastStepTask.get_task_name()}', \
                '{ForecastHourTask.get_task_name()}', '{ApproximationTask.get_task_name()}')"
        )

    # run task
    task.run(
        val_ratio=args.val_ratio,
        random_split=args.random_split,
        num_rounds=args.num_rounds,
        num_early_stopping_rounds=args.num_early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_split_loss=args.min_split_loss,
        batch_size=args.batch_size,
        window_size=args.window_size,
        model_dir=args.model_dir,
        model_name=args.model_name,
        save_first_n_predictions=args.save_first_n_predictions,
        rng_seed=args.rng_seed,
    )


if __name__ == '__main__':
    main()
