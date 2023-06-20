from argparse import ArgumentParser

from tasks import ApproximationTask
from utils import check_paths, read_data


def main() -> None:
    # set up CLI
    parser = ArgumentParser(
        description="XGBoost time series forecasting/approximation model for solar thermal systems"
    )
    parser.add_argument('--data-dir', type=str, default='../data/', help="Path to data files.")
    parser.add_argument('--model-dir', type=str, default='../models/', help="Path to model files.")
    parser.add_argument(
        '--task',
        type=str,
        default='approximation',
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
        help="Fractional size of the validation dataset, compared to the size of the original training dataset.",
    )
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=100,
        help="Max number of rounds (=trees) in the XGB model.",
    )
    parser.add_argument(
        '--num-early-stopping-rounds',
        type=int,
        default=10,
        help="Number of rounds without improvement after which fitting is stopped.",
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
    check_paths(args.data_dir, args.model_dir, args.task, args.test)

    # read data from disk
    source_data, target_data = read_data(args.data_dir, args.test)

    # set up task
    if args.task == ApproximationTask.get_task_name():
        task = ApproximationTask(
            source_data=source_data, target_data=target_data, test=args.test, verbose=args.verbose
        )
    elif args.task == 'forecast_step':
        raise NotImplementedError
    elif args.task == 'forecast_hour':
        raise NotImplementedError
    else:
        raise ValueError(
            f"Unknown task: {args.task}. Choose one of ('forecast_step', 'forecast_hour', '{ApproximationTask.get_task_name()}')"
        )

    # run task
    task.run(
        val_ratio=args.val_ratio,
        num_rounds=args.num_rounds,
        num_early_stopping_rounds=args.num_early_stopping_rounds,
        model_dir=args.model_dir,
        rng_seed=args.rng_seed,
    )


if __name__ == '__main__':
    main()
