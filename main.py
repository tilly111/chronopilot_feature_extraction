import argparse, random, pathlib
import matplotlib
import pandas as pd


if __name__ == '__main__':
    # for interactive plots & debug prints
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    # parser for arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment', type=str, default='helicopter', choices=['helicopter', 'robot_behavior', 'scream', 'eye_tracking'])
    parser.add_argument('--dir_path', type=str, default='/Volumes/Data/chronopilot')
    parser.add_argument('--baseline_subtraction', action='store_true', help="Enable baseline subtraction")
    parser.add_argument('--no_baseline_subtraction', dest='baseline_subtraction', action='store_false',
                        help="Disable baseline subtraction")
    parser.set_defaults(baseline_subtraction=True)
    parser.add_argument('--window_size', type=int, default=60, help='window size for feature extraction in seconds')
    
    # class information
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes for classification')
    parser.add_argument('--target', type=str, default="ppot", help='How to derive the labels', choices=['ppot', 'duration_estimate', 'stress'])
    
    
    config = parser.parse_args()
    
    if config.experiment == 'helicopter':
        from helicopter_features.run import run
        run(config)
    elif config.experiment == 'robot_behavior':
        from robot_behavior_features.run import run
        run(config)
    elif config.experiment == 'scream':
        from scream_features.run import run
        run(config)
    elif config.experiment == 'eye_tracking':
        # from eye_tracking.run import run
        # run(config)
        raise(NotImplementedError)