import argparse
import logging
import os
import time
from pathlib import Path

import yaml

logger = logging.getLogger("Config")
CONFIG_FILE = Path(__file__).parents[1].joinpath('config.yml')


def timed(func):
    """ Decorator to calculate the process time of a function (use: @timed)"""
    def timing(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'Processing time({func.__name__}): {end - start:.5f} sec')
        return result

    return timing


def get_value(*keys):
    """ Read config values from config.yaml """
    with open(CONFIG_FILE, 'r') as stream:
        try:
            config_obj=yaml.safe_load(stream)
            return _get_nested_value(config_obj, keys)
        except yaml.YAMLError as err:
            logger.error(err)


def _get_nested_value(cfg, keys):
    if len(keys) > 1:
        dict_value = cfg.get(keys[0])
        return _get_nested_value(dict_value, keys[1:])
    return cfg.get(keys[0])


def setup_arguments(base_path):
    """
    Utilize the named argument that provided or default them to setup the variable that is needed for customization
    purposes.
    :returns object with attributes
    """
    parser = argparse.ArgumentParser(
        'Arguments to customize and handle the model and to configure the training and evaluation')

    # Arguments to specify the dataset
    dataset_group = parser.add_argument_group('Arguments to specify the dataset related options')
    dataset_group.add_argument('-ds-name', '--dataset-name', type=str, default='traffic_sign_dataset',
                               help='Name for the dataset if you want to download. (default: traffic_sign_dataset)')
    ds_group = dataset_group.add_mutually_exclusive_group()
    ds_group.add_argument('-d', '--download', action="store_true",
                          help='Download the dataset and use it in model-training & evaluation. (default:False)')
    ds_group.add_argument('-ds', '--dataset', type=str, default='traffic_sign_dataset-main',
                          help="Name of the dataset in '~/.keras/datasets' folder to use in training/validation/test")
    ds_group.add_argument('-sub', '--subset', type=str, nargs='+', metavar=('dataset-name', 'SAMPLE-CLASS'),
                          help='''Create a subset from the original dataset and use it for the training. Arguments 
                          are list of strings where the first item will be the name of the new dataset. eg: ".. -sub 
                          new-dataset stop give-way"''')

    # Arguments to configure the training process
    process_group = parser.add_argument_group('Arguments configure the training process')
    process_group.add_argument('-b', '--batch', type=int, default=64,
                               help='Batch size for training the model. (default:64)')
    process_group.add_argument('-ep', '--epoch', type=int, default=20,
                               help='Epoch size for training the model. (default:20)')
    process_group.add_argument('-eval', '--evaluation', action="store_true",
                               help='Run the evaluation after the training. (default:False)')

    # Arguments related to the model
    model_group = parser.add_argument_group('Arguments for the model related options')
    model_group.add_argument('-s', '--save', type=str, default=os.path.join(base_path, 'models'),
                             help='Folder for saving the model')
    model_group.add_argument('-n', '--name', type=str, default='traffic_sign_dataset',
                             help='Name for saving the model')
    args = parser.parse_args()

    if args.subset:
        if args.dataset:
            args.subset = args.subset_name = None
        else:
            args.subset_name = args.subset.pop(0)
    return args
