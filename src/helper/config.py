import argparse
import logging
import time
from pathlib import Path

import yaml

BASE_PATH = Path(__file__).parents[2]       # root path for the application
DIST_PATH = Path(BASE_PATH, 'DIST')         # DIST/
LOG_PATH = Path(DIST_PATH, 'logs')          # DIST/logs
MODELS_PATH = Path(DIST_PATH, 'models')     # DIST/models/
EXPORTED_MODELS_PATH = Path(DIST_PATH, 'mlruns')
SRC_PATH = Path(BASE_PATH.parent, 'src')    # src/
CONFIG_FILE = Path(__file__).parents[1].joinpath('config.yml')

logger = logging.getLogger("Config")


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
            config_obj = yaml.safe_load(stream)
            return _get_nested_value(config_obj, keys)
        except yaml.YAMLError as err:
            logger.error(err)


def _get_nested_value(cfg, keys):
    if len(keys) > 1:
        dict_value = cfg.get(keys[0])
        return _get_nested_value(dict_value, keys[1:])
    return cfg.get(keys[0])


def setup_arguments():
    """
    Utilize the named argument that provided or default them to setup the variable that is needed for customization
    purposes.
    :returns object with attributes
    """
    parser = argparse.ArgumentParser('Arguments to customize and handle the model and to configure the training and '
                                     'evaluation')

    # Arguments to specify the dataset
    dataset_group = parser.add_argument_group(title='Arguments to specify the dataset source. One of them is REQUIRED!')
    ds_group = dataset_group.add_mutually_exclusive_group()
    ds_group.add_argument('-d', '--download', action="store_true",
                          help='Download the dataset and use it in model-training & evaluation. (default:False) The '
                               'data-source need to be specified in the `config.yml` file')
    ds_group.add_argument('-ds', '--dataset', type=str,
                          help="Name of the dataset in '~/.keras/datasets' folder to use in training/validation/test")
    ds_extra = dataset_group.add_argument_group('Other dataset specific options')
    ds_extra.add_argument('-sub', '--subset', type=str, nargs='+', metavar=('dataset-name', 'SAMPLE-CLASS'),
                          help='''Create a subset from the original dataset and use it for the training. Arguments 
                          are list of strings where the first item will be the name of the new dataset. eg: ".. -sub 
                          new-dataset stop give-way"''')

    # Arguments related to the model
    model_group = parser.add_argument_group('Arguments for the model related options')
    model_group.add_argument('-n', '--name', type=str, required=True,
                             help='Name for loading/saving the model')
    model_group.add_argument('-c', '--create', action="store_true",
                             help='Create & train the model with the given name. If this flag is not set, then the '
                                  'previously saved model will be loaded if it exists with the given name.')

    # Arguments to configure the training process
    process_group = parser.add_argument_group('Arguments configure the training process')
    process_group.add_argument('-b', '--batch', type=int, default=64,
                               help='Batch size for training the model. (default:64)')
    process_group.add_argument('-ep', '--epoch', type=int, default=20,
                               help='Epoch size for training the model. (default:20)')

    args = parser.parse_args()
    if not args.dataset and not args.download:
        logger.error('Error! No any data-source has been provided, `download` or `dataset` argument is required!')
        raise ValueError(
            "Because of missing data-source the script is stopped. Please, check the specified CLI arguments.")

    if args.subset and len(args.subset) > 1:
        args.subset_name = args.subset.pop(0)
    else:
        args.sub = args.subset_name = None

    return args
