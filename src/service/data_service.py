import logging
import os
from pathlib import Path
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

from src.helper.config import timed
from src.helper import loader, config

USER_PATH = os.path.expanduser('~')
DATASET_PATH = Path(USER_PATH, '.keras', 'datasets')

logger = logging.getLogger('Data-service')


@timed
def data_sourcing(args):
    logger.info('Data sourcing..')
    # Flag that indicates that data-source is properly set
    dataset_name = dataset_work_path = None
    is_ds_set = False

    # Download and use the original dataset
    if args.download:
        dataset_name = config.get_value('download', 'dataset_name')
        data_source = config.get_value('download', 'dataset_source')
        if not all([dataset_name, data_source]):
            raise ValueError("Error! Missing 'dataset-name' or 'data_source' parameter, check your config.")
        dataset_work_path = loader.download_dataset(args.dataset_name, data_source, dataset_name)
        is_ds_set = True

    # Load use the an existing dataset
    if args.dataset:
        dataset_name = args.dataset
        dataset_work_path = Path(DATASET_PATH, dataset_name)
        if not os.path.isdir(dataset_work_path):
            raise ValueError(f"The given 'dataset-name' argument is invalid, there is no dataset with the given "
                             f"name: {dataset_name}.")
        is_ds_set = True

    # Create and use subset of original dataset if 'subset' and 'subset_name' arg is set.
    if args.subset and args.subset_name and is_ds_set:
        classes = args.subset
        label_set = loader.fetch_labels(dataset_name)
        if all(item in label_set for item in classes):
            new_dataset_path = Path(DATASET_PATH, f'{dataset_name}-{args.subset_name}')
            dataset_work_path = loader.create_filtered_dataset(dataset_work_path, new_dataset_path, classes)
        else:
            raise ValueError('The given subset contains invalid class or classes')

    if is_ds_set:
        return dataset_work_path
    else:
        raise ValueError('Data-source config is invalid!')


@timed
def data_segregation(path, shape, batch_size):
    logger.info(f'Data segregation of dataset: {path.stem}')
    train_dataset = image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        validation_split=0.2,
        subset="training",
        shuffle=True,
        seed=123,
        image_size=(shape[0], shape[1]),
        batch_size=batch_size)

    test_dataset = image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        validation_split=0.2,
        subset="validation",
        shuffle=True,
        seed=123,
        image_size=(shape[0], shape[1]),
        batch_size=batch_size)

    class_labels = train_dataset.class_names

    return train_dataset, test_dataset, class_labels
