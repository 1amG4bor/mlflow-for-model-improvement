import logging
import os
import pathlib

from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

from src.helper.config import timed
from src.helper import loader, config

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
USER_PATH = os.path.expanduser('~')
DATASET_PATH = pathlib.Path(USER_PATH, '.keras', 'datasets')

logger = logging.getLogger('Data-service')


@timed
def data_sourcing(args):
    logger.info('Data sourcing..')
    # Use the predefined dataset in '~/.keras/datasets' folder
    if args.dataset_name:
        dataset_work_path = pathlib.Path(DATASET_PATH, args.dataset_name)
        # Download and use the original dataset
        if args.download:
            data_source = config.get_value('download', 'dataset_source')
            extract_name = config.get_value('download', 'dataset_extracted_name')
            dataset_work_path = loader.download_dataset(args.dataset_name, data_source, extract_name)

        # Create and use subset of original dataset if 'subset' and 'subset_name' arg is set.
        if args.subset and args.subset_name:
            classes = args.subset
            label_set = loader.fetch_labels(args.dataset_name)
            if all(item in label_set for item in classes):
                new_dataset_path = pathlib.Path(DATASET_PATH, f'{args.dataset_name}-{args.subset_name}')
                if not os.path.isdir(dataset_work_path):
                    raise ValueError(f"The given 'dataset-name' argument is invalid, there is no dataset with the given "
                                     f"name: {args.dataset_name}.")
                dataset_work_path = loader.create_filtered_dataset(dataset_work_path, new_dataset_path, classes)
            else:
                logger.error('The given subset contains invalid class or classes')
    else:
        raise ValueError("Error! Missing 'dataset-name' argument.")
    return dataset_work_path


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
        batch_size=batch_size,
    )

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
    return train_dataset, test_dataset
