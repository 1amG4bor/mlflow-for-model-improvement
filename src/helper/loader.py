import logging
import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from typing import List
import tensorflow as tf

from src.helper.config import timed

logger = logging.getLogger("Loader")
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
USER_PATH = os.path.expanduser('~')
DATASET_PATH = Path(USER_PATH, '.keras', 'datasets')


@timed
def download_dataset(dataset_name, dataset_source, extract_name=None):
    """
    Download and extract the given dataset into the chosen sub-folder in .keras/dataset
    :returns: the Path for the extracted dataset (in string)
    """
    existing_datasets = _get_existing_datasets()
    if not dataset_source:
        raise ValueError('Dataset source is not specified, check the `dataset_source` property in `config.yml` file')
    if dataset_name in existing_datasets:
        raise ValueError(f"A dataset is already existed with the given name: '{dataset_name}'")
    url = urlparse(dataset_source)
    file_name = os.path.basename(url.path)
    dataset_images_folder = tf.keras.utils.get_file(
        fname=file_name,
        origin=dataset_source,
        extract=True)
    if extract_name:
        output_folder = Path(DATASET_PATH, dataset_name)
        os.rename(Path(DATASET_PATH, extract_name), output_folder)
        dataset_images_folder = output_folder
    else:
        dataset_images_folder = Path(dataset_images_folder).stem

    logger.info(f"Dataset is downloaded to '{dataset_images_folder}'")
    return Path(dataset_images_folder)


@timed
def create_filtered_dataset(original_dataset: Path, new_dataset: Path, classes: List[str]):
    """ Create new dataset by copying the corresponding images of the given classes """
    label_list = extract_label_set(original_dataset)
    _validate_dataset_classes(classes, label_list)
    imgs = _filter_dataset(original_dataset, classes)
    _copy_images(imgs, new_dataset)
    return new_dataset


def extract_label_set(dataset):
    """ Collect the labels from the given dataset (folder with classified images) """
    labels = [label for label in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, label))]
    return labels


def fetch_labels(dataset_name):
    """ Search for the dataset with the given name and extract the labels """
    datasets = _get_existing_datasets()
    if dataset_name in datasets:
        return extract_label_set(Path(DATASET_PATH, dataset_name))
    return None

def _get_existing_datasets():
    return [ds for ds in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, ds))]


def _validate_dataset_classes(classes: List[str], label_list:  List[str]):
    # Validate if the given class-names can be found in the dataset
    for class_name in classes:
        if class_name not in label_list:
            raise ValueError(f"The given class-name: '{class_name}' is not exist in the dataset")


def _filter_dataset(dataset: Path, classes: List[str]):
    # Collect and filter images form the given path with the provided list of classes that you want to use
    images = []
    for item in classes:
        images.extend(dataset.glob(f'{item}/*.png'))
    return images


def _copy_images(src, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for image in src:
        class_type = image.parent.stem
        destination = os.path.join(output_folder, class_type)
        os.makedirs(destination, exist_ok=True)
        try:
            shutil.copy(image, destination)
        except Exception as err:
            logger.warning(f'Failed to copy the image: {image}, Error: {err}')
    return output_folder
