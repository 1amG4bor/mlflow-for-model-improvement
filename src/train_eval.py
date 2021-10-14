import logging
import os

import mlflow

from src.helper import config
from src.service import data_service

log_format = '%(asctime)s >>%(levelname)s<< %(filename)s|%(funcName)s: ln.%(lineno)d => %(message)s'
logging.basicConfig(format=log_format, level = logging.INFO)
logger = logging.getLogger('Train_predict')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRACKING_URI = 'http://localhost/'


def init(experiment_name, tracking_uri=DEFAULT_TRACKING_URI, run_name=None, version=None):
    """ Initialize parameters and MLflow """
    logger.info('Initialization..')
    ARGS = config.setup_arguments(BASE_PATH)
    img_height = img_width = 128
    input_shape = (img_height, img_width, 1)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag('Version', version)
        mlflow.log_param('input-shape', input_shape)
        mlflow.log_params(dict(vars(ARGS)))
    return ARGS, input_shape


def prepare_data(ARGS, input_shape):
    dataset_path = data_service.data_sourcing(ARGS)
    train_data, test_data = data_service.data_segregation(dataset_path, input_shape, ARGS.batch)
    class_num = len(train_data.class_names)
    logger.info(f'Classes of the dataset: ({class_num}) => {train_data.class_names}')

    return train_data, test_data, class_num


def build_model(ARGS, class_num):
    pass


def train(model, ARGS, train_data):
    pass


def evaluate(model_location, test_data):
    pass


if __name__ == '__main__':

    RUN_ARGS, feature_shape = init(experiment_name='Test experiment')
    # Data Ingestion
    train_ds, test_ds, num_of_classes = prepare_data(RUN_ARGS, feature_shape)
    # Model making
    model = build_model(RUN_ARGS, num_of_classes)
    train(model, RUN_ARGS, train_ds)
    # Evaluation
    accuracy = evaluate(model, test_ds)
    logger.info(f'The `Accuracy` of the model: {accuracy}')
