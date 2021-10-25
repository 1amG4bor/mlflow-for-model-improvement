import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set Tensorflow log-level

import logging
from pathlib import Path
import mlflow.keras

from helper import config
from service import data_service, model_service

DEFAULT_TRACKING_URI = 'http://localhost/'

log_format = '%(asctime)s >>%(levelname)s<< %(filename)s|%(funcName)s: ln.%(lineno)d => %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger('Train_predict')


def init():
    """ Initialize parameters and MLflow
    Params will set the corresponding MLflow parameters
    """
    logger.info('Initialization..')
    ARGS = config.setup_arguments()
    img_height = img_width = 128
    input_shape = (img_height, img_width, 1)
    return ARGS, input_shape


def prepare_data(ARGS, input_shape):
    """ Data Augmentation Ingestion & Segregation
    - Data Ingestion:   gather the data that only need to be fed into the pipeline
    - Data Preparation: assume that the dataset is already prepared, ready-to-use
        (No internal step to analyze and/or modify the dataset)
    - Data Segregation: Split the dataset into subsets of training-set and testing-set
        Validation-set will be separated from the training-dataset (80/20) just before training
    """
    batch = ARGS.batch
    dataset_path = data_service.data_sourcing(ARGS)
    train_data, test_data, class_labels = data_service.data_segregation(dataset_path, input_shape, batch)
    logger.info(f'Classes of the dataset: ({len(class_labels)}) => {class_labels}')

    return train_data, test_data, class_labels


if __name__ == '__main__':
    RUN_ARGS, feature_shape = init()

    # Data Extraction
    train_ds, test_ds, labels = prepare_data(RUN_ARGS, feature_shape)

    # Modelling
    model_name = RUN_ARGS.name
    if RUN_ARGS.create:
        # Run with MLflow
        experiment_name = '4 class experiment'
        run_name = f'for {RUN_ARGS.epoch} epoch'
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        mlflow.tensorflow.autolog()
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('Version', 1.0)
            mlflow.log_param('input-shape', feature_shape)
            params = {i: vars(RUN_ARGS).get(i) for i in ['name', 'subset_name', 'epoch', 'dataset']}
            mlflow.log_params({
                'model_name': params.get('subset_name') or params.get('name'),
                'epochs': params.get('epoch'),
                'dataset': params.get('dataset'),
                'labels': labels,
            })

            # Create, train, and save model
            model = model_service.create(model_name, feature_shape, labels)
            model_service.train(model, train_ds, RUN_ARGS.epoch)
            model_service.save(model, model_name)
            # Validation
            model_service.evaluate_model(model, test_ds)
            acc_trained = model_service.validate_classification(model, test_ds, labels, False)
            mlflow.log_param('evaluation_accuracy', acc_trained)

        logger.info(f'The run has been finished, check: `{DEFAULT_TRACKING_URI}` for the result and for more information!')
    else:
        # Load model
        model_src = Path(model_service.MODELS_PATH, model_name)
        if not os.path.exists(model_src):
            raise ValueError(f"The given model: '{model_name}' in not exist, it cannot be loaded!")
        model = model_service.load(model_src)
        # Validation
        model_service.evaluate_model(model, test_ds)
        acc_loaded = model_service.validate_classification(model, test_ds, labels, False)
