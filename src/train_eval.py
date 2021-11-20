"""Train and evaluate Tool

The script utilizes the 2 main inputs which are the config.yml and the CLI params.
With the combination of these configuration 2 running branches are possible:

- Training workflow:
    - Load the dataset
    - Create a model with the provided configuration
    - Train then save the model
    - Evaluate the model
    - Log the hyper-parameters, values, model, and the train/test result with MLflow

- Load-evaluate workflow:
    - load the dataset and a given model
    - Evaluate the model

For more information about configuration see:
    :py:func:`~setup ARGS from CLI options <helper.config.setup_arguments>`
    :ref:`~config file <config.yml>`

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log-level

import logging
import mlflow.keras

from helper import config
from service import data_service, model_service
from helper.config import DEFAULT_TRACKING_URI, DEFAULT_EXPERIMENT_NAME

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
    train_data, test_data, class_labels = data_service.data_segregation(
        dataset_path, input_shape, batch, ARGS.test_split)
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
        run_name = RUN_ARGS.run_name
        tracking_uri = config.get_value('mlflow', 'tracking_uri') or DEFAULT_TRACKING_URI
        experiment_name = config.get_value('mlflow', 'experiment_name') or DEFAULT_EXPERIMENT_NAME
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.tensorflow.autolog()
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tag('experiment_id', run.info.experiment_id)
            mlflow.set_tag('run_id', run_id)
            mlflow.log_param('input-shape', feature_shape)
            params_to_log = ['name', 'subset_name', 'dataset', 'epoch', 'batch', 'test_split', 'validation_split']
            params = {i: vars(RUN_ARGS).get(i) for i in params_to_log}
            mlflow.log_params({
                'cfg_model_name': params.get('subset_name') or params.get('name'),
                'cfg_dataset_name': params.get('dataset'),
                'cfg_labels': labels,
                'HP_epochs': params.get('epoch'),
                'HP_batch_size': params.get('batch'),
                'HP_test_split': params.get('test_split'),
                'HP_validation_split': params.get('validation_split'),
            })

            # Create, train, and save model
            model = model_service.create(model_name, feature_shape, labels)
            model_service.train(model, train_ds, RUN_ARGS.epoch, RUN_ARGS.validation_split)
            model_service.save(model, model_name)
            # Validation
            model_service.evaluate_model(model, test_ds)

            stat, cumulative_accuracy = model_service.validate_classification(model, test_ds, labels, False)
            for key, value in stat.items():
                acc = round(value['right'] / (value['right'] + value['wrong']) * 100, 1)
                mlflow.log_param(f'accuracy.{key}', acc)
                mlflow.log_param(f'stat.{key}', value)

            # Register the model
            if cumulative_accuracy >= RUN_ARGS.deploy_limit:
                logger.info(f"The '{model_name}' model with runId of '{run_id}' and '{cumulative_accuracy}' accuracy "
                            f"is registered to the model-registry as '{run_name}'.")
                mlflow.register_model(
                    model_uri=f'runs:/{run_id}/model',
                    name=run_name
                )

        logger.info(f'The run has been finished, check: `{tracking_uri}` for the result and for more information!')
    else:
        # Load model
        model = model_service.load(model_name)
        # Validation
        model_service.evaluate_model(model, test_ds)
        model_service.validate_classification(model, test_ds, labels, False)
