import logging
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import numpy as np
import mlflow
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.layers import Resizing, Rescaling

from src.helper import config as cfg
from src.helper.config import LOG_PATH
from src.model.props import ConvolutionProps, DenseProps

logger = logging.getLogger('recognition_model')
DEFAULT_LOG_FOLDER = LOG_PATH


class RecognitionModel(models.Sequential):
    """Traffic sign recognition model
        :parameter input_shape: input shape of the model
        :parameter convolution_props: properties of convolutional layers
        :parameter dense_props: properties of dense layers
        :parameter name: name of the model
        :parameter log_folder: location for logs
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 convolution_props: ConvolutionProps = None,
                 dense_props: DenseProps = None,
                 name: str = 'RecognitionModel',
                 log_folder: str = DEFAULT_LOG_FOLDER,
                 ):
        super().__init__(name=name)
        self.feature_columns = input_shape
        self.convolution_props = convolution_props
        self.dense_props = dense_props
        self.log_folder = log_folder

        if input_shape and convolution_props and dense_props:
            self.__build_layer_structure(input_shape, convolution_props, dense_props)
            for layer in self.layer_list:
                self.add(layer)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_shape = config['input_shape']
        # Create the model (without layers)
        model = cls(input_shape=input_shape,
                    convolution_props=None,
                    dense_props=None,
                    name=config['name'],
                    log_folder=config['log_folder'])
        # Load the layers
        layer_configs = config['layers']
        for layer_config in layer_configs:
            layer = layers.deserialize(layer_config)
            model.add(layer)
        # Build
        build_input_shape = config.get('build_input_shape') or None
        if not model.inputs and build_input_shape and isinstance(build_input_shape, (tuple, list)):
            model.build_model(log_summary=False)
        return model

    def get_config(self):
        return {
            'input_shape': self.feature_columns,
            'convolution_props': self.convolution_props.get_config(),
            'dense_props': self.dense_props.get_config(),
            'name': self.name,
            'log_folder': self.log_folder,
            'layers': self.layers,
            'build_input_shape': self.input_shape,
        }

    def __build_layer_structure(self, input_shape, convolution_props, dense_props):
        # Image pre-processing (resize/rescale)
        structure = [Resizing(*input_shape[:2]), Rescaling(1. / 255)]
        # Create the convolutional layers
        for size in convolution_props.layers:
            structure.append(layers.Conv2D(filters=size,
                                           kernel_size=convolution_props.kernel_size,
                                           padding='same',
                                           activation=convolution_props.activation))
            structure.append(layers.MaxPooling2D())
        # Flatter
        structure.append(layers.Flatten())
        # Create the dense layers
        for size in dense_props.layers:
            structure.append(layers.Dense(units=size, activation=dense_props.activation))
        structure.append(layers.Dense(units=dense_props.final_layer, activation='softmax'))
        self.layer_list = structure

    def compile_model(self, optimizer, loss_fn, metrics):
        self.compile(optimizer, loss_fn, metrics)

    def build_model(self, log_summary=True):
        shape = (0,) + self.feature_columns
        self.build(shape)
        if log_summary:
            logger.info(f'Model summary:\n{self.summary()}')

    def evaluate_accuracy(self, test_input):
        _, accuracy = self.evaluate(test_input)
        return accuracy

    def train_model(self, samples, labels, validation_split, epochs):
        self.fit(
            x=samples,
            y=labels,
            validation_split=validation_split,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(self.log_folder),    # log metrics
            ],
        )

    def save_model(self, model_folder, file_to_save=None):
        if not file_to_save:
            file_to_save = self.name
        destination = Path(model_folder, file_to_save)

        save_format = cfg.get_value('model', 'save_format')
        models.save_model(self, destination, save_format=save_format, overwrite=True)
        mlflow.log_artifacts(destination, file_to_save)
        return destination

    def predict_one(self, sample):
        return self.predict_classes(x=sample, batch_size=1, verbose=1)

    def predict_many(self, samples):
        result = np.argmax(self.predict(samples), axis=-1)
        return result

    @staticmethod
    def load_saved_model(model_location):
        """ Returns a Keras model instance that will be compiled if it was saved that way, otherwise need to compile
        :parameter model_location: destination of the saved model, it could be: `str`, `pathlib.Path`, `h5py.File`
        """
        return models.load_model(model_location, custom_objects={'RecognitionModel': RecognitionModel})

    @staticmethod
    def separate_features_and_labels(dataset):
        features = []
        labels = []
        for sample in dataset.as_numpy_iterator():
            features.extend(sample[0])
            labels.extend(sample[1])
        return features, labels
