import logging
import os
import pathlib
from pathlib import Path
from typing import Tuple

import mlflow
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp

from src.model.props import ConvolutionProps, DenseProps

logger = logging.getLogger('recognition_model')
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_FOLDER = str(Path(BASE_PATH, 'logs'))


class RecognitionModel(Sequential):
    """Traffic sign recognition model
    :parameter input_shape: input shape of the model
    :parameter convolution_props: properties of convolutional layers
    :parameter dense_props: properties of dense layers
    :parameter model_name: name of the model
    :parameter log_folder: location for logs
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 convolution_props: ConvolutionProps,
                 dense_props: DenseProps,
                 model_name: str = 'RecognitionModel',
                 log_folder: str = DEFAULT_LOG_FOLDER
                 ):
        super(RecognitionModel, self).__init__(name=model_name)
        self.log_folder = log_folder
        self.feature_shape = input_shape
        self.dataset_params = [None, input_shape[:-1], 64]

        # Image rescale pre-processing
        self.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape))
        # Create the convolutional layers
        for size in convolution_props.layers:
            self.add(layers.Conv2D(filters=size,
                                   kernel_size=convolution_props.kernel_size,
                                   padding='same',
                                   activation=convolution_props.activation))
            self.add(layers.MaxPooling2D())
        # Flatter
        self.add(layers.Flatten())
        # Create the dense layers
        for size in dense_props.layers:
            self.add(layers.Dense(units=size, activation=dense_props.activation))
        self.add(layers.Dense(dense_props.final_layer))

    def compile_model(self, optimizer, loss_fn, metrics):
        self.compile(optimizer, loss_fn, metrics)

    def evaluate_accuracy(self, test_input):
        _, accuracy = self.evaluate(test_input)
        return accuracy

    def train(self, samples, labels, validation_split, epochs, hparams):
        self.fit(
            x=samples,
            y=labels,
            validation_split=validation_split,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(self.log_folder),  # log metrics
                hp.KerasCallback(self.log_folder, hparams),  # log hparams
            ],
        )

    def save_model(self, model_folder, file_to_save=None):
        if not file_to_save:
            file_to_save = self.name
        destination = Path(model_folder, file_to_save)

        self.save(destination, save_format='h5')
        mlflow.log_artifacts(destination, "model")

    @staticmethod
    def load_model(model_location):
        """ Returns a Keras model instance that will be compiled if it was saved that way, otherwise need to compile
        :parameter model_location: destination of the saved model, it could be: `str`, `pathlib.Path`, `h5py.File`
        """
        return tf.keras.models.load_model(model_location)

    @staticmethod
    def separate_features_and_labels(dataset):
        features =[]
        labels = []
        for sample in dataset.as_numpy_iterator():
            features.extend(sample[0])
            labels.extend(sample[1])
        return features, labels
