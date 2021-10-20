from typing import List


class ConvolutionProps:
    """ Wrapper for properties of convolutional layers
    :parameter layers: list of filter sizes for the convolutional layers
        the number of items determine how many convolutional layer will be created
    :parameter kernel: the kernel size for each convolutional layer
    :parameter activation: the name of activation function that will be used for each convolutional layer
    """

    def __init__(self, layers: List[int], kernel: int, activation: str):
        self.layers = layers
        self.kernel_size = kernel
        self.activation = activation

    def get_config(self):
        return {
            'layers': self.layers,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
        }


class DenseProps:
    """ Wrapper for properties of dense layers
    :parameter layers: list of units for the dense layers
        The number of items determine how many dense layer will be created
        Last item should be the number of classes of the samples (final layer)
    :parameter labels: list of labels that used for classification
    :parameter activation: the name of activation function that will be used for each dense layer
    """

    def __init__(self, layers: List[int], labels: List[str],activation: str):
        self.layers = layers[:-1]
        self.final_layer = layers[-1]
        self.labels = labels
        self.activation = activation

    def get_config(self):
        return {
            'layers': [*self.layers, self.final_layer],
            'labels': self.labels,
            'activation': self.activation,
        }
