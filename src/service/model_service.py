from pathlib import Path
import logging
import numpy as np
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.



from helper import config
from helper.config import MODELS_PATH, LOG_PATH
from model.props import ConvolutionProps, DenseProps
from model.recognition_model import RecognitionModel

logger = logging.getLogger('Model-service')

# Default config values
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LOSS_FN = CategoricalCrossentropy()
DEFAULT_ACTIVATION_FN = 'relu'
DEFAULT_KERNEL = 3


def create(name, input_shape, labels, loss_fn=DEFAULT_LOSS_FN):
    """ Create RecognitionModel based on the predefined config and given arguments """
    logger.info('Create RecognitionModel..')

    # Setup the properties for the model
    convolution_cfg = config.get_value('model', 'convolution')
    dense_cfg = config.get_value('model', 'dense')

    convolution_props = ConvolutionProps(
        layers=convolution_cfg.get('layers', [16]),
        kernel=convolution_cfg.get('kernel', DEFAULT_KERNEL),
        activation=convolution_cfg.get('activation', DEFAULT_ACTIVATION_FN))

    dense_props = DenseProps(
        layers=[*dense_cfg.get('layers', []), len(labels)],
        activation=dense_cfg.get('activation', DEFAULT_ACTIVATION_FN))

    # Initialize the model
    model = RecognitionModel(input_shape, convolution_props, dense_props, name, str(Path(LOG_PATH, name)))

    optimizer = config.get_value('model', 'optimizer') or DEFAULT_OPTIMIZER
    model.compile_model(optimizer=optimizer, loss_fn=loss_fn, metrics=['accuracy'])
    model.build_model()
    return model


def compile_n_build(model, optimizer, loss_fn):
    """ Re-compile and build the model with the given specification """
    model.compile_model(optimizer=optimizer, loss_fn=loss_fn, metrics=['accuracy'])
    model.build_model()


def train(model, train_ds, epochs, validation_split):
    """ Split the given dataset (train/validation) & train the model """
    isolated_ds = train_ds.map(lambda x, y: (x, y))
    image_batch, labels_batch = next(iter(isolated_ds))
    model.train_model(image_batch, labels_batch, validation_split, epochs)


def save(model, name):
    """ Save the model and return the location """
    saved_location = model.save_model(MODELS_PATH, name)
    logger.info(f'Model saved to: {saved_location}')
    return saved_location


def load(model_name, build=False, print_summary=True):
    """ Returns a Keras model instance that will be compiled if it was saved that way, otherwise need to compile
    :parameter model_name: string that represent the name of the model that need to be loaded
    :param build: Boolean, whether to build the model when it is loaded (default `False`).
    :param print_summary: Boolean, whether to print the model summary if the model is re-builded (default `True`).
    """
    model_src = Path(MODELS_PATH, model_name)
    if not model_src.is_dir():
        model_src = model_src.with_suffix('.h5')
    if not model_src.exists():
        raise ValueError(f"The given model: '{model_name}' in not exist, it cannot be loaded!")
    model = RecognitionModel.load_saved_model(model_src)
    if build:
        model.build_model(print_summary)
    return model


def evaluate_model(model, test_ds):
    isolated_ds = test_ds.map(lambda x, y: (x, y))
    samples, labels = next(iter(isolated_ds))
    loss, accuracy = model.evaluate(samples, labels)
    logger.info(f'Evaluation result- loss: {loss}, accuracy: {accuracy * 100}%')
    return accuracy, loss


def validate_classification(model, test_ds, labels, print_detailed=True):
    separator_line = lambda: print(50 * '*')
    num_samples = 0
    num_correct = 0
    stat = {key: {'right': 0, 'wrong': 0} for key in labels}
    for test_batch in test_ds.as_numpy_iterator():
        num_samples += len(test_batch[0])
        label_set = np.argmax(test_batch[1], axis=1)
        result_set = model.predict_many(test_batch[0])
        for label_id, prediction in zip(label_set, result_set):
            class_label = labels[label_id]
            if label_id == prediction:
                num_correct += 1
                stat[class_label]['right'] += 1
            else:
                stat[class_label]['wrong'] += 1
            if print_detailed:
                print(f'Prediction; expected: {class_label}({label_id}), prediction: {labels[prediction]}({prediction})'
                      f' - {"Correct" if label_id == prediction else "Wrong"}')
    cumulative_accuracy = round(num_correct / num_samples * 100, 3)
    # Print the result
    print(">>>   Classification result   <<<")
    separator_line()
    indent = max(len(i) for i in stat.keys())
    for key, value in stat.items():
        correct, wrong = value['right'], value['wrong']
        acc = round(correct / (correct + wrong) * 100, 1)
        print(f"{key:<{indent}} || correct: {correct:<3} - wrong: {wrong:,} => (accuracy: {acc} %)")
    separator_line()
    print(f'Accuracy: {num_correct} is correct out of {num_samples} - {cumulative_accuracy:.3f} %')
    separator_line()
    stat['cumulative'] = {'right': num_correct, 'wrong': num_samples - num_correct}
    return stat, cumulative_accuracy
