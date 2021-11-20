""" Predict Tool

Pre-requirements: Listening mlflow server on the configured port

The script load and convert images to numpy arrays and send them to predict classes.

"""
import argparse
import json
from pathlib import Path

import requests
import numpy as np
from PIL import Image
import tkinter
from tkinter import filedialog as fd

HOST = 'localhost'
PORT = '9000'
BASE_PATH = Path(__file__)
DEFAULT_FOLDER = Path(BASE_PATH.parents[1], 'test-images')


def parse_arguments():
    """Convert CLI options to a dictionary of arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-ct', '--classification-type', type=str, help='Identification for the prediction')
    parser.add_argument('-l', '--labels', type=str, nargs='+', metavar=('dataset-name', 'SAMPLE-CLASS'),
                        required=True, help='Labels for the prediction, otherwise use indices.')
    return parser.parse_args()


def predict(ARGS):
    """Load and convert the selected image then convert it to numpy array to send it for prediction"""
    labels = ARGS.labels

    # User will be asked to select an image from the default folder
    root = tkinter.Tk()
    root.filename = fd.askopenfilename(initialdir=DEFAULT_FOLDER,
                                       title="Select an image file whose class type you want to predict.",
                                       filetypes=(("All file", "*.*"), ("JPG/JPEG Image", ".jpg .jpeg"), ("PNG Image", ".png")))

    # Convert image to the proper format (128x128 pixel, grayscale image)
    img = Image.open(root.filename).resize((128, 128)).convert("L")
    img.show()

    # Convert PIL image into NumPy array
    np_array = np.asarray(img)
    # Reshape the array that required for the model (num_images, img_width, img_height, color_layers)
    np_list = np_array.reshape(1, 128, 128, 1).tolist()

    # Parse to Json then send the request
    json_data = json.dumps({'inputs': np_list})
    r = requests.post(
        url=f'http://{HOST}:{PORT}/invocations',
        headers={'Content-Type': 'application/json'},
        data=json_data)

    predictions = json.loads(r.text)[0]
    accuracies = [round(x * 100, 2) for x in predictions]
    print('Prediction accuracies: ', ', '.join([str(x) for x in accuracies]))

    max_id = accuracies.index(max(accuracies))
    object_class = labels[max_id] if max_id < len(labels) else 'unknown'
    print(f"The object is a(n) '{object_class}' {ARGS.classification_type} with '{accuracies[max_id]}' % accuracy.")


if __name__ == '__main__':
    args = parse_arguments()
    predict(args)