import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.utils import to_categorical


def convert_to_X_Y(images: list, label: str) -> tuple:
    
    """
        Converts a list of images to a X-Y format.

        :param images: The images to convert.
        :param label: The label of the images.
        :return: A tuple containing the images in X-Y format.
    """

    X = []
    Y = []

    for image in images:
        X.append(image)
        Y.append(image)

    return X, Y


def multiple_dataset_conversion(**kwargs) -> tuple:
    
    """
        Converts multiple datasets to an X-Y format.

        :param kwargs: The datasets to convert. Example: yes=[data,data,data], no=[data,data,data]
        :return: A list of covariates and a list of labels.
    """

    X = []
    Y = []

    for classifier, image_lists in kwargs.items():
        for image_list in image_lists:
            for image in image_list:
                X.append(image)
                Y.append(classifier)
    return X, Y


def normalize_dataset(images: list, labels: list) -> tuple:
    
    """
        Normalize images between 0 and 1, and convert labels to binary encoding.

        :param images: The images to normalize.
        :param labels: The labels of the images.
        :return: The normalized images and the labels in binary encoding.
    """

    images = np.array(images) / 255.0
    labels = LabelBinarizer().fit_transform(labels).T[0]
    labels = to_categorical(labels)

    return images, labels
