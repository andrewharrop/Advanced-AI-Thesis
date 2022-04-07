import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import imutils

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

"""
    The purpose of this file is to provide a set of functions that can be used to preprocess images.

    Authors:
        Arsh Lalani, Faculty of Engineering, Western University
        Andrew Harrop, Faculty of Engineering, Western University

"""


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


def crop_images(images: list) -> np.array:
    """
        Find the extreme points along the brain edge and crop the image. Removes blackspace.

        :param images: The images to crop.
        :return: The cropped images.
    """

    cropped_images = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threhold the image and preform erosion and dilation to remove noise
        threshold = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=2)

        # Get max contour
        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        max_contour = max(contours, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
        extRight = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
        extTop = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
        extBot = tuple(max_contour[max_contour[:, :, 1].argmax()][0])

        # crop the image
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
        cropped_images.append(new_image)
    return np.array(cropped_images)

    


def preprocess_images(images: np.array, dimensions:tuple=(224, 224)) -> np.array:
    """
        Preprocess images for VGG16.

        :param images: The images to preprocess.
        :param dimensions: The dimensions to resize the images to.

        :return: The preprocessed images.
    """
    preprocessed_images = []
    for image in images:
        image = cv2.resize(
            image,
            dsize=dimensions,
            interpolation=cv2.INTER_CUBIC
        )
        preprocessed_images.append(preprocess_input(image))
    return np.array(preprocessed_images)


def augment_image_set(X: np.array, Y: np.array, augmentation:ImageDataGenerator, augment_limit:int=3) -> tuple:
    """
        Augment the image set.

        :param X: The images to augment.
        :param Y: The labels of the images.
        :param augmentation: The augmentation to use.
        :return: The augmented images and labels.
    """
    X_augmented = []
    Y_augmented = []
    for image in range(len(X)):
        X_augmented.append(X[image])
        Y_augmented.append(Y[image])
        for i in range(augment_limit):
            augmented_image = augmentation.flow(np.array([X[image]]), np.array([Y[image]]), batch_size=1).next()[0][0]
            X_augmented.append(augmented_image)
            Y_augmented.append(Y[image])
    return np.array(X_augmented), np.array(Y_augmented)


def labels_to_categorical(labels):
    label_binarizer = LabelBinarizer()
    labels = label_binarizer .fit_transform(labels)
    return to_categorical(labels)