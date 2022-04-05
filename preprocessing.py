import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import imutils

from keras.applications.vgg16 import preprocess_input

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

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)
