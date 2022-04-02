import os
import cv2

"""
    The purpose of this file is to provide a set of functions that can be used to load images and convert them into openCV format.

    Authors:
        Arsh Lalani, Faculty of Engineering, Western University
        Andrew Harrop, Faculty of Engineering, Western University
"""

# Possible imporovement: save loaded images so we dont have to load -> resize every time, just load cv2 objects


def load_images(dataset: int = 1) -> tuple:
    """
    Loads images from the specified dataset.

    :param dataset: The dataset to load images from.
    :return: A tuple containing the tumor and normal images.
    """

    # Make sure the dataset is 1 or 2
    if dataset != 1 and dataset != 2:
        raise ValueError("Dataset must be 1 or 2")

    # Determine the path of the dataset [1, 2]
    path = "./Data/dataset_{}".format(dataset)

    # Labeled paths
    tumor_path = "/yes"
    normal_path = "/no"

    # Store the images in two different lists representing the labels
    tumor_images = []
    normal_images = []

    # Iterate through the images in the directory and store them in the labeled lists
    for image in os.listdir(path + tumor_path):
        rndr = cv2.imread(path + tumor_path + "/" + image)
        rndr = cv2.resize(rndr, (224, 224))
        tumor_images.append(rndr)

    for image in os.listdir(path + normal_path):
        rndr = cv2.imread(path + normal_path + "/" + image)
        rndr = cv2.resize(rndr, (224, 224))
        normal_images.append(rndr)

    return tumor_images, normal_images


