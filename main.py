import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


from load_images import load_images

from preprocessing import normalize_dataset
from preprocessing import multiple_dataset_conversion

from plotting import plot_image_classes

############################## Code Formatting Guidelines  #################################################
# Variable name =  var_name (with underscores)
# defining a function = def function_name(parameters):
# Write a small description for each function
# Define Functions for all the repetitive tasks
###########################################################################################################

dataset_1_tumor_images, dataset_1_normal_images = load_images(1)

# Slower because there are 3000 images
#dataset_2_tumor_images, dataset_2_normal_images = load_images(2)
#plot_image_classes(dataset_2_normal_images, dataset_2_tumor_images, n_images=4, title="Dataset 2", figure_title="Dataset_2_Image_Classes")

# Load as many datasets as needed
X, Y = multiple_dataset_conversion(yes=[dataset_1_tumor_images], no=[dataset_1_normal_images])

# Normalize the images and convert labels to binary encoding:
X, Y = normalize_dataset(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9, random_state=42, stratify=Y)

