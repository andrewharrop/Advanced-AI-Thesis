import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from load_images import load_images

############################## Code Formatting Guidelines  #################################################
# Variable name =  var_name (with underscores)
# defining a function = def function_name(parameters):
# Write a small description for each function
# Define Functions for all the repetitive tasks


dataset_1_tumor_images, dataset_1_normal_images = load_images(1)

# Slower because there are 3000 images
#dataset_2_tumor_images, dataset_2_normal_images = load_images(2)


# Plotting Images
def plot_image(image):
    plt.imshow(image)
    plt.show()


# Plotting Images
def plot_images(images):
    for image in images:
        plot_image(image)


plot_images(dataset_1_tumor_images[:3])
