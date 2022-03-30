import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
from imutils import paths

############################## Code Formatting Guidelines  #################################################
#Variable name =  var_name (with underscores)
#defining a function = def function_name(parameters):
#Write a small description for each function
#Define Functions for all the repetitive tasks



#Define image path 
path = "./Data/dataset_1/brain_tumor_dataset"
#Get paths for individual images
image_paths = list(paths.list_images(path))

#Defining lists to store images and labels 
images = []
labels = []

#Iterating over image paths and extracting directory name (yes or no -> labels in our case)
for image_path in image_paths:
    #get the image labels 
    label = image_path.split(os.path.sep)[-2]
    #read image 
    image = cv2.imread(image_path)
    #resize image 
    image = cv2.resize(image, (224, 224))
    images.append(image)
    labels.append(label)


#Plotting Images 
def plot_image(image):
    plt.imshow(image)
    plt.show()

plot_image(images[0])