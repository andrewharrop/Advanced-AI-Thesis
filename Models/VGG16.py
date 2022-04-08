import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping

"""
    The purpose of this file is to test a CNN model.
    Authors:
        Arsh Lalani, Faculty of Engineering, Western University
        Andrew Harrop, Faculty of Engineering, Western University
"""

def build_vgg16():

    base_VGG16_model = VGG16(weights="./Models/pretrained_vgg.h5", include_top=False, input_shape=(224, 224, 3))    

    model = Sequential()
    model.add(base_VGG16_model)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.layers[0].trainable = False

    model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=1e-4),
            metrics=['accuracy'])

    return model


def train_vgg16(model, train_generator, validation_generator, epochs: int=10):

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=6)


    history = model.fit(
        train_generator,
        steps_per_epoch=4,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=2,
        callbacks=[es]
    )

    return history