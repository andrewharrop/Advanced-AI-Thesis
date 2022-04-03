import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

"""
    The purpose of this file is to test a CNN model.

    Authors:
        Arsh Lalani, Faculty of Engineering, Western University
        Andrew Harrop, Faculty of Engineering, Western University
"""


def build_cnn(freeze: bool = False) -> Model:
    """
        Build a CNN model.

        :param freeze: Whether to freeze the weights of the pre-trained model.
        :return: The CNN model.
    
    """

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    base_input = base_model.input
    base_output = base_model.output

    base_output = AveragePooling2D(pool_size=(4, 4))(base_output)
    base_output = Flatten(name="flatten")(base_output)
    base_output = Dense(64, activation='relu')(base_output)
    base_output = Dropout(0.5)(base_output)
    base_output = Dense(2, activation='softmax')(base_output)


    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    

    model = Model(inputs=base_input, outputs=base_output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

    model.summary()

    return model

def train_cnn(model: Model, train_X: list, train_Y: list, test_X: list, test_Y: list, epochs: int = 10, batch_size: int = 8) -> tuple:

    """
        Train a CNN model.
        
        :param model: The CNN model to train.
        :param train_X: The training images.
        :param train_Y: The training labels.
        :param test_X: The testing images.
        :param test_Y: The testing labels.
        :param epochs: The number of epochs to train for.
        :param batch_size: The batch size to use.
        :return: A tuple containing the training and testing accuracy.
    """

    train_generator = ImageDataGenerator(fill_mode='nearest', rotation_range=15)


    train_steps = len(train_X) // batch_size
    test_steps = len(test_X) // batch_size

    history = model.fit_generator(train_generator.flow(train_X, train_Y, batch_size=batch_size),
                                    steps_per_epoch=train_steps, epochs=epochs, validation_data=(test_X, test_Y),
                                        validation_steps=test_steps)


    pred = model.predict(test_X, batch_size=batch_size)
    pred = np.argmax(pred, axis=1)
    actuals = np.argmax(test_Y, axis=1)

    print(classification_report(actuals, pred, target_names=['no', 'yes']))

    print(confusion_matrix(actuals, pred))

    total = sum(sum(confusion_matrix(actuals, pred)))
    accuracy = (confusion_matrix(actuals, pred)[0][0] + confusion_matrix(actuals, pred)[1][1]) / total
    print("Accuracy: %f" % accuracy)

    return history, accuracy







