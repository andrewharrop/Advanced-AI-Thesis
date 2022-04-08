from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D
from tensorflow.keras.optimizers import Adam


def build_cnn():
    """
        Building a CNN model from scratch 
        
        :param input_shape: The shape of the input.
        :return: The CNN model
    """
    #Takes in raw pixel value of input images 
    model_input = Input((224, 224, 3)) 
    #This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor
    model_output = ZeroPadding2D((2, 2))(model_input) 
    #This is a convolutional layer that takes in the model_output and applies a convolution filter to it to extract features from input image 
    model_output = Conv2D(filters = 32, kernel_size = (7, 7), strides = (1, 1))(model_output)
    #This is a normalization layer that applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1
    model_output = BatchNormalization(axis = 3)(model_output)
    #This is an activation layer that produces single output based on weighted sum of inputs
    model_output = Activation('relu')(model_output) 
    #This is a pooling layer that takes the max value over an input window (of size defined by pool_size) for each channel of the input
    model_output = MaxPooling2D(pool_size = (4, 4))(model_output) 
    model_output = MaxPooling2D(pool_size = (4, 4))(model_output) 
    #This is a layer that flattens the input matrix into vectors and does not affect the batch size 
    model_output = Flatten()(model_output) 
    #This is a dense layer that takes in the model_output and runs fully connected neural network layer on it to produce single output
    #We used sigmod activation function for this layer to get outputs in 1 or 0 classifyin brain tumor or not
    model_output = Dense(units = 2, activation='sigmoid')(model_output) 

    model = Model(inputs = model_input, outputs = model_output)
    
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_cnn(model, X_train, Y_train, X_val, Y_val, epochs, batch_size):
    """
        Train a CNN model.
        
        :param model: The CNN model to train.
        :param train_X: The training images.
        :param train_Y: The training labels.
        :param test_X: The testing images.
        :param test_Y: The testing labels.
        :param epochs: The number of epochs to train the model.
        :param batch_size: The batch size to use for training.
        :return: The trained CNN model.
    """
    history = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=epochs, validation_data=(X_val, Y_val), steps_per_epoch=None,workers = 1)

    return history

    