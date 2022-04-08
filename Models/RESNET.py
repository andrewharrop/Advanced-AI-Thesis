from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import SGD


def build_resnet():
    base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


    for layer in base_cnn.layers:
        layer.trainable = False

        
    x = base_cnn.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x) 
    predictions = layers.Dense(2, activation='softmax')(x)
    resnet = Model(base_cnn.input, predictions)
    resnet.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

    return resnet

def train_resnet(model: Model, train_X: list, train_Y: list, test_X: list, test_Y: list, epochs: int = 10, batch_size: int = 8) -> tuple:
    """
        Train a VGG16 model.
        
        :param model: The CNN model to train.
    """
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_Y))
    return history