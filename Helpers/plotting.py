import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""
    The purpose of this file is to provide a set of functions that can be used to plot images/data.

    Authors:
        Arsh Lalani, Faculty of Engineering, Western University
        Andrew Harrop, Faculty of Engineering, Western University

"""

def plot_image(image: np.ndarray, save:str=None) -> None:
    
    """
        Plot an image.

        :param image: The image to plot.
    """

    
    plt.imshow(image)
    plt.axis('off')
    # Remove the x and y ticks
    plt.xticks([])
    plt.yticks([])

    if save is not None:
        plt.savefig("./Figures/"+save+".png")
    plt.show()


def plot_images(images: list) -> None:
    
    """
        Plot a list of images.

        :param images: The images to plot.
    """

    for image in images:
        plot_image(image)


def plot_image_classes(normal_images: list, tumor_images: list, n_images: int = 10, title: str = "Image plot", figure_title: str = None) -> None:
    
    """
        Plots the first n_images from the normal and tumor images on a single plot.

        :param normal_images: The normal images.
        :param tumor_images: The tumor images.
        :param n_images: The number of images to plot.
    """

    fig, axes = plt.subplots(ncols=n_images, nrows=2, figsize=(10, 5))

    fig.subplots_adjust(wspace=0.1)

    fig.suptitle(title, fontsize=16)

    fig.canvas.set_window_title("Image Labels")

    # Use normal images because # normal images < # tumor images
    start_index = np.random.randint(0, len(normal_images) - n_images)
    
    # Plot the first n_images from the normal images
    for i in range(n_images):
        axes[0, i].axis('off')
        axes[0, i].imshow(normal_images[i + start_index])
        axes[0, i].set_title("Normal #{}".format(i + start_index))

    # Plot the first n_images from the tumor images
    for i in range(n_images):
        axes[1, i].axis('off')
        axes[1, i].imshow(tumor_images[i + start_index])
        axes[1, i].set_title("Tumor #{}".format(i + start_index))

    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    if figure_title is not None:
        plt.savefig("./Figures/"+figure_title + ".png")

    plt.show()



def plot_augmented(image: np.ndarray, datagen:ImageDataGenerator ) -> None:
    
    """

        :param image: The image to plot.
        :param label: The label of the image.
    """

    augmented_test_image = image.reshape((1,)+image.shape)
    examples = 0


    # Create a 4x5 grid of augmented images
    fig, axs = plt.subplots(4, 5, figsize=(10, 10))
    # Remove horizontal space between axs
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for batch in datagen.flow(augmented_test_image, batch_size=1):
        coord = examples // 5, examples % 5
        axs[coord].imshow(batch[0])
        axs[coord].axis('off')

        examples += 1
        if examples >= 20:
            break

    plt.show()
    fig.savefig('./Figures/augmented_images.png')


def testing_results(model: object, X_test: list, Y_test:list) -> None:
    
    """
        Plots the accuracy and loss of the model over the epochs.
        :param model: The model to test.
        :param test_generator: The test generator.
        :param test_labels: The test labels.
        :param test_images: The test images.
        :param batch_size: The batch size.
    """

    from sklearn.metrics import classification_report, confusion_matrix
    
    pred = model.predict(X_test, batch_size=32)
    pred = np.argmax(pred, axis=1)
    actuals = np.argmax(Y_test, axis=1)

    print(classification_report(actuals, pred, target_names=['no', 'yes']))

    print(confusion_matrix(actuals, pred))

    total = sum(sum(confusion_matrix(actuals, pred)))
    accuracy = (confusion_matrix(actuals, pred)[0][0] + confusion_matrix(actuals, pred)[1][1]) / total

    print("Accuracy: ", accuracy)


def plot_history(history: dict) -> None:
    
    """
        Plots the accuracy and loss of the model over the epochs.

        :param history: The history of the model.
    """
    history = history.history

    # Plot the training and validation loss in one figure and the training and validation accuracy in another figure.
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], color='red', label='Training loss')
    plt.plot(history['val_loss'], color='blue', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(history['accuracy'], color='red', label='Training accuracy')
    plt.plot(history['val_accuracy'], color='blue', label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('./Figures/history.png')
    plt.show()

