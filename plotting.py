import numpy as np
import matplotlib.pyplot as plt


"""
    The purpose of this file is to provide a set of functions that can be used to plot images/data.

    Authors:
        Arsh Lalani, Faculty of Engineering, Western University
        Andrew Harrop, Faculty of Engineering, Western University

"""

def plot_image(image: np.ndarray) -> None:
    
    """
        Plot an image.

        :param image: The image to plot.
    """

    plt.imshow(image)
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
        plt.savefig(figure_title + ".png")

    plt.show()

def plot_cnn_history(history: dict, epochs: int) -> None:
    
    """
        Plots the accuracy and loss of the model over the epochs.

        :param epochs: The number of epochs.
        :param history: The history of the model.
    """
    print(history.keys(), "\n")
    print(history["loss"], "\n")
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["loss"], label= "train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label= "val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label= "train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label= "val_acc")
    plt.title("Training Loss and Accuracy on Brain Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc= "lower left")
    plt.savefig("plot.jpg")
