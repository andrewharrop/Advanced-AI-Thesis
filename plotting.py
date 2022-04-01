import numpy as np
import matplotlib.pyplot as plt


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


def plot_image_classes(normal_images: list, tumor_images: list, n_images: int = 10) -> None:
    
    """
        Plots the first n_images from the normal and tumor images on a single plot

        :param normal_images: The normal images.
        :param tumor_images: The tumor images.
        :param n_images: The number of images to plot.
    """

    fig, axes = plt.subplots(ncols=n_images, nrows=2, figsize=(12, 4.5))

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Plot the first n_images from the normal images
    for i in range(n_images):
        axes[0, i].axis('off')
        axes[0, i].imshow(normal_images[i])
        axes[0, i].set_title("Normal")

    # Plot the first n_images from the tumor images
    for i in range(n_images):
        axes[1, i].axis('off')
        axes[1, i].imshow(tumor_images[i])
        axes[1, i].set_title("Tumor")

    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    plt.show()
