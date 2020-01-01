from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import blob_dog
from skimage.io import imread
from skimage.util import invert

from img_processing import crop_ui, get_temperature_bounds, show_with_hist


def find_blobs(img):
    '''
    Find blobs in given image and get list of their positions and radiuses.
    '''
    # Detect blobs with Difference of Gaussian
    blobs = blob_dog(img, max_sigma=2, threshold=0.1)
    # Get blobs radiuses from each kernel sigma
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    return blobs


if __name__ == "__main__":
    # Load and show
    img = imread('img/104_E5R_0.jpg')
    img = rgb2gray(img)
    show_with_hist(img, 'Original image')

    # Get temperature bounds
    print(get_temperature_bounds(img))

    # Crop
    img_crop = crop_ui(img)
    show_with_hist(img_crop, 'Cropped image')

    # Invert
    img_inv = invert(img_crop)
    show_with_hist(img_inv, 'Inverted')

    img_prep = img_inv

    blobs = find_blobs(img_prep)

    fig, ax = plt.subplots(1)
    plt.title("Blobs detection with DoH")
    plt.imshow(img_crop, cmap=plt.get_cmap('gray'))
    for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='r', linewidth=0.75, fill=False)
            ax.add_patch(c)
    ax.set_axis_off()
    print(len(blobs))

    plt.show()

