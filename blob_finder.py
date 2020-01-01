'''Find blobs in thermal images of grains.'''

from math import sqrt

import matplotlib.pyplot as plt
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
    SAMPLE_IMG = imread('img/104_E5R_0.jpg')
    SAMPLE_IMG = rgb2gray(SAMPLE_IMG)
    show_with_hist(SAMPLE_IMG, 'Original image')

    # Get temperature bounds
    print(get_temperature_bounds(SAMPLE_IMG))

    # Crop
    IMG_CROP = crop_ui(SAMPLE_IMG)
    show_with_hist(IMG_CROP, 'Cropped image')

    # Invert
    IMG_INV = invert(IMG_CROP)
    show_with_hist(IMG_INV, 'Inverted')

    IMG_PREP = IMG_INV

    BLOBS = find_blobs(IMG_PREP)

    fig, ax = plt.subplots(1)
    plt.title("Blobs detection with DoH")
    plt.imshow(IMG_CROP, cmap=plt.get_cmap('gray'))
    for blob in BLOBS:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='r', linewidth=0.75, fill=False)
        ax.add_patch(c)
    ax.set_axis_off()
    print(len(BLOBS))

    plt.show()
