from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.io import imread
from skimage.util import invert, crop

from img_processing import crop_ui, full_prepare


def compare_detection(img):
    blobs_log = blob_log(img, max_sigma=2, num_sigma=10, threshold=.125)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(img, max_sigma=2, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(img, max_sigma=2, threshold=.01)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)

    return [blobs_log, blobs_dog, blobs_doh]


if __name__ == "__main__":
    img = imread('img/104_E5R_0.jpg')
    img_crop = crop_ui(rgb2gray(img))
    img_prep = full_prepare(img)
    blobs_list = compare_detection(img_prep)

    colors = ('r', 'g', 'b', 'c')
    titles = ('Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes.ravel()
    loop_set = zip(blobs_list, colors, titles)
    for idx, (blobs, color, title) in enumerate(loop_set):
        ax[idx].set_title('{}, number of blobs: {}'.format(title, len(blobs)))
        ax[idx].imshow(img_crop, cmap=plt.get_cmap('gray'))
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
            ax[idx].add_patch(c)

    plt.tight_layout()
    plt.show()

