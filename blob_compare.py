import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.util import invert, crop
from skimage.io import imread

from img_processing import *

setup_matplotlib_params()

# Load and prepare img
img = imread('img/104_E5R_0.jpg')
img_prep = full_prepare(img)
img_crop = crop_ui(rgb2gray(img))

# Find blobs
blobs_log = blob_log(img_prep, max_sigma=2, num_sigma=10, threshold=.125)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(img_prep, max_sigma=2, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(img_prep, max_sigma=2, threshold=.01)
blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)

# Plot comparison
blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['r', 'g', 'b', 'c']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes.ravel()
for idx, (blobs, color, title) in enumerate(zip(blobs_list, colors, titles)):
        ax[idx].set_title('{}, number of blobs: {}'.format(title, len(blobs)))
        ax[idx].imshow(img_crop, cmap=plt.get_cmap('gray'))
        for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
                ax[idx].add_patch(c)

plt.tight_layout()

plt.show()
