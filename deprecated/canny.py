import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature
from skimage.io import imread
from skimage import restoration
from skimage.morphology import *

from skimage.color import rgb2gray
from skimage.filters import median
from img_processing import *
img = imread('img/104_E5R_0.jpg')
img = rgb2gray(img)
im = crop_ui(img)
#im = rescale(im, 4, anti_aliasing=False)

from skimage import exposure
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.morphology import erosion, dilation, opening, closing, white_tophat


setup_matplotlib_params()

# Generate noisy image of a square
im = median(im, np.ones((4, 4)))
im = exposure.equalize_adapthist(im, clip_limit=0.03)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
						sharex=True, sharey=True)
# Compute the Canny filter for two values of sigma
edges2 = feature.canny(im, sigma=2)
ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges2, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)
selem = disk(1)
edges2 = binary_closing(edges2, selem)
from scipy import ndimage as ndi
edges2 = ndi.binary_fill_holes(edges2)
# display results




ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()
