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



from skimage import exposure
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from img_processing import *

img = imread('img/104_E5R_4.jpg')
img = rgb2gray(img)
im = crop_ui(img)
im = median(im, np.ones((4, 4)))
#im = rescale(im, 0.25, anti_aliasing=False)
plt.figure()
plt.imshow(im, cmap='gray')
im = exposure.equalize_adapthist(im, clip_limit=0.03)

im = feature.canny(im, sigma=2)


plt.figure()
plt.imshow(im, cmap='gray')
plt.show()
