import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import try_all_threshold
from skimage.color import rgb2gray
from skimage.filters import unsharp_mask
from skimage.io import imread
from skimage.util import invert

from img_processing import *

setup_matplotlib_params()

# Load image and show hist
img = imread('img/104_E5R_0.jpg')
img = rgb2gray(img)
show_with_hist(img, "Original img")

# Crop FLIR UI
img_cropped  = crop_ui(img)
fig = plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))

# Invert
img_inv = invert(img_cropped)
show_with_hist(img_inv, "Inverted")

# Demo tresholding
fig, ax = try_all_threshold(img_inv, figsize=(6, 7), verbose=False)
plt.show()
