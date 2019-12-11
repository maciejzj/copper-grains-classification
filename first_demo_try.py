import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.exposure import histogram, equalize_hist
from skimage.filters import unsharp_mask
from skimage.morphology import *
from skimage.util import invert, crop
from skimage.io import imread
from skimage import exposure


# Load and show
img = imread('img/103_E5R_0.jpg')
img = rgb2gray(img)
fig = plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap=plt.get_cmap('gray')); plt.title('103_E5R_0')
hist, bins_center = histogram(img)
plt.subplot(1,2,2); plt.plot(bins_center, hist, lw=2);

# Crop
img_cropped  = crop(img, ((27, 19), (25, 37)))
fig = plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))

# Invert
img_inv = invert(img_cropped)
fig = plt.figure()
plt.subplot(1,2,1); plt.imshow(img_inv, cmap=plt.get_cmap('gray')); plt.title('Inv')
hist, bins_center = histogram(img_inv)
plt.subplot(1,2,2); plt.plot(bins_center, hist, lw=2);

# Constrast
#img_contr = unsharp_mask(img_inv, radius=1, amount=1.5)
#
#p2, p98 = np.percentile(img_inv, (2, 98))
#img_contr = exposure.rescale_intensity(img_inv, in_range=(p2, p98))
img_contr = img_inv

ig = plt.figure()
plt.subplot(1,2,1); plt.imshow(img_contr, cmap=plt.get_cmap('gray')); plt.title('Cont')
hist, bins_center = histogram(img_contr)
plt.subplot(1,2,2); plt.plot(bins_center, hist, lw=2);

# Erode
img_closed = area_closing(img_contr)
img_eroded = erosion(img_closed)
fig = plt.figure()
plt.imshow(img_closed, cmap=plt.get_cmap('gray')); plt.title('Morph')
img_prep = img_eroded

# Mark sure front and background
markers = np.zeros_like(img_prep)
markers[img_prep < 0.6] = 1
markers[img_prep > 0.7] = 2
plt.imshow(markers, cmap=plt.get_cmap('gray')); plt.title('Thr')

# Segmentation
from skimage.filters import sobel
elevation_map = sobel(img_prep)
from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)
from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)
fig = plt.figure()
plt.imshow(segmentation); plt.title('Seg')

# Labeling
from scipy import ndimage as ndi
labeled, x = ndi.label(segmentation - 1)
print(x)
fig = plt.figure()
plt.imshow(labeled); plt.title('Lab')

plt.show()
