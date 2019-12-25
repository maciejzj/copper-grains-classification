import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.morphology import watershed
from img_processing import *

# Load and show
img = imread('img/103_E5R_0.jpg')
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

# Segmentation
markers = np.zeros_like(img_prep)
markers[img_prep < 0.6] = 1
markers[img_prep > 0.75] = 2
fig = plt.figure()
plt.imshow(markers, cmap=plt.get_cmap('gray')); plt.title('Thr')

elevation_map = sobel(img_prep)
fig = plt.figure()
plt.imshow(elevation_map); plt.title('Elevation map')
segmentation = watershed(elevation_map, markers)
fig = plt.figure()
plt.imshow(segmentation); plt.title('Segmented')

# Labeling
from scipy import ndimage as ndi
labeled, x = ndi.label(segmentation - 1)
print(x)
fig = plt.figure()
plt.imshow(labeled); plt.title('Labeled')

# Get center of masses
centres = ndi.center_of_mass(segmentation - 1, labeled, list(range(1, x)))
fig = plt.figure()
plt.imshow(img_crop, cmap=plt.get_cmap('gray')); plt.title('Centers')
x,y = zip(*centres)
plt.scatter(y, x, s=4, c='r', marker='X')

plt.show()
