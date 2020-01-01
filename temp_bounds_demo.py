import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.transform import rescale
from skimage.util import invert

from img_processing import get_temperature_bounds


img = imread('img/103_E5R_1.jpg')
img = rgb2gray(img)
img = invert(img)

bounds = (((6, 24), (283, 318)),
          ((219, 236), (283, 318)))

for bound in bounds:
    bound_img = img[slice(*bound[0]), slice(*bound[1])]
    bound_img = rescale(bound_img, 4, anti_aliasing=True)
    thr = threshold_otsu(bound_img)
    img_txt = bound_img > thr

    plt.figure()
    plt.imshow(bound_img, cmap="gray")
    plt.axis('off')
    plt.savefig('temp_bounds_scale.png', dpi=300, transparent=True,
                bbox_inches='tight')
    plt.figure()
    plt.axis('off')
    plt.imshow(img_txt, cmap="gray")
    plt.savefig('temp_bounds_bin.png', dpi=300, transparent=True,
                bbox_inches='tight')

print(get_temperature_bounds(img))

plt.show()
