import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import glob
import natsort
from PIL import Image
import pytesseract
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.util import invert, crop
from skimage.transform import rescale
from skimage.filters import threshold_otsu

def show_with_hist(img, title):
    '''Plot imgage alongside its histogram.'''
    fig = plt.figure()
    plt.subplot(1,2,1); plt.imshow(img, cmap=plt.get_cmap('gray'));
    plt.title(title)
    hist, bins_center = histogram(img)
    plt.subplot(1,2,2); plt.plot(bins_center, hist, lw=2);

def crop_ui(img):
    '''Remove FLIR camera UI from image'''
    img_cropped  = crop(img, ((27, 19), (25, 37)))
    return img_cropped
    
def enhance_sharpness(img):
    '''Enhnace sharpness of FLIR recorded grain image.'''
    return unsharp_mask(img_inv, radius=1, amount=1.5)
        
def enhance_denoising():
    pass

def enhance_contrast():
    pass
    
def get_temperature_bounds(img, bounds=(((6, 24), (283, 318)),
                                       ((219, 236), (283, 318)))):
    '''Extract temperature values from FLIR UI on image.'''
    img = invert(img)
    temp_txt = []
    for bound in bounds:
        bound_img = img[slice(*bound[0]), slice(*bound[1])]
        bound_img = rescale(bound_img, 4, anti_aliasing=True)
        thr = threshold_otsu(bound_img)
        img_txt = bound_img > thr
        img_txt = Image.fromarray(img_txt)
        temp = pytesseract.image_to_string(img_txt, config='digits')
        if temp is not '': 
            temp = float(temp) / 10
        else:
            temp = 0
        temp_txt.append(temp)
    return temp_txt

def full_prepare(img):
    '''Pipeline for FLIR images, convert to grayscale, crop ui and invert.'''
    img_gray = rgb2gray(img)
    img_crop = crop_ui(img_gray)
    img_prep = invert(img_crop)
    return img_prep
    
def load_img_series(path):
    '''Load jpg images containing glob pattern in path and get them in array.'''
    imgs = glob.glob(path + '*.jpg')
    imgs = natsort.natsorted(imgs)
    return [imread(img) for img in imgs]
    
def default_img_set():
    '''
    Get default set of metal grains cooling down recorded with FLIR
    termovision camera.
    '''
    samples_names = ['104_E5R', '113_E5R', '119_E5R',
                     '105_E11R','106_E11R', '115_E11R',
                     '107_E6R', '108_E6R', '117_E6R',
                     '111_E1XP', '112_E1XP', '118_E1XP']
    
    X = [load_img_series('img/' + name) for name in samples_names]
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    return [X, y]
    
def decode_labels(y):
    '''Turn numeric labels into grain samples names.'''
    y_labels = np.array(['E5R', 'E11R', 'E6R', 'E1XP'])
    return y_labels[y]
    
def encode_labels(y_labels):
    '''Turn grain samples names into numeric labels.'''
    y = {'E5R':0, 'E11R':1, 'E6R':2, 'E1XP':3}
    return [y[label] for label in y_labels]
    
def setup_matplotlib_params():
    '''Setup matplotlib to use Latex rendering style.'''
    pgf_with_custom_preamble = {
         "font.family": "serif",
         "text.usetex": True,
         "legend.frameon": False,
         "pgf.rcfonts": False,
         "pgf.preamble": [
            r"\usepackage{units}",
            r"\usepackage{metalogo}",
            r"\usepackage{unicode-math}",
            r"\setmathfont{xits-math.otf}",
            ]
    }
    mpl.rcParams.update(pgf_with_custom_preamble)
