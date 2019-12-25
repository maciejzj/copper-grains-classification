import os
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.util import invert, crop
from skimage.io import imread
import tikzplotlib

from img_processing import *
from blob_detection_compare import *
from blob_series_tracker import *
from blob_analysis import *

def blob_detection_compare_gen():
    img = imread('img/104_E5R_0.jpg')
    img_crop = crop_ui(rgb2gray(img))
    img_prep = full_prepare(img)
    blobs_list = compare_detection(img_prep)

    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    
    for blobs, title in zip(blobs_list, titles):
        fig, ax = plt.subplots()
        plt.title('{}, number of blobs: {}'.format(title, len(blobs)))
        plt.imshow(img_crop, cmap=plt.get_cmap('gray'))
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='r', linewidth=1, fill=False)
            ax.add_patch(c)
       
        title_abbr = ''
        title_abbr = title_abbr.join(word[0] for word in title.split())
        ax.set_axis_off()
        tikzplotlib.save('exports/blob_detection_compare_' + title_abbr)

def blob_count_gen():
    imgs = load_img_series('img/104_E5R')
    imgs_prep = [full_prepare(img) for img in imgs]
    imgs_crop = [crop_ui(rgb2gray(img)) for img in imgs]

    stages_all = find_blob_series(imgs_prep, only_remaining=False)
    stages_rem = find_blob_series(imgs_prep)
    
    # Map stages on first image
    colors = ['blue', 'blueviolet', 'magenta', 'crimson']
    fig, ax = plt.subplots()
    plt.title("Blobs detection with DoH")
    plt.imshow(imgs_crop[0], cmap=plt.get_cmap('gray'))
    for stage, color in zip(stages_rem, colors):
        for blob in stage:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=0.75, fill=False)
            ax.add_patch(c)
    labels = ['Minute 0', 'Minute 1', 'Minute 2', 'Minute 3', 'Minute 4']
    patch_plot_legend_outside(colors, labels)
    ax.set_axis_off()
    tikzplotlib.save('exports/blob_tracker')
    
    # Show two methods combined to compare
    loop_set = enumerate(zip(stages_rem, stages_all, imgs_crop))
    for i, (stage_rem, stage_all, img) in loop_set:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=plt.get_cmap('gray'))
        ax.set_title(
        "Minute: {}, all blobs: {}, rem blobs: {}".
            format(i, len(stage_all), len(stage_rem)))
        for blob_all in stage_all:
            y, x, r = blob_all
            c = plt.Circle((x, y), r, color='b', linewidth=0.75, fill=False)
            ax.add_patch(c)
        for blob_rem in stage_rem:
            y, x, r = blob_rem
            c = plt.Circle((x, y), r, color='r', linewidth=0.75, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        tikzplotlib.save('exports/blob_tracker_min_' + str(i))

def blob_analysis_gen():
    X, y = default_img_set()
    X = [[full_prepare(img) for img in same_sample] for same_sample in X]

    Xa, Xr, Xp = count_blobs_with_all_methods(X)

    colors = ['r', 'g', 'b', 'y']
    labels = ['E5R', 'E11R', 'E6R', 'E1XP']

    plot_blob_stat(Xa, y, colors)
    plt.title('Number of all blobs')
    plt.xlabel('minutes')
    plt.ylabel('number of all detected blobs')
    patch_plot_legend(colors, labels)
    tikzplotlib.save('exports/blob_analysis_all')

    plot_blob_stat(Xr, y, colors)
    plt.title('Number of remaining blobs')
    plt.xlabel('minutes')
    plt.ylabel('remaining blobs')
    patch_plot_legend(colors, labels)
    tikzplotlib.save('exports/blob_analysis_remaining')

    plot_blob_stat(Xp, y, colors)
    plt.title('Percent of remaining blobs')
    plt.xlabel('minutes')
    plt.ylabel('percent of remaining blobs')
    patch_plot_legend(colors, labels)
    tikzplotlib.save('exports/blob_analysis_percent')
    plt.show()

exports_dir = 'exports'
if not os.path.exists(exports_dir):
    os.makedirs(exports_dir)

blob_detection_compare_gen()
blob_count_gen()
blob_analysis_gen()

