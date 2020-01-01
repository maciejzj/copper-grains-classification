'''Plot and analyze data extracted from thermal images of grains.'''

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from img_processing import default_img_set, full_prepare
from blob_series_tracker import count_blobs_with_all_methods


def plot_blob_stat(samples_set, lebels_set, colors):
    '''Plot blob count versus minutes.'''
    plt.figure()
    for sample_series, sample_label_index in zip(samples_set, lebels_set):
        plt.plot(
            [0, 1, 2, 3, 4],
            sample_series,
            c=colors[sample_label_index],
            linewidth=1.2)


def patch_plot_legend(colors, labels):
    '''Make given plots share their legend entry.'''
    legend = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, labels)
    ]
    plt.legend(handles=legend)

def main():
    '''Plot number of detected blobs using three ways of counting.'''
    X, y = default_img_set()
    X = [[full_prepare(img) for img in same_sample] for same_sample in X]

    Xa, Xr, Xp = count_blobs_with_all_methods(X)

    colors = ('r', 'g', 'b', 'y')
    labels = ('E5R', 'E6R', 'E11R', 'E16R')

    plot_blob_stat(Xa, y, colors)
    plt.title('Number of all blobs')
    plt.xlabel('minutes')
    plt.ylabel('number of all detected blobs')
    patch_plot_legend(colors, labels)

    plot_blob_stat(Xr, y, colors)
    plt.title('Number of remaining blobs')
    plt.xlabel('minutes')
    plt.ylabel('remaining blobs')
    patch_plot_legend(colors, labels)

    plot_blob_stat(Xp, y, colors)
    plt.title('Ratio of remaining blobs')
    plt.xlabel('minutes')
    plt.ylabel('Ratio of remaining blobs')
    patch_plot_legend(colors, labels)

    plt.show()


if __name__ == '__main__':
    main()
