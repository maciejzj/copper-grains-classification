'''Demo cross validation of grain classifcation using blob detection.'''

import numpy as np

from blob_series_tracker import count_blobs_with_all_methods
from img_processing import default_img_set, full_prepare
from neural_network import (default_grain_classifier_model,
                            network_cross_validation)


def cross_val_demo(X, y):
    '''Demo cross validation of default grain classfier on given data.'''
    X = np.array(X)
    y = np.array(y)

    model = default_grain_classifier_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    score = network_cross_validation(model, X, y, 3)

    print('Folds scores: (loss, acc)\n', score)
    score = np.array(score)
    print('Cross validation mean score (loss, acc):\n',
          score.mean(axis=0), '\n')

def main():
    '''Demo blob cross validation of grain classifcation.'''
    X, y = default_img_set()
    X = [[full_prepare(img) for img in same_sample] for same_sample in X]

    Xs = count_blobs_with_all_methods(X)

    demo_names = (
        'All blobs detection', 'Detect only remaining blobs',
        'Percentage of remaining blobs'
    )
    for X, demo_name in zip(Xs, demo_names):
        print(demo_name)
        cross_val_demo(X, y)

if __name__ == '__main__':
    main()
