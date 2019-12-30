import numpy as np

from img_processing import *
from blob_series_tracker import *
from neural_network import *

def cross_val_demo(X, y):
    X = np.array(X)
    y = np.array(y)

    model = default_grain_classifier_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    eval = network_cross_validation(model, X, y)

    print('Folds scores: (loss, acc)\n', eval)
    eval = np.array(eval)
    print('Cross validation mean score (loss, acc):\n',
          eval.mean(axis=0), '\n')

X, y = default_img_set()
X = [[full_prepare(img) for img in same_sample] for same_sample in X]

Xs = count_blobs_with_all_methods(X)

demo_names = ['All blobs detection',
              'Detect only remaining blobs',
              'Percentage of remaining blobs']
for X, demo_name in zip(Xs, demo_names):
    print(demo_name)
    cross_val_demo(X, y)

