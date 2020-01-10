'''Demo of grain classifcation using blob detection.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from img_processing import decode_labels, default_img_set, full_prepare
from blob_series_tracker import count_blobs_with_all_methods
from neural_network import default_grain_classifier_model


def classification_demo(X, y):
    '''
    Demo grain classification on given data.
    Train and test default model.
    '''
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.33, random_state=1)

    model = default_grain_classifier_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=300, verbose=0)

    print("Test y:", y_test)
    print("Test y:", [decode_labels(y) for y in y_test])
    print('Test prediction scores:\n', model.predict(X_test))
    prediction = model.predict_classes(X_test)
    print('Test prediction classification:\n', prediction)
    print('Test prediction classification:\n',
          [decode_labels(y) for y in prediction])
    print('Model evaluation loss and accuracy:\n',
          model.evaluate(X_test, y_test, verbose=0), '\n')

    plt.figure()
    ax = plt.subplot()
    plt.title('Model training history')
    plt.xlabel('Epoch')

    lns1 = plt.plot(history.history['accuracy'], c='b', label='Accuracy')
    plt.ylabel('Accuracy')
    plt.twinx()
    lns2 = plt.plot(history.history['loss'], c='r', label='Loss')
    plt.ylabel('Loss')

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right', bbox_to_anchor=(1, 0.5))


def main():
    '''Demo grains classification.'''
    X, y = default_img_set()
    X = [[full_prepare(img) for img in same_sample] for same_sample in X]

    Xs = count_blobs_with_all_methods(X)

    DEMO_NAMES = ('All blobs detection',
                  'Detect only remaining blobs',
                  'Percentage of remaining blobs')
    for X, demo_name in zip(Xs, DEMO_NAMES):
        print(demo_name)
        classification_demo(X, y)

    plt.show()
  

if __name__ == '__main__':
    main()
