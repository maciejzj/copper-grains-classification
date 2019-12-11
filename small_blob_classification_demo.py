import numpy as np
np.set_printoptions(suppress=True, precision=4)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')

from img_processing import *
from blob_series import *
from neural_network import *

setup_matplotlib_params()

def classification_demo(X, y):
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
                                                        test_size=0.33)

    model = default_grain_classifier_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=100, verbose=0)

    print("Test y:", y_test)
    print('Test prediction scores:\n', model.predict(X_test))
    print('Test prediction classification:\n', model.predict(X_test).round())
    print('Model evaluation loss and accuracy:\n', 
          model.evaluate(X_test, y_test, verbose=0),
          '\n')

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title('Model training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

X, y = default_img_set()
X = [[full_prepare(img) for img in same_sample] for same_sample in X]

Xs = []
# Option one: all
Xs.append([
    [len(stage) for stage in find_blob_series(img_series, 
                                              only_remaining = False)]
    for img_series in X
])
# Option two: remaining
Xs.append([
    [len(stage) for stage in find_blob_series(img_series)]
    for img_series in X
])
# Option three: percentage of remaining
Xs.append([
    percent_of_remaining_blobs_in_stages(find_blob_series(img_series)) 
    for img_series in X
])

demo_names = ['All blobs detection',
              'Detect only remaining blobs',
              'Percentage of remaining blobs']
for X, demo_name in zip(Xs, demo_names):
    print(demo_name)
    classification_demo(X, y)

plt.show()
