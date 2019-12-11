from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')

def default_grain_classifier_model():
    '''
    Get default uncompiled model for grain classifcation, based on 5 step
    cooling process.
    '''
    model = keras.Sequential([
        keras.layers.Dense(512, activation='tanh'),
        keras.layers.Dense(256, activation='tanh'),
        keras.layers.Dense(128, activation='tanh'),
        keras.layers.Dense(4, activation='softmax')
    ])
    return model

def network_cross_validation(model, X, y):
    '''Compute cross validation fold scores for given keras model.'''
    eval_scores = []
    for train_index, test_index in StratifiedKFold(n_splits = 3).split(X, y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=250, verbose=0)
        eval_scores.append(model.evaluate(x_test, y_test))
    return eval_scores
