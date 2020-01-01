from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')

def default_grain_classifier_model():
    '''
    Get default uncompiled model for grain classifcation,
    based on 5 step cooling process using number of blobs.
    '''
    model = keras.Sequential([
        keras.layers.Dense(5, activation='tanh'),
        keras.layers.Dense(256, activation='tanh'),
        keras.layers.Dense(128, activation='tanh'),
        keras.layers.Dense(4, activation='softmax')
    ])
    return model

def network_cross_validation(model, X, y, n_splits):
    '''Compute cross validation fold scores for given keras model.'''
    eval_scores = []
    
    folds = StratifiedKFold(n_splits=n_splits).split(X, y)
    for train_index, test_index in folds:
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
        
        model.fit(x_train, y_train, epochs=300, verbose=0)
        eval_scores.append(model.evaluate(x_test, y_test, verbose=0))
    return eval_scores
    
def mean_confusion_matrix(model, X, y, n_splits):
    '''Compute mean confusion matrix using cross validation with n splits.'''
    conf_matrix = np.zeros((4, 4))
    a
    folds = StratifiedKFold(n_splits = n_splits).split(X, y)
    for train_index, test_index in folds:
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
        
        model.fit(x_train, y_train, epochs=300, verbose=0)
        y_pred = model.predict_classes(x_test)
        
        for test, pred in zip(y_test, y_pred):
            conf_matrix[test][pred] = conf_matrix[test][pred] + 1
    
    return conf_matrix / n_splits

