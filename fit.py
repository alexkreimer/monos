import sys
import matplotlib as mpl
print("using MPL version:", mpl.__version__)
mpl.use('TkAgg')

import numpy as np
import pylab
from matplotlib import pyplot as plt
from numpy import linalg as LA

from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold

import multiprocessing as mp

def load_test_data(seq):
    test_data = np.genfromtxt('data/%s_test.txt' % seq, dtype='f8')[1:]

    y = np.array([v[:1] for v in test_data]).ravel()
    X = np.array([v[1:] for v in test_data])
    return (X,y)

def load_train_data(seq):
    train_data = np.genfromtxt('data/%s_train.txt' % seq, dtype='f8')[1:]

    y = np.array([v[:1] for v in train_data]).ravel()
    X = np.array([v[1:] for v in train_data])
    
    return (X,y)

def plt_feature_importance(rf):
    importances = rf.feature_importances_
    importances = importances.reshape((2,20), order='F')
    plt.matshow(importances, cmap=plt.cm.hot)
    plt.title("Feature importances with forests of trees")
    plt.show()

def model_fit(seq, n_estimators):
    X, y = load_train_data(seq)
    
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=mp.cpu_count())
    rf.fit(X, y)
    return rf

def model_eval(seq, rf):
    X, y = load_test_data(seq)

    score = rf.score(X, y)
    print('Test-set R2 score', score)

def fit_and_eval(seq, n_estimators=100):
    rf = model_fit(seq, n_estimators)
    model_eval(seq, rf)
    
def cross_val(seq):
    n_folds = 10
    X, y = load_train_data(seq)

    print('%d-fold cross validation. Dataset: %d samples, %d features' % (n_folds, X.shape[0], X.shape[1]))

    kf = KFold(len(y), n_folds=n_folds)
    n_est = range(10, 100, 10)
    n_est = [10000]
    
    for n_estimators in n_est:
        scores = []
        for i, (train, test) in enumerate(kf):
            rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=mp.cpu_count())
            # the (default) score for each regression tree in the ensemble is regression
            # r2 determination coefficient (e.g., how much variance in y is explained
            # by the model)
            # https://www.khanacademy.org/math/probability/regression/regression-correlation/v/r-squared-or-coefficient-of-determination
            rf.fit(X[train], y[train])

            if False:
                y_pred = rf.predict(X[test])
                score = mean_squared_error(y_pred, y[test])
            else:
                score = rf.score(X[test], y[test])
            scores.append(score)
        scores = np.array(scores)
        print("n_estimators=%d; accuracy (R^2 score): %0.2f (+/- %0.2f)" % (n_estimators, scores.mean(), scores.std() * 2))
    
if __name__ == '__main__':
    seq = sys.argv[1]
    cross_val(seq)
    fit_and_eval(seq)

