import sys
import matplotlib as mpl
print("using MPL version:", mpl.__version__)
mpl.use('TkAgg')

import numpy as np
import pylab
from matplotlib import pyplot as plt
from numpy import linalg as LA
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold

import multiprocessing as mp
import argparse

def load_test_data(seq, ft):
    test_data = np.genfromtxt('data/%s_%s_test.txt' % (ft, seq), dtype='f8')[1:]

    y = np.array([v[:1] for v in test_data]).ravel()
    X = np.array([v[1:] for v in test_data])
    return (X,y)

def load_train_data(seq, ft):
    train_data = np.genfromtxt('data/%s_%s_train.txt' % (ft, seq), dtype='f8')[1:]

    y = np.array([v[:1] for v in train_data]).ravel()
    X = np.array([v[1:] for v in train_data])
    
    return (X,y)

def plt_feature_importance(seq, rf):
    importances = rf.feature_importances_
    importances = importances.reshape((2,int(importances.shape[0]/2)), order='F')
    plt.matshow(importances, cmap=plt.cm.hot)
    plt.title("Feature importances with forests of trees")
    plt.savefig('%s_feat_heatmap.png' % seq, dpi=100)

def model_fit(seq, ft, n_estimators):
    X, y = load_train_data(seq, ft)
    
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=mp.cpu_count())
    rf.fit(X, y)
    return rf

def model_eval(seq, ft, rf):
    X, y = load_test_data(seq, ft)

    score = rf.score(X, y)
    return score

def fit_and_eval(seq, ft, n_estimators=100):
    rf = model_fit(seq, ft, n_estimators)
    plt_feature_importance(seq, rf)
    return model_eval(seq, ft, rf)
    
def cross_val(seq, ft):
    n_folds = 10
    X, y = load_train_data(seq, ft)

    print('%d-fold cross validation. Dataset: %d samples, %d features' % (n_folds, X.shape[0], X.shape[1]))

    kf = KFold(len(y), n_folds=n_folds)
    n_est = range(30, 110, 20)

    results = []
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
        results.append([seq, ft, X.shape[0], n_estimators, scores.mean(), scores.std()*2])
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', help='Sequence to process, defaults to all')
    parser.add_argument('-cv', '--cross_val', help='Run cross-validation and exit', action='store_true')

    features = ['5bins_edges', '200bins']
    args = parser.parse_args()
    
    if args.seq:
        seq = [ args.seq ]
    else:
        seq = ['%02d' % i for i in range(11)]
        seq.append('all')

    res_cv = []
    res = []
    for ft in features:
        for s in seq:
            score = fit_and_eval(s, ft)
            res.append([s, ft, score])
            if args.cross_val:
                res_cv.extend(cross_val(s, ft))

            columns = ['Sequence #', 'Feature type', 'Accuracy']
            df = pd.DataFrame(data=np.array(res), index=range(len(res)),
                      columns=columns)
            with open('fit_results.tex', 'w+') as fd:
                print(df.to_latex(index=False), file=fd)
            
            if args.cross_val:
                columns = ['Sequence #', 'Feature type', '# of samples', '# of estimators',
                           'Accuracy', '$2\sigma$']
                df = pd.DataFrame(data=np.array(res_cv), index=range(len(res_cv)),
                                  columns=columns)
                with open('cv_results.tex', 'w+') as fd:
                    print(df.to_latex(index=False), file=fd)
