# coding: utf-8
import numpy as np
from numpy import linalg as LA
import os, pickle, gzip

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as plticker
from matplotlib import cm
from sklearn import tree
import gzip
from kalman import KalmanFilter

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

KITTI_HOME = '/home/kreimer/KITTI'

def plot_path(xyz):
    plt.figure()
    plt.plot(xyz[:,0], xyz[:,2])
    plt.show()
    return

def read_dts(file):
    dts = []
    with open(file, 'r') as f:
        for num, line in enumerate(f):
            pose = np.array([float(val) for val in line.split()]).reshape(3,4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            try:
                delta = np.dot(LA.inv(ppose), pose)
            except NameError:
                ppose = pose
                continue
            dt = delta[:3,3]
            dts.append(dt)

            ppose = pose
    return dts


# In[20]:

def load_gt(seq):
    poses_file = os.path.join('data', 'paths', '%s.txt' % seq)
    dts = np.array(read_dts(poses_file))
    
    # load the precomputed features
    with gzip.open(os.path.join('data', 'features', '%s_1_6_3_300.pklz' % seq), 'rb') as fd:
        pairs, feat, dts = pickle.load(fd)

    return [np.linalg.norm(dt) for dt in dts], feat, dts

def predict_kalman_seq(seq, feat):
    # load the model
    with gzip.open(os.path.join('data', 'models', 'ETR_1_6_3_300.pklz'), 'rb') as fd:
        rf = pickle.load(fd)

    X = np.array(feat)
    y_pred = rf.predict(X)
    # for i, tree_ in enumerate(rf.estimators_):
    #     with open('tree_' + str(i) + '.dot', 'w') as dotfile:
    #         tree.export_graphviz(tree_, dotfile)

    x = np.array([[0.], [0.]]) # initial state (location and velocity)
    P = np.array([[1000., 0.], [0., 1000.]]) # initial uncertainty
    kf = KalmanFilter()
    x_out = [x]
    xabs = 0
    for measurement in y_pred:
        xabs = xabs + measurement
        x, P = kf.step(x, P, xabs)
        x_out.append(x)
    return x_out

for seq in ['04']:
    dts, feat, xyz = load_gt(seq)
    x_out = predict_kalman_seq(seq, feat)
    xabs = 0
    for dt in dts:
        xabs += dt
        print(xabs)
    for x, xtag in x_out:
        print('loc:', x, 'xtag:', xtag)

    plot_path(np.array(xyz))
