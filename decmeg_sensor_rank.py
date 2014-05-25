__author__ = 'dan'

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from datetime import datetime
from scipy.io import loadmat
import sklearn as sl
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import ProbabilisticPCA
import sklearn.preprocessing as pre
import re
import decmeg_utils as du

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'

# file_suffix = '.mat.pca.csv'
file_suffix = '.mat.csv'

len_ts = 250 # length of time series


def evaluate_sensor_predictive_value_lone(train_dir):
    print('starting evaluate_sensor_predictive_value...')
    # figure out if different sensors are best aligned with the right prediction in each subject
    # data model:  [ subject_id_idx: [ 306 length vector of xval stats ]]

    res = np.zeros((16,306))
    files = os.listdir(train_dir)
    files.sort()
    file_ct = 0

    clf = LogisticRegression(C=1,penalty='l1')

    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        ndata = np.loadtxt(train_dir + '/' + file, delimiter=',')
        print('raw data ' + str(ndata.shape))
        y = ndata[:,-1::]
        # loop through each 250 time series block
        for idx in range(0,306):
            start = idx * len_ts
            end = (idx+1) * len_ts
            X = ndata[:, start:end]
            # print('X = ' + str(X.shape))
            res[file_ct,idx] = np.mean(sl.cross_validation.cross_val_score(clf, X, y.ravel(), scoring='accuracy'))
            print('result for ' + str(file_ct) + ' sensor #' + str(idx) + ' = ' + str(res[file_ct,idx]))
        print('best score for ' + str(file_ct) + ' is ' + str(np.max(res[file_ct,:])))
        file_ct += 1

    np.savetxt(train_dir + '/feature_eval.csv', res, delimiter=',')
    print('results=\n' + str(res))


def evaluate_sensor_predictive_value_ensemble(train_dir):
    print('starting evaluate_sensor_predictive_value...')
    # figure out if different sensors are best aligned with the right prediction in each subject
    # data model:  [ subject_id_idx: [ 306 length vector of xval stats ]]

    res = np.zeros((16,306))
    files = os.listdir(train_dir)
    files.sort()
    file_ct = 0

    clf = LogisticRegression(C=1,penalty='l1')

    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        ndata = np.loadtxt(train_dir + '/' + file, delimiter=',')
        print('raw data ' + str(ndata.shape))
        X = ndata[:, 0:-1]
        y = ndata[:,-1::]
        clf = RandomForestClassifier(max_features=1.0)
        clf.fit(X,y.ravel())
        res[file_ct] = clf.feature_importances_

        file_ct += 1

    np.savetxt(train_dir + '/feature_eval_02.csv', res, delimiter=',')
    print('results=\n' + str(res))


def evaluate_best_sensors_model(train_dir):
    print('evaluate best sensor models...')
    # using the output of the method above, xval all train subjects by the best N sensors
    max_sensors = 300
    sensor_step = 10
    num_samples = max_sensors/sensor_step
    res = np.zeros((16,num_samples+1))
    files = os.listdir(train_dir)
    files.sort()
    file_ct = 0

    clf = LogisticRegression(C=1,penalty='l1')

    # load output from above
    print('loading and calculating rankings...')
    eval = np.loadtxt(train_dir + '/feature_eval_02.csv', delimiter=',')

    # create ranked matrix
    reval = np.zeros(eval.shape)
    for idx in range(0,len(eval)):
        temp = eval[idx].argsort()
        ranks = np.empty(len(temp),int)
        reval[idx][temp] = np.arange(len(temp))

    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        ndata = np.loadtxt(train_dir + '/' + file, delimiter=',')
        print('raw data ' + str(ndata.shape))
        y = ndata[:,-1::]

        # baseline score based on all features
        print('calculating baseline...')
        X = du.get_features_all(ndata)
        res[file_ct,num_samples] = np.mean(sl.cross_validation.cross_val_score(clf, X, y.ravel(), scoring='accuracy'))
        print('baseline subject ' + str(file_ct) + ' = ' + str(res[file_ct,num_samples]))

        for num_sensors in range(0,max_sensors+1, sensor_step):
            if num_sensors == 0:
                continue
            # get best sensors for this subject
            sbest = np.where(reval[file_ct]<=num_sensors)[0] # this is a tuple, so select first and only element
            X = du.get_features_subset_01(ndata,sbest)
            res[file_ct,(num_sensors/sensor_step)-1] = np.mean(sl.cross_validation.cross_val_score(clf, X, y.ravel(), scoring='accuracy'))
            print('evaluated subject ' + str(file_ct) + ' on nbest sensors ' + str(num_sensors) + ' X.shape=' + str(X.shape) + ' | res=' + str(res[file_ct,(num_sensors/sensor_step)-1]))
        file_ct += 1

    np.savetxt(train_dir + '/best_sensors_results.csv', res, delimiter=',')
    print('results=\n' + str(res))


def visualize_best_and_worst_sensors(train_dir):
    print('visualize_best_and_worst_sensors...')

    num_sensors = 10

    num_samples = 10

    files = os.listdir(train_dir)
    files.sort()
    file_ct = 0

    # load output from above
    print('loading and calculating rankings...')
    eval = np.loadtxt(train_dir + '/feature_eval.csv', delimiter=',')

    # create ranked matrix
    reval = np.zeros(eval.shape)
    for idx in range(0,len(eval)):
        temp = eval[idx].argsort()
        ranks = np.empty(len(temp),int)
        reval[idx][temp] = np.arange(len(temp))

    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        ndata = np.loadtxt(train_dir + '/' + file, delimiter=',')
        print('raw data ' + str(ndata.shape))
        y = ndata[:,-1::]

        # get best and worst sensors
        idx_best = np.where(reval[file_ct] < num_sensors)[0] # N best
        idx_worst = np.where(reval[file_ct] >= (reval.shape[1] - num_sensors))[0]

        # select N 1s and 10 0s
        iy0 = np.where(y == 0)[0][0:num_samples]
        iy1 = np.where(y == 1)[0][0:num_samples]

        # create a matrix to hold the various combinations of (0, 1) and (best, worst)
        ndatas = np.zeros((2, 2, num_samples, num_sensors*len_ts))
        ndatas[0,0] = du.get_features_subset_01(ndata[iy0], idx_best)
        ndatas[1,0] = du.get_features_subset_01(ndata[iy1], idx_best)
        ndatas[0,1] = du.get_features_subset_01(ndata[iy0], idx_worst)
        ndatas[1,1] = du.get_features_subset_01(ndata[iy1], idx_worst)

        # create container for average of all sensors on each sample by timeseries
        navgs = np.zeros((2, 2, num_samples, len_ts))

        for i in [0,1]:
            for j in [0,1]:
                for k in range(0, num_samples):
                    d0 = ndatas[i][j][k]
                    d0m = np.reshape(d0, (num_sensors, len_ts))
                    navgs[i][j][k] = np.mean(d0m, axis=0)

        res = np.zeros((num_samples*4, len_ts))
        ct = 0
        for i in [0,1]:
            for j in [0,1]:
                res[(ct*num_samples):(ct*num_samples+num_samples)] = navgs[i][j]
                ct += 1
        np.savetxt(train_dir + 'best_worst_' + file, res, delimiter=',')

    print('done')


evaluate_sensor_predictive_value_ensemble(in_dir + 'train/out/')
evaluate_best_sensors_model(in_dir + 'train/out/')
# visualize_best_and_worst_sensors(in_dir + 'train/out/')


