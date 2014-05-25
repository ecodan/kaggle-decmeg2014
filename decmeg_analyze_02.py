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

'''
Trying out a completely different approach.

Step 1: capture statistics about the shape of the sensor curves broken out as follows:
- break each sensor timeline into t segments
- assign each sensor a bin number s by rank of standalone prediction ability
- gather some aggregate stats about each prediction bin and time segment that describe that segment
- create a flattened feature vector for each sample that is s * t(*stats) long (+ label)

Step 2: look at the difference

Step 3: see how well segment can be predicted

Step 4: see how well the observed image can be predicted

Note: didn't work so well...

'''
in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'

# file_suffix = '.mat.csv'
file_suffix = '.mat.csv.binned.csv'
# file_suffix = '.mat.short.csv'

len_ts = 250 # length of time series
num_sensors = 306

def load_data(dir, num_files=100, num_samples=None):
    print('loading data from ' + dir)
    files = os.listdir(dir)
    files.sort()
    # concat all files
    data = []
    file_ct = 0
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        ndata = np.loadtxt(dir + '/' + file, delimiter=',')
        if len(data) == 0:
            data = ndata if num_samples == None else ndata[0:num_samples, :]
        else:
            data = np.vstack([data, ndata if num_samples == None else ndata[0:num_samples, :]])

        file_ct += 1
        if file_ct >= num_files:
            break

    return data


# pulls all columns except last (y) column from data matrix
def get_features_all(data):
    return data[:,0:-1]

# pulls subset of columns in non-contiguous segments from data matrix
# this is based on analysis that not all sensors are equal in predictive value
def get_features_subset_01(data, sensors):
    # create a container to hold n rows of # sensors * len_ts columns
    ret = np.zeros((len(data), len(sensors)*len_ts))
    sensor_array = sensors.tolist()
    for idx, val in enumerate(sensor_array):
        ret[:,(idx*len_ts):(idx*len_ts+len_ts)] = data[:, (val*len_ts):(val*len_ts+len_ts)]
    return ret


def segment_and_bin(train_dir):
    print('starting segment_and_bin...')

    files = os.listdir(train_dir)
    files.sort()
    file_ct = 0

    s = 10
    t = 25
    m = 6
    sensor_bin_size = num_sensors / s
    time_bin_size = len_ts / t

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
        # data heirarchy is sample -> sensor -> time
        ndata = np.loadtxt(train_dir + '/' + file, delimiter=',')
        print('raw data ' + str(ndata.shape))
        X = ndata[:, 0:-1]
        y = ndata[:,-1::]

        # new heirarchy is sample -> sensor bin -> time bin -> measurement
        res = np.zeros((len(y), s, t, m))
        res_o = np.zeros((len(y), s * t * m))

        num_samples = len(X)

        for idx in range(0, num_samples):
            sensor_data = X[idx]

            sensor_data = sensor_data.reshape((num_sensors, len_ts))

            # group sensors into bins and loop
            for iS in range(0,s):
                r_start = iS * sensor_bin_size
                r_stop = iS * sensor_bin_size + sensor_bin_size
                bsensors = np.where(np.all([reval[file_ct] >= r_start, reval[file_ct] < r_stop], axis=0))
                dsensors = sensor_data[bsensors]

                # loop through time bins and collect metrics
                for iT in range(0,t):
                    t_start = iT * time_bin_size
                    t_end = iT * time_bin_size + time_bin_size
                    tsensors = dsensors[:, t_start:t_end]
                    res[idx][iS][iT][0] = tsensors.mean()
                    res[idx][iS][iT][1] = tsensors.std()
                    res[idx][iS][iT][2] = tsensors.min()
                    res[idx][iS][iT][3] = tsensors.max()
                    res[idx][iS][iT][4] = np.percentile(tsensors, 25)
                    res[idx][iS][iT][5] = np.percentile(tsensors, 75)


        file_ct += 1

        for idx in range(0, num_samples):
            res_o[idx] = res[idx].ravel()
        res_o = np.hstack((res_o, y))
        np.savetxt(train_dir + file + '.binned.csv', res_o, delimiter=',')
        print('results=\n' + str(res))


def evaluate_models(train_dir):
    print('reading files')

    # data = np.loadtxt(train_dir + '/train_subject01.mat.csv.binned.csv', delimiter=',')
    data = load_data(train_dir)

    print('raw data ' + str(data.shape))
    X = get_features_all(data)
    print('X ' + str(X.shape))
    y = data[:,-1::]
    print('y ' + str(y.shape))

    # for C in [.01,.1,1,10]:
    #     for p in ['l1','l2']:
    #         print('C=' + str(C) + ' p=' + p)
    #         clf = LogisticRegression(C=C,penalty=p)
    #         print('cross validating')
    #         scores = sl.cross_validation.cross_val_score(clf, X, y.ravel(), scoring='accuracy')
    #         mscores = np.mean(scores)
    #         print('ref xval scores=' + str(scores) + ' | mean=' + str(mscores))

    # random forest - not working so well on this
    # for e in [5,10,25]:
    #     for c in ['gini','entropy']:
    #         for f in [.5,.75,1.0]:
    #             print (': evaluating e=' + str(e) + ' c=' + c + ' f=' + str(f))
    #             clf = RandomForestClassifier(n_estimators=e,criterion=c, max_features=f)
    #             scores = sl.cross_validation.cross_val_score(clf, X, y.ravel(), scoring='roc_auc')
    #             mscores = np.mean(scores)
    #             print(': RF xval scores=' + str(scores) + ' | mean=' + str(mscores))
    #             # if mscores > best_auc:
    #             #     best_auc = mscores

    # gradient boost
    # for e in [50,100,200]:
    #         for f in [.5,.75,1.0]:
    #             print (': evaluating e=' + str(e) + ' f=' + str(f))
    #             clf = GradientBoostingClassifier(n_estimators=e, max_features=f, verbose=True)
    #             scores = sl.cross_validation.cross_val_score(clf, X, y.ravel(), scoring='roc_auc')
    #             mscores = np.mean(scores)
    #             print(': GB xval scores=' + str(scores) + ' | mean=' + str(mscores))
    #             # if mscores > best_auc:
    #             #     best_auc = mscores

    # create training and test data (.75/.25 split)
    clf = LogisticRegression(C=10,penalty='l1')
    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y)
    clf.fit(X_train, y_train.ravel())
    z = clf.predict(X_test)
    print ('TRAIN conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
    print ('TRAIN class report:\n' + str(sl.metrics.classification_report(y_test, z)))


# segment_and_bin(in_dir + 'train/out/')

evaluate_models(in_dir + 'train/out/')