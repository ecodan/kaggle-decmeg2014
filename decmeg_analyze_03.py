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

'''
This approach tries to reorder the sensors by individual ranking to keep the feature columns intact
'''

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'

# file_suffix = '.mat.pca.csv'
file_suffix = '.mat.csv'
reord_file_suffix = '.reorder.csv'

len_ts = 250 # length of time series
num_subjects=8
num_samples=200
num_sensors=306



# play-around method for trying different models
def evaluate_models(train_dir):
    print('reading files from ' + train_dir)
    data = du.load_data(train_dir, file_suffix, num_subjects, num_samples)

    print('raw data ' + str(data.shape))
    X = du.get_features_all(data)
    print('X ' + str(X.shape))
    y = data[:,-1::]
    print('y ' + str(y.shape))

    # create training and test data (.75/.25 split)
    clf = LogisticRegression(C=10,penalty='l1')
    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y)
    clf.fit(X_train, y_train.ravel())
    z = clf.predict(X_test)
    print ('TRAIN conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
    print ('TRAIN class report:\n' + str(sl.metrics.classification_report(y_test, z)))

    print('reordering by rank')
    reval = du.load_ranking(train_dir)

    # loop though the dataset and reorder the sensors based on rank
    newX = np.zeros(X.shape)
    for nsubj in range(0, num_subjects):
        idx_start = (nsubj*num_samples)
        idx_stop = (nsubj*num_samples) + num_samples
        for i, v in enumerate(reval[nsubj]):
            # print('DIAG i=' + str(i) + ' v=' + str(v) + ' idx_start=' + str(idx_start) + ' idx_stop=' + str(idx_stop))
            newX[idx_start:idx_stop, (int(v)*len_ts):((int(v)*len_ts) + len_ts)] = X[idx_start:idx_stop, (i*len_ts):((i*len_ts) + len_ts)]
    print('validation: non-0>' + str(np.count_nonzero(X)) + '=' + str(np.count_nonzero(newX)) + ' | zero>' + str(X.size - np.count_nonzero(X)) + '=' + str(newX.size - np.count_nonzero(newX)) )

    X = newX
    clf = LogisticRegression(C=10,penalty='l1')
    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y)
    clf.fit(X_train, y_train.ravel())
    z = clf.predict(X_test)
    print ('TRAIN2 conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
    print ('TRAIN2 class report:\n' + str(sl.metrics.classification_report(y_test, z)))


def evaluate_rank_prediction_models(train_dir, file_suffix):
    print('evaluate_rank_prediction_models')

    # create a matrix of time series by rank for 1-N people and x-val
    reval = du.load_ranking(train_dir)

    files = os.listdir(train_dir)
    files.sort()
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        X = du.get_features_all(data)
        print('X ' + str(X.shape))
        y = data[:,-1::]
        print('y ' + str(y.shape))



def reorder_features(train_dir, file_suffix):
    print('reorder_features...')

    reval = du.load_ranking(train_dir)

    files = os.listdir(train_dir)
    files.sort()
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        X = du.get_features_all(data)
        print('X ' + str(X.shape))
        y = data[:,-1::]
        print('y ' + str(y.shape))

        newX = np.zeros(X.shape)
        for i, v in enumerate(range(0,num_sensors)):
            newX[:, (v*len_ts):((v*len_ts) + len_ts)] = X[:, (i*len_ts):((i*len_ts) + len_ts)]
        print('validation: non-0>' + str(np.count_nonzero(X)) + '=' + str(np.count_nonzero(newX)) + ' | zero>' + str(X.size - np.count_nonzero(X)) + '=' + str(newX.size - np.count_nonzero(newX)) )
        ndata = np.hstack((newX, y))
        np.savetxt(train_dir + file + '.reorder.csv', ndata, delimiter=',',fmt='%1.3f')



def train(train_dir):
    print ('training...')
    data = du.load_data(train_dir, file_suffix, 8, 200)

    X = data[:,0:-1]
    print('X ' + str(X.shape))
    y = data[:,-1::]
    print('y ' + str(y.shape))

    clf = LogisticRegression(C=1,penalty='l2')
    clf.fit(X, y.ravel())

    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y)
    z = clf.predict(X_test)
    print ('TRAIN conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
    print ('TRAIN class report:\n' + str(sl.metrics.classification_report(y_test, z)))

    return clf



def predict(test_dir, clf):
    print ('predicting...')

    files = os.listdir(test_dir)
    files.sort()
    output = np.zeros((4058,2), dtype=np.int)
    idx = 0
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(test_dir + '/' + file, delimiter=',')
        z = clf.predict(data)
        m = re.match("test_subject(.*)\.mat", file)
        uid = m.group(1)
        userid = int(uid)
        for i in range(0, len(z)):
            output[idx][0] = userid * 1000 + i
            output[idx][1] = z[i]
            idx += 1
    df = pd.DataFrame(output)
    df.columns = ['Id', 'Prediction']
    df.to_csv(test_dir + '/predictions.csv', index=False, header=True)

evaluate_models(in_dir + 'train/out/')

# reorder_features(in_dir + 'train/out/', file_suffix)

# clf = train(in_dir + '/train/out')

# predict(in_dir + '/test/out', clf)
