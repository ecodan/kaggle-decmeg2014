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
This approach tries to follow the gist of the research paper.

1) Train a classifier on each train subject
2) Predict each row of ALL subjects on every classifier (resulting in a matrix [total trials] x [num classifiers])
3) Train a meta-classifier on the features above
4) Predict each trial of the test subjects (resulting in a matrix [total trials] x [num classifiers])
5) Predict the test subjects with the meta-classifier

'''

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'

file_suffix = '.mat.csv'

len_ts = 250 # length of time series


def train(train_dir):
    print ('training...')

    classifiers = []
    meta_clf = LogisticRegression(C=1,penalty='l2')

    meta_test = np.zeros((1600, 76501))
    # load each file and train the subject classifier
    files = os.listdir(train_dir)
    files.sort()
    rows = 0
    fidx = 0
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')

        X = data[:,0:-1]
        print('X ' + str(X.shape))
        y = data[:,-1]
        print('y ' + str(y.shape))

        rows += len(y)
        print(file + ' training L1 classifier...')
        clf = LogisticRegression(C=1,penalty='l2')
        # clf.fit(X, y.ravel())

        X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y, test_size=100)
        clf.fit(X_train, y_train)
        z = clf.predict(X_test)
        print (file + 'TRAIN conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
        print (file + 'TRAIN class report:\n' + str(sl.metrics.classification_report(y_test, z)))

        i1 = fidx*100
        i2 = (fidx*100) + 100
        # print('populating meta_test [' + str(i1) + ':' + str(i2) + ']' )
        meta_test[i1:i2,0:-1] = X_test
        meta_test[i1:i2,-1] = y_test

        classifiers.append(clf)
        fidx += 1

    # create placeholders for meta features - [one row for each training row] x [column for each classifier + a label]
    meta_features = np.zeros((rows, len(classifiers)+1))

    # go back through and predict each trial against each classifier
    row_ct = 0
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        num_file_rows = len(data)
        print(file + ' building L2 features...')
        for idx, clf in enumerate(classifiers):
            meta_features[row_ct:(num_file_rows + row_ct),idx] = clf.predict(data[:,0:-1])
        meta_features[row_ct:(num_file_rows + row_ct),-1] = data[:,-1]
        row_ct += num_file_rows

    print('training L2 classifier')
    X = meta_features[:,0:-1]
    print('L2 X shape=' + str(X.shape))
    y = meta_features[:,-1]
    print('L2 y shape=' + str(y.shape))
    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y, test_size=0.2)
    meta_clf.fit(X_train, y_train)
    z = meta_clf.predict(X_test)
    print ('L2 conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
    print ('L2 class report:\n' + str(sl.metrics.classification_report(y_test, z)))

    return classifiers, meta_clf



def predict(test_dir, classifiers, meta_clf):
    print ('predicting...')

    files = os.listdir(test_dir)
    files.sort()
    output = np.zeros((4058,2), dtype=np.int)
    idx = 0
    meta_features = []
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(test_dir + '/' + file, delimiter=',')
        f_features = np.zeros((len(data),(len(classifiers) + 1)))
        print('performing L1 predictions on ' + file)
        for idx, clf in enumerate(classifiers):
            X = data[:,0:-1]
            f_features[:,idx] = clf.predict(X)
        f_features[:, -1] = data[:,-1] # set the user ids column
        if len(meta_features) == 0:
            meta_features = f_features
        else:
            meta_features = np.vstack((meta_features, f_features))

    print('performing L2 predictions...')
    output = np.zeros((len(meta_features), 2), dtype=np.int)
    output[:,1] = meta_clf.predict(meta_features[:,0:-1])
    output[:,0] = meta_features[:,-1]

    df = pd.DataFrame(output)
    df.columns = ['Id', 'Prediction']
    df.to_csv(test_dir + '/predictions.csv', index=False, header=True)




classifiers, meta_clf = train(in_dir + '/train/out')

predict(in_dir + '/test/out', classifiers, meta_clf)
