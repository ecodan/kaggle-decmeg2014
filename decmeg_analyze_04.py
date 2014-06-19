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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import ProbabilisticPCA
import sklearn.preprocessing as pre
import re
import decmeg_utils as du
from sklearn import grid_search
from sklearn import svm


'''
This approach tries to follow the gist of the research paper.

1) Train a classifier on each train subject
2) Predict each row of ALL subjects on every classifier (resulting in a matrix [total trials] x [num classifiers])
3) Train a meta-classifier on the features above
4) Predict each trial of the test subjects (resulting in a matrix [total trials] x [num classifiers])
5) Predict the test subjects with the meta-classifier


So far the best results are with:
- LR for first pass clf
- AdaBoost for second pass clf
- 125 (.5 sec) data

'''

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'

file_suffix = '.mat.csv'

len_ts = 250 # length of time series
num_sensors = 306
incl_ts = 125 # the length of the time series elements to include
reserve_test_size = 25

# exclude sensors that consistently score low across all train subjects
sensors_below_100 = np.array([211, 216, 220, 234, 264, 269])
sensors_below_75 = np.array([78,135,150,211,214,216,220,234,253,264,266,269,276,281,285,288])
sensors_below_50 = np.array([48,78,123,135,150,165,181,192,195,211,214,216,219,220,222,225,234,253,263,264,266,269,270,274,275,276,279,281,282,285,288,297])
sensors_below_25 = np.array([22,23,27,34,37,39,44,47,48,63,64,66,68,74,78,99,107,123,129,132,135,141,143,150,155,159,165,167,176,181,186,192,195,199,200,201,205,209,211,212,214,215,216,219,220,222,224,225,228,234,235,239,245,252,253,254,259,260,263,264,265,266,269,270,274,275,276,278,279,280,281,282,285,286,287,288,291,294,297,300,305])

# which sensor exclusions to use on this run
selected_threshold = sensors_below_75

# adjusted count of sensors
num_sensors_adj = num_sensors - len(selected_threshold)

# shift the value of the sensor numbers down one so that they are indexes
sensors_idxs = selected_threshold - 1

def train(train_dir):
    print ('training...')

    classifiers = []

    meta_test = np.zeros((16*reserve_test_size, (num_sensors_adj * incl_ts)+1))
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
        # strip off the latter half of the time run (since most of the action appears in the first half)
        Xr = X.reshape((len(X),num_sensors,len_ts))
        Xrs = Xr[:,:,0:incl_ts]
        Xrs = np.delete(Xrs, sensors_idxs, 1)
        X = Xrs.reshape(len(X),num_sensors_adj * incl_ts)
        print('X ' + str(X.shape))
        y = data[:,-1]
        print('y ' + str(y.shape))

        rows += len(y)
        print(file + ' training L1 classifier...')
        clf = LogisticRegression(C=1,penalty='l2')
        # params = {'penalty':['l1','l2'], 'C':[.1,1,10]}
        # gclf = grid_search.GridSearchCV(clf, params, scoring='accuracy', n_jobs=2, verbose=1)
        # gclf.fit(X, y)
        # print(file + ' L1 grid scores:\n' + str(gclf.grid_scores_))
        # clf = gclf.best_estimator_

        # cross validate to see how well the model will do
        X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y, test_size=reserve_test_size)
        clf.fit(X_train, y_train)
        z = clf.predict(X_test)
        print (file + ' TRAIN conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
        print (file + ' TRAIN class report:\n' + str(sl.metrics.classification_report(y_test, z)))

        i1 = fidx*reserve_test_size
        i2 = (fidx*reserve_test_size) + reserve_test_size
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
        # put the data back into 3-D to strip off time series and sensors
        Xr = data[:,0:-1].reshape((len(data),num_sensors,len_ts))
        # only include specific time segments
        Xrs = Xr[:,:,0:incl_ts]
        # remove low value sensors
        Xrs = np.delete(Xrs, sensors_idxs, 1)
        # put back to 2-D
        X = Xrs.reshape(len(data),num_sensors_adj * incl_ts)
        num_file_rows = len(data)
        print(file + ' building L2 features...')
        for idx, clf in enumerate(classifiers):
            meta_features[row_ct:(num_file_rows + row_ct),idx] = clf.predict(X)
        meta_features[row_ct:(num_file_rows + row_ct),-1] = data[:,-1]
        row_ct += num_file_rows

    print('training L2 classifier')
    X = meta_features[:,0:-1]
    print('L2 X shape=' + str(X.shape))
    y = meta_features[:,-1]
    print('L2 y shape=' + str(y.shape))

    # meta_clf = LogisticRegression(C=1,penalty='l2')
    meta_clf = AdaBoostClassifier()
    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y, test_size=0.2)
    meta_clf.fit(X_train, y_train)
    z = meta_clf.predict(X_test)
    print ('L2 conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
    print ('L2 class report:\n' + str(sl.metrics.classification_report(y_test, z)))

    # re-train with all of the data
    # meta_clf.fit(X, y)

    # test against reserved data
    meta_features = np.zeros((len(meta_test),(len(classifiers))))
    print('META TEST performing L1 predictions on meta_test data')
    for idx, clf in enumerate(classifiers):
        X = meta_test[:,0:-1]
        meta_features[:,idx] = clf.predict(X)
    print('META TEST performing L2 predictions...')
    z = meta_clf.predict(meta_features)
    print ('META TEST conf matrix:\n' + str(sl.metrics.confusion_matrix(meta_test[:,-1], z)))
    print ('META TEST class report:\n' + str(sl.metrics.classification_report(meta_test[:,-1], z)))


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
            Xr = X.reshape((len(X),num_sensors,len_ts))
            Xrs = Xr[:,:,0:incl_ts]
            Xrs = np.delete(Xrs, sensors_idxs, 1)
            X = Xrs.reshape(len(X),num_sensors_adj * incl_ts)
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
