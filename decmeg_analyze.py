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


in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'

# file_suffix = '.mat.pca.csv'
file_suffix = '.mat.csv'

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


def evaluate_models(train_dir):
    print('reading files')
    data = load_data(train_dir, 3, 100)

    print('raw data ' + str(data.shape))
    X = data[:,0:-1]
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


def train(train_dir):
    print ('training...')
    data = load_data(train_dir)

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

# clf = train(in_dir + '/train/out')

# predict(in_dir + '/test/out', clf)