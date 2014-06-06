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
from sklearn import grid_search
from sklearn import svm
import pickle

'''
This approach does the following:

1) Trains a classifier for each of the 16 train subjects
2) On a trial by trial basis, compare each test subject with each train subject and try to identify which train subject has the highest correlation
3) Use that train subject's classifier to predict the test subject

'''

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'

file_suffix = '.mat.csv'

len_ts = 250 # length of time series


def train(train_dir):
    print ('training...')

    classifiers = []
    comps = []

    # load each file and train the subject classifier
    files = os.listdir(train_dir)
    files.sort()
    rows = 0
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
        # params = {'penalty':['l1','l2'], 'C':[.1,1,10]}
        # gclf = grid_search.GridSearchCV(clf, params, scoring='accuracy', n_jobs=2, verbose=1)
        # gclf.fit(X, y)
        # print(file + ' L1 grid scores:\n' + str(gclf.grid_scores_))
        # clf = gclf.best_estimator_
        clf.fit(X, y.ravel())

        X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y, test_size=100)
        # clf.fit(X_train, y_train)
        z = clf.predict(X_test)
        print (file + ' TRAIN conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test, z)))
        print (file + ' TRAIN class report:\n' + str(sl.metrics.classification_report(y_test, z)))

        classifiers.append(clf)

    s = pickle.dumps(classifiers)
    f = open(train_dir + '/classifiers.pkl', 'w')
    f.write(s)
    f.close()

    return classifiers


def match_subject_to_comp(data, comps):
    print('match_subject_to_comp')
    comp_corrs = np.zeros([len(comps),2])
    idx = 0
    for comp in comps:
        # run correlation between the 50 comp and 50 test
        # print('DIAG shapes = ' + str(comp.shape) + ' ' + str(data.shape))
        combo = np.vstack((comp[:,0:-1], data[:,0:-1]))
        corr = np.corrcoef(combo)
        # print('corr.shape=' + str(corr.shape))
        # figure out if each test trial best correlates with the train 0 or 1; assumes 0's are first in data file
        corr0s = []
        corr1s = []
        for j in range(50,100): # the test data is 50:100; compare the correlation with the train data in 0:50
            corr0 = np.mean(corr[j,0:25])
            corr1 = np.mean(corr[j,25:50])
            if (corr0 > corr1):
                corr0s.append(corr0)
            else:
                corr1s.append(corr1)
        comp_corrs[idx] = [np.mean(corr0s), np.mean(corr1s)]
        idx += 1

    # pick best performing comp
    comp_dev = (comp_corrs - np.mean(comp_corrs,axis=0)) # calculate deviation from mean for the two columns
    best_dev = 0.0
    best_dev_idx = 0
    for idx, comp_row in enumerate(comp_dev):
        comp_row_sum = np.sum(comp_row)
        if comp_row_sum > best_dev:
            best_dev = comp_row_sum
            best_dev_idx = idx

    return best_dev, best_dev_idx

def comp_predict(X, comp):

    return y


def predict(test_dir, train_dir):
    print ('predicting...')

    # load the classifiers
    f = open(train_dir + '/classifiers.pkl', 'r')
    s = f.read()
    classifiers = pickle.loads(s)
    print('classifiers: ' + str(classifiers))

    # load the comp file
    comps = []
    files = os.listdir(train_dir)
    files.sort()
    for file in files:
        if file.endswith('s10-sampled.csv') == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        comps.append(data)

    output = np.zeros((4058,2), dtype=np.int)
    idx = 0

    # iterate through the test files
    files = os.listdir(test_dir)
    files.sort()
    for file in files:
        if file.endswith('s10-xsampled.csv') == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(test_dir + '/' + file, delimiter=',')

        best_dev, best_dev_idx = match_subject_to_comp(data, comps)

        print('best match for ' + file + ' is comp ' + str(best_dev_idx) + ' with score ' + str(best_dev))

        # run prediction against that model
        m = re.match("(.*)\.s10-xsampled\.csv", file)
        main_file = m.group(1)
        print('loading and predicting ' + main_file)
        data = np.loadtxt(test_dir + '/' + main_file, delimiter=',')
        ids = data[:,-1]
        z = classifiers[best_dev_idx].predict(data[:,0:-1])
        for i in range(0, len(z)):
            output[idx][0] = ids[i]
            output[idx][1] = z[i]
            idx += 1

    df = pd.DataFrame(output)
    df.columns = ['Id', 'Prediction']
    df.to_csv(test_dir + '/predictions.csv', index=False, header=True)



if __name__=='__main__':

    classifiers = train(in_dir + '/train/out')

    predict(in_dir + '/test/out', in_dir + '/train/out')
    # predict(in_dir + '/train/out', in_dir + '/train/out')

