__author__ = 'dan'

'''
Step 1 in pipeline

This approach averages the brain data for each time slice accross all sensors

DEC: this didn't work at all - ignore this file

Input: decmeg matlab files

Output:
1) diff'd time series (delta between each two time frames)
2) descretized time series (deltas converted to binary with threshold N)

'''

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from datetime import datetime
from scipy.io import loadmat

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'
num_components = 5000

def munge(in_dir):

    pca = PCA(n_components=num_components)

    # loop through all matlab files in in_dir
    i = 0
    for dir in ['/train','/test']:
        out_dir = in_dir + dir + '/out/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        files = os.listdir(in_dir + dir)
        files.sort()
        for file in files:
            if file.endswith('.mat'):
                i += 1

                # read matlab file
                in_file = in_dir + dir + '/' + file
                print('reading file ' + in_file + ' at ' + str(datetime.now()))

                fmat = loadmat(in_file)

                sfreq = fmat['sfreq']
                tmin = fmat['tmin']
                tmax = fmat['tmax']
                print('f=' + str(sfreq) + ' | min=' + str(tmin) + ' | max=' + str(tmax))

                X = np.array(fmat['X'])
                Xs = np.array(X[:,:,75::])
                Xf = np.zeros((np.shape(Xs)[0], np.shape(Xs)[1]))
                for idx in range(0,np.shape(Xs)[0]):
                    Xf[idx] = np.mean(Xs[idx], axis=1)
                Xf -= Xf.mean(0)
                Xf = np.nan_to_num(Xf / Xf.std(0))
                print('X=' + str(np.shape(X)) + ' | Xs=' + str(np.shape(Xs)) + ' | Xf=' + str(np.shape(Xf)))

                y = []
                data = []

                if dir == '/train':
                    y = np.array(fmat['y'])
                    print('y=' + str(np.shape(y)))
                    data = np.hstack([Xf, y])
                else:
                    data = Xf

                print('data=' + str(np.shape(data)))
                np.savetxt(out_dir + file + '2.csv', data, delimiter=',',fmt='%1.3f')

                if i == 1:
                    # train PCA
                    print('training PCA')
                    pca.fit(Xf)

                print('transforming to PCA')
                Xp = pca.transform(Xf)
                Xp -= Xp.mean(0)
                Xp = np.nan_to_num(Xp / Xp.std(0))
                if dir == '/train':
                    data = np.hstack([Xp, y])
                else:
                    data = Xp
                print('data PCA=' + str(np.shape(data)))
                np.savetxt(out_dir + file + '2.pca.csv', data, delimiter=',',fmt='%1.3f')


munge(in_dir)