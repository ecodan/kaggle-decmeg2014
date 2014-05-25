__author__ = 'dan'

'''
Step 1 in pipeline

Input: decmeg matlab files

Output:
- CSV file for each subject with:
 -- m rows = # Samples (test runs)
 -- n cols = # Sensors x # Time Series datapoints
note: the time series datapoints before the application of stimulus have been removed

'''

import os
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
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

                # calculate range
                start = int(abs(tmin)*sfreq)
                stop = start + int((tmax * sfreq))

                # X = 3D raw feature data (n_Samples x n_Sensors x n_TimePoints)
                X = np.array(fmat['X'])
                # Xs = 3D raw feature data from stimulus forward (n_Samples x n_Sensors x n_TimePoints)
                Xs = np.array(X[:,:,start:stop])
                # Xf = 2D scaled feature data from stimulus forward (n_Samples x (n_Sensors x n_TimePoints))
                Xf = np.reshape(Xs, (np.shape(Xs)[0], np.shape(Xs)[1] * np.shape(Xs)[2]))
                # scale/normalize the data
                Xf -= Xf.mean(0)
                Xf = np.nan_to_num(Xf / Xf.std(0))
                print('X=' + str(np.shape(X)) + ' | Xs=' + str(np.shape(Xs)) + ' | Xf=' + str(np.shape(Xf)))

                y = []
                data = []

                # append labels for train data
                if dir == '/train':
                    y = np.array(fmat['y'])
                    print('y=' + str(np.shape(y)))
                    data = np.hstack([Xf, y])
                else:
                    data = Xf

                print('data=' + str(np.shape(data)))
                np.savetxt(out_dir + file + '.csv', data, delimiter=',',fmt='%1.3f')

                # dont think PCA will work due to feture shift between subjects, so disabling
                # if i == 1:
                #     # train PCA
                #     print('training PCA')
                #     pca.fit(Xf)
                #
                # print('transforming to PCA')
                # Xp = pca.transform(Xf)
                # Xp -= Xp.mean(0)
                # Xp = np.nan_to_num(Xp / Xp.std(0))
                # if dir == '/train':
                #     data = np.hstack([Xp, y])
                # else:
                #     data = Xp
                # print('data PCA=' + str(np.shape(data)))
                # np.savetxt(out_dir + file + '.pca.csv', data, delimiter=',',fmt='%1.3f')


munge(in_dir)