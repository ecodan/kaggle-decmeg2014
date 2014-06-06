__author__ = 'dan'

'''
Step 1 in pipeline

Input: decmeg matlab files

Output:
- CSV file for each subject with:
 -- m rows = # Trials (test runs)
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
                    id = np.array(fmat['Id'])
                    print('id=' + str(np.shape(id)))
                    data = np.hstack([Xf, id])

                print('data=' + str(np.shape(data)))
                np.savetxt(out_dir + file + '.csv', data, delimiter=',',fmt='%1.3f')


def apply_shrinkage(in_dir, shrinkage):
    print('apply_shrinkage')
    for dir in ['/train','/test']:
        train_dir = in_dir + dir
        out_dir = train_dir + '/out/'
        files = os.listdir(out_dir)
        files.sort()
        file_suffix = '.mat.csv'
        for file in files:
            if file.endswith(file_suffix) == False:
                continue
            print('loading ' + file)
            data = np.loadtxt(out_dir + '/' + file, delimiter=',')
            X = data[:,0:-1]
            y = data[:,-1]

            # shrink
            len_ts = 250
            num_sensors = 306
            clumps = len_ts / shrinkage
            data_s = np.zeros((len(data), (num_sensors*clumps)+1))
            for sh in range(0, num_sensors*clumps):
                data_s[:,sh] = np.mean(X[:,(sh*shrinkage):((sh*shrinkage) + shrinkage)], axis=1)
            data_s[:,-1] = y
            np.savetxt(out_dir + file + '.s' + str(shrinkage) + '.csv', data_s, delimiter=',',fmt='%1.3f')

            if dir == '/train':
                # select 25 1s and 25 0's
                zero_rows = data_s[data_s[:,-1]== 0,:]
                ones_rows = data_s[data_s[:,-1]== 1,:]
                sample = np.vstack((zero_rows[0:25,:],ones_rows[0:25,:]))
                np.savetxt(out_dir + file + '.s' + str(shrinkage) + '-sampled.csv', sample, delimiter=',',fmt='%1.3f')
                xsample = np.vstack((zero_rows[-25:,:],ones_rows[-25:,:]))
                np.savetxt(out_dir + file + '.s' + str(shrinkage) + '-xsampled.csv', xsample, delimiter=',',fmt='%1.3f')
            else:
                # for test subjects, just get first 50 and hope they are fairly random
                np.savetxt(out_dir + file + '.s' + str(shrinkage) + '-sampled.csv', data_s[0:50,:], delimiter=',',fmt='%1.3f')
                np.savetxt(out_dir + file + '.s' + str(shrinkage) + '-xsampled.csv', data_s[-50:,:], delimiter=',',fmt='%1.3f')


# munge(in_dir)
apply_shrinkage(in_dir, 10)
