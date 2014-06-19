__author__ = 'dan'

import os
import numpy as np
import sklearn as sl
import decmeg_analyze_05 as dma05
from pylab import *
from sklearn.linear_model import LogisticRegression

'''
These methods explore the similarities and differences between sensors and trials in the training set.

- The similarity between each sensor, grouped by image type (2x2x306)
- The similarity between each trial, groupes by image type (2x2)
'''

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'
file_suffix = '.mat.csv'

len_ts = 250 # length of time series
num_sensors = 306

def explore_sensor_correlation(train_dir):

    print ('explore_sensors...')
    files = os.listdir(train_dir)
    files.sort()
    rows = 0
    fidx = 0
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        y = data[:,-1]
        for s in range(0, num_sensors):
            print('starting sensor ' + str(s))
            start = s * len_ts
            end = start + len_ts
            ds = data[:,start:end]
            corr = np.corrcoef(ds)
            print('corr: ' + str(corr.shape))
            sensor_grid = [[[],[]],[[],[]]] # 2x2 matrix of lists
            for i in range(0,num_sensors):
                for j in range(0, num_sensors):
                    il = int(y[i])
                    jl = int(y[j])
                    sensor_grid[il][jl].append(corr[i,j])
            sensor_res = np.zeros([2,2])
            sensor_res[0,0] = np.mean(sensor_grid[0][0])
            sensor_res[0,1] = np.mean(sensor_grid[0][1])
            sensor_res[1,0] = np.mean(sensor_grid[1][0])
            sensor_res[1,1] = np.mean(sensor_grid[1][1])
            print('sensor ' + str(s) + ':\n' + str(sensor_res))

        break


def find_best_sensors(train_dir):

    print ('explore_sensors...')
    files = os.listdir(train_dir)
    files.sort()
    rows = 0
    fidx = 0

    res = np.zeros((num_sensors, 16))
    res_rank = np.zeros((num_sensors, 16))
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        y = data[:,-1]
        for s in range(0, num_sensors):
            print('starting sensor ' + str(s))
            start = s * len_ts
            end = start + len_ts
            X = data[:,start:end]
            X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y, test_size=100)
            clf = LogisticRegression(C=1,penalty='l2')
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            res[s, fidx] = score
        res_rank[:,fidx] = res[:,fidx].argsort().argsort() # rank the sensors
        fidx += 1

    np.savetxt( train_dir + '/sensor_compare.csv', res, delimiter=',')
    np.savetxt( train_dir + '/sensor_compare_ranks.csv', res_rank, delimiter=',')

    sensors = np.arange(1,307)
    best_ranks = res_rank.min(axis=1)

    print('excludes 100:\n' + str(sensors[best_ranks > 100]))
    print('excludes 75:\n' + str(sensors[best_ranks > 75]))
    print('excludes 50:\n' + str(sensors[best_ranks > 50]))
    print('excludes 25:\n' + str(sensors[best_ranks > 25]))

# see comment below for results
def explore_trials(train_dir, shrinkage=25):

    print ('explore_trials...')
    files = os.listdir(train_dir)
    files.sort()
    rows = 0
    fidx = 0
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        # try making it all positive
        data = np.abs(data)
        y = data[:,-1]
        # shrink
        clumps = len_ts / shrinkage
        data_s = np.zeros((len(data), num_sensors*clumps))
        for sh in range(0, num_sensors*clumps):
            data_s[:,sh] = np.mean(data[:,(sh*shrinkage):((sh*shrinkage) + shrinkage)], axis=1)
        corr = np.corrcoef(data_s[:,0:-1])
        print('corr: ' + str(corr.shape))
        sensor_grid = [[[],[]],[[],[]]] # 2x2 matrix of lists
        for i in range(0,num_sensors):
            for j in range(0, num_sensors):
                il = int(y[i])
                jl = int(y[j])
                sensor_grid[il][jl].append(corr[i,j])
        sensor_res = np.zeros([2,2])
        sensor_res[0,0] = np.mean(sensor_grid[0][0])
        sensor_res[0,1] = np.mean(sensor_grid[0][1])
        sensor_res[1,0] = np.mean(sensor_grid[1][0])
        sensor_res[1,1] = np.mean(sensor_grid[1][1])
        print('file ' + str(file) + ':\n' + str(sensor_res))

        figure()
        boxplot([sensor_grid[1][1],sensor_grid[1][0],sensor_grid[0][1],sensor_grid[0][0]])
        show()

        # break

def explore_trials_accross_subjects(train_dir):

    print ('explore_trials_accross_subjects...')
    files = os.listdir(train_dir)
    files.sort()
    rows = 0
    fidx = 0
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        y = data[:,-1]
        for comp in files:
            if comp.endswith(file_suffix) == False:
                continue
            if comp == file:
                continue
            print('loading comp ' + file)
            comp_data = np.loadtxt(train_dir + '/' + comp, delimiter=',')
            comp_y = comp_data[:,-1]
            sensor_grid = [[[],[]],[[],[]]] # 2x2 matrix of lists
            for i in range(0, len(data)):
                for j in range(0, len(comp_data)):
                    corr = np.corrcoef(data[i,:], comp_data[j,:])
                    il = int(y[i])
                    jl = int(comp_y[j])
                    sensor_grid[il][jl].append(corr[0,1]) # this will be a 2x2 gris and we need the value not on the idenitity axis

            sensor_res = np.zeros([2,2])
            sensor_res[0,0] = np.mean(sensor_grid[0][0])
            sensor_res[0,1] = np.mean(sensor_grid[0][1])
            sensor_res[1,0] = np.mean(sensor_grid[1][0])
            sensor_res[1,1] = np.mean(sensor_grid[1][1])
            print('file ' + str(file) + ' vs. comp ' + str(comp) +':\n' + str(sensor_res))

        break

# how well can subjects be matched to themselves?
def xval_correlation_01(train_dir):
    print ('xval_correlation_01...')

    # load comps
    comps = []
    files = os.listdir(train_dir)
    files.sort()
    for file in files:
        if file.endswith('s10-sampled.csv') == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        comps.append(data)

    # xval with samples
    files = os.listdir(train_dir)
    files.sort()
    file_suffix = 's10.csv'
    for file in files:
        if file.endswith(file_suffix) == False:
            continue
        print('loading ' + file)
        data = np.loadtxt(train_dir + '/' + file, delimiter=',')
        X = data[:,0:-1]
        y = data[:,-1:]
        X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y, test_size=0.2)
        bc, bc_idx = dma05.match_subject_to_comp(np.hstack((X_test,y_test)),comps)
        print('best match for ' + file + ' is comp ' + str(bc_idx) + ' with score ' + str(bc))


# explore_sensor_correlation(in_dir + '/train/out')
# explore_trials(in_dir + '/train/out')
# explore_trials_accross_subjects(in_dir + '/train/out')
# xval_correlation_01(in_dir + '/train/out')
find_best_sensors(in_dir + '/train/out')

'''
result of full subject/trial correlation (no shrinkage):
loading train_subject01.mat.csv
corr: (594, 594)
file train_subject01.mat.csv:
[[ 0.0058443  -0.00363393]
 [-0.00363393  0.00875742]]
loading train_subject02.mat.csv
corr: (586, 586)
file train_subject02.mat.csv:
[[ 0.00729953 -0.00347951]
 [-0.00347951  0.0100766 ]]
loading train_subject03.mat.csv
corr: (578, 578)
file train_subject03.mat.csv:
[[ 0.00640798 -0.00348557]
 [-0.00348557  0.00676228]]
loading train_subject04.mat.csv
corr: (594, 594)
file train_subject04.mat.csv:
[[ 0.00607254 -0.00161016]
 [-0.00161016  0.0065517 ]]
loading train_subject05.mat.csv
corr: (586, 586)
file train_subject05.mat.csv:
[[ 0.00609804 -0.00182342]
 [-0.00182342  0.00768737]]
loading train_subject06.mat.csv
corr: (588, 588)
file train_subject06.mat.csv:
[[ 0.00722001 -0.00229381]
 [-0.00229381  0.00555176]]
loading train_subject07.mat.csv
corr: (588, 588)
file train_subject07.mat.csv:
[[ 0.00812266 -0.00175001]
 [-0.00175001  0.0065889 ]]
loading train_subject08.mat.csv
corr: (592, 592)
file train_subject08.mat.csv:
[[ 0.00905871 -0.00608198]
 [-0.00608198  0.00964307]]
loading train_subject09.mat.csv
corr: (594, 594)
file train_subject09.mat.csv:
[[ 0.00796905 -0.00142001]
 [-0.00142001  0.00653381]]
loading train_subject10.mat.csv
corr: (590, 590)
file train_subject10.mat.csv:
[[ 0.00739908 -0.00289525]
 [-0.00289525  0.00753868]]
loading train_subject11.mat.csv
corr: (592, 592)
file train_subject11.mat.csv:
[[ 0.00662843 -0.00302629]
 [-0.00302629  0.00657087]]
loading train_subject12.mat.csv
corr: (586, 586)
file train_subject12.mat.csv:
[[ 0.00809768 -0.00251515]
 [-0.00251515  0.00583272]]
loading train_subject13.mat.csv
corr: (588, 588)
file train_subject13.mat.csv:
[[ 0.0056017  -0.00240262]
 [-0.00240262  0.00600275]]
loading train_subject14.mat.csv
corr: (588, 588)
file train_subject14.mat.csv:
[[ 0.00562359 -0.00237471]
 [-0.00237471  0.00632174]]
loading train_subject15.mat.csv
corr: (580, 580)
file train_subject15.mat.csv:
[[ 0.00942691 -0.00257616]
 [-0.00257616  0.00826248]]
loading train_subject16.mat.csv
corr: (590, 590)
file train_subject16.mat.csv:
[[ 0.00729127 -0.00356036]
 [-0.00356036  0.00664637]]

here's the first four with shrinkage = 10:
loading train_subject01.mat.csv
corr: (594, 594)
file train_subject01.mat.csv:
[[ 0.00932276 -0.00532143]
 [-0.00532143  0.00958278]]
loading train_subject02.mat.csv
corr: (586, 586)
file train_subject02.mat.csv:
[[ 0.0095259  -0.00447714]
 [-0.00447714  0.00964076]]
loading train_subject03.mat.csv
corr: (578, 578)
file train_subject03.mat.csv:
[[ 0.00928884 -0.00447499]
 [-0.00447499  0.00838645]]
loading train_subject04.mat.csv
corr: (594, 594)
file train_subject04.mat.csv:
[[ 0.00963971 -0.00204904]
 [-0.00204904  0.00986369]]

shrinkage = 25:
loading train_subject01.mat.csv
corr: (594, 594)
file train_subject01.mat.csv:
[[ 0.01040455 -0.00596468]
 [-0.00596468  0.01164763]]
loading train_subject02.mat.csv
corr: (586, 586)
file train_subject02.mat.csv:
[[ 0.01107396 -0.00508452]
 [-0.00508452  0.01153015]]
loading train_subject03.mat.csv
corr: (578, 578)
file train_subject03.mat.csv:
[[ 0.00759692 -0.00276397]
 [-0.00276397  0.0073951 ]]
loading train_subject04.mat.csv
corr: (594, 594)
file train_subject04.mat.csv:
[[  1.23571702e-02  -5.98663224e-05]
 [ -5.98663224e-05   1.19807950e-02]]
'''