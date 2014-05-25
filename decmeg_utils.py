__author__ = 'dan'

import os
import numpy as np

def load_data(dir, file_suffix, num_subjects=100, num_samples=None):
    print('loading data from ' + dir + ' | suff=' + file_suffix + ' | #subj=' + str(num_subjects) + ' | #samp=' + str(num_samples))
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
        if file_ct >= num_subjects:
            break

    return data


# returns 0-based array of rankings in order of sensor number (0 = worst)
def load_ranking(dir):
    # load output from above
    print('loading and calculating rankings...')
    eval = np.loadtxt(dir + '/feature_eval.csv', delimiter=',')

    # create ranked matrix
    reval = np.zeros(eval.shape)
    for idx in range(0,len(eval)):
        temp = eval[idx].argsort()
        ranks = np.empty(len(temp),int)
        reval[idx][temp] = np.arange(len(temp))

    np.savetxt(dir + '/ranks.csv', reval, delimiter=',')
    return reval


# pulls all columns except last (y) column from data matrix
def get_features_all(data):
    return data[:,0:-1]


# pulls subset of columns in non-contiguous segments from data matrix
# this is based on analysis that not all sensors are equal in predictive value
def get_features_subset_01(data, sensors, len_ts=250):
    # create a container to hold n rows of # sensors * len_ts columns
    ret = np.zeros((len(data), len(sensors)*len_ts))
    sensor_array = sensors.tolist()
    for idx, val in enumerate(sensor_array):
        ret[:,(idx*len_ts):(idx*len_ts+len_ts)] = data[:, (val*len_ts):(val*len_ts+len_ts)]
    return ret
