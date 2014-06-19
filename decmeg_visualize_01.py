__author__ = 'dan'

import os
import numpy as np
import sklearn as sl
import decmeg_analyze_05 as dma05
from pylab import *
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

'''
Bunch of utility methods to visualize elements of the data.

'''

in_dir = '/Users/dan/dev/datasci/kaggle/decmeg2014/'
file_suffix = '.mat.csv'

len_ts = 250 # length of time series
num_sensors = 306


def vis_best_worst_sensors(train_dir):

    print ('vis_best_worst_sensors...')

    for subj_num in range(1,17):
        subject = subj_num
        tmin = 0.0 # in sec.
        tmax = 0.5 # in sec.
        cv = 5 # numbers of fold of cross-validation
        filename = train_dir + '/train_subject%02d.mat' % subject
        scores_file = train_dir + '/channel_scores_subject%02d.csv' % subject
        layout_filename = '/Users/dan/dev/datasci/dspython/refcode/DecMeg2014/additional_files/Vectorview-all.lout'

        print "Loading %s" % filename
        data = loadmat(filename, squeeze_me=True)
        X = data['X']
        y = data['y']
        sfreq = data['sfreq']

        print "Applying the desired time window: [%s, %s] sec." % (tmin, tmax)
        time = np.linspace(-0.5, 1.0, 375)
        time_window = np.logical_and(time >= tmin, time <= tmax)
        X = X[:,:,time_window]
        time = time[time_window]

        # downsample
        shrinkage = 5
        len_ts = len(time)
        num_sensors = 306
        clumps = len_ts / shrinkage
        data_s = np.zeros((len(X), num_sensors, clumps))
        for sens in range(0, num_sensors):
            for sh in range(0, clumps):
                data_s[:,sens,sh] = np.mean(X[:,sens,(sh*shrinkage):((sh*shrinkage) + shrinkage)], axis=1)
        X = data_s
        time = np.linspace(0.0, 0.5, 25)

        print "Loading channels name."
        channel_name = np.loadtxt(layout_filename, skiprows=1, usecols=(5,), delimiter='\t', dtype='S')

        print "Computing cross-validated accuracy for each channel."
        clf = LogisticRegression(random_state=0)
        score_channel = np.zeros(X.shape[1])

        if os.path.isfile(scores_file):
            score_channel = np.loadtxt(scores_file, delimiter=',')
        else:
            for channel in range(X.shape[1]):
                print "Channel %d (%s) :" % (channel, channel_name[channel]),
                X_channel = X[:,channel,:].copy()
                X_channel -= X_channel.mean(0)
                X_channel = np.nan_to_num(X_channel / X_channel.std(0))
                scores = cross_val_score(clf, X_channel, y, cv=cv, scoring='accuracy')
                score_channel[channel] = scores.mean()
                print score_channel[channel]

            np.savetxt(scores_file, score_channel, delimiter=',')

        print
        plt.interactive(True)

        print
        print "Channels with the highest accuracy:",
        n_best = 1
        best_channels = np.argsort(score_channel)[-n_best:][::-1]
        print best_channels

        print "Plotting the average signal of each class."
        X_best_face = X[:,best_channels,:][y==1] * 1.0e15
        X_best_scramble = X[:,best_channels,:][y==0] * 1.0e15
        plt.figure()
        num_trails = 5
        for i, channel in enumerate(best_channels):
            plt.subplot(n_best*2,1,i+1)
            plt.yticks(fontsize='small')
            plt.xticks(fontsize='small')
            for trial in range(0, num_trails):
                plt.plot(time, X_best_face[trial][i], 'r-')

            plt.subplot(n_best*2,1,i+2)
            plt.yticks(fontsize='small')
            plt.xticks(fontsize='small')
            for trial in range(0, num_trails):
                plt.plot(time, X_best_scramble[trial][i], 'b-')

            plt.axis('tight')
            tmp = min(X_best_face[:][i].min(), X_best_scramble[:][i].min())
            text_y = (max(X_best_face[:][i].max(), X_best_scramble[:][i].max()) - tmp)*0.9 + tmp
            plt.text(0.6, text_y, str(i+1)+') '+str(channel_name[channel])+' = '+("%0.2f" % score_channel[channel]), bbox=dict(facecolor='white', alpha=1.0), fontsize='small')
            if i == (len(best_channels) - 1):
                plt.xlabel('Time (sec)', fontsize='small')

            if i == (len(best_channels) / 2):
                plt.ylabel('Magnetic Field (fT)', fontsize='small')

        plt.savefig(train_dir + '/subject_%02d_best_N_channels_signals.png' % subject)



vis_best_worst_sensors(in_dir + '/train')
