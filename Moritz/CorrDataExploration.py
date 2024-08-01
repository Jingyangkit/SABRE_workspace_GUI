# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:16:39 2023

@author: Pierre Labouré
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

import os
import sys
import json
import glob
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#from sklearn import svm
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import re
import seaborn as sns

import scipy
from scipy import stats

directoryStr = str(Path(__file__).parents[0])
directoryStr += '/data/CeDR/dataStats/preLoaded'
print(directoryStr)
directory = os.fsencode(directoryStr)



def get_norm(D1, D2):
    N1 = np.linalg.norm(D1, axis = -1)
    N2 = np.linalg.norm(D2, axis = -1)
    return N1, N2

def shape_match(s1, s2, n1, n2):
    corr = signal.correlate(s1, s2, 'same')
    corr_norm = corr/(n1*n2)
    
    shift = 1024 - np.argmax(corr_norm)
    mismatch = np.max(corr_norm)
    scale = n2/n1
    '''
    fig, axs = plt.subplots(1, 2, figsize = (15, 15))
    axs[0].plot(corr_norm)
    axs[1].plot(s1)
    axs[1].plot(s2)
    
    plt.show()
    print(shift, mismatch, scale)
    '''
    return shift, mismatch, scale
    

count = 1
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    
    if filename.find('MONITOR') != -1:
        
        with open(directoryStr + '/' + filename, "rb") as input_file:
            [data, labels] = pickle.load(input_file)
        if count == 1:
            data1 = data
        else:
            data2 = data
        count +=1


N1, N2 = get_norm(data1, data2)

idx = 296
shape_match(data1[idx], data2[idx], N1[idx], N2[idx])
'''
N = 1000
Shifts = np.zeros(N)
Mismatchs = np.zeros(N)
Scales = np.zeros(N)

for idx in range(N):
    shift, mismatch, scale = shape_match(data1[idx], data2[idx], N1[idx], N2[idx])
    
    Shifts[idx] = shift
    Mismatchs[idx] = mismatch
    Scales[idx] = scale


fig, axs = plt.subplots(1, 3, figsize = (15, 15))
axs[0].plot(Shifts)
axs[0].set_title('Shifts')
axs[1].plot(Mismatchs)
axs[1].set_title('Mismatchs')
axs[2].plot(Scales)
axs[2].set_title('Scale ratio')
plt.show()

'''

with open(directoryStr + '/220616-084959_dataset_XYZZ2_1k.pickle', "rb") as input_file:
            [data1, labels1] = pickle.load(input_file)
with open(directoryStr + '/230525-152857_dataset_XYZZ2_Ref_range50_MONITOR.pickle', "rb") as input_file:
            [data2, labels2] = pickle.load(input_file)
with open(directoryStr + '/230525-192751_dataset_XYZZ2_Ref_range50_MONITOR.pickle', "rb") as input_file:
            [data3, labels3] = pickle.load(input_file)

N1, N2 = get_norm(data1, data2)
N1, N3 = get_norm(data1, data3)

idx = 0
shape_match(data1[idx], data2[idx], N1[idx], N2[idx])

N = 100
Shifts12 = np.zeros(N)
Shifts13 = np.zeros(N)
Shifts23 = np.zeros(N)

for idx in range(N):
    shift, mismatch, scale = shape_match(data1[idx], data2[idx], N1[idx], N2[idx])
    Shifts12[idx] = shift

    shift, mismatch, scale = shape_match(data1[idx], data3[idx], N1[idx], N3[idx])
    Shifts13[idx] = shift
 
    shift, mismatch, scale = shape_match(data2[idx], data3[idx], N2[idx], N3[idx])
    Shifts23[idx] = shift
    
fig, axs = plt.subplots(1, 3, figsize = (15, 15))
axs[0].plot(Shifts12)
axs[0].set_title('Shifts 1-2')
axs[1].plot(Shifts13)
axs[1].set_title('Shifts 1-3')
axs[2].plot(Shifts23)
axs[2].set_title('shifts 2-3')
fig.suptitle('1:1K; 2:MONITOR15; 3:MONITOR19')
plt.show()











