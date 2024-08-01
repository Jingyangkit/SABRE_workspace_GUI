# -*- coding: utf-8 -*-
"""
Created on Tue 15 June 08:26:01 2021

@author: morit

Deep Regression with Ensembles
Script to call Simplex with maxiter and step
"""

import os
import sys
import json
import glob
import pickle
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from scipy import signal, optimize
from numpy.polynomial.polynomial import polyfit
MYPATH = 'C:/Users/Magritek/Documents/Moritz/'
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data/'
sys.path.append(MYPATH+'Utils/')
import utils_Spinsolve

plt.style.use(['science', 'grid'])
plt.rcParams.update({"font.family": "sans-serif",})

initial_config = {
        "sample": 'H2OCu',           #H2OCu or ethanol
        "set":'coarse',
        "base_models": 0,  # RAY_RESULTS_COARSE_NAS or [RAY_RESULTS_COARSE_NAS,RAY_RESULTS_COARSE_HPO]
        "nr_models": 50,                        # 10 or 50
        "downsample_factor": 16,
        "label_scaling": 100,
        "max_data": 1e5,
        "filters": 32,
        "meta_type": 'fc',          # fc, linear, average, none or none_tuned
        "drop_p_ensemble": 0.0,
    }     

seed = 45612

max_data = 1e5
downsamplefactor = 16
channels = 4
offset_value = 1000
label_scaling = 100
sampling_points = 32768
averages = 10 # nr of averages for ensemble
ensemble_range = 5 # range for z0-shift augmentation for ensemble
device = "cuda:0" if torch.cuda.is_available() else "cpu"


nr_evaluations = 2
np.random.seed(seed)
random_distortions = np.random.randint(-10000,10000,size=(nr_evaluations,3)) #discrete uniform

import argparse
parser = argparse.ArgumentParser(description="Run Simplex Comparison")
parser.add_argument("--verbose",type=int,default=0) # 0 for no output, 1 for minimal output, 2 for max output
parser.add_argument("--method",type=str,default='parabola') # 'parabola' or 'simplex'
input_args = parser.parse_args()

import warnings
if input_args.verbose == 2: warnings.filterwarnings("ignore")

class Arguments():
    def __init__(self):
        self.count = 0

my_arg = Arguments()

def seed_everything(seed):
    #random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed = 5
seed_everything(seed)

def getSNR(array):   
    ry = array.real
    sig = max(ry)
    ns = ry[int(3*len(array)/4):-1]
    return sig/np.std(ns)

# values: height, width
def criterion_one_peak(spectrum, min_width, max_peak_height, lamda=[1,1]):
    N = spectrum
    # CAUTION with height and distance
    peak_index = signal.find_peaks(N, height = N.max()*0.7, distance=1000)[0]
    [width, height_of_evaluation,_,_] = signal.peak_widths(N, peak_index)
    # normalize criterion such that height and width are proportional to their optimal value and in range [0,1]
    return 1/2*(lamda[0]*(min_width/width.item())
                + lamda[1]*N.max()/max_peak_height).item()   # only for 1 peak! 

                
 
com = utils_Spinsolve.init( verbose=(input_args.verbose>0), gui=(input_args.verbose>1) )

results_array = []
success_rate = 0
sign_rate = 0
mean_c = []
mae = []
mean_steps = []

for d in random_distortions:
    xaxis, initial_spectrum, shimmed_spectrum, shims, info = utils_Spinsolve.setShimsAndStartComparison(com, my_arg.count, d, 'simplex', maxiter=0, return_shimvalues=True, verbose=(input_args.verbose>1))
    linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
    linewidth_shimmed = utils_Spinsolve.get_linewidth_Hz(shimmed_spectrum)
    initial = initial_spectrum[::initial_config["downsample_factor"]]/initial_config["max_data"]
    shimmed = shimmed_spectrum[::initial_config["downsample_factor"]]/initial_config["max_data"]
    min_width = signal.peak_widths(initial, signal.find_peaks(initial, height = initial.max()*0.7, distance=1000)[0])[0].item()
    max_peak_height = initial.max()
    c1 = criterion_one_peak(shimmed, min_width, max_peak_height)

    for key,val in info.items():
        exec(key + '=' + val)

    pred = np.multiply([int(xbefore)-int(xafter), int(ybefore)-int(yafter), int(zbefore)-int(zafter)], -1)
    print(d, pred)


    success_rate += (c1>1)
    sign_rate += (np.sign(d)==np.sign(pred)).sum()
    mean_c.append(c1)
    mae.append(mean_absolute_error(d,pred))
    results_array.append(['dist: {}, pred: {}, lw0.5: {} -> {}'.format(d, pred, linewidth_initial, linewidth_shimmed)])
    mean_steps.append(stepcounter)

    print('Nr. function eval.', stepcounter)
    print(linewidth_initial)
    print(linewidth_shimmed)
    print(c1)

print("Success rate: ", success_rate/len(random_distortions))
print("Correct prediction rate: ", round(sign_rate/(len(random_distortions)*3),2) )
print("Mean criterion improvement: {} {} % +/- {}".format( ('+' if np.mean(mean_c)>1 else '-'), round((np.mean(mean_c)-1)*100, 2), round(np.abs((np.std(mean_c)-1)*100),2)) )
print("Averaged MAE: {} +/- {}".format(round(np.mean(mae),1), round(np.std(mae),1)) )
print("Mean steps: {} +/- {}".format(round(np.mean(mean_steps),1), round(np.std(mean_steps),1)) )

utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0)) 