import os
import sys
import json
import glob
import pickle
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nmrglue as ng
from datetime import datetime
from torch import nn
import torch.nn.functional as F
#from sklearn import svm
from sklearn.metrics import mean_absolute_error
from scipy import signal, optimize
from numpy.polynomial.polynomial import polyfit
MYPATH = 'C:/Users/Magritek/Documents/Moritz/'
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data/'
sys.path.append(MYPATH)
import utils_Spinsolve

import os

class Arguments():
    def __init__(self):
        self.count = 0

my_arg = Arguments()

com = utils_Spinsolve.init(gui=True)

ID = 19
SAMPLES = [82, 88, 25] 
distortions = np.array([    [-9,-35,-3,256], [10,33,-4,-131], [-15,1,-59,49] ]) 
predictions = np.array([    [13,33,-7,-185], [-10,-32,0,142], [ 13,-7,39,-44]])

global spectra_memory
spectra_memory = []
global results_data
results_data = []

INVERT_PRED = False
RANDOM = -1
sc_factor = 1 

for i in range(len(distortions)):
    
    nr = SAMPLES[i]
    
    distortion = distortions[i]
    xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, i+1, distortion,
                                                return_shimvalues=True, return_fid=True) 
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(initial_spectrum[16000:16000+2048]/1e5,sampling_points=32768, bandwidth = 5000)
    spectra_memory.append([nr, -1, 'initial', lw_shimmed_50, fid])
    batched_spectra = np.concatenate((initial_spectrum/sc_factor, np.zeros(4)))[np.newaxis,:]
    
    METHOD = 'parabola'
    RANDOM = -1  
    distortion = np.add( distortions[i], predictions[i] )
    xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, i+1, distortion,
                                                return_shimvalues=True, return_fid=True)  
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(initial_spectrum[16000:16000+2048]/1e5,sampling_points=32768, bandwidth = 5000)
    spectra_memory.append([nr, 9, 'ours', lw_shimmed_50, fid])
    print('We took 9 iterations for lw {}'.format(lw_shimmed_50))
    # start parabola run
    xaxis, _, _, info = utils_Spinsolve.setShimsAndStartComparisonV3(com, my_arg.count, distortion, method=METHOD)
    for key,val in info.items():
        exec(key + '=' + val)
    pred_sim = np.multiply([int(xbefore)-int(xafter), int(ybefore)-int(yafter), int(zbefore)-int(zafter),int(z2before)-int(z2after)], -1)
    print(distortion, pred_sim)
    #print('Steps to LW ', stepsToLW)
    s = stepcounter
    #s = stepsToLW  
    # apply 
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                            np.add(-pred_sim,distortion), True,True, verbose=2)
    my_arg.count += 1
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp[16000:16000+2048]/1e5,sampling_points=32768, bandwidth = 5000)
    spectra_memory.append([nr, s, 'ours+parabola', lw_shimmed_50, fid])                                                                        
    #save
    print('Parabola took {} iterations for lw {}'.format(len(lw50arr), lw_shimmed_50))
        
    
    ###############################
    distortion = distortions[i]
    METHOD = 'simplex'
    RANDOM = -1
    xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, i+1, distortion,
                                                return_shimvalues=True, return_fid=True) 
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(initial_spectrum[16000:16000+2048]/1e5,sampling_points=32768, bandwidth = 5000)
    spectra_memory.append([nr, -1, 'initial2', lw_shimmed_50, fid])
    # start simplex run
    xaxis, _, _, info = utils_Spinsolve.setShimsAndStartComparisonV3(com, my_arg.count, distortion, 
                                                                     method=METHOD, maxiter=150)
    for key,val in info.items():
        exec(key + '=' + val)
    pred_sim = np.multiply([int(xbefore)-int(xafter), int(ybefore)-int(yafter), int(zbefore)-int(zafter),int(z2before)-int(z2after)], -1)
    print(distortion, pred_sim)
    #print('Steps to LW ', stepsToLW)
    s = stepcounter
    #s = stepsToLW
    # apply 
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                            np.add(-pred_sim,distortion), True,True, verbose=2)
    my_arg.count += 1
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp[16000:16000+2048]/1e5,sampling_points=32768, bandwidth = 5000)
    spectra_memory.append([nr, s, 'simplex', lw_shimmed_50, fid])
    print('Simplex took {} iterations for lw {}'.format(len(lw50arr), lw_shimmed_50))
    
columns_data = ['Nr', 'Steps', 'Method', 'lw50', 'FID']
df_spectra = pd.DataFrame(spectra_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/DRR/custom_comparison_id{}.pickle'.format(
    ID,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

                                                
utils_Spinsolve.shutdown(com)                                                
