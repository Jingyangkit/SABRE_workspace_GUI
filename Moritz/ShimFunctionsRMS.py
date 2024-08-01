# -*- coding: utf-8 -*-
"""
Created on Tue 08 February 2022
@author: morit
Measure shim influence as function of RMS(FID) 
"""

import os
import sys
import json
import glob
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from scipy import signal

# Import utils script
MYPATH = 'C:/Users/Magritek/Documents/Moritz/'             
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data/'      
#CAUTION: DATAPATH should include ray_results folder !
sys.path.append(MYPATH+'Utils/')
import utils_Spinsolve

import argparse
parser = argparse.ArgumentParser(description="Run 1shot ensemble")
parser.add_argument("--verbose",type=int,default=0)         # 0 for no output, 1 for minimal output, 2 for max output
input_args = parser.parse_args()


plt.style.use(['science', 'nature', 'high-contrast'])
#plt.rcParams.update({"font.family": "sans-serif",})

#%% Define user and constant variables

class Arguments():
    def __init__(self):
        self.count = 0                      # experiment counter

        self.downsample_factor = 16         # downsample factor
        self.max_data = 1e5                 # max data used in preprocessing
        self.sampling_points = 32768
        self.device = "cpu"                 # Inference device
        
        self.nr_shims = 15                  
        

        self.seed = 45612                   # Seed to set random distortions

my_arg = Arguments()

#%%

def getSNR(array):
    ry = array.real
    sig = max(ry)
    ns = ry[int(3*len(array)/4):-1]
    return sig/np.std(ns)


def criterion_one_peak(spectrum, min_width, max_peak_height, lamda=[1,1]):
    N = spectrum
    # CAUTION with height and distance
    peak_index = signal.find_peaks(N, height = N.max()*0.7, distance=1000)[0]
    [width, height_of_evaluation,_,_] = signal.peak_widths(N, peak_index)
    # normalize criterion such that height and width are proportional to their optimal value and in range [0,1]
    return 1/2*(lamda[0]*(min_width/width.item()) + lamda[1]*N.max()/max_peak_height).item()   # only for 1 peak!

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power) 

# Initialize Python-Prospa interface
com = utils_Spinsolve.init( verbose=(input_args.verbose>0), gui = True )


SPECTRA_MEMORY = []

shimNames = ["xshim", "yshim",  "zshim",
                       "z2shim", "zxshim", "zyshim", "x2y2shim",
                       "xyshim", "z3shim", "z2xshim", "z2yshim",
                       "zx2y2shim", "zxyshim", "x3shim", "y3shim"]

#weighting = [1,1,1,10,10,10,10,10,50,50,50,50,50,50,50] # simplex weighting scaled
#weighting = [1,1,3,35,15,15,12,12,250,200,200,100,100,75,75] # quick rearranged and scaled

# weighting = [1,1,1,20,20,20,20,20,100,100,100,100,100,100,100] # run 1
# weighting = [1,1,2,15,10,10,10,10,70,50,50,50,50,50,50] # run 2
weighting = [1.2,1,2,18,0,0,0,0,0,0,0,0,0,0,0] # run 2. X,Y,Z,Z2 only. Values for range [-50,50] !
#weighting = [1.2,1,2,18,10,10,10,10,70,50,50,50,50,50,50] # run 4. ALL shims. (Used run 2 values for higher order)
weighting = [1.2,1,2,18,8,10,8,8,80,50,60,40,60,40,60] # run 5. modified values

for shim_idx in range(my_arg.nr_shims):
    
    if weighting[shim_idx] == 0: break
    
    ref_rms = []
    for i in range(10): # avg for reference RMS
        xaxis, spectrum, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, np.zeros(my_arg.nr_shims, dtype=np.int), return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
        my_arg.count += 1
        ref_rms.append(utils_Spinsolve.get_RMS(fid))
    ref_rms = np.mean(ref_rms)
    
    # create offset array with higher density around 0
    pos = powspace(start=0, stop=50, power=3, num=20) # num*2 -1 acq.
    neg = pos[::-1]*-1
    offsets = np.append(neg[:-1],pos) # append and cut double 0
    offsets = np.unique((offsets*weighting[shim_idx]).astype(int)) # cut to unique and scale by weighting

    lws = np.zeros([len(offsets)]) # linewidths array
    rmss = np.zeros([len(offsets)]) # rms array
    rmss_percentage = np.zeros([len(offsets)]) # rms percentage array

    for offset_idx, o in enumerate(offsets): 
        
        distortion = np.zeros(my_arg.nr_shims, dtype=np.int)
        distortion[shim_idx] = o
        if input_args.verbose>1: print(distortion)
        
        xaxis, spectrum, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, distortion, return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
        # measure FWHM in Hz
        linewidth = utils_Spinsolve.get_linewidth_Hz(spectrum, bandwidth = 5000)
        rms = utils_Spinsolve.get_RMS(fid)
        
        lws[offset_idx] = linewidth
        rmss[offset_idx] = rms
        rmss_percentage[offset_idx] = rms/ref_rms
        
        my_arg.count += 1
        SPECTRA_MEMORY.append(spectrum)      
        
    first_over = np.argwhere(rmss_percentage>0.5)
    print('Scaling {}: {}'.format(shimNames[shim_idx], offsets[first_over[0]]/50 ))
    
    np.savetxt(DATAPATH + '/ShimFunctions/shim_values_scaled_{}.out'.format(shimNames[shim_idx]), (offsets, lws))
    np.savetxt(DATAPATH + '/ShimFunctions/shim_values_scaled_rmsp_{}.out'.format(shimNames[shim_idx]), (offsets, rmss_percentage))
        
    plt.figure()
    plt.plot(offsets, lws, 'x--', label='shim {}'.format(shimNames[shim_idx]))
    plt.legend()
    plt.ylabel('Linewidth [Hz]')
    plt.xlabel('Shim offset')
    plt.title('Shim value function')
    plt.savefig(DATAPATH + '/ShimFunctions/img3_shimfunction_scaled_lw_{}.png'.format(shimNames[shim_idx]))
    #plt.savefig(DATAPATH + '/ShimFunctions/img_shimfunction_scaled_{}.pdf'.format(shimNames[shim_idx]))
    
    plt.figure()
    plt.plot(offsets, rmss, 'x--', label='shim {}'.format(shimNames[shim_idx]))
    plt.legend()
    plt.ylabel('FID RMS')
    plt.xlabel('Shim offset')
    plt.title('Shim value function')
    plt.savefig(DATAPATH + '/ShimFunctions/img3_shimfunction_scaled_rms_{}.png'.format(shimNames[shim_idx]))
    
    plt.figure()
    plt.plot(offsets, rmss_percentage, 'x--', label='shim {}'.format(shimNames[shim_idx]))
    plt.legend()
    plt.ylabel('(FID RMS)/(Max RMS) ')
    plt.xlabel('Shim offset')
    plt.title('Shim value function')
    plt.savefig(DATAPATH + '/ShimFunctions/img3_shimfunction_scaled_rmsp_{}.png'.format(shimNames[shim_idx]))
    #plt.savefig(DATAPATH + '/ShimFunctions/img2_shimfunction_scaled_rmsp_{}.pdf'.format(shimNames[shim_idx]))

    #if shim_idx==1: break ############################################



with open(DATAPATH + '/ShimFunctions/SPECTRA_MEMORY_shimfunction_scaled.txt'.format(shimNames[shim_idx]), 'w') as f:
    for item in SPECTRA_MEMORY:
        for i in item:
            f.write(str(i) + ' ')
        f.write("\n")

# Shutdown Python-Prospa interface
utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))