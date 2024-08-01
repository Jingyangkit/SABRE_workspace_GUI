# -*- coding: utf-8 -*-
"""
Created on Tue 15 June 08:26:01 2021

@author: morit

live Deep Regression with Ensembles

run over 100 random distortions
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
from datetime import datetime
#from torch import nn
#import torch.nn.functional as F
#from sklearn import svm
#from sklearn.metrics import mean_absolute_error
from scipy import signal, optimize
#from numpy.polynomial.polynomial import polyfit
MYPATH = 'C:/Users/Magritek/Documents/Moritz/'
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data/'
sys.path.append(MYPATH+'Utils/')
import utils_Spinsolve

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
parser = argparse.ArgumentParser(description="Run 1shot ensemble")
parser.add_argument("--verbose",type=int,default=0) # 0 for no output, 1 for minimal output, 2 for max output
parser.add_argument("--meta",type=str,default='fc')
input_args = parser.parse_args()


plt.style.use(['science', 'nature', 'high-contrast'])
plt.rcParams.update({"font.family": "sans-serif",})

initial_config = {
        "sample": 'H2OCu',           #H2OCu, EtOH-0.1, EtOH-0.5, gluc-0.01, toluene, acetoneH20, acetoneCHCl, CHCl-1, isoprop
        "set":'coarse',
        "downsample_factor": 16,
        "label_scaling": 100,
        "max_data": 1e5,#*0.68312625, # max_data * max_dataset_value ###################################################################
        "sweep": int(32768/20000*3000)        # 1000 for H20, 3000 for EtOH & isoprop
    }     

channels = 4
offset_value = 1000
sampling_points = 32768
pred_averages = 0 # nr of pred_averages for ensemble
pred_shift_range = 5 # range for z0-shift augmentation for ensemble
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import warnings
if input_args.verbose == 2: warnings.filterwarnings("ignore")

class Arguments():
    def __init__(self):
        self.count = 0

my_arg = Arguments()

    
com = utils_Spinsolve.init( verbose=(input_args.verbose>0) )

xaxis, initial_spectrum, ref_shims = utils_Spinsolve.setShimsAndRun(com, my_arg.count, [0,0,0], return_shimvalues=True, verbose=(input_args.verbose>1))
linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
ref_shims = ref_shims[:3]
my_arg.count += 1

np.savetxt(DATAPATH+'/DRE/spectrum_dre_{}_best.txt'.format(initial_config["sample"]), initial_spectrum)

tmp = initial_spectrum
global min_i
global max_i 
ns = tmp[int(3*len(tmp)/4):-1]
min_i, max_i = np.where(tmp>(ns.mean()+0.5*ns.mean()))[0][0], np.where(tmp>(ns.mean()+0.5*ns.mean()))[0][-1]

#min_i = min_i - 1500
#max_i = 32767
#print(min_i, max_i)

w,h,start,stop = signal.peak_widths(tmp, signal.find_peaks(tmp, height = tmp.max()*0.9, distance=1000)[0])
start, stop = (start-2**15/2)/2**15*2e4, (stop-2**15/2)/2**15*2e4
plt.hlines(h,start,stop, colors="grey", linestyles='dotted')
plt.annotate('\scriptsize{}Hz'.format(int(linewidth_initial.item())), [stop+w/2/2**15*2e4,h+50], va='center')
plt.ticklabel_format(axis='y', scilimits=(0,0))
plt.plot(xaxis[min_i:max_i], initial_spectrum[min_i:max_i])
plt.xlim(xaxis[min_i], xaxis[min_i+initial_config["sweep"]])
plt.legend()
#plt.title('Optimal spectrum without distortions')
plt.ylabel("Signal [a.u.]")
plt.xlabel("Frequency [Hz]")
plt.savefig(DATAPATH + '/DRE/img_dre_{}_best_v2.pdf'.format(initial_config["sample"]))
plt.savefig(DATAPATH + '/DRE/img_dre_{}_best_v2.png'.format(initial_config["sample"]))

utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))

with open(DATAPATH + '/DRE/spectra_memory_dre_best_{}_{}_{}.txt'.format(initial_config["sample"],input_args.meta, datetime.now().timestamp()), 'w') as f:
    for i in initial_spectrum: 
        f.write(str(i) + ' ')
    f.write("\n")