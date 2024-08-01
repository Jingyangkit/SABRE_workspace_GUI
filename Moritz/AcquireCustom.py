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

com = utils_Spinsolve.init(gui=True)

distortions = np.array([    [0,0,0,0],
                            [-20,0,0,0],
                            [0,-20,0,0],
                            [0,0,-20,0],
                            [0,0,0,-50],
                            [20,20,0,0],
                            [20,0,20,0],
                            [20,0,0,20],
                            [0,20,20,0],
                            [0,0,20,20], ]) 

# =============================================================================
# range_ = 40
# nr_exps = (2*range_+1)**2
# distortions = np.zeros([nr_exps+1,3])
# counter = 0
# for i in range(-range_,range_+1):
#     for j in range(-range_,range_+1):
#         distortions[counter,0] = i
#         distortions[counter,1] = j
#         counter += 1
# =============================================================================
        
# =============================================================================
# range_ = 10
# nr_exps = (2*range_+1)**3
# distortions = np.zeros([nr_exps+1,3])
# counter = 0
# for i in range(-range_,range_+1):
#     for j in range(-range_,range_+1):
#         for k in range(-range_,range_+1):
#             distortions[counter,0] = i
#             distortions[counter,1] = j
#             distortions[counter,2] = k*2    # !!!! manual weight
#             counter += 1
# =============================================================================

for i in range(len(distortions)):

    distortion = distortions[i]
    xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, i+1, distortion,
                                                return_shimvalues=True, return_fid=True)
    
    time.sleep(2)
                                                
utils_Spinsolve.shutdown(com)                                                
