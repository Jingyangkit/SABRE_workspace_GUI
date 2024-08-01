# -*- coding: utf-8 -*-
"""
Created on Tue 29 Mar 08:26:01 2022

@author: morit

live Deep Regression with random shim values and LSTM

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
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data'
sys.path.append(MYPATH)
import utils_Spinsolve

import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
parser = argparse.ArgumentParser(description="Run enhanced deep regression")
parser.add_argument("--verbose",type=int,default=2) # 0 for no output, 1 for minimal output, 2 for max output
input_args = parser.parse_args()

#plt.style.use(['science', 'nature', 'high-contrast'])
##plt.rcParams.update({"font.family": "sans-serif",})

import warnings
if input_args.verbose == 2: warnings.filterwarnings("ignore")

#https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

initial_config = {
        'set': 'random',
        'id': 'Real2_64',
        'compressionModel': 'VAEsmooth',                  # model's id
        'compressionExperiment':'TuneReal2_64',
        "sample": 'Ref',     # Ref (=5%), Ref10 (10%), H2OCu (+CuSO4), MeOH, H2OTSP (=6mg), H2OTSP50 (50mg,1ml), 
                                    #H2OTSP100 (100mg,1ml) (if TSP100: use TSP peak), H2ONic400 (1ml D2O with 400mg Nicotinamid)
        'tta': False,                # test-time augmentation
        'postshim': True,           # shim after experiment to guarantee same starting points
        'full_memory': False,         # store whole spectrum in additional memory
        "downsample_factor": 1,
        'ROI_min': 16000,           # calculated via mean_idx(signal(2048p)>2*noise) * downsamplefactor
        "max_data": 1e5,            # *0.68312625, # max_data * max_dataset_value (scaling)
        'scale_sample': True, # scale first sample in sequence to one (instead of global max)
        'shim_weightings': [1.2,1,2,18,0,0,0,0,0,0,0,0,0,0,0],  # Shim weighting. range * weighting = shim values
        'acq_range': 50,            # range of absolute shim offsets for highest-impact shim (weighting=1)
        "drop_p_fc": 0.0,
        "drop_p_conv": 0.0,
        "DRE_gridlike": False,
    }

# VARIABLES
INVERT_PRED = False
STEPS_RAND = 10


with open(DATAPATH + '/CeDR/models/RNN/config_DR_Z2_{}_id{}.json'.format(initial_config['set'],initial_config["id"])) as f:
    config_RNN = json.load(f)

config_RNN = merge_two_dicts(initial_config, config_RNN)   #merge configs

with open(DATAPATH + '/CeDR/models/Encoder/_{}_{}.json'.format(initial_config['compressionModel'], initial_config['compressionExperiment'])) as f:
    config_encoder = json.load(f)



# =============================================================================
# # overwrite 
# =============================================================================

if 'H2O' in config_RNN['sample'] or 'Ref' in config_RNN['sample']:    
    config_RNN['ROI_min'] = 16000
    ROLL = 0
    PLT_X = [80,120]
    if 'Ref' in config_RNN['sample'] or config_RNN['sample'] == 'H2O100': pass
    initial_config['phasecorrect'] = False # shut off for singlet reference peak 


sampling_points = 32768
#pred_averages = 0 # nr of pred_averages for ensemble
#pred_shift_range = 5 # range for z0-shift augmentation for ensemble
device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 45612
nr_evaluations = 1
np.random.seed(seed)
gauss_noise = np.random.normal(0, 1/3,size=(nr_evaluations,config_RNN['nr_shims']))
#random_distortions = (np.random.randint(-config['acq_range'],config['acq_range'],size=(nr_evaluations,config['nr_shims']))*config['shim_weightings'][:config['nr_shims']]).astype(int) #discrete uniform
random_distortions = (config_RNN['acq_range']*gauss_noise*config_RNN['shim_weightings'][:config_RNN['nr_shims']]).astype(int) #discrete uniform

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

#%% DL part
from models import ConvLSTM, ConvTransformer, LSTMcompressedMeanOnly, VAESmooth

def get_single_model_encoder(filename):

    if config_encoder['model'] == 'VAESmooth':
        model = VAESmooth(config_encoder['past_observations'], config_encoder['latent_dim'], config_encoder['hidden_dims'],
                          config_encoder['kernel_size'], config_encoder['loss'], config_encoder['gaussian_kernel_size'])
    model_state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

#%% Functions

def save(spectrum_tmp, prediction_scaled):
    if INVERT_PRED: pass
    # TODO change success definition
    success = (spectrum_tmp/sc_factor).max() > batched_spectra[0,:-config['nr_shims']].max()
    sign_d = np.sign(distortion)
    sign_p = np.sign(-prediction_scaled)
    sign_d[np.where(sign_d==0)[0]] = 1 # count 0 as + sign
    sign_p[np.where(sign_p==0)[0]] = 1
    sign = (sign_d==sign_p).sum()/len(distortion)
    mae = mean_absolute_error(distortion/config_RNN['shim_weightings'][:config_RNN['nr_shims']]/config_RNN['acq_range'],
                       -prediction_scaled/config_RNN['shim_weightings'][:config_RNN['nr_shims']]/config_RNN['acq_range'])
    lw_init_50 = utils_Spinsolve.get_linewidth_Hz(batched_spectra[0,:-config_RNN['nr_shims']], sampling_points=32768, bandwidth = 5000)
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000)    
    lw_init_055 = utils_Spinsolve.get_linewidth_Hz(batched_spectra[0,:-config_RNN['nr_shims']], sampling_points=32768, bandwidth = 5000, height=0.9945)
    lw_shimmed_055 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000, height=0.9945)
  
    results_data.append([nr, s, int(INVERT_PRED), int(success), sign, int(RANDOM), distortion, prediction_scaled, 
                         lw_init_50.item(), lw_shimmed_50.item(), lw_init_055.item(), lw_shimmed_055.item(), mae, sc_factor])



def random_step(iteration, distortion, config):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    #offset_rand = np.random.normal(0,1/3, size=config['nr_shims'])
    offset_rand = np.random.uniform(-.5,.5, size=config['nr_shims'])
    offset_rand_scaled = (offset_rand*config['acq_range']*config['shim_weightings'][:config['nr_shims']]).astype(int)
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                         np.add(offset_rand_scaled,distortion), True,True, verbose=(input_args.verbose>1))

    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]
    norm_spectrum = np.linalg.norm(spectrum_tmp, axis = -1, keepdims = True)
    spectrum_tmp = np.divide(spectrum_tmp, norm_spectrum)
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],spectrum_tmp,'--', label='random{}'.format(iteration), alpha=0.4, color = colors[iteration-1])

    inputs = torch.tensor(spectrum_tmp).unsqueeze(0).float()
    outputs = model_encoder(inputs.view(-1, 1, 2048))
    prediction = outputs[0]
    prediction = prediction.detach().numpy().squeeze()
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],prediction,'-', label='reconstruction{}'.format(iteration), alpha=1, color = colors[iteration-1])

#%%       
com = utils_Spinsolve.init( verbose=(input_args.verbose>0))

# for DRE comparison: 4+1 & DRE_gridlike !

# TODO 
# Fix STEPS for REAL and RAND

# loop over rand steps to measure influence
# "TAB" code if used
#for STEPS_RAND in range(6,11):

global spectra_memory
spectra_memory = []
global full_memory # big memory for whole fid (not only ROI)
full_memory = [] 
global results_data
results_data = []

# loop over all random distortions and track performance
for nr, distortion in enumerate(random_distortions):

    model_encoder = get_single_model_encoder(DATAPATH + '/CeDR/models/Encoder/model_{}_{}.pt'.format(initial_config['compressionModel'], initial_config['compressionExperiment']))
    
    plt.figure(figsize = (15, 15))
    # random steps
    for s in range(1, STEPS_RAND+1):
        RANDOM = True
        if initial_config['DRE_gridlike']:  
            batched_spectra = grid_step(s, distortion, batched_spectra, config)
        else:
            random_step(s, distortion, config_RNN)

    plt.xlim(PLT_X) # crop for plotting
    plt.legend()
    plt.title('Shimming with eDR')
    plt.ylabel("Signal [a.u.]")
    plt.xlabel("Frequency [Hz]")
    plt.savefig(DATAPATH + '/CeDR/plots/img_RECONSTRUCTION_{}_id{}.png'.format(
        initial_config["compressionModel"],initial_config["compressionExperiment"]))
    plt.show()

utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))
