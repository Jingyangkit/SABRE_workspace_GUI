# -*- coding: utf-8 -*-
"""
Created on Tue 15 June 08:26:01 2021

@author: morit

Deep Regression with Ensembles
Comparison of Simplex for i iterations to reach DRE's criterion --> Compare steps / function evaluations

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

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import argparse
parser = argparse.ArgumentParser(description="Run 1shot ensemble")
parser.add_argument("--verbose",type=int,default=0) # 0 for no output, 1 for minimal output, 2 for max output
parser.add_argument("--meta",type=str,default='fc') # none or fc
input_args = parser.parse_args()

plt.style.use(['science', 'grid'])
plt.rcParams.update({"font.family": "sans-serif",})

ENSEMBLE_COARSE = '/ensemble_multiregression_coarse_trainLarge_v2'

#RAY_RESULTS_COARSE_NAS = '/raytune_results_coarse_2021-07-29-v2.pickle'
#RAY_RESULTS_COARSE_HPO = '/raytune_results_coarse_2021-08-25_Bay.pickle'

RAY_RESULTS_COARSE_NAS = '/raytune_results_coarse_2021-12-17.pickle' ##############################################################################################

initial_config = {
        "sample": 'H2OCu',           #H2OCu or ethanol
        "set":'coarse',
        "base_models": RAY_RESULTS_COARSE_NAS,  # RAY_RESULTS_COARSE_NAS or [RAY_RESULTS_COARSE_NAS,RAY_RESULTS_COARSE_HPO]
        "nr_models": 50,                        # 10 or 50
        "downsample_factor": 16,
        "label_scaling": 100,
        "max_data": 1e5,#*0.68312625, # max_data * max_dataset_value      #######################################################################
        "filters": 32,
        "meta_type": input_args.meta,          # fc, linear, average, none or none_tuned
        "drop_p_ensemble": 0.0,
    }     


seed = 45612

channels = 4
offset_value = 1000
sampling_points = 32768
pred_averages = 0 # nr of pred_averages for ensemble
pred_shift_range = 5 # range for z0-shift augmentation for ensemble
device = "cuda:0" if torch.cuda.is_available() else "cpu"

nr_evaluations = 100
np.random.seed(seed)
random_distortions = np.random.randint(-10000,10000,size=(nr_evaluations,3)) #discrete uniform

#random_distortions = [np.array([0,0,0])]    # artificial distortion
#distortion = np.array([-2538, 2342, -921])
#distortion = np.array([-700,-1300,400])    # artificial distortion

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

# values: height, width, snr
def criterion_one_peak(spectrum, min_width, max_peak_height, lamda=[1,1]):
    N = spectrum
    # CAUTION with height and distance
    peak_index = signal.find_peaks(N, height = N.max()*0.9, distance=1000)[0]
    [width, height_of_evaluation,_,_] = signal.peak_widths(N, peak_index)
    # normalize criterion such that height and width are proportional to their optimal value and in range [0,1]
    return 1/2*(lamda[0]*(min_width/width.item())
                + lamda[1]*N.max()/max_peak_height).item()   # only for 1 peak! 
                
class MyCNNflex_Regr(nn.Module):
    def __init__(self,  input_shape, num_classes=3, drop_p_conv=.2, drop_p_fc=.5, kernel_size=49, stride=2, pool_size=1, filters=32, num_layers=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_size = pool_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.filters = filters     
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p) )
            return conv           
        layers = []     
        layers.append( one_conv(input_shape[1], self.filters, self.kernel_size, self.stride, self.drop_p_conv) )
        self.outshape = int( (input_shape[2]-self.kernel_size)/self.stride +1 )
        for i in range(num_layers-1):
            block = one_conv(self.filters, self.filters, self.kernel_size, self.stride, self.drop_p_conv)
            layers.append(block)
            self.outshape = int( (self.outshape-self.kernel_size)/self.stride +1)
            if self.pool_size > 1:
                layers.append( nn.MaxPool1d(2,stride=self.pool_size) )
                self.outshape = int(self.outshape/self.pool_size)
        self.features = nn.Sequential(*layers)
        fc = []
        fc.append( nn.Dropout(self.drop_p_fc) )
        fc.append( nn.Linear(self.outshape*self.filters, self.filters) ) 
        fc.append( nn.Linear(self.filters, num_classes) )
        self.fc_block = nn.Sequential(*fc)            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.fc_block(x)
    
class MyEnsemble(nn.Module):
    def __init__(self, model_list, outshape=3):
        super(MyEnsemble, self).__init__()
        self.models = model_list
        # Remove last linear layer
        for idx,m in enumerate(self.models):
            features = self.models[idx].fc_block[-1].in_features
            self.models[idx].fc_block[-1] = nn.Identity()      
        # Create new classifier
        self.gate = nn.Linear(features*len(self.models), features) # allow non-linear dependecies with ReLU
        self.drop = nn.Dropout(initial_config["drop_p_ensemble"])
        self.regressor = nn.Linear(features, outshape)  
    def forward(self, x):
        tmp = torch.Tensor().to(device)
        for m in self.models:
            tmp = torch.cat((tmp, m(x)), dim=1).to(device)
        x = self.gate(F.relu(tmp))
        x = self.drop(x)
        x = self.regressor(x)
        return x

class MyLinearEnsemble(nn.Module):
    def __init__(self, model_list, outshape=3):
        super(MyLinearEnsemble, self).__init__()
        self.models = model_list
        self.regressor = nn.Linear(3*len(self.models), outshape)
    def forward(self, x):
        tmp = torch.Tensor().to(device)
        for m in self.models:
            tmp = torch.cat((tmp, m(x)), dim=1).to(device)
        x = self.regressor(F.relu(tmp))
        return x

class MyAverageEnsemble(nn.Module):
    def __init__(self, model_list, outshape=3):
        super(MyAverageEnsemble, self).__init__()
        self.models = model_list
    def forward(self, x):
        tmp = torch.Tensor(x.shape[0], 3).to(device)
        for m in self.models:
            tmp = (tmp + m(x))
        return tmp / len(self.models)

def get_single_model(set_str, ray_results):
    with open(DATAPATH + ray_results, 'rb') as f:
        res = pickle.load(f)    
    res_ascending = res.sort_values('loss')
    dir_top1 = res_ascending['logdir'][:1]   
    # change paths to Magritek PC
    dir_top1 = [DATAPATH + '/ray_results/raytune_{}/'.format(set_str) + os.path.basename(d) for d in dir_top1][0]
    with open(dir_top1+'\params.json', 'rb') as f:
        config = json.load(f)
    model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),channels,sampling_points/initial_config["downsample_factor"]),
                        num_classes = 3, kernel_size=int(config["kernel_size"]),stride=int(config["stride"]),
                        pool_size=int(config["pool_size"]), num_layers=int(config["num_layers"]),
                        drop_p_conv = config["drop_p_conv"], drop_p_fc = config["drop_p_fc"])   
    last_checkpoint = glob.glob(dir_top1+'\checkpoint_*')[-1]
    checkpoint_path = os.path.join(last_checkpoint, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model
    

def get_ensemble(set_str, ray_results, nr_models=50):
    model_list = []
    if len(ray_results) > 2 : ray_results = [ray_results]  
    for sub in ray_results:
        with open(DATAPATH + sub, 'rb') as f:
            res = pickle.load(f)    
        res_ascending = res.sort_values('loss')
        dirs_top10 = res_ascending['logdir'][:nr_models]        
        # change paths to Magritek PC
        dirs_top10 = [DATAPATH + '/ray_results/raytune_{}/'.format(set_str) + os.path.basename(d) for d in dirs_top10]     
        for d in dirs_top10:
            with open(d+'\params.json', 'rb') as f:
                config = json.load(f)

            model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),channels,sampling_points/initial_config["downsample_factor"]),
                                num_classes = 3, kernel_size=int(config["kernel_size"]),stride=int(config["stride"]),
                                pool_size=int(config["pool_size"]), num_layers=int(config["num_layers"]),
                                drop_p_conv = config["drop_p_conv"], drop_p_fc = config["drop_p_fc"])
            
            last_checkpoint = glob.glob(d+'\checkpoint_*')[-1]
            checkpoint_path = os.path.join(last_checkpoint, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state)
            model.to(device)
            model.eval()     
            for param in model.parameters():
                param.requires_grad_(False)               
            model_list.append(model)        
    if initial_config["meta_type"] == 'linear': model = MyLinearEnsemble(model_list)
    if initial_config["meta_type"] == 'fc': model = MyEnsemble(model_list)
    if initial_config["meta_type"] == 'average': model = MyAverageEnsemble(model_list)
    return model
    

def get_batched_spectrum(distortion, ref_shims, initial_spectrum, channels, offset_value):
    batch = np.zeros([1, channels, int(sampling_points/initial_config["downsample_factor"])])   
    def apply_and_measure(offset, distortion=distortion):
            xaxis, spectrum, shims = utils_Spinsolve.setShimsAndRun(com, my_arg.count, np.add(offset,distortion), True, verbose=(input_args.verbose>1))
            if input_args.verbose == 2: print('shims: ', shims)
            my_arg.count += 1
            if input_args.verbose == 2: print(my_arg.count)
            return spectrum    
    batch[0,0] = initial_spectrum[::initial_config["downsample_factor"]]
    batch[0,1] = apply_and_measure([offset_value,0,0])[::initial_config["downsample_factor"]]/initial_config["max_data"]
    batch[0,2] = apply_and_measure([0,offset_value,0])[::initial_config["downsample_factor"]]/initial_config["max_data"]
    batch[0,3] = apply_and_measure([0,0,offset_value])[::initial_config["downsample_factor"]]/initial_config["max_data"]       
    return batch


def shot(distortion, set_as_string, model_set, offset_value, ray_results, pred_shift_range=pred_shift_range, return_initial=False):
    global spectra_memory
    
    if initial_config['meta_type'] != 'none' and initial_config['meta_type'] != 'none_tuned':
        model = get_ensemble(set_as_string, ray_results, nr_models=initial_config["nr_models"])
    else: model = get_single_model(set_as_string, ray_results)
    if initial_config['meta_type'] != 'average' and initial_config['meta_type'] != 'none': 
        state = torch.load(DATAPATH + model_set + "_" + initial_config["meta_type"] + ".pt", map_location=torch.device('cpu'))
        model.load_state_dict(state)
    model.eval()

    xaxis, initial_spectrum, ref_shims = utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion, return_shimvalues=True, verbose=(input_args.verbose>1))
    linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
    initial_out = initial_spectrum
    initial_spectrum = initial_spectrum / initial_config["max_data"] # scale to dataset 
    ref_shims = ref_shims[:3]
    my_arg.count += 1

    if input_args.verbose == 2: print('ref shims: ', ref_shims)

    batched_spectra = get_batched_spectrum(distortion, ref_shims, initial_spectrum, channels, offset_value)
    for tmp in batched_spectra[0]: spectra_memory.append(tmp)
    if input_args.verbose == 2: print('shape batched', batched_spectra.shape)

    # run "ensemble" of spectra with different augmentation trough the model
    running_pred = []
    prediction = model(torch.tensor(batched_spectra).float()) # first prediction without augmentation
    running_pred.append(prediction.detach().numpy()[0]*sampling_points/initial_config["label_scaling"])
    # average over predictions with augmented/shifted input
    for i in range(pred_averages):
        # homogen shift
        #prediction = model(torch.roll(torch.tensor(batched_spectra), np.random.randint(-int(pred_shift_range), int(pred_shift_range))).float())
        #heterogen shift
        batch = torch.tensor(np.zeros([1, channels, int(sampling_points/initial_config["downsample_factor"])])).float()
        shifts =np.random.randint(-int(pred_shift_range), int(pred_shift_range), size=4)
        for ix, s in enumerate(shifts):
            batch[0,ix] = torch.roll(torch.tensor(batched_spectra[0,ix]), s).float()
        prediction = model(batch)
        running_pred.append(prediction.detach().numpy()[0]*sampling_points/initial_config["label_scaling"])       
        if input_args.verbose >= 1: print('Prediction', running_pred[-1])
        
    tx,ty,tz = np.mean(running_pred, axis=0).astype(int)

    #scale to real value and clip to prevent eddy currents
    tx = sorted((-10000,tx,10000))[1]
    ty = sorted((-10000,ty,10000))[1]
    tz = sorted((-10000,tz,10000))[1]

    if input_args.verbose >= 1: 
        print('artificial distortion (x,y,z): ', distortion)
        print('predicted correction (x,y,z): ',tx,ty,tz)

    xaxis, shimmed_spectrum = utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion+[-tx,-ty,-tz], verbose=(input_args.verbose>0))
    linewidth_shimmed = utils_Spinsolve.get_linewidth_Hz(shimmed_spectrum)
    spectra_memory.append(shimmed_spectrum[::initial_config["downsample_factor"]]/initial_config["max_data"])

    return [tx,ty,tz], linewidth_initial, linewidth_shimmed, initial_out

# if prediction is not improving criterion, undo and check whether spectra in batch is better.
def checkup(prediction, distortion, offset_value):
    global spectra_memory
    shims = [[0,0,0], [offset_value,0,0], [0,offset_value,0], [0,0,offset_value]] # standard shims for batch creation
    recent_set = np.array(spectra_memory[-5:])
    initial = recent_set[0]

    min_width = signal.peak_widths(initial, signal.find_peaks(initial, height = initial.max()*.9, distance=1000)[0])[0].item()
    max_peak_height = initial.max()
    if input_args.verbose >= 1: print("Ref. width, height: {} {}".format(min_width,max_peak_height))

    criteria = np.empty([5])
    for idx, spectrum in enumerate(recent_set): 
        try:
            criteria[idx] = criterion_one_peak(spectrum, min_width, max_peak_height)
        except ValueError:
            criteria[idx] = 0

    if input_args.verbose >= 1: print("Criteria (ref, +x, +y, +z, pred): ", [round(num, 4) for num in criteria])

    best = np.argmax(criteria)
    if best != 4: # if best criterion is not predicted shim setting
        if input_args.verbose >= 1: print("best criterion != predicted shims. Resetting.")
        # undo shimming
        xaxis, spectrum = utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion-prediction, verbose=(input_args.verbose>0))
        # apply shims if offsetting in one direction improved the shimming
        if best != 0: 
            utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion+shims[best], verbose=(input_args.verbose>0))   
        if input_args.verbose >= 1: print("Taking {}. offset for best setting. ".format(np.array(['ref', 'x', 'y', 'z'])[best]))
        if input_args.verbose >= 1: print("Distortion before ", distortion)
        distortion += shims[best]
        if input_args.verbose >= 1: print("Distortion after ", distortion)

    return (best == 4), criteria[-1]

global spectra_memory
spectra_memory = []

com = utils_Spinsolve.init( verbose=(input_args.verbose>0), gui = (input_args.verbose>0) )

mean_lw_initial = []
mean_lw_simplex = []
mean_lw_dre = []
mean_fe_simplex = [] # function evaluations
results_array = []

# loop over all random distortions and track performance
for d in random_distortions:
    print(my_arg.count)
    #CAREFUL
    # maxiter and step !

    pred_dre, lw_init, lw_shimmed, initial_spectrum = shot(d, 'coarse', ENSEMBLE_COARSE, ray_results=initial_config["base_models"], offset_value=1000, return_initial=True)
    success, criterion_after_dre = checkup(pred_dre, d, offset_value=1000)
    print(d, pred_dre)
    print("DRE lw50: {} -> {}".format(lw_init.item(),lw_shimmed.item()))
    mean_lw_initial.append(lw_init)
    mean_lw_dre.append(lw_shimmed)
    if success: print("Success")
    
    xaxis, _, shimmed_spectrum, shims, info = utils_Spinsolve.setShimsAndStartComparison(com, my_arg.count, d, 'simplex', maxiter=50, step=1000, lw_stopping_val = lw_shimmed.item(),
                                                                                         return_shimvalues=True, verbose=(input_args.verbose>1))
    linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
    linewidth_shimmed = utils_Spinsolve.get_linewidth_Hz(shimmed_spectrum)
    initial = initial_spectrum[::initial_config["downsample_factor"]]/initial_config["max_data"]
    shimmed = shimmed_spectrum[::initial_config["downsample_factor"]]/initial_config["max_data"]
    min_width = signal.peak_widths(initial, signal.find_peaks(initial, height = initial.max()*0.9, distance=1000)[0])[0].item()
    max_peak_height = initial.max()
    c1 = criterion_one_peak(shimmed, min_width, max_peak_height)
    for key,val in info.items():
        exec(key + '=' + val)
    pred_sim = np.multiply([int(xbefore)-int(xafter), int(ybefore)-int(yafter), int(zbefore)-int(zafter)], -1)
    print(d, pred_sim)
    print('Nr. function eval.', stepcounter)
    mean_fe_simplex.append(stepcounter)
    print('Simplex lw50: {} -> {}'.format(linewidth_initial.item(),linewidth_shimmed.item()))
    mean_lw_simplex.append(linewidth_shimmed)
    print('\n')

    results_array.append(['dist: {}, pred DRE: {}, pred simplex: {}, lw50 DRE: {} -> {}, lw50 Simplex: {}->{}, simplex function evaluations: {}'
                        .format(d, pred_dre, pred_sim, lw_init, lw_shimmed, linewidth_initial, linewidth_shimmed, stepcounter)])

print('\n')
print('Mean lw initial: {} +/- {}'.format(round(np.mean(mean_lw_initial),2), round(np.std(mean_lw_initial),2)))
print('Mean lw Simplex: {} +/- {}'.format(round(np.mean(mean_lw_simplex),2), round(np.std(mean_lw_simplex),2)))
print('Mean lw DRE: {} +/- {}'.format(round(np.mean(mean_lw_dre),1), round(np.std(mean_lw_dre),1)))
print('Mean function evaluations Simplex: {} +/- {}'.format(round(np.mean(mean_fe_simplex),1), round(np.std(mean_fe_simplex),1)))
print('Mean function evaluations DRE: 5')


with open(DATAPATH + '/DRE/results_comparison_iterations_{}_{}_{}.txt'.format(initial_config["sample"],initial_config["meta_type"], datetime.now().timestamp()), 'w') as f:
    f.write('Mean lw initial: {} +/- {}'.format(round(np.mean(mean_lw_initial),2), round(np.std(mean_lw_initial),2)))
    f.write("\n")
    f.write('Mean lw Simplex: {} +/- {}'.format(round(np.mean(mean_lw_simplex),2), round(np.std(mean_lw_simplex),2)))
    f.write("\n")
    f.write('Mean lw DRE: {} +/- {}'.format(round(np.mean(mean_lw_dre),1), round(np.std(mean_lw_dre),2)))
    f.write("\n")
    f.write('Mean function evaluations Simplex: {} +/- {}'.format(round(np.mean(mean_fe_simplex),2), round(np.std(mean_fe_simplex),2)))
    f.write("\n")
    f.write('Mean function evaluations DRE: 5')
    f.write("\n")
    for item in results_array:
        f.write(str(item))
        f.write("\n")

utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))