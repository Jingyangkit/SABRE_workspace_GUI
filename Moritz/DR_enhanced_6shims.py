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
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data/'
sys.path.append(MYPATH+'Utils/')
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
        'id': 28,                    # model's id
        "sample": 'Ref',           #Ref, H2O, MeOH, H2OTSP
        "set":'random6',
        "downsample_factor": 1,
        'ROI_min': 16000,           # calculated via mean_idx(signal(2048p)>2*noise) * downsamplefactor
        "max_data": 1e5,            # *0.68312625, # max_data * max_dataset_value (scaling)
        'scale_sample': True, # scale first sample in sequence to one (instead of global max)
        'shim_weightings': [1.2,1,2,18,10,10,0,0,0,0,0,0,0,0,0],  # Shim weighting. range * weighting = shim values
        'acq_range': 20,            # range of absolute shim offsets for highest-impact shim (weighting=1)
        "drop_p_fc": 0.0,
        "drop_p_conv": 0.0,
        'full_memory': True,         # store whole spectrum in additional memory
        'tta': False,                # test-time augmentation
    }     

with open(DATAPATH + '/DRR/models/config_eDR_6s_{}_id{}.json'.format(initial_config['set'],initial_config["id"])) as f:
    config = json.load(f)

config = merge_two_dicts(initial_config, config)   #merge configs


# =============================================================================
# # overwrite 
# =============================================================================

if 'H2O' in config['sample'] or config['sample'] == 'Ref':    
    config['ROI_min'] = 16000
    ROLL = 0
    PLT_X = [80,120]
if 'TSP' in config['sample']:    
    config['ROI_min'] = 18500
    ROLL = 0
    PLT_X = [450,550]
elif config['sample'] == 'MeOHp1': 
    config['ROI_min'] = 14552 # MeOH first peak
    ROLL = -500
    PLT_X = [-100,0]
elif config['sample'] == 'MeOHp2': 
    config['ROI_min'] = 16600 # MeOH first peak
    ROLL = +500
    PLT_X = [75,125]


seed = 45612

sampling_points = 32768
#pred_averages = 0 # nr of pred_averages for ensemble
#pred_shift_range = 5 # range for z0-shift augmentation for ensemble
device = "cuda:0" if torch.cuda.is_available() else "cpu"

nr_evaluations = 50 #100
np.random.seed(seed)
gauss_noise = np.random.normal(0, 1/3,size=(nr_evaluations,config['nr_shims']))
#random_distortions = (np.random.randint(-config['acq_range'],config['acq_range'],size=(nr_evaluations,config['nr_shims']))*config['shim_weightings'][:config['nr_shims']]).astype(int) #discrete uniform
random_distortions = (config['acq_range']*gauss_noise*config['shim_weightings'][:config['nr_shims']]).astype(int) #discrete uniform

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
    peak_index = signal.find_peaks(N, height = N.max()*0.7, distance=1000)[0]
    [width, height_of_evaluation,_,_] = signal.peak_widths(N, peak_index)
    # normalize criterion such that height and width are proportional to their optimal value and in range [0,1]
    return 1/2*(lamda[0]*(min_width/width.item())
                + lamda[1]*N.max()/max_peak_height).item()   # only for 1 peak! 

#%% DL part
from models import ConvLSTM, ConvLSTMf2fc, ConvTransformer

def get_single_model(filename):
    
    if config['model'] == 'ConvLSTM':
        model = ConvLSTM(spectrum_size=2048, action_size=config["nr_shims"], hidden_size=config["hidden_size"], 
                         output_size=config["nr_shims"], num_layers_lstm=config['num_layers_lstm'], 
                     num_layers_cnn=config["num_layers_cnn"], filters_conv=config["filters_conv"], 
                     stride=config["stride"], kernel_size=config["kernel_size"], spectra_content=config['spectra_content'])
    if config['model'] == 'ConvTransformer':
        model = ConvTransformer(spectrum_size=2048, action_size=config["nr_shims"], hidden_size=config["hidden_size"], 
                     output_size=config["nr_shims"], num_layers_transformer=config["num_layers_transformer"], num_heads_transformer=config['num_heads_transformer'],
                 drop_p_trans=config['drop_p_trans'], num_layers_cnn=config["num_layers_cnn"], filters_conv=config["filters_conv"], 
                 stride=config["stride"], kernel_size=config["kernel_size"])
    model_state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model
 
#%% Functions

def save(spectrum_tmp, prediction_scaled):
    if config['spectra_content'] == 'abs': # reload spectrum, as lw is different for abs!
        spectrum_tmp = ng.proc_base.fft(full_memory[-1][3])[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"] # load from current fid (or spectra_memory_full?)
        spectrum_tmp = spectrum_tmp.real
        initial_spec_tmp = initial_spectrum_LW_CALC/sc_factor
    else: 
        initial_spec_tmp = batched_spectra[0,:-config['nr_shims']]
    # TODO change success definition
    success = (spectrum_tmp/sc_factor).max() > batched_spectra[0,:-config['nr_shims']].max()
    sign_d = np.sign(distortion)
    sign_p = np.sign(-prediction_scaled)
    sign_d[np.where(sign_d==0)[0]] = 1 # count 0 as + sign
    sign_p[np.where(sign_p==0)[0]] = 1
    sign = (sign_d==sign_p).sum()/len(distortion)
    mae = mean_absolute_error(distortion/config['shim_weightings'][:config['nr_shims']]/config['acq_range'],
                       -prediction_scaled/config['shim_weightings'][:config['nr_shims']]/config['acq_range'])
    lw_init_50 = utils_Spinsolve.get_linewidth_Hz(initial_spec_tmp, sampling_points=32768, bandwidth = 5000)
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000)    
    lw_init_055 = utils_Spinsolve.get_linewidth_Hz(initial_spec_tmp, sampling_points=32768, bandwidth = 5000, height=0.9945)
    lw_shimmed_055 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000, height=0.9945)
  
    results_data.append([nr, s, int(0), int(success), sign, int(RANDOM), distortion, prediction_scaled, 
                         lw_init_50.item(), lw_shimmed_50.item(), lw_init_055.item(), lw_shimmed_055.item(), mae, sc_factor])

# test time augmentation 
def tta(inputs, pred_averages = 10):
    
    shift = 8
    noise = 0.1 / config['acq_range']
    shims = config['nr_shims']
    
    running_pred = []
    if config['model']=='ConvLSTM':
        (h0,c0) = model.initHidden(batch_size=inputs.shape[0]) # TODO CHECK
        h0, c0 = h0.to(device), c0.to(device)     
        outputs, hidden = model(inputs, (h0,c0))  
    if config['model']=='ConvTransformer':    
                outputs = model(inputs)               
    prediction = outputs[:,-1]
    running_pred.append(prediction.detach().numpy()[0])
    
    # average over predictions with augmented/shifted input
    for i in range(pred_averages):
        d, t = inputs[:,:,:2048], inputs[:,:,2048:]
        
        # normal shift
        d = torch.roll(d, np.random.randint(-shift,shift))
        
        # normal label noise
        t=t+np.concatenate((np.zeros([1,shims]),
                                np.tile( np.random.uniform(-noise,noise,size=(1,shims)),
                                inputs.shape[1]-1).reshape(inputs.shape[1]-1,-1)))
        # awgn
        awgn = 0
        
        # interaction noise
        range_interactions = noise
        interactions = np.random.uniform(-range_interactions,range_interactions,
                                              size = (shims, shims))
        interactions = (interactions + interactions.T)/2     # make symmetric
        np.fill_diagonal(interactions, 1)  
        
        augm_inp = torch.cat(((d+awgn),
                    torch.tensor(np.matmul(t,interactions))), dim=-1).float()
                    
        if config['model']=='ConvLSTM':             
            (h0,c0) = model.initHidden(batch_size=inputs.shape[0]) # TODO CHECK
            h0, c0 = h0.to(device), c0.to(device)     
            outputs, hidden = model(augm_inp, (h0,c0))   
        if config['model']=='ConvTransformer':    
            outputs = model(inputs)                
        prediction = outputs[:,-1]
        running_pred.append(prediction.detach().numpy()[0])
        
    return np.mean(running_pred, axis=0)


def random_step(iteration, distortion, batched_spectra, config):
    #offset_rand = np.random.normal(0,1/3, size=config['nr_shims'])
    offset_rand = np.random.uniform(-.5,.5, size=config['nr_shims'])
    offset_rand_scaled = (offset_rand*config['acq_range']*config['shim_weightings'][:config['nr_shims']]).astype(int)
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                         np.add(offset_rand_scaled,distortion), True,True, verbose=(input_args.verbose>1))
    if config['full_memory']: full_memory.append([nr, s, sc_factor, fid])
    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    if config['spectra_content']!='real': 
        spectrum_tmp = ng.proc_base.fft(fid)[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
        if config['spectra_content']=='abs': spectrum_tmp = np.abs(spectrum_tmp)
    # - vs + offset
    batched_spectra = np.append( batched_spectra, np.concatenate((spectrum_tmp/sc_factor, -offset_rand))[np.newaxis,:], axis=0)   
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],spectrum_tmp,'--', label='random{}'.format(iteration), alpha=0.4)
    
    save(spectrum_tmp, (offset_rand*config['acq_range']*config['shim_weightings'][:config['nr_shims']]).astype(int))
    
    return batched_spectra

def step(iteration, distortion, batched_spectra, config):
    rolled_batch = np.roll(batched_spectra, ROLL)
    inputs = torch.tensor(rolled_batch).unsqueeze(0)#.float()
    if config['spectra_content'] != 'complex': inputs = inputs.float()
    
    if not config['tta']:
        if config['model']=='ConvLSTM':   
            (h0,c0) = model.initHidden(batch_size=inputs.shape[0]) # TODO CHECK
            h0, c0 = h0.to(device), c0.to(device)     
            outputs, hidden = model(inputs, (h0,c0))        
        #print(outputs.detach().numpy())    
        if config['model']=='ConvTransformer':    
            outputs = model(inputs)                  
        prediction = outputs[:,-1]
        prediction = prediction.detach().numpy()[0]
    else: 
        prediction = tta(inputs)

    #clip to prevent bad currents
    prediction_scaled = np.clip(-10000, (prediction*config['shim_weightings'][:config['nr_shims']]*config['acq_range']).astype(int), 10000)
             
    if input_args.verbose >= 1: 
        print('artificial distortion (x,y,z,z2): ', distortion)
        print('predicted correction (x,y,z,z2): ', prediction_scaled)

    # take prediction as "random" 
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                            np.add(prediction_scaled,distortion), True,True, verbose=(input_args.verbose>1))
    if config['full_memory']: full_memory.append([nr, s, sc_factor, fid])
    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    if config['spectra_content']!='real': 
        spectrum_tmp = ng.proc_base.fft(fid)[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
        if config['spectra_content']=='abs': spectrum_tmp = np.abs(spectrum_tmp)
    if input_args.verbose == 2: print('shims: ', shims)
    else: plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp,
            label='$u({})$'.format(iteration))
    
    # CARE  
    # - pred or + pred ?
    # - prediction gives better results (5.5.2022)
    batched_spectra = np.append( batched_spectra, np.concatenate((spectrum_tmp/sc_factor, -prediction))[np.newaxis,:], axis=0)
    
    save(spectrum_tmp, prediction_scaled)
    
    return batched_spectra, prediction_scaled
#%%       
com = utils_Spinsolve.init( verbose=(input_args.verbose>0) )

# VARIABLES
STEPS_REAL = 1
STEPS_RAND = 10

# TODO 
# Fix STEPS for REAL and RAND

# loop over rand steps to measure influence
# "TAB" code if used
#for STEPS_RAND in range(6,11):

global spectra_memory
spectra_memory = []
global results_data
results_data = []
global full_memory # big memory for whole fid (not only ROI)
full_memory = [] 

# Get best spectrum
xaxis, best_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, np.zeros(config['nr_shims']),
                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
my_arg.count += 1
spectrum_tmp = best_spectrum[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
plt.figure()
plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp)
plt.title('Best spectrum with {} shims'.format(config['nr_shims']))
plt.xlim(PLT_X) # crop for plotting
#plt.savefig(DATAPATH + '/DRR/img_eDR_{}_best.png'.format(config["sample"]))
plt.show()

# loop over all random distortions and track performance
for nr, distortion in enumerate(random_distortions):

    model = get_single_model(DATAPATH+'/DRR/models/model_eDR_6s_{}_id{}.pt'.format(config['set'],config["id"]))

    xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, distortion,
                                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
    if config['full_memory']: full_memory.append([nr, 0, 1, fid])
    my_arg.count += 1
    linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
    initial_spectrum = initial_spectrum[config["ROI_min"]:config["ROI_min"]+2048] / config["max_data"] # scale to dataset 
    if config['spectra_content']!='real': 
        initial_spectrum = ng.proc_base.fft(fid)[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
        initial_spectrum_LW_CALC = initial_spectrum # save spectrum before abs to variable for save() function and lw calculation
        if config['spectra_content']=='abs': initial_spectrum = np.abs(initial_spectrum)
    ref_shims = ref_shims[:config['nr_shims']]
    
    if config['scale_sample']: sc_factor = initial_spectrum.max()
    else: sc_factor = 1
    
    plt.figure()
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],initial_spectrum, label='initial')
    
    batched_spectra = np.concatenate((initial_spectrum/sc_factor, np.zeros(config['nr_shims'])))[np.newaxis,:] # first 
    
    # random steps
    for s in range(1, STEPS_RAND+1):
        RANDOM = True
        batched_spectra = random_step(s, distortion, batched_spectra, config)

    # real steps
    for s in range(STEPS_RAND+1,STEPS_REAL+STEPS_RAND+1):
        RANDOM = False
        batched_spectra, prediction_scaled = step(s, distortion, batched_spectra, config)

    plt.xlim(PLT_X) # crop for plotting
    plt.legend()    
    plt.title('Shimming with eDR')
    plt.ylabel("Signal [a.u.]")
    plt.xlabel("Frequency [Hz]")
    plt.savefig(DATAPATH + '/DRR/plots/img_eDR_6s_{}_{}_id{}_r{}s{}_{}.png'.format(initial_config["set"],config["sample"],initial_config["id"],STEPS_RAND,STEPS_REAL,nr))
    plt.show()
    
    for i, tmp in enumerate(batched_spectra): 
        spectra_memory.append([nr, i, int(RANDOM), sc_factor, list(tmp[:2048]), list(tmp[-config['nr_shims']:])])


# Convert results to pd df
columns_data = ['Nr', 'Step', 'Inverted', 'Success', 'Sign', 'Random', 'Distortion', 'Prediction', 
                'lw50_init', 'lw50_step', 'lw55_init', 'lw55_step', 'MAE', 'ScalingFactor']
df = pd.DataFrame(results_data, columns=columns_data)
df.to_excel(DATAPATH + '/DRR/results_eDR_{}_id{}_{}_r{}s{}_{}.xlsx'.format(initial_config["set"],initial_config["id"],config["sample"],STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

    
# print results
print("Success rate: ", df.loc[df['Step']==STEPS_REAL+STEPS_RAND]['Success'].mean())
print("Correct prediction rate: ", round( df.loc[df['Step']==STEPS_REAL+STEPS_RAND]['Sign'].mean() , 3))
#print("Mean criterion improvement: {} {} % +/- {}".format( ('+' if np.mean(mean_c)>1 else '-'), round((np.mean(mean_c)-1)*100, 2), round(np.abs((np.std(mean_c)-1)*100),2)) )
print("Averaged MAE: {} +/- {}".format(round(df.loc[df['Step']==STEPS_REAL+STEPS_RAND]['MAE'].mean(),2), round(df.loc[df['Step']==STEPS_REAL+STEPS_RAND]['MAE'].mean(),2)) )

# convert memory to pd df
# Excel cell limit is 32767 --> Store as pickle
columns_data = ['Nr', 'Step', 'Random', 'ScalingFactor', 'Spectrum', 'ShimOffsets']
df_spectra = pd.DataFrame(spectra_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/DRR/spectra_memory_eDR_{}_id{}_{}_r{}s{}_{}.pickle'.format(initial_config["set"],initial_config["id"],config["sample"],STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

# convert memory to pd df
columns_data = ['Nr', 'Step', 'ScalingFactor', 'Spectrum']
df_spectra = pd.DataFrame(full_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/DRR/spectra_memoryFULL_eDR_{}_id{}_{}_r{}s{}_{}.pickle'.format(initial_config["set"],initial_config["id"],config["sample"],STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))


import json
with open(DATAPATH + '/DRR/config_eDR_{}_id{}_{}_r{}s{}_{}.json'.format(
    initial_config["set"],initial_config["id"],config["sample"],
    STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')), 'w') as f:
    json.dump(config,f)

utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))
