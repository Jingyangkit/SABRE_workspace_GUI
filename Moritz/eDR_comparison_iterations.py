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

# TODO
# CHECK simplex steps/weighting of Z2
# !!!


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
        "sample": 'Ref',           #Ref, H2OCu, EtOH-0.1, EtOH-0.5, gluc-0.01, toluene, acetoneH20, acetoneCHCl, CHCl, isoprop
        "set":'random',
        "downsample_factor": 1,
        'ROI_min': 16000,           # calculated via mean_idx(signal(2048p)>2*noise) * downsamplefactor
        "max_data": 1e5,            # *0.68312625, # max_data * max_dataset_value (scaling)
        'scale_sample': True, # scale first sample in sequence to one (instead of global max)
        'shim_weightings': [1.2,1,2,18,0,0,0,0,0,0,0,0,0,0,0],  # Shim weighting. range * weighting = shim values
        'acq_range': 50,            # range of absolute shim offsets for highest-impact shim (weighting=1)
        "drop_p_fc": 0.0,
        "drop_p_conv": 0.0,
    }     

with open(DATAPATH + '/DRR/models/config_DR_Z2_{}.json'.format(initial_config['set'])) as f:
    config = json.load(f)

config = merge_two_dicts(initial_config, config)   #merge configs


# =============================================================================
# # overwrite 
# =============================================================================

if config['sample'] == 'H2O' or config['sample'] == 'Ref':    
    config['ROI_min'] = 16000
    ROLL = 0
    PLT_X = [80,120]
elif config['sample'] == 'MeOHp1': 
    config['ROI_min'] = 14552 # MeOH first peak
    ROLL = -500
    PLT_X = [-100,0]
elif config['sample'] == 'MeOHp2': 
    config['ROI_min'] = 16600 # MeOH first peak
    ROLL = +500
    PLT_X = [75,125]


seed = 45612

channels = 4
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
class ConvLSTM(nn.Module): # convolutional LSTM  
    def __init__(self, spectrum_size, action_size, hidden_size, output_size, num_layers_lstm=2, past_obs=4,
                 num_layers_cnn=5, filters_conv=32, stride=2, kernel_size=51, drop_p_conv=0, drop_p_fc=0):
        super().__init__()       
        self.hidden_size = hidden_size
        self.num_layers_lstm = num_layers_lstm
        self.num_layers_cnn = num_layers_cnn
        self.filters_conv = filters_conv
        self.stride = stride
        self.kernel_size = kernel_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.spectrum_size = spectrum_size
        self.action_size = action_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        #self.past_obs = past_obs # variable!      
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p))
            return conv
        
        block_conv = [] # for i2f (input to features)
        block_conv.append( one_conv(1, self.filters_conv, self.kernel_size, stride=self.stride, drop_p=self.drop_p_conv) )
        self.feature_shape = int( (self.spectrum_size-self.kernel_size)/self.stride +1 )
        for i in range(self.num_layers_cnn-1):
            conv = one_conv(self.filters_conv, self.filters_conv, self.kernel_size, self.stride, drop_p=self.drop_p_conv)
            block_conv.append(conv)
            self.feature_shape = int( (self.feature_shape-self.kernel_size)/self.stride +1)
        self.i2f = nn.Sequential(*block_conv) # input to features  
        self.ln_features = nn.LayerNorm(self.feature_shape*self.filters_conv+self.action_size)    
        self.lstm = nn.LSTM(self.feature_shape*self.filters_conv+self.action_size, hidden_size, self.num_layers_lstm,
                            batch_first=True)      
        self.ln_lstm = nn.LayerNorm(hidden_size)      
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p_fc)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)           
    def forward(self, inputs, hidden):
        past_obs = inputs.shape[1]
        conv_part = inputs[:,:, :self.spectrum_size]  # spectra
        fc_part = inputs[:,:, self.spectrum_size:]     # shim actions/values
        features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.feature_shape*self.filters_conv])).float().to(self.DEVICE)
        for k in range(past_obs): # convolve each spectrum for its own to keep temporal nature
            features[:,k] = self.i2f(conv_part[:,k].unsqueeze(1)).view(inputs.shape[0],-1)
        #features = self.i2f(conv_part)
        combined = torch.cat((features, fc_part), 2)      
        combined = self.ln_features(combined)       
        out, (h0,c0) = self.lstm(combined, hidden)       
        out = self.ln_lstm(out)       
        x = self.relu(self.fc1(out))
        x = self.drop(x)
        x = self.tanh(self.fc2(x))
        return x, (h0,c0)
    def initHidden(self, batch_size=1):
        return ( torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size), torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size) )

def get_single_model(filename):
    model = ConvLSTM(spectrum_size=2048, action_size=config["nr_shims"], hidden_size=config["hidden_size"], 
                         output_size=config["nr_shims"], num_layers_lstm=2, past_obs=config["input_channels"],
                     num_layers_cnn=config["num_layers_cnn"], filters_conv=config["filters_conv"], 
                     stride=config["stride"], kernel_size=config["kernel_size"]) 
    model_state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model
 
#%% Functions

def invert(prediction_scaled, s): 
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                                np.add(prediction_scaled,distortion), True,True, verbose=(input_args.verbose>1))
    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    if input_args.verbose == 2: print('shims: ', shims)
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp, '--',
            label='$u({})$'.format(s), alpha=0.5)
    return spectrum_tmp
    

def save(spectrum_tmp, prediction_scaled):
    if INVERT_PRED: pass
    # TODO change success definition
    success = (spectrum_tmp/sc_factor).max() > batched_spectra[0,:-config['nr_shims']].max()
    sign_d = np.sign(distortion)
    sign_p = np.sign(-prediction_scaled)
    sign_d[np.where(sign_d==0)[0]] = 1 # count 0 as + sign
    sign_p[np.where(sign_p==0)[0]] = 1
    sign = (sign_d==sign_p).sum()/len(distortion)
    mae = mean_absolute_error(distortion/config['shim_weightings'][:config['nr_shims']]/config['acq_range'],
                       -prediction_scaled/config['shim_weightings'][:config['nr_shims']]/config['acq_range'])
    lw_init_50 = utils_Spinsolve.get_linewidth_Hz(batched_spectra[0,:-config['nr_shims']], sampling_points=32768, bandwidth = 5000)
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000)    
    lw_init_055 = utils_Spinsolve.get_linewidth_Hz(batched_spectra[0,:-config['nr_shims']], sampling_points=32768, bandwidth = 5000, height=0.9945)
    lw_shimmed_055 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000, height=0.9945)
  
    results_data.append([METHOD, nr, s, INVERT_PRED, success, sign, RANDOM, distortion, prediction_scaled, 
                         lw_init_50.item(), lw_shimmed_50.item(), lw_init_055.item(), lw_shimmed_055.item(), mae, sc_factor])

# first offset = random 
def random_step(iteration, distortion, batched_spectra, config):
    #offset_rand = np.random.normal(0,1/3, size=config['nr_shims'])
    offset_rand = np.random.uniform(-.5,.5, size=config['nr_shims'])
    offset_rand_scaled = (offset_rand*config['acq_range']*config['shim_weightings'][:config['nr_shims']]).astype(int)
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                         np.add(offset_rand_scaled,distortion), True,True, verbose=(input_args.verbose>1))
    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    # - vs + offset
    batched_spectra = np.append( batched_spectra, np.concatenate((spectrum_tmp/sc_factor, -offset_rand))[np.newaxis,:], axis=0)   
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],spectrum_tmp,'--', label='random{}'.format(iteration), alpha=0.4)
    
    save(spectrum_tmp, (offset_rand*config['acq_range']*config['shim_weightings'][:config['nr_shims']]).astype(int))
    
    return batched_spectra

def step(iteration, distortion, batched_spectra, config):
    rolled_batch = np.roll(batched_spectra, ROLL)
    inputs = torch.tensor(rolled_batch).unsqueeze(0).float()
    (h0,c0) = model.initHidden(batch_size=inputs.shape[0]) # TODO CHECK
    h0, c0 = h0.to(device), c0.to(device)     
    outputs, hidden = model(inputs, (h0,c0))        
    #print(outputs.detach().numpy())          
    prediction = outputs[:,-1]
    
    prediction = prediction.detach().numpy()[0]
    if INVERT_PRED: prediction = -prediction    # invert to cancel

    #clip to prevent bad currents
    prediction_scaled = np.clip(-10000, (prediction*config['shim_weightings'][:config['nr_shims']]*50).astype(int), 10000)
             
    if input_args.verbose >= 1: 
        print('artificial distortion (x,y,z,z2): ', distortion)
        print('predicted correction (x,y,z,z2): ', prediction_scaled)

    # take prediction as "random" 
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                            np.add(prediction_scaled,distortion), True,True, verbose=(input_args.verbose>1))
    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    if input_args.verbose == 2: print('shims: ', shims)
    if INVERT_PRED: 
        pass
        #plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp, '--', label='$-u({})$'.format(iteration), alpha=0.5)
    else: plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp,
            label='$u({})$'.format(iteration))
    
    # CARE  
    # - pred or + pred ?
    # - prediction gives better results (5.5.2022)
    batched_spectra = np.append( batched_spectra, np.concatenate((spectrum_tmp/sc_factor, -prediction))[np.newaxis,:], axis=0)
    
    if INVERT_PRED: 
        spectrum_tmp = invert(-prediction_scaled, s) 
        save(spectrum_tmp, -prediction_scaled)
    else: save(spectrum_tmp, prediction_scaled)
    
    return batched_spectra, prediction_scaled
#%%       

# VARIABLES
INVERT_PRED = False
STEPS_RAND = 4
STEPS_REAL = 3

global spectra_memory
spectra_memory = []
global results_data
results_data = []

com = utils_Spinsolve.init( verbose=(input_args.verbose>0), gui=True )

# Get best spectrum
xaxis, best_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, np.zeros(config['nr_shims']),
                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
my_arg.count += 1
spectrum_tmp = best_spectrum[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
plt.figure()
plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp)
plt.title('Best spectrum with {} shims'.format(config['nr_shims']))
plt.xlim(PLT_X) # crop for plotting
#plt.savefig(DATAPATH + '/DRR/img_dre_{}_best.png'.format(config["sample"]))
plt.show()

# loop over all random distortions and track performance
for nr, distortion in enumerate(random_distortions):

    # prediction part
    METHOD = 'ours'
    model = get_single_model(DATAPATH+'/DRR/models/model_DR_Z2_{}.pt'.format(config['set']))
    xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, distortion,
                                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
    my_arg.count += 1
    #linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
    initial_spectrum = initial_spectrum[config["ROI_min"]:config["ROI_min"]+2048] / config["max_data"] # scale to dataset 
    #ref_shims = ref_shims[:config['nr_shims']]
    
    if config['scale_sample']: sc_factor = initial_spectrum.max()
    else: sc_factor = 1
    
    plt.figure()
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],initial_spectrum, label='initial')
    
    batched_spectra = np.concatenate((initial_spectrum/sc_factor, np.zeros(config['nr_shims'])))[np.newaxis,:] # first 
    
    # random steps
    for s in range(STEPS_RAND):
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
    #plt.savefig(DATAPATH + '/DRR/img_dre_{}_{}.png'.format(config["sample"],nr))
    plt.show()
    
    for tmp in batched_spectra: spectra_memory.append(tmp)


    # Comparison part
    METHOD = 'parabola'
    RANDOM = None
    lw50 = results_data[-1][10] # last item, lw50
    
    xaxis, _, shimmed_spectrum, info = utils_Spinsolve.setShimsAndStartComparisonV3(com, my_arg.count, distortion, method='parabola',  lw_stopping_val = lw50)
    spectrum_tmp = shimmed_spectrum[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    for key,val in info.items():
        exec(key + '=' + val)
    pred_sim = np.multiply([int(xbefore)-int(xafter), int(ybefore)-int(yafter), int(zbefore)-int(zafter),int(z2before)-int(z2after)], -1)
    print(distortion, pred_sim)
    print('Steps to LW ', stepsToLW)
    #s = stepcounter
    s = stepsToLW
    save(spectrum_tmp, -pred_sim)
    results_data[-1][10] = lw50arr # overwrite last lw50 value with array of parabola shimming values
        
# Convert results to pd df
columns_data = ['Method', 'Nr', 'Step', 'Inverted', 'Success', 'Sign', 'Random', 'Distortion', 'Prediction', 
                'lw50_init', 'lw50_step', 'lw55_init', 'lw55_step', 'MAE', 'ScalingFactor']
df_ = pd.DataFrame(results_data, columns=columns_data)
    
# print results
print("Success rate: ", df_.loc[df_['Step']==STEPS_REAL+STEPS_RAND-1]['Success'].mean())
print("Correct prediction rate: ", round( df_.loc[df_['Step']==STEPS_REAL+STEPS_RAND-1]['Sign'].mean() , 3))
#print("Mean criterion improvement: {} {} % +/- {}".format( ('+' if np.mean(mean_c)>1 else '-'), round((np.mean(mean_c)-1)*100, 2), round(np.abs((np.std(mean_c)-1)*100),2)) )
print("Averaged MAE: {} +/- {}".format(round(df_.loc[df_['Step']==STEPS_REAL+STEPS_RAND-1]['MAE'].mean(),2), round(df_.loc[df_['Step']==STEPS_REAL+STEPS_RAND-1]['MAE'].mean(),2)) )

df_.to_excel(DATAPATH + '/DRR/results_comparison_iterations_{}_{}.xlsx'.format(config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

with open(DATAPATH + '/DRR/spectra_memory_comparison_iterations_{}_{}.txt'.format(config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')), 'w') as f:
    for item in spectra_memory:
        for i in item: 
            f.write(str(i) + ' ')
        f.write("\n")


utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))
