# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:15:43 2023

@author: Pierre Labouré
"""


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

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
iteration = 0
markers = ['.', '*', 'd']
'''
plt.figure(figsize = (15, 15))

idx_test = 2

for file in os.listdir(directory):
    
    filename = os.fsdecode(file)
    print(filename)
    if filename.find('MONITOR') != -1:
    
        with open(directoryStr + '/' + filename, "rb") as input_file:
            [data, labels] = pickle.load(input_file)
            idx_best = np.argmin(np.linalg.norm(labels, axis = -1))
        plt.plot(data[idx_best], label = filename + str(labels[idx_best]))

plt.legend(fontsize = 5)
plt.show()

plt.figure(figsize = (15, 15))

for file in os.listdir(directory):
    
    filename = os.fsdecode(file)
    print(filename)
    if filename.find('MONITOR') != -1:

    
        with open(directoryStr + '/' + filename, "rb") as input_file:
            [data, labels] = pickle.load(input_file)
            idx_best = np.argmax(np.max(data, axis = -1))
        plt.plot(data[idx_best], label = filename + str(labels[idx_best]))

plt.legend(fontsize = 5)
plt.show()



plt.figure(figsize = (15, 15))

for file in os.listdir(directory):
    
    filename = os.fsdecode(file)
    print(filename)
    if filename.find('MONITOR') != -1:

        with open(directoryStr + '/' + filename, "rb") as input_file:
            [data, labels] = pickle.load(input_file)
        plt.plot(data[0], label = filename + str(labels[0]))
        plt.plot(data[1], label = filename + str(labels[1]))
        plt.plot(data[2], label = filename + str(labels[2]))

plt.legend(fontsize = 5)
plt.show()


fig, axs = plt.subplots(2, 2, figsize = (15, 15))
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.find('MONITOR') != -1:

        with open(directoryStr + '/' + filename, "rb") as input_file:
            [data, labels] = pickle.load(input_file)
        axs[0, 0].scatter(labels[:, 0], np.max(data, axis = -1), label = filename + 'max_0')
        axs[0, 1].scatter(labels[:, 1], np.max(data, axis = -1), label = filename + 'max_1')
        axs[1, 0].scatter(labels[:, 2], np.max(data, axis = -1), label = filename + 'max_2')
        axs[1, 1].scatter(labels[:, 3], np.max(data, axis = -1), label = filename + 'max_3')

plt.legend()
plt.show()
'''

def FWHM(labels, data):
    half_max = max(data, axis = -1)/2
    d = sign(half_max - data[:, 0:-1]) - sign(half_max - data[:, 1:])
    left_idx = find(d > 0, axis = -1)

'''
fig, axs = plt.subplots(2, 2, figsize = (15, 15))

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.find('MONITOR') != -1:

        with open(directoryStr + '/' + filename, "rb") as input_file:
            [data, labels] = pickle.load(input_file)
        axs[0, 0].scatter(labels[:, 0], np.max(data, axis = -1), label = filename + 'max_0')
        axs[0, 1].scatter(labels[:, 1], np.max(data, axis = -1), label = filename + 'max_1')
        axs[1, 0].scatter(labels[:, 2], np.max(data, axis = -1), label = filename + 'max_2')
        axs[1, 1].scatter(labels[:, 3], np.max(data, axis = -1), label = filename + 'max_3')

plt.legend()
plt.show()
'''

from DTW import *

idx = 118

#fig, axs = plt.subplots(2, 2, figsize = (15, 15))
count = 1
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    
    if filename.find('MONITOR') != -1:
        
        with open(directoryStr + '/' + filename, "rb") as input_file:
            [data, labels] = pickle.load(input_file)
        if count == 1:
            data1 = data
            label1 = labels
        else:
            data2 = data
            label2 = labels
        count +=1
'''
s1 = data1[idx]
s2 = data2[idx]

mask = sakoe(s1, s2, 50)
DTW_mat = DTW_plot_links(s1, s2, mask, 'regular')
DTW_show_matrix(DTW_mat)

DDTW_mat = DTW_plot_links(s1, s2, mask, 'derivative')
DTW_show_matrix(DDTW_mat)

offset_histogram(s1, s2, DDTW_mat)
'''
'''
n1 = 280
n2 = 320
N = n2 - n1

A = np.zeros(N)
Loc = np.zeros(N)
Scale = np.zeros(N)
R = np.zeros(N)

for i in range(n1, n2):
    
    print(i)
    
    s1 = data1[i]
    s2 = data2[i]
    
    DDTW_mat = DDTW(s1, s2, mask)
    a, loc, scale = hist_fit_skewed(s1, s2, DDTW_mat, 20)
    
    A[i-n1] = a
    Loc[i-n1] = loc
    Scale[i-n1] = scale
    
    R[i-n1] = ROI_compare_amp(s1, s2, DDTW_mat, 20)
    
plt.figure()
plt.plot(A)
plt.title('Alpha')
plt.show()

plt.figure()
plt.plot(Loc)
plt.title('Loc')
plt.show()

plt.figure()
plt.plot(Scale)
plt.title('Scale')
plt.show()

plt.figure()
plt.plot(R)
plt.title('R')
plt.show()

'''

from models import ConvLSTM, ConvTransformer, LSTMcompressedMeanOnly, VAESmooth
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
config = {
    "nr_shims" : 4,
    "acq_range" : 50,
    "label_noise" : .1,
    "label_noise_type" : 'normal',
    "input_channels" : 10,
    "set" : 'random',
    "id" : 'Real2_16'
    }

from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels, config, transform = False):
        self.data = data
        self.labels = labels
        self.config = config
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        noise_step = np.ones(self.config["nr_shims"])/self.config['acq_range']     # define noise step as 1 discrete increment of shims
        arr = np.arange(self.data.shape[0])                                     # allocate array for indices
        idxs_rand = np.random.choice(np.delete(arr,idx), replace=False, size=(self.config['input_channels']-1)) # get random indices of data to load for sequence
        idxs = np.append(idx, idxs_rand)                                         # append to idx of __getitem__

        # Change labels and concat with data for DQN_Fuse
        labels_rand = self.labels[idxs_rand] - self.labels[idx]                             # relative to unshimmed
        labels_out = np.concatenate((np.zeros([1,self.config['nr_shims']]), labels_rand))     # Optional: change architecture?

        if self.transform:             
            # add noise to the labels (here to relative shim offsets. target label is modified at return)
            if self.config["label_noise"]!=0:
                # target noise
                noise = noise_step*np.random.uniform(-(self.config["label_noise"]),(self.config["label_noise"]),
                                          size=self.config["nr_shims"])
                # offset noise (step not uniform)
                if self.config['label_noise_type']=='complex':
                    labels_out=labels_out+self.config["label_noise"]*np.concatenate((np.zeros([1,self.config['nr_shims']]),
                                np.random.uniform(-noise_step,noise_step,size=(self.config['input_channels']-1,self.config['nr_shims']))))
                elif self.config['label_noise_type']=='normal':
                    labels_out=labels_out+self.config["label_noise"]*np.concatenate((np.zeros([1,self.config['nr_shims']]),
                                np.tile( np.random.uniform(-noise_step,noise_step,size=(1,self.config['nr_shims'])),
                                        self.config['input_channels']-1).reshape(self.config['input_channels']-1,-1)))
            else:
                noise = 0  
        return torch.cat((torch.tensor(self.data[idxs]),torch.tensor(labels_out)),dim=-1), torch.tensor(self.labels[idx]).float()


RNN1 = MyDataset(data1, label1, config)
RNN2 = MyDataset(data2, label2, config)

MYPATH = 'C:/Users/Magritek/Documents/Moritz/'
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data'
sys.path.append(MYPATH)

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

with open(DATAPATH + '/CeDR/models/RNN/config_DR_Z2_{}_id{}.json'.format(config['set'],config["id"])) as f:
    config_RNN = json.load(f)

config = merge_two_dicts(config, config_RNN)   #merge configs

with open(DATAPATH + '/CeDR/models/Encoder/_{}_{}.json'.format(config_RNN['compressionModel'], config_RNN['compressionExperiment'])) as f:
    config_encoder = json.load(f)

def get_single_model_RNN(filename):
    
    if config['model'] == 'ConvLSTM':
        model = ConvLSTM(spectrum_size=2048, action_size=config["nr_shims"], hidden_size=config["hidden_size"], 
                     output_size=config["nr_shims"], num_layers_lstm=config["num_layers_lstm"],
                     num_layers_cnn=config["num_layers_cnn"], filters_conv=config["filters_conv"], 
                     stride=config["stride"], kernel_size=config["kernel_size"],
                     pool_size=int(config["pool_size"]), dilation=config["dilation"])
    elif config['model'] == 'LSTMcompressedMeanOnly':
        model = LSTMcompressedMeanOnly(feature_shape = config['latent_dim'], action_size = config['nr_shims'], 
                                       hidden_size = config['hidden_size'], output_size = config['nr_shims'], 
                                       num_layers_lstm = config['num_layers_lstm'])

    model_state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def get_single_model_encoder(filename):

    if config_encoder['model'] == 'VAESmooth':
        model = VAESmooth(config_encoder['past_observations'], config_encoder['latent_dim'], config_encoder['hidden_dims'],
                          config_encoder['kernel_size'], config_encoder['loss'], config_encoder['gaussian_kernel_size'])
    model_state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

model = get_single_model_RNN(DATAPATH+'/CeDR/models/RNN/model_DR_Z2_{}_id{}.pt'.format(config['set'],config["id"]))
model_encoder = get_single_model_encoder(DATAPATH + '/CeDR/models/Encoder/model_{}_{}.pt'.format(config['compressionModel'], config['compressionExperiment']))

loader1 = torch.utils.data.DataLoader(RNN1, batch_size = len(RNN1), shuffle = False)
loader2 = torch.utils.data.DataLoader(RNN2, batch_size = len(RNN2), shuffle = False)


for i_batch, (inputs, targets) in enumerate(loader1):
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)

        targets = torch.div(targets, torch.max(targets, dim = 0).values)
        
        encoded_inputs = model_encoder.encode(inputs[:, :, :2048].view(-1, 1, 2048))
        inputs_encoder = torch.cat((encoded_inputs[0].view(1, inputs.shape[1], -1),encoded_inputs[1].view(1, inputs.shape[1], -1), inputs[:, :, 2048:]), 2)
        (h0,c0) = model.initHidden(batch_size=inputs.shape[0]) # TODO CHECK
        h0, c0 = h0.to(device), c0.to(device)
        outputs, hidden = model(inputs_encoder, (h0,c0))
        outputs = outputs[:,-1]

        mae = np.append(mae, abs(outputs.cpu().detach().numpy())-abs(targets.cpu().detach().numpy()))

















































































