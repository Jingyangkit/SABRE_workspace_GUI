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
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn
import torch.nn.functional as F
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
#parser.add_argument("--meta",type=str,default='mlp')         # mlp, fc, average, none or none_tuned
parser.add_argument("--sample",type=str,default='H2OCu')    #H2OCu, EtOH-0.1, EtOH-0.5
input_args = parser.parse_args()


plt.style.use(['science', 'nature', 'high-contrast'])
plt.rcParams.update({"font.family": "sans-serif",})

#%% Define user and constant variables

class Arguments():
    def __init__(self):
        self.count = 0                      # experiment counter

        self.sample = input_args.sample     # sample type
        #self.base_models = '/cascade_files/' # name of ray results pickle file
        #self.nr_models = 50                 # nr weak learners
        self.downsample_factor = 16         # downsample factor
        self.label_scaling = 100            # label scaling used to avoid exploding and vanishing gradients
        self.max_data = 1e5                 # max data used in preprocessing
        #self.meta_type = input_args.meta          # mlp or fc
        self.channels = 4                   # 4 channels per input
        self.offset_values = [1000, 100, 10]            # step size s
        self.sampling_points = 32768
        self.device = "cpu"                 # Inference device
        
        self.configs = []
        self.sets = ['coarse', 'fine', 'finest']
        for i,s in enumerate(self.sets):
            with open(DATAPATH + '/cascade_files/config_DR_{}.json'.format(s)) as f:
                c = json.load(f)
            self.configs.append(c)

        self.nr_evaluations = 2           # Nr. averages for random distortions
        self.seed = 45612                   # Seed to set random distortions

my_arg = Arguments()

np.random.seed(my_arg.seed)
random_distortions = np.random.randint(-10000,10000,size=(my_arg.nr_evaluations,3)) #discrete uniform

def seed_everything(seed):
    #random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SEED = 5
seed_everything(SEED)

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
                nn.Dropout(drop_p))
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

def get_single_model(set_str, ray_results):
    with open(DATAPATH + ray_results, 'rb') as f:
        res = pickle.load(f)
    res_ascending = res.sort_values('loss')
    dir_top1 = res_ascending['logdir'][:1]
    # change paths to Magritek PC
    dir_top1 = [DATAPATH + '/ray_results/'+d for d in dir_top1][0]
    with open(dir_top1+'\params.json', 'rb') as f:
        config = json.load(f)
    model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),my_arg.channels,my_arg.sampling_points/my_arg.downsample_factor),
                        num_classes = 3, kernel_size=int(config["kernel_size"]),stride=int(config["stride"]),
                        pool_size=int(config["pool_size"]), num_layers=int(config["num_layers"]),
                        drop_p_conv = config["drop_p_conv"], drop_p_fc = config["drop_p_fc"])
    last_checkpoint = glob.glob(dir_top1+'\checkpoint_*')[-1]
    checkpoint_path = os.path.join(last_checkpoint, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device(my_arg.device))
    model.load_state_dict(model_state)
    model.to(my_arg.device)
    model.eval()
    return model

def get_batched_spectrum(distortion, ref_shims, initial_spectrum, channels, offset_value):
    batch = np.zeros([1, channels, int(my_arg.sampling_points/my_arg.downsample_factor)])
    def apply_and_measure(offset, label, distortion=distortion):
            # distort shims and do standard proton experiment. returns x-axis in Hz, spectrum and shim values
            xaxis, spectrum, shims = utils_Spinsolve.setShimsAndRun(com, my_arg.count, np.add(offset,distortion), True, verbose=(input_args.verbose>1))
            if input_args.verbose == 2: print('shims: ', shims)
            plt.plot(xaxis[MIN_IDX:MAX_IDX], spectrum[MIN_IDX:MAX_IDX], '--', label='$u+{}$'.format(label), alpha=0.5) if offset!=[0,0,0] else plt.plot(xaxis, spectrum, ':', label='{}'.format(offset))
            my_arg.count += 1
            if input_args.verbose == 2: print(my_arg.count)
            return spectrum
    batch[0,0] = initial_spectrum[::my_arg.downsample_factor]
    batch[0,1] = apply_and_measure([offset_value,0,0],'x')[::my_arg.downsample_factor]/my_arg.max_data
    batch[0,2] = apply_and_measure([0,offset_value,0],'y')[::my_arg.downsample_factor]/my_arg.max_data
    batch[0,3] = apply_and_measure([0,0,offset_value],'z')[::my_arg.downsample_factor]/my_arg.max_data
    return batch


def shot(distortion, my_arg, step):
    global SPECTRA_MEMORY
    
    set_as_string = my_arg.sets[step]
    config = my_arg.configs[step]
    offset_value = my_arg.offset_values[step]
    
    if input_args.verbose == 2: print('config', config)

    #model = get_single_model(set_as_string, ray_results)
    
    # load fixed model !
    model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),my_arg.channels,my_arg.sampling_points/my_arg.downsample_factor),
                        num_classes = 3, kernel_size=int(config["kernel_size"]),stride=int(config["stride"]),
                        pool_size=int(config["pool_size"]), num_layers=int(config["num_layers"]),
                        drop_p_conv = config["drop_p_conv"], drop_p_fc = config["drop_p_fc"])
    state_dict = torch.load(DATAPATH+'/cascade_files/model_DR_{}.pt'.format(set_as_string), map_location=torch.device(my_arg.device))
    # create new OrderedDict that does not contain `module.
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.to("cpu")
    model.eval()

    # distort shims and do standard proton experiment. returns x-axis in Hz, spectrum and reference shim values
    xaxis, initial_spectrum, ref_shims = utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion, return_shimvalues=True, verbose=(input_args.verbose>1))
    # measure FWHM in Hz
    linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
    initial_spectrum = initial_spectrum / my_arg.max_data # scale to dataset
    ref_shims = ref_shims[:3]
    my_arg.count += 1

    if input_args.verbose == 2: print('ref shims: ', ref_shims)

    plt.figure()
    plt.plot(1,1,alpha=0) # add 1 to color cycle
    tmp = initial_spectrum*my_arg.max_data
    ns = tmp[int(3*len(tmp)/4):-1]
    global MIN_IDX
    global MAX_IDX
    MIN_IDX, MAX_IDX = np.where(tmp>(ns.mean()+0.25*ns.mean()))[0][0], np.where(tmp>(ns.mean()+0.25*ns.mean()))[0][-1]
    plt.plot(xaxis[MIN_IDX:MAX_IDX], tmp[MIN_IDX:MAX_IDX], label = 'unshimmed $u$')

    batched_spectra = get_batched_spectrum(distortion, ref_shims, initial_spectrum, my_arg.channels, offset_value)
    for tmp in batched_spectra[0]: SPECTRA_MEMORY.append(tmp)
    if input_args.verbose == 2: print('shape batched', batched_spectra.shape)

    # make prediction
    running_pred = []
    prediction = model(torch.tensor(batched_spectra).float())
    running_pred.append(prediction.detach().numpy()[0]*my_arg.sampling_points/my_arg.label_scaling)
    tx,ty,tz = np.mean(running_pred, axis=0).astype(int)

    #scale to real value and clip to prevent currents to harm the hardware.
    tx = sorted((-10000,tx,10000))[1]
    ty = sorted((-10000,ty,10000))[1]
    tz = sorted((-10000,tz,10000))[1]

    if input_args.verbose >= 1:
        print('artificial distortion (x,y,z): ', distortion)
        print('predicted correction (x,y,z): ',tx,ty,tz)

    # distort shims and do standard proton experiment. returns x-axis in Hz and spectrum
    xaxis, shimmed_spectrum = utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion+[-tx,-ty,-tz], verbose=(input_args.verbose>0))
    # measure FWHM in Hz
    linewidth_shimmed = utils_Spinsolve.get_linewidth_Hz(shimmed_spectrum)
    SPECTRA_MEMORY.append(shimmed_spectrum[::my_arg.downsample_factor]/my_arg.max_data)

    # plot FWHM
    try:
        if True:
            if False:
                tmp = initial_spectrum*my_arg.max_data
                w,h,start,stop = signal.peak_widths(tmp, signal.find_peaks(tmp, height = tmp.max()*0.9, distance=1000)[0])
                start, stop = (start-2**15/2)/2**15*2e4, (stop-2**15/2)/2**15*2e4
                plt.hlines(h,start,stop, colors="grey", linestyles='dotted')
                plt.annotate('\scriptsize{}Hz'.format(int(linewidth_initial.item())), [stop+w/2/2**15*2e4,h+50], va='center')
            w,h,start,stop = signal.peak_widths(shimmed_spectrum, signal.find_peaks(shimmed_spectrum, height = shimmed_spectrum.max()*0.9, distance=1000)[0])
            start, stop = (start-2**15/2)/2**15*2e4, (stop-2**15/2)/2**15*2e4
            plt.hlines(h,start,stop, colors="grey", linestyles='dotted')
            plt.annotate('\scriptsize{}Hz'.format(int(linewidth_shimmed.item())), [stop+w/10/2**15*2e4,h], va='center')
        plt.plot(xaxis[MIN_IDX:MAX_IDX], shimmed_spectrum[MIN_IDX:MAX_IDX], label='shimmed')
        plt.legend()
        plt.title('Shimming with DRE using meta type {}'.format( my_arg.meta_type)) if my_arg.meta_type != 'none_tuned' else plt.title('Shimming with DRE using meta type none tuned') # latex formating errors
        plt.ylabel("Signal [a.u.]")
        plt.xlabel("Frequency [Hz]")
        plt.savefig(DATAPATH + '/DRE/img_dre_{}_{}_{}.pdf'.format(my_arg.sample,my_arg.meta_type, round(my_arg.count/4)))
        plt.savefig(DATAPATH + '/DRE/img_dre_{}_{}_{}.png'.format(my_arg.sample,my_arg.meta_type, round(my_arg.count/4)))
        if input_args.verbose > 0: plt.show()
    except TypeError:
        pass
    except AttributeError:
        pass
    except ValueError:
        pass


    return [tx,ty,tz], linewidth_initial, linewidth_shimmed

# if prediction is not improving criterion, undo and check whether spectra in batch is better.
def checkup(prediction, distortion, offset_value):
    global SPECTRA_MEMORY
    shims = [[0,0,0], [offset_value,0,0], [0,offset_value,0], [0,0,offset_value]] # standard shims for batch creation
    recent_set = np.array(SPECTRA_MEMORY[-5:])
    initial = recent_set[0]

    min_width = signal.peak_widths(initial, signal.find_peaks(initial, height = initial.max()*0.9, distance=1000)[0])[0].item()
    max_peak_height = initial.max()
    if input_args.verbose >= 1: print("Ref. width, height: {} {}".format(min_width, max_peak_height))

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
        # distort shims and do standard proton experiment. returns x-axis in Hz, spectrum and shim values
        xaxis, spectrum = utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion-prediction, verbose=(input_args.verbose>0))
        # apply shims if offsetting in one direction improved the shimming
        if best != 0:
            # use best offsets for shim correction and do standard proton experiment. returns x-axis in Hz, spectrum and shim values
            utils_Spinsolve.setShimsAndRun(com, my_arg.count, distortion+shims[best], verbose=(input_args.verbose>0))
        if input_args.verbose >= 1: print("Taking {}. offset for best setting. ".format(np.array(['ref', 'x', 'y', 'z'])[best]))
        if input_args.verbose >= 1: print("Distortion before ", distortion)
        distortion += shims[best]
        if input_args.verbose >= 1: print("Distortion after ", distortion)

    return (best == 4), criteria[-1]

global SPECTRA_MEMORY
SPECTRA_MEMORY = []

# Initialize Python-Prospa interface
com = utils_Spinsolve.init( verbose=(input_args.verbose>0) )

# Problem: When to switch? 
# Cannot use distortion range as black box
# Use lw ?


# check fine alone
random_distortions = np.random.randint(-2000,2000,size=(my_arg.nr_evaluations,3)) #discrete uniform
step = 1
for d in random_distortions:    
    pred, lw_init, lw_shimmed = shot(d, my_arg, step)
    success, criterion_after = checkup(pred, d, offset_value=my_arg.offset_values[step])
    utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))
    exit()
    
###############################################################################################
###############################################################################################

step = 0
for d in random_distortions:    
    pred, lw_init, lw_shimmed = shot(d, my_arg, step)
    success, criterion_after = checkup(pred, d, offset_value=my_arg.offset_values[step])
    step += 1
    d = d - pred
    pred, lw_init, lw_shimmed = shot(d, my_arg, step)
    success, criterion_after = checkup(pred, d, offset_value=my_arg.offset_values[step])
    step += 1
    d = d - pred
    pred, lw_init, lw_shimmed = shot(d, my_arg, step)
    success, criterion_after = checkup(pred, d, offset_value=my_arg.offset_values[step])

    utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))
    exit()


###############################################################################################
###############################################################################################

results_array = []
success_rate = 0
sign_rate = 0
mean_c = []
mae = []

# loop over all random distortions and track performance
for d in random_distortions:
    step = 1
    pred, lw_init, lw_shimmed = shot(d, my_arg, step)
    success, criterion_after = checkup(pred, d, offset_value=my_arg.offset_values[step])
    success_rate += success
    sign_rate += (np.sign(d)==np.sign(pred)).sum()
    mean_c.append(criterion_after)
    mae.append(mean_absolute_error(d,pred))
    results_array.append(['dist: {}, pred: {}, lw0.5: {} -> {}'.format(d, pred, lw_init, lw_shimmed)])

# print results
print("Success rate: ", success_rate/len(random_distortions))
print("Correct prediction rate: ", round(sign_rate/(len(random_distortions)*3),3) )
print("Mean criterion improvement: {} {} % +/- {}".format( ('+' if np.mean(mean_c)>1 else '-'), round((np.mean(mean_c)-1)*100, 2), round(np.abs((np.std(mean_c)-1)*100),2)) )
print("Averaged MAE: {} +/- {}".format(round(np.mean(mae),1), round(np.std(mae),1)) )


# save results to file
with open(DATAPATH + '/DRE/results_dre_{}_{}_{}.txt'.format(my_arg.sample,my_arg.meta_type, datetime.now().timestamp()), 'w') as f:
    f.write("Success rate: {}".format(success_rate/len(random_distortions)))
    f.write("\n")
    f.write("Correct prediction rate: {}".format( round(sign_rate/(len(random_distortions)*3),3)))
    f.write("\n")
    f.write("Mean criterion improvement: {} {} % +/- {}".format( ('+' if np.mean(mean_c)>1 else '-'), round((np.mean(mean_c)-1)*100, 2), round(np.abs((np.std(mean_c)-1)*100),2)))
    f.write("\n")
    f.write("Averaged MAE: {} +/- {}".format(round(np.mean(mae),1), round(np.std(mae),1)))
    f.write("\n")
    for item in results_array:
        f.write(str(item))
        f.write("\n")

with open(DATAPATH + '/DRE/SPECTRA_MEMORY_dre_{}_{}_{}.txt'.format(my_arg.sample,my_arg.meta_type, datetime.now().timestamp()), 'w') as f:
    for item in SPECTRA_MEMORY:
        for i in item:
            f.write(str(i) + ' ')
        f.write("\n")

# Shutdown Python-Prospa interface
utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))