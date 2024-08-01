# -*- coding: utf-8 -*-
"""
Created on Mon 07 Aug 2023

@author: morit

live Reinforcement Learning based shimming with compressed spectra

    - XYZ shims
    - TRAIN script
    
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
from pathlib import Path
from datetime import datetime
from torch import nn
import torch.nn.functional as F
#from sklearn import svm
from sklearn.metrics import mean_absolute_error
from scipy import signal, optimize
from numpy.polynomial.polynomial import polyfit
MYPATH = 'C:/Users/Magritek/Documents/Moritz/'
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data'
DATAPATH_RL = DATAPATH + '/RLshim'
sys.path.append(MYPATH)
import utils_Spinsolve
import utils_RL
from actor_models import Actor_DDPG


import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
parser = argparse.ArgumentParser(description="Run RLshim")
parser.add_argument("--verbose",type=int,default=0) # 0 for no output, 1 for minimal output, 2 for max output
input_args = parser.parse_args()

import warnings
if input_args.verbose < 2: warnings.filterwarnings("ignore")

#https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

initial_config = {
        'mode':                 'train',                                # 'train' or 'eval'
        'nr_shims':             3,
        'id':                   'Real2_16',                             # model's id
        'experiment':           'RLshim_online_10',                      #write n° of experiment
        "sample":               'Ref',                                  # Ref (=5%), Ref10 (10%), H2OCu (+CuSO4), MeOH, H2OTSP (=6mg), H2OTSP50 (50mg,1ml), 
                                                                        #H2OTSP100 (100mg,1ml) (if TSP100: use TSP peak), H2ONic400 (1ml D2O with 400mg Nicotinamid)
        'postshim':             True,                                   # shim after experiment to guarantee same starting points
        'full_memory':          False,                                  # store whole spectrum in additional memory
        "set":                  '3shims',
        "downsample_factor":    1,
        'ROI_min':              16000,                                  # calculated via mean_idx(signal(2048p)>2*noise) * downsamplefactor
        "max_data":             1e5,                                    # *0.68312625, # max_data * max_dataset_value (scaling)
        'scale_sample':         True,                                   # scale first sample in sequence to one (instead of global max)
        'shim_weightings':      [1.2,1,2,0,0,0,0,0,0,0,0,0,0,0,0],      # Shim weighting. range * weighting = shim values
        'acq_range':            50,                                     # range of absolute shim offsets for highest-impact shim (weighting=1)
        
        'compressionModel':     'VAESmooth',
        'compressionExperiment':'TuneReal2_16',
        'actor_model':          'Actor_DDPG',
        
        #Environement
        'optimum':              'lw',                                   # lw or fid
        'target_tolerance':     0.5,                                    # lower threshold for lw !
        #'full_range':          True,                                   #True if max action is 2*ZLIM instead of 1*
        'action_range':         0.3,                                    #2 if we want full range, 1 if we want half range of the whole space
        'bad_action_scale':     2,                                      # scaling of the punishement if we increase the linewidth
        'concat_observations':  4,
        
        #agent parameters
        'hidden_size':          512,
        'critic_learning_rate': 1e-3,
        'actor_learning_rate':  1e-4,                                   #Normaly 1e-4 but I used a specific scheduler, check in agent
        'gamma':                0.55,
        'tau':                  1e-2,
    
        #Reward
        'reward':               'log',                                  #classic or log
        'doneReward':           100,
        'OOBpunishement':       -10,                                    #Out Of Bounds
        'logPunishement':       2,                                      #Only for log reward
        
        #Training
        'episodes':             1500,
    
        #Memory
        'PER' : True,
    }

# VARIABLES

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#with open(DATAPATH + '/RLshim/models/agent/config_DR_Z2_{}_id{}.json'.format(initial_config['set'],initial_config["id"])) as f:
#    config = json.load(f)

#config = merge_two_dicts(initial_config, config)   #merge configs
config = initial_config

with open(DATAPATH + '/CeDR/models/Encoder/_{}_{}.json'.format(config['compressionModel'], config['compressionExperiment'])) as f:
    config_encoder = json.load(f)


# =============================================================================
# # overwrite 
# =============================================================================

if 'H2O' in config['sample'] or 'Ref' in config['sample']:    
    config['ROI_min'] = 16000
    ROLL = 0
    PLT_X = [80,120]
    if 'Ref' in config['sample'] or config['sample'] == 'H2O100': pass

sampling_points = 32768
#pred_averages = 0 # nr of pred_averages for ensemble
#pred_shift_range = 5 # range for z0-shift augmentation for ensemble
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def get_actor(filename):
    
    # TODO

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
    
    
# %%
import gymnasium as gym
from gymnasium import spaces

XLIM = config['shim_weightings'][0]*config['acq_range']
YLIM = config['shim_weightings'][1]*config['acq_range']
ZLIM = config['shim_weightings'][2]*config['acq_range']    

dMax = np.sqrt(XLIM**2 + YLIM**2 + ZLIM**2)


class MyEnvironment(gym.Env):
    def __init__(self, config, npoints=2048, npointsCompressed = config_encoder['latent_dim'], optimum=None):
        super().__init__()
        self.config = config
        self.npoints = npoints
        self.npointsCompressed = npointsCompressed
        
        self.model_encoder = get_single_model_encoder(DATAPATH + '/CeDR/models/Encoder/model_{}_{}.pt'.format(config['compressionModel'], config['compressionExperiment']))

        self.step_counter = 0
        
        # reward
        self.optimal_fidSurface = optimum if config['optimum'] == 'fid' else 1-optimum       # pos fid, neg lw
        self.target_tolerance = self.config['target_tolerance']
        self.max_reward = 100
        self.time_slope = 0     # time penalty. Unused here.

        # initialize first guesses
        self.x_guess = 0
        self.y_guess = 0
        self.z_guess = 0
        #self.z2_guess = 0
        #self.z3_guess = 0      
        
        nr_actions = self.config['nr_shims']
        
        self.action_space = spaces.Box( low=-1, high=1,shape=(nr_actions,), dtype=np.float32)
        # spectrum + action = observation
        self.observation_space = spaces.Box(low=0,high=1, shape=(npointsCompressed+nr_actions,),dtype=np.float32) # TODO check high
        
        
    def step(self, action):
        scaled_action = action *self.config['action_range']*np.array([XLIM, YLIM, ZLIM])
        
        # clip for hardware safety
        scaled_action = np.clip(-1000, scaled_action.astype(int), 1000)
        
        previous_distortion = [self.x_guess, self.y_guess, self.z_guess]
        
        self.x_guess+=scaled_action[0]
        self.y_guess+=scaled_action[1]
        self.z_guess+=scaled_action[2]
        
        # TODO clip guesses to acq_range ?
        
        # SUM of actions/predictions!
        sum_actions_scaled = [self.x_guess, self.y_guess, self.z_guess]
        
        
        if input_args.verbose >= 1: 
            print('artificial distortion (x,y,z,z2): ', self.distortion)
            print('previous distortion (x,y,z,z2): ', previous_distortion)
            print('predicted action (x,y,z,z2): ', scaled_action)
            print('new distortion (x,y,z,z2): ', sum_actions_scaled)
            
        
        
        # important! apply guess and not action!
        xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                                np.add(sum_actions_scaled,self.distortion), True,True, verbose=(input_args.verbose>1))
        if config['full_memory'] and config['mode'] == 'eval': full_memory.append([nr, self.step_counter, self.sc_factor, fid])
        my_arg.count += 1
        spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
        
        # CARE: - pred vs + pred AND guess not action
        self.batched_spectra = np.append( self.batched_spectra, np.concatenate((spectrum_tmp/self.sc_factor, 
                                                            -(sum_actions_scaled/np.array([XLIM, YLIM, ZLIM]))))[np.newaxis,:], axis=0)
        
        self.spectrum = spectrum_tmp
        #######    
        
        if config['optimum'] == 'fid':
            next_fidSurface = np.sqrt(np.mean(np.square(fid.real)))
        else:
            next_fidSurface = 1 - utils_Spinsolve.get_linewidth_Hz(spectrum_tmp, sampling_points=32768, bandwidth = 5000).item()
            
        print('fidSurface after step is {}, optimal is {}'.format(next_fidSurface, self.optimal_fidSurface))#diagnosis
            
        if self.config['reward'] == 'classic':
            # immediate reward as improvement to previous FID surface
            reward = (next_fidSurface/self.optimal_fidSurface - self.current_fidSurface/self.optimal_fidSurface) * self.max_reward
            reward = self.config['bad_action_scale']*reward if reward < 0 else reward       #multiply reward if negative

            done = True if next_fidSurface > self.target_tolerance*self.optimal_fidSurface else False
            
            if done: 
                reward = self.config['doneReward'] + reward       # give additional done reward. (Caution! Seems to cause problems.)
            
            # punish border
            if abs(self.x_guess)>XLIM or abs(self.y_guess)>YLIM or abs(self.z_guess)>ZLIM:
                # DISCUSS: stop or no stop
                d1 = (np.abs(self.x_guess) - XLIM).clip(min=0)
                d2 = (np.abs(self.y_guess) - YLIM).clip(min=0)
                d3 = (np.abs(self.z_guess) - ZLIM).clip(min=0)

                if self.config["stop_at_border"]: 
                    done = True
                    reward = -100
                reward = self.config['OOBpunishement']*np.sqrt(d1**2 + d2**2 + d3**2)/dMax + reward # carry on "badness" of exceeding border
                # reset to lim
                # can be replace with action clipping
                if abs(self.x_guess)>XLIM: self.x_guess = XLIM * np.sign(self.x_guess)
                if abs(self.y_guess)>YLIM: self.y_guess = YLIM * np.sign(self.y_guess)
                if abs(self.z_guess)>ZLIM: self.z_guess = ZLIM * np.sign(self.z_guess)
        
        if self.config['reward'] == 'log':                      #log reward from https://proceedings.mlr.press/v162/kaiser22a.html
            reward = utils_RL.log_objectif(self.current_fidSurface, self.optimal_fidSurface) - utils_RL.log_objectif(next_fidSurface, self.optimal_fidSurface)
            reward = self.config['bad_action_scale']*reward if reward < 0 else reward
            if abs(self.x_guess)>XLIM or abs(self.y_guess)>YLIM or abs(self.z_guess)>ZLIM:
                d1 = (np.abs(self.x_guess) - XLIM).clip(min=0)
                d2 = (np.abs(self.y_guess) - YLIM).clip(min=0)
                d3 = (np.abs(self.z_guess) - ZLIM).clip(min=0)
                if abs(self.x_guess)>XLIM: self.x_guess = XLIM * np.sign(self.x_guess)
                if abs(self.y_guess)>YLIM: self.y_guess = YLIM * np.sign(self.y_guess)
                if abs(self.z_guess)>ZLIM: self.z_guess = ZLIM * np.sign(self.z_guess)
                reward = self.config['OOBpunishement']*np.sqrt(d1**2 + d2**2 + d3**2)/dMax + reward
            done = True if next_fidSurface > self.target_tolerance*self.optimal_fidSurface else False
            
        self.step_counter += 1
        info = {}
        
        if done:
            print('Episode successful, last line width is {}'.format(utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/self.sc_factor, sampling_points=32768, bandwidth = 5000)))#diagnosis
            print('Episode successful, Optimal line width is {}'.format(utils_Spinsolve.get_linewidth_Hz(best_spectrum/self.sc_factor, sampling_points=32768, bandwidth = 5000)))#diagnosis
            print('Corrected Distortion for success', self.distortion + sum_actions_scaled)
        #Save in XL
        self.distortion_success = self.distortion + sum_actions_scaled
        
        if config['mode'] == 'eval': save(spectrum_tmp, sum_actions_scaled, done, np.sqrt(np.mean(np.square(fid.real))), reward)
        
        
        # compress
        spectrum = spectrum_tmp.reshape((1, 1, self.npoints))
        spectrum = torch.tensor(np.divide(spectrum, np.linalg.norm(spectrum, axis = 2, keepdims= True))).float() #normalize spectrum
        encoded = self.model_encoder.encode(spectrum)
        mu = encoded[0].detach().numpy().squeeze()

        self.current_fidSurface = next_fidSurface
        
        return np.concatenate( (mu, np.array(action)) ), reward, done, info
        
    def reset(self, deterministic_distortion=[]):
        self.x_guess = 0
        self.y_guess = 0
        self.z_guess = 0
        self.step_counter = 0
        
        self.distortion = deterministic_distortion if deterministic_distortion != [] else (config['acq_range']*np.random.normal(0, 1/3,size=(config['nr_shims']))*config['shim_weightings'][:config['nr_shims']]).astype(int)
        #self.distortion = deterministic_distortion if deterministic_distortion != [] else (config['acq_range']*np.random.uniform(-1, 1,size=(config['nr_shims']))*config['shim_weightings'][:config['nr_shims']]).astype(int)
        
              
        xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, self.distortion,
                                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
        if config['full_memory'] and config['mode'] == 'eval': full_memory.append([nr, 0, 1, fid])
        my_arg.count += 1
        linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
        initial_spectrum = initial_spectrum[config["ROI_min"]:config["ROI_min"]+2048] / config["max_data"] # scale to dataset

        # norm to first spectrum
        if config['scale_sample']: self.sc_factor = initial_spectrum.max()
        else: self.sc_factor = 1
        
        self.batched_spectra = np.concatenate((initial_spectrum/self.sc_factor, np.zeros(config['nr_shims'])))[np.newaxis,:] # first         
        
        #Save in XL
        self.max_initial_fid = np.max(fid.real)

        if config['optimum'] == 'fid':
            self.current_fidSurface = np.sqrt(np.mean(np.square(fid.real)))
        else:
            self.current_fidSurface = 1 - utils_Spinsolve.get_linewidth_Hz(initial_spectrum, sampling_points=32768, bandwidth = 5000).item()
        
        self.initial_lineWidth = utils_Spinsolve.get_linewidth_Hz(initial_spectrum/self.sc_factor, sampling_points=32768, bandwidth = 5000)
        self.distortion = self.distortion
        
        if config['mode'] == 'eval': save(initial_spectrum, np.zeros(config['nr_shims']), -1, self.current_fidSurface, None)
        
        # compress
        spectrum = initial_spectrum.reshape((1, 1, self.npoints))
        spectrum = torch.tensor(np.divide(spectrum, np.linalg.norm(spectrum, axis = 2, keepdims= True))).float() #normalize spectrum
        encoded = self.model_encoder.encode(spectrum)
        mu = encoded[0].detach().numpy().squeeze()

        return np.concatenate( (mu, np.array([0,0,0])[:self.config['nr_shims']]) )


#%% DDPG
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

# =============================================================================
#         # Care: Still distribution over actions ? 
# =============================================================================
        return x

#https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/44
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('linear'))
        torch.nn.init.constant_(m.bias, 0)   
        
def make_gif(batched_spectra, episode):
    import imageio
    plt.figure()

    steps = 0
    filenames = []
    for i, spectra_aug in enumerate(batched_spectra):
    
        filename = './data/f{}.png'.format(steps)
        filenames.append(filename)
        plt.plot(spectra_aug[:2048])
        plt.title('Frame {}'.format(steps))
        plt.ylim([0,2])
        plt.savefig(filename)
        plt.close()
        steps+=1
    
    with imageio.get_writer('data/RLshim/data/mygif_online_{}_{}.gif'.format(initial_config['experiment'], episode), mode='I', fps=1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

class DDPGagent:
    def __init__(self, env, hidden_size=512, actor_learning_rate=1e-5,
                 critic_learning_rate=2e-4, gamma=0.9, tau=1e-2, max_memory_size=100000):
        # Params
        # Currently: fully connected over flattened input spectra. Change!
        self.num_states = env.observation_space.shape[0]*env.observation_space.shape[1]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.curr_step = 0
        self.hidden_size = hidden_size

        self.critic_loss = -1
        self.policy_loss = -1

        #Diagnosis material
        self.next_Q = -1
        self.Qprime = -1
        self.Qvals = -1
        self.rewards = -1

        self.numSuccessSampled = 0
        self.numSampled = 0
        # Networks

        #calling class from str class name
        #https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object

        self.actor = eval(initial_config['actor_model'])(self.num_states, hidden_size, self.num_actions).to(DEVICE)
        self.actor_target = eval(initial_config['actor_model'])(self.num_states, hidden_size, self.num_actions).to(DEVICE)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1).to(DEVICE)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1).to(DEVICE)

        self.actor.apply(weights_init)
        self.actor_target.apply(weights_init)
        self.critic.apply(weights_init)
        self.critic_target.apply(weights_init)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        #self.memory = utils_RL.ReplayMemory(max_memory_size) 
        self.alpha = 0.9
        self.beta = 0.9
        self.prioritized_replay_eps = 1e-6
        self.memory = utils_RL.PrioritizedReplayMemory(capacity=max_memory_size, alpha=self.alpha) 
  
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        self.curr_step+=1
        
        state = torch.autograd.Variable(torch.from_numpy(state).float().unsqueeze(0)).to(DEVICE)
        action = self.actor.forward(state.flatten()) # TODO change! 
        action = action.detach().cpu().numpy()
        return action
    
    def update(self, batch_size):
# =============================================================================
#         # Care: A2C vs A3C (synchronous vs async. updates)
# =============================================================================
        #states, next_states, actions, rewards, doneState = self.memory.sample(batch_size)
        states, next_states, actions, rewards, doneState, weight, batch_idx = self.memory.sample(batch_size, self.beta)
        self.numSuccessSampled += (rewards >= torch.full((batch_size, 1), 100)).sum()
        self.numSampled += batch_size
        # Care! --> flattened states for FC.
        states, next_states = states.reshape(batch_size, -1), next_states.reshape(batch_size, -1) # !!!
        
        # Critic loss        
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        next_Q = next_Q*(torch.ones(batch_size) - doneState).view(batch_size,1)

        self.next_Q = next_Q
        self.rewards = rewards
        Qprime = rewards + self.gamma * next_Q
        self.Qprime = Qprime

        Qvals = self.critic.forward(states, actions)
        self.Qvals = Qvals

        self.critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        self.policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean() # TODO check! (Checked: similar in spinningup)

        td_errors = (self.Qvals - self.Qprime).cpu().detach().numpy()
        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
        self.memory.update_priorities(batch_idx, new_priorities)


        # update networks
        self.actor_optimizer.zero_grad()
        self.policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        self.critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks (by polyak averaging)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


#%%       
com = utils_Spinsolve.init( verbose=(input_args.verbose>0))

#Evaluate best fid Surface based on 5 experiments
n_eval_ = 5
criteria = np.empty(n_eval_)
for i in range(n_eval_):
# Get best spectrum
    xaxis, best_spectrum, best_fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, np.zeros(config['nr_shims']),
                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
    my_arg.count += 1
    
    if initial_config['optimum'] == 'fid':
        criteria[i] = np.sqrt(np.mean(np.square(best_fid.real)))
    else:
        pass
        criteria[i] = utils_Spinsolve.get_linewidth_Hz(best_spectrum, sampling_points=32768, bandwidth = 5000)
        
optimum = np.max(criteria) if initial_config['optimum'] == 'fid' else np.min(criteria)


#%% TRAIN RL
from collections import deque
from statistics import mean, stdev


env = MyEnvironment(initial_config, optimum = optimum)
env = utils_RL.ConcatObservations(env, 4)

agent = DDPGagent(env,
                   hidden_size=initial_config['hidden_size'],
                   actor_learning_rate=initial_config['actor_learning_rate'],
                   critic_learning_rate=initial_config['critic_learning_rate'], 
                   gamma = initial_config['gamma'],
                   tau = initial_config['tau'])           #!!!
nb_actions = env.action_space.shape[-1]
action_noise = utils_RL.OrnsteinUhlenbeckActionNoise( mu=np.zeros(nb_actions), sigma=([.1] * np.ones(nb_actions)))

initial_config['target_tolerance'] = env.target_tolerance
initial_config['actor_input'] = agent.num_states
initial_config['actor_hidden'] = agent.hidden_size
initial_config['actor_output'] = agent.num_actions

batch_size = 128
rewards = []
critic_loss = []
actor_loss = []
avg_rewards = []
ratioSuccessDrawn = []
LRs = []
linewidths = []
fids = []
best_fids = []

x_success = []
y_success = []
z_success = []

initial_fidSurfaces = []
initial_lineWidths = []
initial_distortions = []
final_distortions = []
maxs_initial_fid = []
successes = []

# XL Analysis
columns_data = ['initialFidSurface', 'initialLineWidth','initialDistortion', 'finalDistortion', 'maxInitialFid', 'done']
data_analysis = {
    'initialFidSurface':initial_fidSurfaces,
    'initialLineWidth':initial_lineWidths,
    'initialDistortion':initial_distortions,
    'finalDistortion':final_distortions,
    'maxInitialFid':maxs_initial_fid,
    'done':successes,
    }
df_analysis = pd.DataFrame(data_analysis)

best_mean_length = float('inf')

episodes = initial_config['episodes']
max_steps = 20
print_every = 10
explore_start = 0.99 # check via explor_decay**episodes
explore_decay = explore_start #current explore factor. 

ep_lengths = deque([],maxlen=100)

logger = utils_RL.MetricLogger(Path(DATAPATH+'/RLshim/data/'))

numSuccess = 0

for episode in range(episodes):
    state = env.reset()
    #Save in XL
    initial_fidSurfaces.append(env.current_fidSurface)
    initial_lineWidths.append(env.initial_lineWidth)
    initial_distortions.append(list(env.distortion))
    
    action_noise.reset() # # Reset internal state after an episode is complete.
    episode_reward = 0
    ep_critic_loss = 0
    ep_actor_loss = 0

    for step in range(max_steps):
        #2D
        print(step, ' step')
        action = agent.get_action(state).clip(-1, 1) # scaling moved to env! 
        
        
        # add noise
        noise = action_noise() # or gaussian noise via action + np.random.randn
        action += noise * explore_decay
        
        # clip actions that would exceed border!
        # replaces termination and negative reward of -100 for border values in env class
        # no "leakage" because shim currents should be know. 
        # However, absolute prediction of currents would loose RL nature.
        #min_possible_action = -np.array([Z1LIM, Z2LIM, Z3LIM])-[env.z_guess,env.z2_guess,env.z3_guess]
        #max_possible_action =  np.array([Z1LIM, Z2LIM, Z3LIM])-[env.z_guess,env.z2_guess,env.z3_guess]
        #action = np.clip(action, min_possible_action, max_possible_action)  

        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, new_state, action, reward, done)
        #print(reward, 'STEP REWARD')
        #env.render()
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)   
            if step == 0 and episode%10==0:
                print(agent.next_Q[:8], 'next_Q') 
                print(agent.rewards[:8], 'reward') 
                print(agent.Qprime[:8], 'Qprime')
                print(agent.Qvals[:8], 'Qvals') 
                print((agent.Qprime[:8] - agent.Qvals[:8]), 'Qprime - Qvals')
                print(agent.policy_loss, 'policy_loss') 
                print(agent.numSuccessSampled, 'Number of Success Drawn')
                print(agent.numSuccessSampled/agent.numSampled, 'ratio')
                print(numSuccess, 'Number of succesfull episodes')
                ratioSuccessDrawn.append(agent.numSuccessSampled/agent.numSampled)
                agent.numSampled = 0
                agent.numSuccessSampled = 0
        
        state = new_state
        episode_reward += reward
        
        logger.log_step(reward,  
                        agent.critic_loss.detach().cpu().numpy() if type(agent.critic_loss) != int else agent.critic_loss,
                        agent.policy_loss.detach().cpu().numpy() if type(agent.policy_loss) != int else agent.policy_loss) # TODO log loss
        
        ep_critic_loss += agent.critic_loss.detach().cpu().numpy() if type(agent.critic_loss) != int else agent.critic_loss
        ep_actor_loss += agent.policy_loss.detach().cpu().numpy() if type(agent.policy_loss) != int else agent.policy_loss

        if done:
            fids.append(env.current_fidSurface)

            best_fids.append(env.optimal_fidSurface)
            plt.plot(fids, label = 'successful fid Surfaces')
            plt.plot(best_fids, label = 'best fid Surfaces line')
            plt.legend()
            plt.xlabel('success episodes')
            plt.ylabel('fidSurfaces')
            plt.savefig(DATAPATH + '/RLshim/data/fidSurfaces_{}'.format(initial_config['experiment']))
            plt.clf()
            break

    #agent.scheduler.step()
    #LRs.append(agent.scheduler.optimizer.param_groups[0]['lr'])
    ep_lengths.append(step)
    if episode%50 == 0:
        make_gif(env.batched_spectra, episode)
    final_distortions.append(list(env.distortion_success))
    if done:
        success = 1
    else:
        success = 0
    
    row_analysis = {
        'initialFidSurface':env.current_fidSurface,
        'initialLineWidth':env.initial_lineWidth,
        'initialDistortion':list(env.distortion),
        'finalDistortion':list(env.distortion_success),
        'maxInitialFid':env.max_initial_fid,
        'done':success,
        }
    df_analysis = df_analysis.append(row_analysis, ignore_index=True)
    df_analysis.to_excel(DATAPATH + '/RLshim/result_analysis_RLshim_id{}_{}_{}.xlsx'.format(
        initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))
    
    explore_decay = explore_start**episode 
    logger.log_episode()
    if mean(ep_lengths) < best_mean_length and episode > 0:
        print('SAVING', mean(ep_lengths))
        best_mean_length = mean(ep_lengths)
        with open(DATAPATH + '/RLshim/_{}_{}_{}D.json'.format(initial_config["actor_model"],
                                                        initial_config["experiment"], 
                                                        initial_config['nr_shims']), 'w') as handle:
            json.dump(str(initial_config), handle)

        torch.save(agent.actor.state_dict(), DATAPATH + '/RLshim/model_{}_{}_{}D.pt'.format(initial_config["actor_model"],
                                                                                        initial_config["experiment"], 
                                                                                        initial_config['nr_shims']))
        
    if episode % print_every == 0: 
        #print("episode: {}, reward: {}, average _reward: {} \n".format(
        #            episode,np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
        logger.record(episode=episode, epsilon=explore_decay, step=agent.curr_step)
        
        
        #reevaluate optimum
        for i in range(n_eval_):
        # Get best spectrum
            xaxis, best_spectrum, best_fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, np.zeros(config['nr_shims']),
                                        return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
            my_arg.count += 1
            if initial_config['optimum'] == 'fid':
                criteria[i] = np.sqrt(np.mean(np.square(best_fid.real)))
            else:
                criteria[i] = utils_Spinsolve.get_linewidth_Hz(best_spectrum, sampling_points=32768, bandwidth = 5000)
                
        optimum = np.max(criteria) if initial_config['optimum'] == 'fid' else np.min(criteria)
        if optimum > env.env.optimal_fidSurface and initial_config['optimum'] == 'fid':
            env.env.optimal_fidSurface = optimum
            print('new best Optimum is {}, former is {}'.format(optimum, env.env.optimal_fidSurface))
        if optimum < env.env.optimal_fidSurface and initial_config['optimum'] == 'lw':
            #env.env.optimal_fidSurface = optimum
            pass 
            # do not update optimum for lw criterion!
        
    rewards.append(episode_reward)
    critic_loss.append(np.round(ep_critic_loss/(step+1), 5))
    actor_loss.append(np.round(ep_actor_loss/(step+1), 5))
    avg_rewards.append(np.mean(rewards[-10:]))
    
    linewidths.append(utils_Spinsolve.get_linewidth_Hz(env.spectrum/env.sc_factor, sampling_points=32768, bandwidth = 5000))
    plt.plot(linewidths)
    plt.xlabel('episode')
    plt.ylabel('linewidths')
    plt.savefig(DATAPATH + '/RLshim/data/linewidths_{}'.format(initial_config['experiment']))
    plt.savefig(DATAPATH + '/RLshim/data/linewidths_{}.svg'.format(initial_config['experiment']))
    plt.close()
    
    maxs_initial_fid.append(env.max_initial_fid)
    plt.plot(maxs_initial_fid/env.env.optimal_fidSurface)
    plt.xlabel('episode')
    plt.ylabel('max_initial_fid')
    plt.savefig(DATAPATH + '/RLshim/data/maxInitialFid_{}'.format(initial_config['experiment']))
    plt.savefig(DATAPATH + '/RLshim/data/maxInitialFid_{}.svg'.format(initial_config['experiment']))
    plt.close()

    plt.figure()
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(DATAPATH + '/RLshim/data/3D_reward_{}'.format(initial_config['experiment']))
    plt.savefig(DATAPATH + '/RLshim/data/3D_reward_{}.svg'.format(initial_config['experiment']))
    plt.close()

    fig, axs = plt.subplots(nrows=2, ncols = 2)
    axs[0, 0].plot(critic_loss)
    axs[0, 0].set_title('critic_loss')
    axs[0, 0].set_xlabel('episodes')
    axs[0, 1].plot(actor_loss)
    axs[0, 1].set_title('actor_loss')
    axs[0, 1].set_xlabel('episodes')
    axs[1, 0].plot(logger.moving_avg_ep_lengths)
    axs[1, 0].set_title('mean_length')
    axs[1, 0].set_xlabel('episodes*{}'.format(print_every))
    axs[1, 1].plot(ratioSuccessDrawn)
    axs[1, 1].set_title('ratio of Success Transition Drawn')
    axs[1, 1].set_xlabel('episodes*{}'.format(print_every))
    plt.savefig(DATAPATH + '/RLshim/data/3D_metrics_{}'.format(initial_config['experiment']))
    plt.close()


shim_weightings = initial_config['shim_weightings']
initial_config['shim_weightings'] = str(initial_config['shim_weightings'])
df = pd.DataFrame(initial_config, index = [0])
df['batch_size'] = batch_size
df['max_steps'] = max_steps
df['explore_start'] = explore_start
df['alpha'] = agent.alpha
df['prioritized_replay_eps'] = agent.prioritized_replay_eps

df['mean_length'] = best_mean_length

df.to_excel(DATAPATH + '/RLshim/data' + '/experiment_results_{}_{}.xlsx'.format(initial_config["experiment"],datetime.today().strftime('%Y-%m-%d %H-%M-%S')))


#utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))

###########################################################################################################################################################################
###########################################################################################################################################################################
#%%
def save(spectrum_tmp, sum_actions_scaled, done, fid_area, reward):
    # TODO change success definition
    success = (spectrum_tmp/env.sc_factor).max() > env.batched_spectra[0,:-config['nr_shims']].max()
    sign_d = np.sign(env.distortion)
    sign_p = np.sign(-np.array(sum_actions_scaled))
    sign_d[np.where(sign_d==0)[0]] = 1 # count 0 as + sign
    sign_p[np.where(sign_p==0)[0]] = 1
    sign = (sign_d==sign_p).sum()/len(env.distortion)
    
    # modified! Conversion problems if configs are not merged properly.
    tmp = config['shim_weightings'].replace('[','').replace(']','')
    tmp = [float(x) for x in tmp.split(', ')][:config['nr_shims']]
    
    mae = mean_absolute_error(env.distortion/np.array(tmp)/config['acq_range'],
                       -np.array(sum_actions_scaled)/np.array(tmp)/config['acq_range'])
    lw_init_50 = utils_Spinsolve.get_linewidth_Hz(env.batched_spectra[0,:-config['nr_shims']], sampling_points=32768, bandwidth = 5000)
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/env.sc_factor, sampling_points=32768, bandwidth = 5000)    
    lw_init_055 = utils_Spinsolve.get_linewidth_Hz(env.batched_spectra[0,:-config['nr_shims']], sampling_points=32768, bandwidth = 5000, height=0.9945)
    lw_shimmed_055 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/env.sc_factor, sampling_points=32768, bandwidth = 5000, height=0.9945)

    results_data.append([nr, step, int(done), int(success), sign, fid_area, reward, env.distortion, sum_actions_scaled, 
                         lw_init_50.item(), lw_shimmed_50.item(), lw_init_055.item(), lw_shimmed_055.item(), mae, env.sc_factor])


['Nr', 'Step', 'Done', 'Success', 'Sign', 'FID_Area', 'Reward', 'Distortion', 'Prediction', 
                'lw50_init', 'lw50_step', 'lw55_init', 'lw55_step', 'MAE', 'ScalingFactor']

#%% TEST RL
# Test RL agent on same random distortions as eDR and CeDR to allow comparison

# set to eval mode
initial_config['mode']='eval'	
env.config=initial_config

with open(DATAPATH_RL + '/_Actor_DDPG_{}_3D.json'.format(initial_config['experiment']), 'rb') as handle:
    config_actor = eval(json.load(handle))

	
actor = eval(config_actor['actor_model'])(config_actor['actor_input'], 
                                          config_actor['hidden_size'],
                                          config_actor['nr_shims']).to(DEVICE)
actor.load_state_dict(torch.load(DATAPATH_RL + '/model_{}_{}_{}D.pt'.format(config_actor["actor_model"],
                                                                            config_actor["experiment"], 
                                                                            config_actor['nr_shims'])))


#config = merge_two_dicts(initial_config, config)   #merge configs
config = initial_config

# load compression model

with open(DATAPATH + '/CeDR/models/Encoder/_{}_{}.json'.format(config['compressionModel'], config['compressionExperiment'])) as f:
    config_encoder = json.load(f)


seed = 45612
nr_evaluations = 50
np.random.seed(seed)
gauss_noise = np.random.normal(0, 1/3,size=(nr_evaluations,config['nr_shims']))
#random_distortions = (np.random.randint(-config['acq_range'],config['acq_range'],size=(nr_evaluations,config['nr_shims']))*config['shim_weightings'][:config['nr_shims']]).astype(int) #discrete uniform
tmp = config['shim_weightings'].replace('[','').replace(']','')
tmp = [float(x) for x in tmp.split(', ')][:config['nr_shims']]
random_distortions = (config['acq_range']*gauss_noise*tmp).astype(int) #discrete uniform

# allow RL agent to take more than 20 max steps
max_steps = 50


global spectra_memory
spectra_memory = []
global full_memory # big memory for whole fid (not only ROI)
full_memory = [] 
global results_data
results_data = []


# Get best spectrum
xaxis, best_spectrum, best_fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, np.zeros(config['nr_shims']),
                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
my_arg.count += 1
spectrum_tmp = best_spectrum[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
plt.figure()
plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp)
plt.title('Best spectrum with {} shims'.format(config['nr_shims']))
plt.xlim(PLT_X) # crop for plotting♠
plt.savefig(DATAPATH + '/RLshim/img_RLshim_{}_best.png'.format(config["sample"]))
plt.show()

# store best before experiment
best50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp, sampling_points=32768, bandwidth = 5000)   
best055 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp, sampling_points=32768, bandwidth = 5000, height=0.9945)
columns_data = ['lw50', 'lw055','Spectrum']
df_spectra = pd.DataFrame([[best50.item(), best055.item(), list(spectrum_tmp)]], columns=columns_data)   

print(best50)
df_spectra.to_excel(DATAPATH + '/RLshim/best_spectrum_RLshim_id{}_{}_{}.xlsx'.format(
    initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))



# loop over all random distortions and track performance
for nr, distortion in enumerate(random_distortions):
    
    step = -1
    state = env.reset(deterministic_distortion=distortion)

    # TODO let agent interact 
    for step, _ in enumerate(range(max_steps)):
        print(step, ' step')
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
		# query only trained actor from RL agent
        action = actor.forward(state.flatten()) # TODO change! 
        action = action.detach().cpu().numpy().clip(-1, 1)
		
        new_state, reward, done, _ = env.step(action) 
        state = new_state
        
        if done: 
			# TODO read out batched_spectra
            break

    # !! I change step-1 to step-2 in np.repeat back to step
    random_array = np.append(np.append(np.array([-1]),np.repeat(0,step)), np.array([1]))
    for i, tmp in enumerate(env.batched_spectra):
        #print(i, random_array[i])
        spectra_memory.append([nr, i, random_array[i], env.sc_factor, list(tmp[:2048]), list(tmp[-config['nr_shims']:])])


# Convert results to pd df
columns_data = ['Nr', 'Step', 'Done', 'Success', 'Sign', 'FID_Area', 'Reward', 'Distortion', 'Prediction', 
                'lw50_init', 'lw50_step', 'lw55_init', 'lw55_step', 'MAE', 'ScalingFactor']
df = pd.DataFrame(results_data, columns=columns_data)
df.to_excel(DATAPATH + '/RLshim/results_RLshim_id{}_{}_{}.xlsx'.format(
        initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

    
# print results
print("Success rate: ", df.loc[df['Done']==1]['Success'].mean())
print("Correct prediction rate: ", round( df.loc[df['Done']==1]['Sign'].mean() , 3))
#print("Mean criterion improvement: {} {} % +/- {}".format( ('+' if np.mean(mean_c)>1 else '-'), round((np.mean(mean_c)-1)*100, 2), round(np.abs((np.std(mean_c)-1)*100),2)) )
print("Averaged MAE: {} +/- {}".format(round(df.loc[df['Done']==1]['MAE'].mean(),2), round(df.loc[df['Done']==1]['MAE'].mean(),2)) )

# convert memory to pd df
# Excel cell limit is 32767 --> Store as pickle
columns_data = ['Nr', 'Step', 'Done', 'ScalingFactor', 'Spectrum', 'ShimOffsets']
df_spectra = pd.DataFrame(spectra_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/RLshim/spectra_memory_RLshim_id{}_{}_{}.pickle'.format(
    initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

# convert memory to pd df
columns_data = ['Nr', 'Step', 'ScalingFactor', 'Spectrum']
df_spectra = pd.DataFrame(full_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/RLshim/spectra_memoryFULL_RLshim_id{}_{}_{}.pickle'.format(
    initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

import json
with open(DATAPATH + '/RLshim/config_RLshim_id{}_{}_{}.json'.format(
    initial_config["id"],config["sample"],
    datetime.today().strftime('%Y-%m-%d-%HH-%mm')), 'w') as f:
    json.dump(str(config),f)

#%% run automated shim to guarantee same starting point for next iteration
if config['postshim']:
    print('Post-shimming #1. Please wait...')
    com.RunProspaMacro(b'QuickShim()')
    com.RunProspaMacro(b'gExpt->runExperiment()')
    print('Post-shimming #2. Please wait...')
    com.RunProspaMacro(b'QuickShim()')
    com.RunProspaMacro(b'gExpt->runExperiment()')
    

#%% Test run2 after shimming
# Store with 'EVAL0'

spectra_memory = []
full_memory = [] 
results_data = []


# loop over all random distortions and track performance
for nr, distortion in enumerate(random_distortions):
    
    step = -1
    state = env.reset(deterministic_distortion=distortion)

    # TODO let agent interact 
    for step, _ in enumerate(range(max_steps)):
        print(step, ' step')
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
		# query only trained actor from RL agent
        action = actor.forward(state.flatten()) # TODO change! 
        action = action.detach().cpu().numpy().clip(-1, 1)
		
        new_state, reward, done, _ = env.step(action) 
        state = new_state
        
        if done: 
			# TODO read out batched_spectra
            break

    # !! I change step-1 to step-2 in np.repeat back to step
    random_array = np.append(np.append(np.array([-1]),np.repeat(0,step)), np.array([1]))
    for i, tmp in enumerate(env.batched_spectra):
        #print(i, random_array[i])
        spectra_memory.append([nr, i, random_array[i], env.sc_factor, list(tmp[:2048]), list(tmp[-config['nr_shims']:])])


# Convert results to pd df
columns_data = ['Nr', 'Step', 'Done', 'Success', 'Sign', 'FID_Area', 'Reward', 'Distortion', 'Prediction', 
                'lw50_init', 'lw50_step', 'lw55_init', 'lw55_step', 'MAE', 'ScalingFactor']
df = pd.DataFrame(results_data, columns=columns_data)
df.to_excel(DATAPATH + '/RLshim/results_RLshim_EVAL0_id{}_{}_{}.xlsx'.format(
        initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

    
# convert memory to pd df
# Excel cell limit is 32767 --> Store as pickle
columns_data = ['Nr', 'Step', 'Done', 'ScalingFactor', 'Spectrum', 'ShimOffsets']
df_spectra = pd.DataFrame(spectra_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/RLshim/spectra_memory_RLshim_EVAL0_id{}_{}_{}.pickle'.format(
    initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

# convert memory to pd df
columns_data = ['Nr', 'Step', 'ScalingFactor', 'Spectrum']
df_spectra = pd.DataFrame(full_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/RLshim/spectra_memoryFULL_RLshim_EVAL0_id{}_{}_{}.pickle'.format(
    initial_config["id"],config["sample"],datetime.today().strftime('%Y-%m-%d-%HH-%mm')))


utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))
