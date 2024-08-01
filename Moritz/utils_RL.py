import gymnasium as gym
import time
import torch
import random 
import numpy as np
import datetime
import matplotlib.pyplot as plt
from collections import deque


if torch.cuda.is_available():
    DEVICE = "cuda:0"
else: DEVICE = "cpu"

#%%
# Ornstein-Ulhenbeck Process
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# (Adding noise to the actions makes sure that the deterministic policy explores!)
# (Original DDPG paper recommends OU noise, but uncorrelated 0-mean Gaussian noise works well and is easier)
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

#%% minimal Replay Memory
class ReplayMemory(object):
    def __init__(self, capacity, torch=True):
        self.memory = deque([],maxlen=capacity) #Deque (Doubly Ended Queue), prefered over list
        self.torch =torch
    def push(self, state, next_state, action, reward, done, info):          # cache
        """Save a transition"""
        state = state.__array__()
        next_state = next_state.__array__()
    
        state = torch.tensor(state).to(DEVICE, dtype=torch.float)
        next_state = torch.tensor(next_state).to(DEVICE, dtype=torch.float)
        action = torch.tensor(action).to(DEVICE, dtype=torch.float)
        reward = torch.tensor([reward]).to(DEVICE, dtype=torch.float)
        done = torch.tensor(done).to(DEVICE, dtype=torch.float)
        current_idx = torch.tensor(info['idx_before_action']).to(DEVICE, dtype=torch.float)
    
        self.memory.append((state, next_state, action, reward, done, current_idx))
    
    def sample(self, batch_size):   # recall     
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done
    
    def __len__(self):
        return len(self.memory)

#%% Memory for LSTM with history instead of hidden states
#https://github.com/Amin-Sz/Deep-Reinforcement-Learning/blob/master/LSTM-TD3/LSTM-TD3_AntPyBulletEnv-v0.py

class ReplayMemoryLSTM(object):
    def __init__(self, mem_size, state_dims, action_dims, history_length):
        self.size = mem_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.history_length = history_length
        self.counter = 0
        self.states = torch.zeros((mem_size, state_dims), dtype=torch.float)
        self.next_state = torch.zeros((mem_size, state_dims), dtype=torch.float)
        self.actions = torch.zeros((mem_size, action_dims), dtype=torch.float)
        self.rewards = torch.zeros((mem_size, 1), dtype=torch.float)
        self.dones = torch.zeros((mem_size, 1), dtype=torch.float)

    def push(self, state, next_state, action, reward, done):
        index = self.counter % self.size
        self.states[index, :] = torch.tensor(state).to(DEVICE, dtype=torch.float)
        self.actions[index, :] = torch.tensor(action).to(DEVICE, dtype=torch.float)
        self.rewards[index] = torch.tensor(reward).to(DEVICE, dtype=torch.float)
        self.next_state[index, :] = torch.tensor(next_state).to(DEVICE, dtype=torch.float)
        self.dones[index] = torch.tensor(done).to(DEVICE, dtype=torch.float)
        self.counter = self.counter + 1

    def sample(self, batch_size):
        indices = np.random.choice(range(self.history_length, np.min([self.size, self.counter])),
                                   size=batch_size, replace=False)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_state = self.next_state[indices]
        dones = self.dones[indices]

        if self.history_length == 0:
            hist_state = torch.zeros((batch_size, 1, self.state_dims))
            hist_action = torch.zeros((batch_size, 1, self.action_dims))
            hist_next_state = torch.zeros((batch_size, 1, self.state_dims))
            hist_next_action = torch.zeros((batch_size, 1, self.action_dims))
            hist_length = torch.ones((batch_size, 1), dtype=torch.int64)
        else:
            hist_state = torch.zeros((batch_size, self.history_length, self.state_dims))
            hist_action = torch.zeros((batch_size, self.history_length, self.action_dims))
            hist_next_state = torch.zeros((batch_size, self.history_length, self.state_dims))
            hist_next_action = torch.zeros((batch_size, self.history_length, self.action_dims))
            hist_length = self.history_length * torch.ones((batch_size, 1), dtype=torch.int64)

            for i, index in enumerate(indices):
                start_index = index - self.history_length
                if start_index < 0:
                    start_index = 0
                if True in self.dones[start_index:index]:
                    start_index = start_index + torch.where(self.dones[start_index:index] == True)[0][-1] + 1

                length = index - start_index
                hist_state[i, 0:length] = self.states[start_index:index, :]
                hist_action[i, 0:length] = self.actions[start_index:index, :]
                hist_next_state[i, 0:length] = self.next_state[start_index:index, :]
                hist_next_action[i, 0:length] = self.actions[start_index + 1:index + 1]
                if length <= 0:
                    length = 1
                hist_length[i] = length

        return states, next_state, actions, rewards, dones, \
               hist_length, hist_state, hist_next_state, hist_action, hist_next_action
       
        
#%% minimal Replay Memory
class ReplayMemoryRNN(object):
    def __init__(self, capacity, torch=True):
        self.memory = deque([],maxlen=capacity) #Deque (Doubly Ended Queue), prefered over list
    def push(self, state, next_state, action, reward, done, hidden):          # cache
        """Save a transition"""
        state = state.__array__()
        next_state = next_state.__array__()
    
        state = torch.tensor(state).to(DEVICE, dtype=torch.float)
        next_state = torch.tensor(next_state).to(DEVICE, dtype=torch.float)
        action = torch.tensor(action).to(DEVICE, dtype=torch.float)
        reward = torch.tensor([reward]).to(DEVICE, dtype=torch.float)
        done = torch.tensor(done).to(DEVICE, dtype=torch.float)
        hidden = torch.tensor(hidden).to(DEVICE, dtype=torch.float)
    
        self.memory.append((state, next_state, action, reward, done, hidden,))
    
    def sample(self, batch_size):   # recall     
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done, hidden = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done, hidden
    
    def __len__(self):
        return len(self.memory)
        
#%% Replay Memory for RNN with TD3 -> hidden for actor, critic1, critic2
class ReplayMemoryTD3(object):
    def __init__(self, capacity, torch=True):
        self.memory = deque([],maxlen=capacity) #Deque (Doubly Ended Queue), prefered over list
        self.torch =torch
    def push(self, state, next_state, action, reward, done, hidden_A, hidden_Q1, hidden_Q2):         # cache
        """Save a transition"""
        state = state.__array__()
        next_state = next_state.__array__()
    
        state = torch.tensor(state).to(DEVICE, dtype=torch.float)
        next_state = torch.tensor(next_state).to(DEVICE, dtype=torch.float)
        action = torch.tensor(action).to(DEVICE, dtype=torch.float)
        reward = torch.tensor([reward]).to(DEVICE, dtype=torch.float)
        done = torch.tensor(done).to(DEVICE, dtype=torch.float)
        hidden_A = torch.tensor(hidden_A).to(DEVICE, dtype=torch.float)
        hidden_Q1 = torch.tensor(hidden_Q1).to(DEVICE, dtype=torch.float)
        hidden_Q2 = torch.tensor(hidden_Q2).to(DEVICE, dtype=torch.float)
    
        self.memory.append((state, next_state, action, reward, done, hidden_A, hidden_Q1, hidden_Q2))
    
    def sample(self, batch_size):   # recall     
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done, hidden_A, hidden_Q1, hidden_Q2 = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done, hidden_A, hidden_Q1, hidden_Q2
    
    def __len__(self):
        return len(self.memory)
 
#%% Modified from  https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# Copyright (c) 2017 OpenAI (http://openai.com)
class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._storage = []
        self._maxsize = capacity
        self._next_idx = 0
        
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        
    def __len__(self): # from parent ReplayMemory
        return len(self._storage)

    def push(self, state, next_state, action, reward, done):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx

        state = state.__array__()
        next_state = next_state.__array__()
    
        state = torch.tensor(state).to(DEVICE, dtype=torch.float)
        next_state = torch.tensor(next_state).to(DEVICE, dtype=torch.float)
        action = torch.tensor(action).to(DEVICE, dtype=torch.float)
        reward = torch.tensor([reward]).to(DEVICE, dtype=torch.float)
        done = torch.tensor(done).to(DEVICE, dtype=torch.float)
        
        data = (state, next_state, action, reward, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        batch = map(self._storage.__getitem__, idxes)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return state, next_state, action, reward, done, weights, idxes

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

class PrioritizedReplayMemoryOffline(object):
    def __init__(self, capacity, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._storage = []
        self._maxsize = capacity
        self._next_idx = 0
        
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        
    def __len__(self): # from parent ReplayMemory
        return len(self._storage)

    def push(self, state, next_state, action, reward, done, info):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx

        state = state.__array__()
        next_state = next_state.__array__()
    
        state = torch.tensor(state).to(DEVICE, dtype=torch.float)
        next_state = torch.tensor(next_state).to(DEVICE, dtype=torch.float)
        action = torch.tensor(action).to(DEVICE, dtype=torch.float)
        reward = torch.tensor([reward]).to(DEVICE, dtype=torch.float)
        done = torch.tensor(done).to(DEVICE, dtype=torch.float)
        current_idx = torch.tensor(info['idx_before_action']).to(DEVICE, dtype=torch.float)
        
        data = (state, next_state, action, reward, done, current_idx)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        batch = map(self._storage.__getitem__, idxes)
        state, next_state, action, reward, done, current_idx = map(torch.stack, zip(*batch))

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return state, next_state, action, reward, done, current_idx, weights, idxes

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
 
#%% 
 
# Used originally in my python DDPG implementation   
# class Memory:
    # def __init__(self, max_size):
        # self.max_size = max_size
        # self.buffer = deque(maxlen=max_size)
    
    # def push(self, state, next_state, action, reward, done):
        # experience = (state, next_state, action, np.array([reward]), done)
        # self.buffer.append(experience)

    # def recall(self, batch_size):
        # state_batch = []
        # action_batch = []
        # reward_batch = []
        # next_state_batch = []
        # done_batch = []

        # batch = random.sample(self.buffer, batch_size)

        # for experience in batch:
            # state, next_state, action, reward, done = experience
            # state_batch.append(state)
            # action_batch.append(action)
            # reward_batch.append(reward)
            # next_state_batch.append(next_state)
            # done_batch.append(done)
        
        # return state_batch, next_state_batch, action_batch, reward_batch, done_batch

    # def __len__(self):
        # return len(self.buffer)
        
        
        
#%% Tree structures 

# From https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
# Copyright (c) 2017 OpenAI (http://openai.com)
import operator

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)        
        
#%% Wrapper for Env to concat n last states 
# https://blog.paperspace.com/getting-started-with-openai-gym/

class ConcatObservations(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k                          # num_past_frames
        self.frames = deque([], maxlen=k)   # double ended queue
        shp = env.observation_space.shape
        self.observation_space = \
            gym.spaces.Box(low=0, high=1, shape=((k,) + shp), dtype=env.observation_space.dtype)

    # mob: modified for determinisitc distortion!
    def reset(self, deterministic_distortion=[]): # concatenate just the initial observations repeatedly
        ob = self.env.reset(deterministic_distortion) if deterministic_distortion != [] else self.env.reset()
        #ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info
    
    def _get_ob(self):
        return np.array(self.frames)
        
# Take last for observations and flatten
class FlattenObservations(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k                          # num_past_frames
        self.frames = deque([], maxlen=k)   # double ended queue
        shp = env.observation_space.shape
        self.observation_space = \
            gym.spaces.Box(low=0, high=1, shape=(int(k * shp[0]),), dtype=env.observation_space.dtype)

    def reset(self): # concatenate just the initial observations repeatedly
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info
    
    def _get_ob(self):
        return np.array(self.frames).flatten()

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedActions(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env): # added by mob
        super().__init__(env)

    def action(self, action): # line changed by mob
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b 
  
#%% Logging
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0 
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3) # 10 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Policy Loss {mean_ep_loss} - "
            f"Mean Actor Loss {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for idx, metric in enumerate(["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]):
            ylabels = [   'Average Reward', 'Average episode length', 'Policy loss', 'Actor loss']
                        
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.xlabel('Episodes x10')
            plt.ylabel(ylabels[idx])
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


#%% with initial = random 
class ConcatObservationsRand(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k                          # num_past_frames
        self.frames = deque([], maxlen=k)   # double ended queue
        shp = env.observation_space.shape
        self.observation_space = \
            gym.spaces.Box(low=0, high=1, shape=((k,) + shp), dtype=env.observation_space.dtype)
            
    # mob: modified for determinisitc distortion!
    def reset(self, deterministic_distortion=[]):
        self.frames = deque([], maxlen=self.k)   # empty history!!
        ob = self.env.reset(deterministic_distortion) if deterministic_distortion != [] else self.env.reset()
        self.frames.append(ob)  # first unshimmed
        while len(self.frames) < self.k:
            # added random actions for reset
            action = np.random.normal(0,1/3,self.env.action_space.shape[0]).clip(-2,2)
            #action = np.random.uniform(-2,2,env.action_space.shape[0])
            ob, reward, done, info = self.env.step(action)
            # catch not in dataset error          
            if not done:
                self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info
    
    def _get_ob(self):
        return np.array(self.frames)


def sigmoid_exploration(x, r_p):
    return 1/(1+np.exp(x-r_p))


def log_objectif(fidSurface, optimal_fidSurface):
    objective = np.log(np.abs(fidSurface - optimal_fidSurface)/np.abs(optimal_fidSurface)).clip(-100, 100)
    return objective