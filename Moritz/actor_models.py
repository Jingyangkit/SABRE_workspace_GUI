import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class Actor_DDPG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor_DDPG, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        #x = self.linear4(x)
        x = torch.tanh(self.linear4(x))

        return x
    

class Actor_LSTM(nn.Module):
    def __init__(self, state_shape, hidden_size, output_size, num_layers_lstm=2, 
                 drop_p_fc=0):
        super().__init__()
        self.num_layers_lstm = num_layers_lstm
        self.drop_p_fc = drop_p_fc
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.state_shape = state_shape
        
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ln_features = nn.LayerNorm(self.state_shape)
        
        self.lstm = nn.LSTM(self.state_shape, hidden_size, self.num_layers_lstm,
                            batch_first=True)
        self.ln_lstm = nn.LayerNorm(hidden_size)
        
        # i2o
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p_fc)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state, hidden):
        #Input has shape (batch, sequence_length, latent_dim+label_size)
        #in this short version, the STD of the VAE has been removed
        past_obs = state.shape[1]

        mean_part = state[:, :, :self.state_shape]
        fc_part = state[:,:, self.state_shape:]          # shim actions/values

        sampled_features = torch.from_numpy(np.zeros([state.shape[0], past_obs, self.state_shape])).float().to(self.DEVICE)
        for k in range(past_obs):                           # sample from each distribution
            sampled_features[:, k] = mean_part[:, k]

        #features = self.i2f(conv_part)
        combined = torch.cat((sampled_features, fc_part), 2)
        combined = self.ln_features(combined)
        
        # LSTM + i2h
        out, (h0,c0) = self.lstm(combined, hidden)
        out = self.ln_lstm(out)
        
        #i2o
        x = self.relu(self.fc1(out))
        x = self.drop(x)
        x = self.tanh(self.fc2(x))
        return x, (h0,c0)

    def initHidden(self, batch_size=1):
        return ( torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size), torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size) )


#pnn
#https://github.com/hengdashi/pnn/blob/main/src/model/pnn.py
#I don't need v and alpha for sim to real it seems
#https://arxiv.org/abs/1610.04286

class PNNColumn(nn.Module):
    def __init__(self, cid, input_size, hidden_size, output_size, previous_size):
        super(PNNColumn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cid = cid
        self.previous_size = previous_size  # [[512, 512, 512, 3]]
        # 6 layers neural network
        self.nlayers = 4

        # init normal nn, lateral connection, adapter layer and alpha
        self.w = nn.ModuleList()  # normal network
        self.u = nn.ModuleList()

        # normal neural network
        self.w.extend(
            [
                nn.Linear(input_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, output_size),
            ]
        )

        # only add lateral connections and adapter layers if not first column
        # v[col][layer][(nnList on that layer)]
        for i in range(self.cid):
            # lateral connection
            self.u.append(nn.ModuleList())
            self.u[i].extend(
                [
                    nn.Linear(previous_size[i][k], self.w[k].out_features)
                    for k in range(self.nlayers)
                ]
            )

        # init weights
        self._init_new_column()

    def forward(self, x, pre_out):
        """feed forward process for a single column"""
        # put a placeholder to occupy the first layer spot
        next_out, w_out, u_out = [], x, []

        # pass input layer by layer
        out = None
        for i in range(self.nlayers):
            # pass into normal network
            w_out = self.w[i](w_out)
            # u, alpha, v are only valid if cid is not zero
            # summing over for all networks from previous cols
            # u[k][i]: u network for ith layer kth column
            u_out = [
                self.u[k][i](pre_out[k][i]) if self.cid else torch.zeros(w_out.shape)
                for k in range(self.cid)
            ]
            if i == self.nlayers - 1:
                w_out = w_out + sum(u_out)
                next_out.append(w_out)
                w_out = torch.tanh(w_out)
            else:
                w_out = w_out + sum(u_out)
                next_out.append(w_out)
                w_out = self._activate(w_out)


        # TODO: do we need information from previous columns or not?
        return w_out, next_out
        #  return out, next_out

    def _activate(self, x):
        return F.relu(x)

    def _init_new_column(self):
        for feedForward in self.w:
            nn.init.zeros_(feedForward.weight)
            nn.init.zeros_(feedForward.bias)
            if self.cid == 0:
                nn.init.xavier_uniform_(feedForward.weight)

        for i in range(self.cid):
            for j in range(self.nlayers):
                nn.init.zeros_(self.u[i][j].weight)
                nn.init.zeros_(self.u[i][j].bias)
        if self.cid > 0:
            nn.init.eye_(self.u[-1][-1].weight)

class PNN(nn.Module):
    """Progressive Neural Network"""

    def __init__(self, sizes):
        # sizes = [
        # [input_size1, hidden_size_1, output_size1],
        # [input_size2, hidden_size_2, output_size2]
        # ]
        super(PNN, self).__init__()
        # current column index
        self.current = 0
        # complete network
        self.columns = nn.ModuleList()

        input_size, hidden_size, output_size = 0, 0, 0

        for i, list in enumerate(sizes):
            previous_size = [3 * [hidden_size] + [output_size]]
            input_size = list[0]
            hidden_size = list[1]
            output_size = list[2]
            self.columns.append(
                PNNColumn(i, input_size, hidden_size, output_size, previous_size)
            )
            # freeze parameters that is not on first column
            if i != 0:
                for params in self.columns[i].parameters():
                    params.requires_grad = False

    def forward(self, X, Y):
        """
        PNN forwarding method
        X is the state of the current environment being trained
        """
        pre_out = []
        
        '''
        for i in range(self.current + 1):
            out, next_out = self.columns[i](X, pre_out)
            pre_out.append(next_out)
        '''
        out, next_out = self.columns[0](X, pre_out)
        pre_out.append(next_out)

        out, next_out = self.columns[1](Y, pre_out)
        pre_out.append(next_out)
        
        return out

    def freeze(self):
        """freeze previous columns"""
        self.current += 1

        if self.current >= len(self.columns):
            return

        # freeze previous columns
        for i in range(self.current + 1):
            for params in self.columns[i].parameters():
                params.requires_grad = False

        # enable grad on next column
        for params in self.columns[self.current].parameters():
            params.requires_grad = True

    def parameters(self, cid=None):
        """return parameters of the current column"""
        if cid is None:
            return super(PNN, self).parameters()
        return self.columns[cid].parameters()

'''
sizes = [[76, 512, 3], [76, 128, 3]]
pnn = PNN(sizes)
ten = torch.ones(76)
pnn(ten)
pnn.freeze()
pnn(ten)
'''

