import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

#https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/44
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('tanh'))
        torch.nn.init.constant_(m.bias, 0)


class MyCNN_Class(nn.Module):
    def __init__(self, num_classes, input_shape, drop_p=.5): # in: (1,1,32768)
        super().__init__()
        self.drop_p = drop_p
        self.drop = torch.nn.Dropout(drop_p)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=4), # out (1,32,8190)
            #nn.BatchNorm1d(32),
            nn.ReLU())
        self.outshape = int( (input_shape[2]-11)/4 +1)
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, stride=4), # out (1,64,2045)
            #nn.BatchNorm1d(64),
            nn.ReLU())
        self.outshape = int( (self.outshape-11)/4 +1)
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, stride=4), # out ?
            #nn.BatchNorm1d(128),
            nn.ReLU())
        self.outshape = int( (self.outshape-11)/4 +1)
        self.linear1 = nn.Linear(self.outshape*128, num_classes)
    def forward(self, x):
        x = self.layer1(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = self.drop(x)
        x = self.layer3(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)
        return self.linear1(x)


class MyCNN3v2_Regr(nn.Module):
    def __init__(self,  input_shape, num_classes=3, drop_p=.5): # in: (1,1,32768)
        super().__init__()
        self.kernel_size = 11
        self.stride = 4
        self.drop_p = drop_p
        self.drop = torch.nn.Dropout(drop_p)
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_shape[1], 64, kernel_size=self.kernel_size, stride=self.stride), # out (1,32,8190)
            #nn.BatchNorm1d(32),
            nn.ReLU())
        self.outshape = int( (input_shape[2]-self.kernel_size)/self.stride +1)
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=self.kernel_size, stride=self.stride), # out (1,64,2045)
            #nn.BatchNorm1d(64),
            nn.ReLU())
        self.outshape = int( (self.outshape-self.kernel_size)/self.stride +1)
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=self.kernel_size, stride=self.stride), # out ?
            #nn.BatchNorm1d(128),
            nn.ReLU())
        self.outshape = int( (self.outshape-self.kernel_size)/self.stride +1)
        self.linear1 = nn.Linear(self.outshape*256, num_classes)
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = self.drop(x)
        x = self.layer3(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)
        return self.linear1(x)

class MyCNN3v3_Regr(nn.Module):
    def __init__(self,  input_shape, num_classes=3, drop_p=.2, kernel_size=49, stride=2, pool_size=2, filters=64):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_size = pool_size
        self.drop_p = drop_p
        self.filters = filters
        self.drop = torch.nn.Dropout(drop_p)
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_shape[1], 64, kernel_size=self.kernel_size, stride=self.stride), 
            #nn.BatchNorm1d(32),
            nn.ReLU())
        self.outshape = int( (input_shape[2]-self.kernel_size)/self.stride +1 )
        if self.pool_size > 1:
            self.pool = nn.MaxPool1d(self.pool_size)
            self.outshape = int(self.outshape/2)
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.filters, self.filters, kernel_size=self.kernel_size, stride=self.stride),
            #nn.BatchNorm1d(64),
            nn.ReLU())
        self.outshape = int( (self.outshape-self.kernel_size)/self.stride +1)
        if self.pool_size > 1: self.outshape = int(self.outshape/2)
        self.layer3 = nn.Sequential(
            nn.Conv1d(self.filters, self.filters, kernel_size=self.kernel_size, stride=self.stride),
            #nn.BatchNorm1d(128),
            nn.ReLU())
        self.outshape = int( (self.outshape-self.kernel_size)/self.stride +1)
        if self.pool_size > 1: self.outshape = int(self.outshape/2)
        self.linear1 = nn.Linear(self.outshape*self.filters, self.filters)
        self.linear2 = nn.Linear(self.filters, num_classes)
    def forward(self, x):
        #x = x.unsqueeze(1)
        if self.pool_size > 1: x = self.pool(self.layer1(x))
        else: x = self.layer1(x)
        x = self.drop(x)
        if self.pool_size > 1: x = self.pool(self.layer2(x))
        else: x = self.layer2(x)
        x = self.drop(x)
        if self.pool_size > 1: x = self.pool(self.layer3(x))
        else: x = self.layer3(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        return self.linear2(x)
        
  
  
           
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
                nn.Dropout(drop_p)
            )
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
        

# same as models.MyCNNflex_Regr but with bugfix of dropout layer order and bugfix of pooling
# purpose: DQN for RL
class MyCNNflex_DQN(nn.Module):
    def __init__(self,  input_shape, num_classes=3, drop_p_conv=.2, drop_p_fc=.1, kernel_size=51, stride=2, pool_size=1, filters=32, num_layers=5, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = 0
        self.pool_size = pool_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.filters = filters
        
        # block of conv, relu, drop
        def one_conv(in_c, out_c, kernel_size, stride, drop_p, dilation):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
            return conv
            
        layers = [] # all conv layers 
        layers.append( one_conv(input_shape[0], self.filters, self.kernel_size, self.stride, self.drop_p_conv, self.dilation) )
        self.outshape = int( (input_shape[1]+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride +1 )
        for i in range(num_layers-1):
            block = one_conv(self.filters, self.filters, self.kernel_size, self.stride, self.drop_p_conv, self.dilation)
            layers.append(block)
            self.outshape = int( (self.outshape+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride +1)
            if self.pool_size > 1:
                layers.append( nn.MaxPool1d(2,stride=self.pool_size) )
                self.outshape = int(self.outshape/self.pool_size)
        self.features = nn.Sequential(*layers)
        
        fc = [] # all fc layer
        fc.append( nn.Linear(self.outshape*self.filters, self.filters) ) 
        fc.append( nn.Dropout(self.drop_p_fc) )
        fc.append( nn.ReLU() )
        fc.append( nn.Linear(self.filters, num_classes) )
        self.fc_block = nn.Sequential(*fc)     
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.fc_block(x)
        
        
        
        
# same as models.MyCNNflex_DQN but FC is infused with history of shim_values (or past actions?)
# purpose: DQN for RL
class DQN_Fuse(nn.Module):
    def __init__(self,  input_shape, num_classes=3, drop_p_conv=.2, drop_p_fc=.1, kernel_size=51, stride=2, pool_size=1, 
                 filters_conv=32, filters_fc=256, num_layers=5, mode=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_size = pool_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.filters_fc = filters_fc
        self.filters_conv = filters_conv
        
        self.mode = mode # 'actor', or 'critic' or None
        
        # input_shape is tuple of ( spectrum_shape, shim_values_shape )
        self.conv_input_shape = input_shape[0]
        self.fc_input_shape = input_shape[1]
        
        # block of conv, relu, drop
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
            return conv
            
        layers = [] # all conv layers 
        layers.append( one_conv(self.conv_input_shape[0], self.filters_conv, self.kernel_size, self.stride, self.drop_p_conv) )
        self.outshape = int( (self.conv_input_shape[1]-self.kernel_size)/self.stride +1 )
        for i in range(num_layers-1):
            block = one_conv(self.filters_conv, self.filters_conv, self.kernel_size, self.stride, self.drop_p_conv)
            layers.append(block)
            self.outshape = int( (self.outshape-self.kernel_size)/self.stride +1)
            if self.pool_size > 1:
                layers.append( nn.MaxPool1d(2,stride=self.pool_size) )
                self.outshape = int(self.outshape/self.pool_size)
        self.features = nn.Sequential(*layers)
        
        self.ln_features = nn.LayerNorm(self.outshape*self.filters_conv+np.prod(self.fc_input_shape)+self.fc_input_shape[1]) if self.mode=='critic' else nn.LayerNorm(self.outshape*self.filters_conv+np.prod(self.fc_input_shape))
        
        fc = [] # all fc layer
        if self.mode=='critic': fc.append( nn.Linear(self.outshape*self.filters_conv + np.prod(self.fc_input_shape) + self.fc_input_shape[1], self.filters_fc) ) #last fc of critic additionally takes action as input
        else: fc.append( nn.Linear(self.outshape*self.filters_conv + np.prod(self.fc_input_shape), self.filters_fc) ) # fused FC here
        fc.append( nn.Linear(self.filters_fc, self.filters_fc) ) # proxy layer to avoid changes to fc_input (both drop + Relu!)
        fc.append( nn.Dropout(self.drop_p_fc) )
        fc.append( nn.ReLU() )
        fc.append( nn.Linear(self.filters_fc, num_classes) )
        self.fc_block = nn.Sequential(*fc)     
        
    # expects input of ( batch_size, past_observations,  2048 + nr_shims ) -> flattened!
    def forward(self, state, action=None): # allow to input actions for critic architecture
        nr_shims = self.fc_input_shape[-1]
        fc_x = state[:,:,-nr_shims:].contiguous().view(state.shape[0],-1) # get last (nr_shims) values and flatten to (batch_size, :)
        conv_x = state[:,:,:-nr_shims].contiguous()
        conv_x = self.features(conv_x)
        conv_x = conv_x.view(conv_x.shape[0], -1) 
        # infuse additional values into FC layer
        out = torch.cat( (conv_x, fc_x), 1)
        out = self.ln_features(out)
        if action!=None: out = torch.cat( (out, action), 1) # infuse action for critic 
        if self.mode=='actor': return F.hardtanh( self.fc_block(out) )
        return self.fc_block(out)


#%% 

class ConvLSTM(nn.Module): # convolutional LSTM  for complex valued input
    # treat complex input as 2 channels
    def __init__(self, spectrum_size, action_size, hidden_size, output_size, num_layers_lstm=2,
                 num_layers_cnn=5, filters_conv=32, stride=2, kernel_size=51, dilation=1, pool_size=1, drop_p_conv=0, drop_p_fc=0, spectra_content='real', use_tanh=True):
        super().__init__()       
        self.hidden_size = hidden_size
        self.num_layers_lstm = num_layers_lstm
        self.num_layers_cnn = num_layers_cnn
        self.filters_conv = filters_conv
        self.stride = stride
        self.dilation = dilation
        self.padding = 0
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.spectrum_size = spectrum_size
        self.action_size = action_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.spectra_content = spectra_content # real, abs or complex spectrum
        self.use_tanh = use_tanh # last activation tanh. Not useful for RL
        
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        def one_conv(in_c, out_c, kernel_size, stride, drop_p, dilation):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(drop_p))
            return conv
        
        if self.spectra_content == 'complex':
            in_channels = 2
        else:
            in_channels = 1
        
        block_conv = [] # for i2f (input to features)
        block_conv.append( one_conv(in_channels, self.filters_conv, self.kernel_size, self.stride, self.drop_p_conv, self.dilation) )
        self.feature_shape = int( (self.spectrum_size+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride +1 )
        for i in range(num_layers_cnn-1):
            block = one_conv(self.filters_conv, self.filters_conv, self.kernel_size, self.stride, self.drop_p_conv, self.dilation)
            block_conv.append(block)
            self.feature_shape = int( (self.feature_shape+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride +1)
            if self.pool_size > 1:
                block_conv.append( nn.MaxPool1d(2,stride=self.pool_size) )
                self.feature_shape = int(self.feature_shape/self.pool_size)
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
        fc_part = inputs[:,:, self.spectrum_size:].real if self.spectra_content == 'complex' else inputs[:,:, self.spectrum_size:]    # shim actions/values
        features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.feature_shape*self.filters_conv])).to(self.DEVICE)
        for k in range(past_obs): # convolve each spectrum for its own to keep temporal nature
            if self.spectra_content == 'complex':
                complex_2channel = torch.stack([conv_part[:,k].real, conv_part[:,k].imag],1).float().to(self.DEVICE)
                features[:,k] = self.i2f(complex_2channel).view(inputs.shape[0],-1)
            else: 
                features[:,k] = self.i2f(conv_part[:,k].unsqueeze(1)).view(inputs.shape[0],-1)
        #features = self.i2f(conv_part)
        combined = torch.cat((features, fc_part), 2).float()
        
        combined = self.ln_features(combined)
        
        (h0,c0) = hidden
        out, (h0,c0) = self.lstm(combined, (h0.contiguous(), c0.contiguous()))
        
        out = self.ln_lstm(out)
        
        x = self.relu(self.fc1(out))
        x = self.drop(x)
        x = self.fc2(x)
        if self.use_tanh: x = self.tanh(x)
        return x, (h0,c0)
        
    def forward_trunc(self, inputs, hidden): # without last layer 
        past_obs = inputs.shape[1]
        conv_part = inputs[:,:, :self.spectrum_size]  # spectra
        fc_part = inputs[:,:, self.spectrum_size:].real if self.spectra_content == 'complex' else inputs[:,:, self.spectrum_size:]    # shim actions/values
        features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.feature_shape*self.filters_conv])).to(self.DEVICE)
        for k in range(past_obs): # convolve each spectrum for its own to keep temporal nature
            if self.spectra_content == 'complex':
                complex_2channel = torch.stack([conv_part[:,k].real, conv_part[:,k].imag],1).float().to(self.DEVICE)
                features[:,k] = self.i2f(complex_2channel).view(inputs.shape[0],-1)
            else: 
                features[:,k] = self.i2f(conv_part[:,k].unsqueeze(1)).view(inputs.shape[0],-1)
        #features = self.i2f(conv_part)
        combined = torch.cat((features, fc_part), 2).float()  
        combined = self.ln_features(combined)      
        (h0,c0) = hidden
        out, (h0,c0) = self.lstm(combined, (h0.contiguous(), c0.contiguous()))       
        out = self.ln_lstm(out)
        x = self.relu(self.fc1(out))
        return x, (h0,c0)

    def initHidden(self, batch_size=1):
        return ( torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size), torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size) )
 
 
# FIX: FC after i2f
class ConvLSTMf2fc(nn.Module): # convolutional LSTM  for complex valued input
    # treat complex input as 2 channels
    def __init__(self, spectrum_size, action_size, hidden_size, output_size, num_layers_lstm=2,
                 num_layers_cnn=5, filters_conv=32, stride=2, kernel_size=51, dilation=1, pool_size=1, drop_p_conv=0, drop_p_fc=0, spectra_content='real', use_tanh=True):
        super().__init__()       
        self.hidden_size = hidden_size
        self.num_layers_lstm = num_layers_lstm
        self.num_layers_cnn = num_layers_cnn
        self.filters_conv = filters_conv
        self.stride = stride
        self.dilation = dilation
        self.padding = 0
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.spectrum_size = spectrum_size
        self.action_size = action_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.spectra_content = spectra_content # real, abs or complex spectrum
        self.use_tanh = use_tanh # last activation tanh. Not useful for RL
        
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        def one_conv(in_c, out_c, kernel_size, stride, drop_p, dilation):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(drop_p))
            return conv
        
        if self.spectra_content == 'complex':
            in_channels = 2
        else:
            in_channels = 1
        
        block_conv = [] # for i2f (input to features)
        block_conv.append( one_conv(in_channels, self.filters_conv, self.kernel_size, self.stride, self.drop_p_conv, self.dilation) )
        self.feature_shape = int( (self.spectrum_size+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride +1 )
        for i in range(num_layers_cnn-1):
            block = one_conv(self.filters_conv, self.filters_conv, self.kernel_size, self.stride, self.drop_p_conv, self.dilation)
            block_conv.append(block)
            self.feature_shape = int( (self.feature_shape+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride +1)
            if self.pool_size > 1:
                block_conv.append( nn.MaxPool1d(2,stride=self.pool_size) )
                self.feature_shape = int(self.feature_shape/self.pool_size)
        self.i2f = nn.Sequential(*block_conv) # input to features
        
        self.f2fc = nn.Linear(self.feature_shape*self.filters_conv,self.feature_shape*self.filters_conv)
        
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
        fc_part = inputs[:,:, self.spectrum_size:].real if self.spectra_content == 'complex' else inputs[:,:, self.spectrum_size:]    # shim actions/values
        features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.feature_shape*self.filters_conv])).to(self.DEVICE)
        for k in range(past_obs): # convolve each spectrum for its own to keep temporal nature
            if self.spectra_content == 'complex':
                complex_2channel = torch.stack([conv_part[:,k].real, conv_part[:,k].imag],1).float().to(self.DEVICE)
                features[:,k] = self.i2f(complex_2channel).view(inputs.shape[0],-1)
            else: 
                features[:,k] = self.i2f(conv_part[:,k].unsqueeze(1)).view(inputs.shape[0],-1)
        # FIX f2fc
        features = self.relu(self.f2fc(features.float()))
        #features = self.i2f(conv_part)
        combined = torch.cat((features, fc_part), 2).float()
        
        combined = self.ln_features(combined)
        
        (h0,c0) = hidden
        out, (h0,c0) = self.lstm(combined, (h0.contiguous(), c0.contiguous()))
        
        out = self.ln_lstm(out)
        
        x = self.relu(self.fc1(out))
        x = self.drop(x)
        x = self.fc2(x)
        if self.use_tanh: x = self.tanh(x)
        return x, (h0,c0)


    def initHidden(self, batch_size=1):
        return ( torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size), torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size) )
 

    
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
class ConvTransformer(nn.Module):
    def __init__(self, spectrum_size, action_size, hidden_size, output_size, num_layers_transformer=4, num_heads_transformer = 8,
                 num_layers_cnn=5, filters_conv=32, stride=2, kernel_size=51, drop_p_conv=0, drop_p_fc=0, drop_p_trans=0, spectra_content='real'):
        super().__init__()       
        self.hidden_size = hidden_size
        self.num_layers_transformer = num_layers_transformer
        self.num_heads_transformer = num_heads_transformer
        self.num_layers_cnn = num_layers_cnn
        self.filters_conv = filters_conv
        self.stride = stride
        self.kernel_size = kernel_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.drop_p_trans = drop_p_trans
        self.spectrum_size = spectrum_size
        self.action_size = action_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.spectra_content = spectra_content # real, abs or complex spectrum
        
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p))
            return conv
        
        if self.spectra_content == 'complex':
            in_channels = 2
        else:
            in_channels = 1
        
        block_conv = [] # for i2f (input to features)
        block_conv.append( one_conv(in_channels, self.filters_conv, self.kernel_size, stride=self.stride, drop_p=self.drop_p_conv) )
        self.feature_shape = int( (self.spectrum_size-self.kernel_size)/self.stride +1 )
        for i in range(self.num_layers_cnn-1):
            conv = one_conv(self.filters_conv, self.filters_conv, self.kernel_size, self.stride, drop_p=self.drop_p_conv)
            block_conv.append(conv)
            self.feature_shape = int( (self.feature_shape-self.kernel_size)/self.stride +1)
        self.i2f = nn.Sequential(*block_conv) # input to features
        
        self.dim_trans = self.feature_shape*self.filters_conv+self.action_size
        
        self.ln_features = nn.LayerNorm(self.dim_trans)
        
        #self.lstm = nn.LSTM(self.feature_shape*self.filters_conv+self.action_size, hidden_size, self.num_layers_lstm, batch_first=True)
        
        # full transformer
# =============================================================================
#         self.transformer = nn.Transformer(d_model = self.dim_trans,
#                                           nhead=4, # must be divisible by dim (12 for kernel 19)
#                                           # TODO add hyperparameters, 
#                                           batch_first=True)
# =============================================================================
        
        # encoder only 
        self.pos_encoder = PositionalEncoding(self.dim_trans, dropout=self.drop_p_trans, max_len=self.dim_trans)
        encoder_layers = torch.nn.TransformerEncoderLayer(self.dim_trans, batch_first=True,
                                                          nhead=self.num_heads_transformer,
                                                          dim_feedforward=self.hidden_size, 
                                                          dropout=self.drop_p_trans)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, 
                                                               num_layers=self.num_layers_transformer)
        #self.encoder = nn.Embedding(ntoken, d_model)
        #self.decoder = nn.Linear(self.dim_trans, self.action_size)
        
        self.ln_lstm = nn.LayerNorm(self.dim_trans)
        
        self.fc1 = nn.Linear(self.dim_trans, self.hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p_fc)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
            
    def forward(self, inputs, dec_seq_len=1):
        past_obs = inputs.shape[1]
        conv_part = inputs[:,:, :self.spectrum_size]  # spectra
        fc_part = inputs[:,:, self.spectrum_size:].real if self.spectra_content == 'complex' else inputs[:,:, self.spectrum_size:]    # shim actions/values
        features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.feature_shape*self.filters_conv])).to(self.DEVICE)
        for k in range(past_obs): # convolve each spectrum for its own to keep temporal nature
            if self.spectra_content == 'complex':
                complex_2channel = torch.stack([conv_part[:,k].real, conv_part[:,k].imag],1).float().to(self.DEVICE)
                features[:,k] = self.i2f(complex_2channel).view(inputs.shape[0],-1)
            else: 
                features[:,k] = self.i2f(conv_part[:,k].unsqueeze(1)).view(inputs.shape[0],-1)
        #features = self.i2f(conv_part)
        combined = torch.cat((features, fc_part), 2).float()
        
        combined = self.ln_features(combined)
        
        #TODO 
        # positional encoding ? 
        # Decoder input should contain last src, and all rest values except last
        # src = combined[:, :(past_obs-dec_seq_len)] * math.sqrt(self.dim_trans)
        # tgt = combined[:, (past_obs-dec_seq_len-1):(past_obs-1)] * math.sqrt(self.dim_trans)
        # mask 'combined' and shift -> target seq
        # 
        #tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1])
        
        # full transformer
        #out = self.transformer(src.to(self.DEVICE), tgt.to(self.DEVICE), tgt_mask=tgt_mask.to(self.DEVICE))
        
        # trans encoder only
        # no pos encoding as order is irrelevant
        src = combined #* math.sqrt(self.dim_trans)
        #src = self.pos_encoder(src)
        out = self.transformer_encoder(src
                                       #, mask = generate_square_subsequent_mask(src.shape[1]).to(self.DEVICE)
                                       ) # with or without mask?
        
        #out = self.ln_lstm(out)
        
        x = self.relu(self.fc1(out))
        x = self.drop(x)
        x = self.tanh(self.fc2(x))
        return x


class VAESmooth(nn.Module):
    #Added a smoothing kernel at the output during testing phase only
    #In order to have beautiful plots
    def __init__(self, in_channels, latent_dim, hidden_dims, kernel_size, f_loss, gaussian_kernel_size):
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels_static = in_channels  
        if kernel_size is None:
            self.kernel_size = 51
        else:
            self.kernel_size = kernel_size

        self.stride = 2
        self.padding = 0
        self.dilation = 1
        self.gaussian_kernel_size = gaussian_kernel_size
        print(self.gaussian_kernel_size, 'gaussina kernel size')

        # Problems with stride, dilation & output dim!

        self.outshape=2048

        modules = []
        if hidden_dims is None:
            #self.hidden_dims = [32, 64, 128, 256, 512]
            self.hidden_dims = [512, 256, 128, 64, 32] # mob: inverted!
        else: self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=self.kernel_size, stride=self.stride,
                              padding  = self.padding, dilation = self.dilation),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            self.outshape = int( (self.outshape+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride +1) # changed!
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.outshape, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*self.outshape, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * self.outshape)
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i],self.hidden_dims[i + 1],kernel_size=self.kernel_size,stride=self.stride, 
                                       padding  = self.padding, dilation=self.dilation),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        modules.append( nn.Sequential( # last layer
                nn.ConvTranspose1d(self.hidden_dims[i + 1],self.in_channels_static,kernel_size=self.kernel_size,stride=self.stride,
                                   padding  = self.padding, output_padding=1, dilation = self.dilation),
                #nn.BatchNorm1d(self.in_channels_static),
                nn.Linear(2048, 2048), nn.Sigmoid()))
        self.decoder = nn.Sequential(*modules)

        self.loss = f_loss["name"]
        self.loss_param = f_loss["param"]

        #Adding blurring filter to smooth the testing and plots
        self.blur = False
        #self.smooth_kernel = self.gaussianKernel(self.gaussian_kernel_size)

        self.hidden_dims.reverse()

    def train_mode(self):
        # Deactivate the smoothening filter for training
        self.blur = False
    def test_mode(self):
        # Activate the smoothening filter to test and plot
        self.blur = True

    def gaussianKernel(size):
        std = (size - 1)/2
        mean = (size - 1)/2
        kernel = torch.exp(torch.tensor(-((np.arange(size) - mean)/std)**2, dtype = torch.float))
        kernel/=kernel/sum()
        return kernel

    def encode(self, inputs):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(inputs)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(result.shape[0], self.hidden_dims[-1], -1)
        result = self.decoder(result)

        if self.blur:
            result = F.conv1d(result, weight = self.smooth_kernel.reshape((1, 1, -1)), padding = (self.gaussian_kernel_size-1)//2)

        #result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), inputs, mu, log_var]

    def loss_function(self,*args,kld_weight=0.005):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        if self.loss == "mse" : recons_loss =F.mse_loss(recons, inputs)
        elif self.loss == "huber" : recons_loss = F.huber_loss(recons, inputs, delta = self.loss_param)
        elif self.loss == "mae" : recons_loss = F.l1_loss(recons, inputs)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}

    def sample(self,num_samples, current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class LSTMcompressedMeanOnly(nn.Module): # convolutional LSTM  
    #LSTM network using compressed data but using only the mean value
    def __init__(self, feature_shape, action_size, hidden_size, output_size, num_layers_lstm=2, 
                 drop_p_fc=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers_lstm = num_layers_lstm
        self.drop_p_fc = drop_p_fc
        self.action_size = action_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.feature_shape = feature_shape

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.ln_features = nn.LayerNorm(self.feature_shape+self.action_size)

        self.lstm = nn.LSTM(self.feature_shape+self.action_size, hidden_size, self.num_layers_lstm,
                            batch_first=True)
        self.ln_lstm = nn.LayerNorm(hidden_size)

        # i2o
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p_fc)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        #Input has shape (batch, sequence_length, latent_dim*2+label_size)
        past_obs = inputs.shape[1]

        mean_part = inputs[:, :, :self.feature_shape]
        std_part = inputs[:, :, self.feature_shape:2*self.feature_shape]

        fc_part = inputs[:,:, 2*self.feature_shape:]          # shim actions/values

        sampled_features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.feature_shape])).float().to(self.DEVICE)
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
         