import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,dropout=0.1,factorize=True):
        super(MLPNetwork, self).__init__()

        self.factorize = factorize
        self.output_dim = output_dim
        # expect hidden dims to be a list
        if isinstance(hidden_dims,int):
            hidden_dims = [hidden_dims]

        mlp_layers = []
        if factorize:
            mlp_layers.append(nn.Linear(3*input_dim, hidden_dims[0]))
        else:
            mlp_layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # mlp_layers.append(nn.LayerNorm(hidden_dims[0]))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))

        input_dim = hidden_dims[0]
        if len(hidden_dims)>1:
            for hidden_dim in hidden_dims[1:]:
                mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                # mlp_layers.append(nn.LayerNorm(hidden_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        self.fc = nn.Linear(hidden_dims[-1],output_dim)

        # print(f'init mlp networks, hidden_dims:{hidden_dims}')

    def _factorize_forward(self,x):

        x = torch.flatten(x,start_dim=1)
        x = self.mlp(x)
        out = self.fc(x)
        return out
    
    def _direct_forward(self,x):
        
        # x = torch.flatten(x,start_dim=1)
        x = self.mlp(x)
        out = self.fc(x)
        return out


    def forward(self,x):

        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)

        seq_length = x.shape[-2]

        if self.factorize:
            y = [[]]*(seq_length-2)
            for i in range(1, seq_length-1):
                y[i-1] = self._factorize_forward(x[:,i-1:i+2])
                
            y = torch.stack(y,dim=-2)
        
        else:
            y = [[]]*(seq_length)
            for i in range(0, seq_length):
                y[i] = self._direct_forward(x[:,i])
                
            y = torch.stack(y,dim=-2)

        return y

class LSTMBaseline(nn.Module):
    def     __init__(self, input_dim, hidden_dims, output_dim,bidirectional=False):

        super(LSTMBaseline, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dims[0]
        self.num_layers = len(hidden_dims)
        self.out_dim = output_dim

        # LSTM for processing sequences
        self.seq_lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim , num_layers=self.num_layers,
                                     bidirectional=bidirectional,batch_first=True)

        # Fully connected layer for output
        if bidirectional:
            self.output_fc = nn.Linear(in_features=2*self.hidden_dim , out_features=output_dim)
        else:
            self.output_fc = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

        # print(f'lstm baseline model: hidden_dim:{self.hidden_dim} num_layers:{self.num_layers}')

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)

        out, _  = self.seq_lstm(x)
        out = self.output_fc(out)
        ##  the first 2 input of x are conditions, start from 2: is output for future actions
        return out[:,2:,:]

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,bidirectional=False):

        super(LSTMNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dims[0]

        self.num_layers = len(hidden_dims)
        self.out_dim = output_dim

        # LSTM for processing sequences

        self.seq_lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim , num_layers=self.num_layers,
                                     bidirectional=bidirectional,batch_first=True)

        # Fully connected layer for output
        if bidirectional:
            self.output_fc = nn.Linear(in_features=2*self.hidden_dim , out_features=output_dim)
        else:
            self.output_fc = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

        # print(f'idm decoder hidden dims: {self.hidden_dim}')
    def forward(self, x):

        out, _  = self.seq_lstm(x)
        out = self.output_fc(out)

        return out

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    import time
    network = LSTMBaseline(input_dim=20,hidden_dims=[256,256], output_dim=2)
    input = torch.randn(64,12,20)
    tgt = torch.randn(64, 12, 2)

    start_time = time.time()
    output = network.forward(input)
    done_time = time.time()
    print(output.shape, done_time-start_time)

