import torch
import torch.nn as nn

from utils.model_utils import Linear

class LinearAggregate(nn.Module):
    def __init__(self, config):
        super(LinearAggregate, self).__init__()

        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        self.para_init = config.para_init

        seq_in_size = 4*config.d_hidden

        lin_config = [seq_in_size]*2

        self.aggregator = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, config.d_out))
        
        self.init_params()
    
    def init_params(self):
        '''initialize parameters'''
        for m in self.modules():
            # print m
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                m.bias.data.normal_(0, self.para_init)

    def forward(self, x, y):
        x_and_y = torch.cat([x,y,x-y,x*y],-1)
        return [self.aggregator(x_and_y)]
