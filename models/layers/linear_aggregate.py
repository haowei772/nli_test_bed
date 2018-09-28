import torch
import torch.nn as nn

from utils.model_utils import Linear

class LinearAggregate(nn.Module):
    def __init__(self, config):
        super(LinearAggregate, self).__init__()

        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

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

    def forward(self, x, y):
        x_and_y = torch.cat([x,y,x-y,x*y],-1)
        return [self.aggregator(x_and_y)]
