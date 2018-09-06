import torch
import torch.nn as nn

from utils.model_utils import Linear

class LinearAggregate(nn.Module):
    def __init__(self, config):
        super(LinearAggregate, self).__init__()

        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

        seq_in_size = 2*config.d_hidden
        if config.birnn:
            seq_in_size *= 2
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

    def forward(self, x):
        return self.aggregator(x)
