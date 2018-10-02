import torch
import torch.nn as nn

class Sum(nn.Module):
    def __init__(self, config):
        super(Sum, self).__init__()
        pass

    def forward(self, input, dim=1):
        output = torch.sum(input, dim)
        return [output]