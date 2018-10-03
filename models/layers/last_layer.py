import torch
import torch.nn as nn

class LastLayer(nn.Module):
    def __init__(self, config):
        super(LastLayer, self).__init__()
        pass
    
    def forward(self, input):
        output = input[:,-1,:]
        return [output]