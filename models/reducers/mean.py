import torch
import torch.nn as nn

class Mean(nn.Module):
    def __init__(self, config):
        super(Mean, self).__init__()
        pass

    def forward(self, input, dim=1):
        output = torch.mean(input, dim)
        return output