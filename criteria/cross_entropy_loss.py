import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x, target):
        return self.criterion(x, target)