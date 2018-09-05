import torch.nn as nn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def clones_sequential(module, N):
    "Produce N identical layers."
    return nn.Sequential(copy.deepcopy(module) for _ in range(N))