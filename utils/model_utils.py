import torch.nn as nn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def clones_sequential(module, N):
    "Produce N identical layers."
    return nn.Sequential(copy.deepcopy(module) for _ in range(N))

class Bottle(nn.Module):
    
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass