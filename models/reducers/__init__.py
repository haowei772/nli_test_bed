import torch.nn as nn


from .mean import Mean
from .last_layer import LastLayer


'''
Dict containing all the reducers
Add new reducers to this dict
'''
reducers = {
    'mean': Mean,
    'last_layer': LastLayer
}


'''
Main reducer class
'''
class Reducer(nn.Module):
    def __init__(self, reducer_name, config):
        super(Reducer, self).__init__()
        if reducer_name not in reducers:
            raise NotImplementedError(f"reducer '{reducer_name}' not implemented")
        
        self.reducer = reducers[reducer_name](config)
    
    def forward(self, x):
        """
        x (batch_size, _ ,d_hidden)
        output (batch_size, d_hidden)
        """
        return self.reducer(x)