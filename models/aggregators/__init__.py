import torch
import torch.nn as nn
from .linear_aggregate import LinearAggregate


'''
Dict containing all the aggreagtors
Add new aggregators to this dict
'''
aggregators = {
    'linear_aggregate': LinearAggregate
}


'''
Main aggreagtor class
'''
class Aggregator(nn.Module):
    def __init__(self, aggregator_name, config):
        super(Aggregator, self).__init__()
        if aggregator_name not in aggregators:
            raise NotImplementedError(f"aggregator '{aggregator_name}' not implemented")
        
        self.aggregator = aggregators[aggregator_name](config)
    
    def forward(self, x, y):
        """
        x (batch_size, d_hidden)
        y (batch_size, d_hidden)
        output (batch_size, d_out)
        """
        return self.aggregator(torch.cat([x, y], 1))