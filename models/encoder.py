import torch.nn as nn
from .encoders import encoders

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        if not config.encoder:
            raise NotImplementedError("'encoder' not defined in config")
        
        if config.encoder not in encoders:
            raise NotImplementedError(f"encoder '{config.encoder}' not implemented")
        
        self.encoder = encoders[config.encoder](config)
    
    def forward(self, x, x_embed):
        """
        x (batch_size, seq_len)
        x_embed (batch_size, seq_len, d_embed)
        output (batch_size, d_hidden)
        """
        return self.encoder(x, x_embed)