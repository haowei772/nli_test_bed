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
    
    def forward(self, x, y=None):
        """
        x (batch_size, seq_len)
        x_embed (batch_size, seq_len, d_embed)
        output (batch_size, d_hidden)
        """
        if y is not None:
            return self.encoder(x, y)
        else:
            return self.encoder(x)
    
    def draw_attentions(self, sent1, sent2):
        """
        drawing attention maps
        """
        if hasattr(self.encoder, 'draw_self_attentions'):
            self.encoder.draw_self_attentions(sent1)
        
        if hasattr(self.encoder, 'draw_attentions'):
            self.encoder.draw_attentions(sent1, sent2)
