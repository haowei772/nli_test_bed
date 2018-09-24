import torch.nn as nn

from .transformer import Transformer, TransformerInterAttn
from .rnn import RNN
from .rnn_inter_attn import RNNInterAttn


'''
Dict containing all the encoders
Add new encoders to this dict
'''
encoders = {
    'rnn': RNN,
    'transformer': Transformer,
    'rnn_inter_attn': RNNInterAttn,
    'transformer_inter_attn': TransformerInterAttn,
}


'''
Main encoder class
'''
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        if not config.encoder:
            raise NotImplementedError("'encoder' not defined in config")
        
        if config.encoder not in encoders:
            raise NotImplementedError(f"encoder '{config.encoder}' not implemented")
        
        self.encoder = encoders[config.encoder](config)
    
    def forward(self, x):
        """
        x (batch_size, seq_len, d_embed)
        output (batch_size, seq_len, d_hidden)
        """
        return self.encoder(x)
    
    def draw_attentions(self, sent1, sent2):
        """
        drawing attention maps
        """
        if hasattr(self.encoder, 'draw_self_attentions'):
            self.encoder.draw_self_attentions(sent1)
        
        if hasattr(self.encoder, 'draw_attentions'):
            self.encoder.draw_attentions(sent1, sent2)


'''
Main encoder with inter attention class
'''
class EncoderInterAttn(nn.Module):
    def __init__(self, config):
        super(EncoderInterAttn, self).__init__()
        if not config.encoder_inter_attn:
            raise NotImplementedError("'encoder_inter_attn' not defined in config")
        
        if config.encoder_inter_attn not in encoders:
            raise NotImplementedError(f"encoder '{config.encoder_inter_attn}' not implemented")
        
        self.encoder = encoders[config.encoder_inter_attn](config)
    
    def forward(self, x, y):
        """
        x (batch_size, seq_len, d_embed)
        y (batch_size, seq_len, d_embed)
        output (batch_size, seq_len, d_hidden)
        """
        return self.encoder(x, y)

    
    def draw_attentions(self, sent1, sent2):
        """
        drawing attention maps
        """
        if hasattr(self.encoder, 'draw_self_attentions'):
            self.encoder.draw_self_attentions(sent1)
        
        if hasattr(self.encoder, 'draw_attentions'):
            self.encoder.draw_attentions(sent1, sent2)