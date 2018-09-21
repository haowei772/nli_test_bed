import torch.nn as nn

from .encoders import encoders

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
