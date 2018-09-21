import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, config, vocab):
        super(Embedding, self).__init__()

        # embedding dimension
        self.d_model = config.d_embed

        self.lut = nn.Embedding(len(vocab), self.d_model)

        # copy weights from vocab vectors to lut
        self.lut.weight.data.copy_(vocab.vectors)

        # if embedding weights should be updated
        if not config.train_embed:
            self.lut.weight.requires_gard = False

        # if apply weights on the output
        self.apply_weight_embed = config.apply_weight_embed
    
    def forward(self, x):
        # ----- lookup -----
        lut_x = self.lut(x)

        # ----- apply weights -----
        if self.apply_weight_embed:
            lut_x = lut_x * math.sqrt(self.d_model)

        return lut_x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, config): #d_model, dropout, max_len=5000
        super(PositionalEncoding, self).__init__()
        self.dropout = config.dropout_pe
        self.d_model = config.d_embed
        self.max_len = config.max_len

        self.dropout = nn.Dropout(p=self.dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
        div_term = div_term.float()
        div_term = torch.exp(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)