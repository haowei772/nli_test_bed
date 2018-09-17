import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
# from utils.utils import draw
from utils.model_utils import clones, draw


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config

        c = copy.deepcopy

        # multi head attention
        attn = MultiHeadedAttention(config)

        # positionwise feed forward
        ff = PositionwiseFeedForward(config)

        # encoder
        self.encoder = Encoder(config, EncoderLayer(config, c(attn), c(ff)))

        self.pad = 0
    
    def forward(self, x, x_embed):
        """
        x: (batch_size, seq_len)
        x_embed: (batch_size, seq_len, embed_size)
        encoder_output: (batch_size, seq_len, embed_size)
        output: (batch_size, embed_size)
        """
        x_mask = (x != self.pad).unsqueeze(-2)
        encoder_output = self.encoder(x_embed, x_mask)
        output = torch.mean(encoder_output, 1)
        return output
    
    def draw_self_attentions(self, sent):
        for layer in range(len(self.encoder.layers)):
            fig, axs = plt.subplots(1,self.config.h_transformer, figsize=(20, 10))
            for h in range(self.config.h_transformer):
                draw(self.encoder.layers[layer].self_attn.attn[0, h].data, 
                    sent, sent if h ==0 else [], ax=axs[h])
            fig.savefig(self.config.save_path + "/" + f"{strftime('%H:%M:%S', gmtime())}_self_attn_layer_{layer}.png")
            plt.close(fig)


class TransformerInterAttention(nn.Module):
    def __init__(self, config):
        super(TransformerInterAttention, self).__init__()

        self.config = config

        c = copy.deepcopy

        # multi head attention
        attn = MultiHeadedAttention(config)

        # positionwise feed forward
        ff = PositionwiseFeedForward(config)

        # encoder
        self.encoder = EncoderInterAttention(
            config, 
            DoubleEncoderInterAttentionLayer(config, c(attn), c(attn), c(ff)),
            )

        self.pad = 0
    
    def forward(self, x, x_embed, y, y_embed):
        x_mask = (x != self.pad).unsqueeze(-2)
        y_mask = (y != self.pad).unsqueeze(-2)

        encoder_output = self.encoder(x_embed, x_mask, y_embed, y_mask)
        output = torch.mean(encoder_output, 1)
        return output
    
    def draw_self_attentions(self, sent):
        for layer in range(len(self.encoder.layers)):
            fig, axs = plt.subplots(1,self.config.h_transformer, figsize=(20, 10))
            for h in range(self.config.h_transformer):
                draw(self.encoder.layers[layer].self_attn.attn[0, h].data, 
                    sent, sent if h ==0 else [], ax=axs[h])
            fig.savefig(self.config.save_path + "/" + f"{strftime('%H:%M:%S', gmtime())}_self_attn_layer_{layer}.png")
            plt.close(fig)
    
    def draw_attentions(self, sent1, sent2):
        for layer in range(len(self.encoder.layers)):
            fig, axs = plt.subplots(1,self.config.h_transformer, figsize=(20, 10))
            for h in range(self.config.h_transformer):
                draw(self.encoder.layers[layer].src_attn.attn[0, h].data, 
                    sent1, sent2 if h ==0 else [], ax=axs[h])
            fig.savefig(self.config.save_path + "/" + f"{strftime('%H:%M:%S', gmtime())}_attn_layer_{layer}.png")
            plt.close(fig)



class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, config, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, config.n_layers)
        self.norm = LayerNorm(config)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderInterAttention(nn.Module):
    "Encoder with inter attention"
    def __init__(self, config, layer):
        super(EncoderInterAttention, self).__init__()
        self.layers = clones(layer, config.n_layers)
        self.norm = LayerNorm(config)
        
    def forward(self, x, x_mask, y, y_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, x_mask, y, y_mask)
        return self.norm(x)

class DoubleEncoderInterAttention(nn.Module):
    "Encoder with inter attention"
    def __init__(self, config, layer, layer_second):
        super(DoubleEncoderInterAttention, self).__init__()
        self.layers = clones(layer, config.n_layers)
        self.layers_second = clones(layer_second, config.n_layers)
        self.norm = LayerNorm(config)
        
    def forward(self, x, x_mask, y, y_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers_second:
            y = layer(y, y_mask)

        for layer in self.layers:
            x = layer(x, x_mask, y, y_mask)

        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, config):
        super(LayerNorm, self).__init__()
        features = config.d_transformer
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = config.eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        dropout = config.dropout_pe

        self.norm = LayerNorm(config)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, config, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()

        dropout = config.dropout_pe
        size = config.d_transformer

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(config), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class EncoderInterAttentionLayer(nn.Module):
    """
    Encoder with inter attention is made up of three sublayers, 
    self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, config, self_attn, src_attn, feed_forward):
        super(EncoderInterAttentionLayer, self).__init__()
        self.dropout = config.dropout_pe 
        self.size = config.d_transformer

        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(config), 3)
 
    def forward(self, x, x_mask, y, y_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, x_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, y, y, y_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()

        h = config.h_transformer
        d_model = config.d_transformer
        dropout = config.dropout_pe

        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        d_model = config.d_transformer
        d_ff = config.d_ff_transformer
        dropout = config.dropout_pe
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, premise, hypothesis, pad=0):
        self.premise = premise
        self.premise = (premise != pad).unsqueeze(-2)
        self.hypothesis = hypothesis
        self.hypothesis = (hypothesis != pad).unsqueeze(-2)

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn