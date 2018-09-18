import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.parameter import Parameter
from time import gmtime, strftime
# from utils.utils import draw
from utils.model_utils import clones, draw

class Transformer1(nn.Module):
    def __init__(self, config):
        super(Transformer1, self).__init__()
        self.config = config
        self.encoder = TransformerEncoder(self.config)
        self.encoder_w_input = TransformerEncoderWInput(self.config)
    
    def forward(self, x, y):

        x_encoded = self.encoder(x)
        y_attn_over_x_encoded = self.encoder_w_input(y, x)
        output = torch.mean(y_attn_over_x_encoded, 1)
        return output


class TransformerEncoderWInput(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderWInput, self).__init__()

        self.config = config

        c = copy.deepcopy

        block = BlockWInput(config, scale=True)

        self.layers = clones(block, config.n_layers)
    
    def forward(self, x, y):
        """
        x: (batch_size, seq_len, embed_size)
        y: (batch_size, seq_len, embed_size)
        h: (batch_size, seq_len, embed_size)
        """

        h = x
        for block in self.layers:
            h = block(h, y)

        return h


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.config = config

        c = copy.deepcopy

        block = Block(config, scale=True)

        self.layers = clones(block, config.n_layers)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_size)
        h: (batch_size, seq_len, embed_size)
        """
        
        h = x
        for block in self.layers:
            h = block(h)

        return h
    
    # def draw_self_attentions(self, sent):
    #     for layer in range(len(self.encoder.layers)):
    #         fig, axs = plt.subplots(1,self.config.h_transformer, figsize=(20, 10))
    #         for h in range(self.config.h_transformer):
    #             draw(self.encoder.layers[layer].self_attn.attn[0, h].data, 
    #                 sent, sent if h ==0 else [], ax=axs[h])
    #         fig.savefig(self.config.save_path + "/" + f"{strftime('%H:%M:%S', gmtime())}_self_attn_layer_{layer}.png")
    #         plt.close(fig)



def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=d_embed)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)
        self.attn = None

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v), w

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, query, key=None, value=None):

        if key is None or value is None:
            query = self.c_attn(query)
            query, key, value = query.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a, self.attn = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * d_embed)
        super(MLP, self).__init__()
        nx = cfg.d_embed
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.d_embed
        self.attn = Attention(nx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h

class BlockWInput(nn.Module):
    def __init__(self, cfg, scale=False):
        super(BlockWInput, self).__init__()
        nx = cfg.d_embed
        self.attn = Attention(nx, cfg, scale)
        self.attn_w_input = Attention(nx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
        self.ln_3 = LayerNorm(nx)

    def forward(self, x, y):
        a1 = self.attn(x)
        n1 = self.ln_1(x + a1)

        a2 = self.attn_w_input(n1, y, y)
        n2 = self.ln_2(n1 + a2)

        m = self.mlp(n2)
        h = self.ln_2(n2 + m)

        return h


class ClfHead(nn.Module):
    """Classification Head for the transformer

    TODO: test this class."""
    def __init__(self, clf_token, cfg, n_class):
        super(ClfHead, self).__init__()
        self.d_embed = cfg.d_embed
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.d_embed, n_class)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.d_embed)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)

        return clf_logits