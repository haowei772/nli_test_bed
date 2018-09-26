import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from time import gmtime, strftime
from utils.model_utils import clones, draw

class AttentionMultihead(nn.Module):
    def __init__(self, cfg, scale=False):
        super(AttentionMultihead, self).__init__()
        n_state = cfg.d_hidden  # in Attention: n_state=768 (nx=d_embed)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, n_state)
        self.c_proj = Conv1D(n_state, 1, n_state)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)
        self.attn = None
        self.config = cfg

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
    
    def draw_attentions(self, sent1, sent2, name=""):
        if self.attn is None: return

        for h in range(self.n_head):
            fig, axs = plt.subplots(1,1, figsize=(10, 10))
            draw(self.attn[0, h].data,sent1, sent2, ax=axs)
            fig.savefig(self.config.save_path + "/" + f"{strftime('%H:%M:%S', gmtime())}_attn_head_{h}_{name}.png")
            plt.close(fig)

    def forward(self, x, context=None):
        query = x
        key = value = context

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