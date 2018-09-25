import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionVanilla(nn.Module):
    """
    attend to a context.
    """
    def __init__(self, config):
        super(AttentionVanilla, self).__init__()
        dim = config.hidden_size
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None
        self.attn = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, x, context):
        batch_size = x.size(0)
        hidden_size = x.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(x, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, x), dim=2)

        # x -> (batch, out_len, dim)
        x = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        self.attn = attn
        return x
