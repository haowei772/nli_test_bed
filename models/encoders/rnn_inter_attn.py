import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNInterAttn(nn.Module):
    def __init__(self, config):
        super(RNNInterAttn, self).__init__()

        rnn_cell = config.rnn_cell
        input_size = config.input_size
        hidden_size = config.hidden_size
        n_layers = config.n_layers
        birnn = config.birnn

        dropout = 0 if n_layers == 1 else config.dp_ratio

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.rnn = self.rnn_cell(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=n_layers,batch_first=True,
                                bidirectional=birnn, dropout=dropout)
        
        self.attn = Attention(hidden_size)

    def forward(self, inputs, context):

        output, hidden = self.rnn(inputs)
        output, attn = self.attn(output, context)

        return output


class Attention(nn.Module):
    """
    attend to a context.
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, input, context):
        batch_size = input.size(0)
        hidden_size = input.size(2)
        input_size = input.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(input, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
