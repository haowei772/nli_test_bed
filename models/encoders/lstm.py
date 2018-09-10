import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()

        self.config = config

        # input size
        input_size = config.d_proj if config.projection else config.d_embed

        # dropout
        dropout = 0 if config.n_layers == 1 else config.dp_ratio

        # LSTM
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=dropout,
                        bidirectional=config.birnn)
    
    def forward(self, inputs, inputs_embed):
        """
        inputs (batch_size, seq_len, input_size)
        output (batch_size, d_hidden)
        """
        batch_size = inputs_embed.size()[0]
        inputs_embed = inputs_embed.transpose(0,1)
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 =  inputs_embed.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0,1).contiguous().view(batch_size, -1)

