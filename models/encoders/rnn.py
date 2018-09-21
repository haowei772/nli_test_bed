import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()

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

    def forward(self, x):
        output_x, (hx, cx) = self.rnn(x)
        return output_x
