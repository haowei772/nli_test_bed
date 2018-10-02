import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DecomposableEncoder(nn.Module):
    
    def __init__(self, config):
        super(DecomposableEncoder, self).__init__()
        self.config = config
        self.embedding_size = config.d_embed
        self.hidden_size = config.hidden_size
        self.para_init = config.para_init

        self.input_linear = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                # m.bias.data.uniform_(-0.01, 0.01)

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length (Long tensor)
        '''
        batch_size = sent1.size(0)

        sent1 = sent1.view(-1, self.embedding_size)
        sent2 = sent2.view(-1, self.embedding_size)

        sent1_linear = self.input_linear(sent1).view(
            batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear(sent2).view(
            batch_size, -1, self.hidden_size)

        return [sent1_linear, sent2_linear]