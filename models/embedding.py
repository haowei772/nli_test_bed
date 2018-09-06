import math
import torch.nn as nn

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
