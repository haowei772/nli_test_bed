import torch.nn as nn

from .embedding import Embedding
from .encoder import Encoder
from .positional_encoding import PositionalEncoding
from .aggregator import Aggregator

class NLI(nn.Module):

    def __init__(self, config, vocab):
        super(NLI, self).__init__()

        # embedding
        self.embed = Embedding(config, vocab)

        # pos encoding
        self.pos_encoding = None
        if config.positional_encoding:
            self.pos_encoding = PositionalEncoding(config)

        # encoder
        self.encode_p = Encoder(config)
        self.encode_h = Encoder(config)

        # aggregator
        self.aggregate = Aggregator(config)




    def forward(self, batch):

        # ----- embedding -----
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)

        # ----- pos encoding -----
        if self.pos_encoding:
            prem_embed = self.pos_encoding(prem_embed)
            hypo_embed = self.pos_encoding(hypo_embed)


        # ----- encoder -----
        premise = self.encode_p(batch.premise, prem_embed, batch.hypothesis, 
            hypo_embed)
        hypothesis = self.encode_h(batch.hypothesis, hypo_embed, batch.premise, 
            prem_embed)

        # ----- aggregator -----
        scores = self.aggregate(premise, hypothesis)


        return scores

