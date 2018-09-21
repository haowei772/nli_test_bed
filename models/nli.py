import torch.nn as nn

from .embedding import Embedding, PositionalEncoding
from .encoders import Encoder
from .aggregators import Aggregator
from .reducers import Reducer

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
        self.encode = Encoder(config)

        self.encode_p = self.encode_h = self.encode

        # reducer
        self.reduce = Reducer(config)

        self.reduce_p = self.reduce_h = self.reduce

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
        premise = self.encode_p(prem_embed)
        hypothesis = self.encode_h(hypo_embed)


        # ----- reducer -----
        premise_reduced = self.reduce_p(premise)
        hypothesis_reduced = self.reduce_h(hypothesis)


        # ----- aggregator -----
        scores = self.aggregate(premise_reduced, hypothesis_reduced)


        return scores

