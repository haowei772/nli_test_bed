import torch.nn as nn

from .embedding import Embedding, PositionalEncoding
from .encoders import Encoder, EncoderInterAttn
from .aggregators import Aggregator
from .reducers import Reducer

class NLIInterAttn(nn.Module):

    def __init__(self, config, vocab):
        super(NLIInterAttn, self).__init__()

        # embedding
        self.embed = Embedding(config, vocab)

        # pos encoding
        self.pos_encoding = None
        if config.positional_encoding:
            self.pos_encoding = PositionalEncoding(config)

        # encoder
        self.encode_p = Encoder(config)
        self.encode_h = Encoder(config)

        # encoder with inter attention
        self.encode_p_inter = EncoderInterAttn(config)
        self.encode_h_inter = EncoderInterAttn(config)

        # reducer
        self.reduce_p = Reducer(config)
        self.reduce_h = Reducer(config)


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
        premise_encoded = self.encode_p(prem_embed)
        hypothesis_encoded = self.encode_h(hypo_embed)


        # ----- encoder inter attention -----
        premise_encoded_inter = self.encode_p_inter(premise_encoded, hypothesis_encoded)
        hypothesis_encoded_inter = self.encode_h_inter(hypothesis_encoded, premise_encoded)


        # ----- reducer -----
        premise_reduced = self.reduce_p(premise_encoded_inter)
        hypothesis_reduced = self.reduce_h(hypothesis_encoded_inter)


        # ----- aggregator -----
        scores = self.aggregate(premise_reduced, hypothesis_reduced)


        return scores

