import copy
import torch.nn as nn

from .embedding import Embedding, PositionalEncoding
from .encoders import Encoder
from .attentions import Attention
from .aggregators import Aggregator
from .reducers import Reducer

class Model(nn.Module):

    def __init__(self, config, vocab):
        super(Model, self).__init__()

        c = copy.deepcopy

        self.siamese = config.siamese

        # embedding
        self.embed = Embedding(config, vocab)

        # pos encoding
        self.pos_encoding = None
        if config.positional_encoding:
            self.pos_encoding = PositionalEncoding(config)

        # encoder
        self.encoders_p = []
        self.encoders_h = []
        for encoder_name in config.encoder:
            if config.siamese:
                this_encoder_p = this_encoder_h = Encoder(encoder_name, config)
            else:
                this_encoder_h = Encoder(encoder_name, config)
                this_encoder_p = Encoder(encoder_name, config)

            self.encoders_p.append(this_encoder_p)
            self.encoders_h.append(this_encoder_h)

        self.encoders_p = nn.ModuleList(self.encoders_p)
        self.encoders_h = nn.ModuleList(self.encoders_h)

        # attention
        self.attentions_p = []
        self.attentions_h = []
        for attention_name in config.attention:
            if config.siamese:
                this_attention_p = this_attention_h = Attention(attention_name, config)
            else:
                this_attention_h = Attention(attention_name, config)
                this_attention_p = Attention(attention_name, config)
            self.attentions_p.append(this_attention_p)
            self.attentions_h.append(this_attention_h)

        self.attentions_p = nn.ModuleList(self.attentions_p) if self.attentions_p else None
        self.attentions_h = nn.ModuleList(self.attentions_h) if self.attentions_h else None


        # reducer
        self.reduce = Reducer(config.reducer, config)

        # aggregator
        self.aggregate = Aggregator(config.aggregator, config)
            

    
    def forward(self, batch):

        # ----- embedding -----
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)


        # ----- pos encoding -----
        if self.pos_encoding:
            prem_embed = self.pos_encoding(prem_embed)
            hypo_embed = self.pos_encoding(hypo_embed)


        # ----- encoder -----
        premise_encoded = prem_embed
        for module in self.encoders_p:
            premise_encoded = module(premise_encoded)

        hypothesis_encoded = hypo_embed
        for module in self.encoders_h:
            hypothesis_encoded = module(hypothesis_encoded)


        # ----- attention -----
        if self.attentions_p is not None and self.attentions_h is not None:
            premise_encoded_attended = premise_encoded
            hypothesis_encoded_attended = hypothesis_encoded
            for attn_p, attn_h in zip(self.attentions_p, self.attentions_h):
                premise_encoded_attended = attn_p(premise_encoded_attended, hypothesis_encoded)
                hypothesis_encoded_attended = attn_p(hypothesis_encoded_attended, premise_encoded)
            premise_encoded = premise_encoded_attended
            hypothesis_encoded = hypothesis_encoded_attended


        # ----- reducer -----
        premise_reduced = self.reduce(premise_encoded)
        hypothesis_reduced = self.reduce(hypothesis_encoded)


        # ----- aggregator -----
        scores = self.aggregate(premise_reduced, hypothesis_reduced)


        return scores

