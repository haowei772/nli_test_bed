import copy
import torch
import torch.nn as nn
import numpy as np
from utils.model_utils import clones

from .embedding import Embedding, PositionalEncoding
from .layers import layers

class ModelTraversal(nn.Module):

    def __init__(self, config, vocab):
        super(ModelTraversal, self).__init__()

        c = copy.deepcopy
        self.config = config
        self.vocab = vocab
        self.nx = config.nx
        self.structure = config.structure


        self.create_model()


    def create_model(self):
        # ----- embedding -----
        self.embed = Embedding(self.config, self.vocab)


        # ----- pos encoding -----
        self.pos_encoding = None
        if self.config.positional_encoding:
            self.pos_encoding = PositionalEncoding(self.config)


        # ----- encoder ----- 
        nx_layers = []

        # TODO: using deepcopy gives an error
        # nx times
        for _ in range(self.config.nx):

            this_layers = []

            # creating layer
            for layer_n in self.config.structure:
                layer_info = self.config.layers[layer_n]
                layer = layers[layer_info['name']](dotdict(layer_info['config']))
                this_layers.append(layer)

            nx_layers.append(nn.ModuleList(this_layers))

        self.nx_layers = nn.ModuleList(nx_layers)


        # ----- reducer ----- 
        self.reduce = None
        if self.config.reducer:
            self.reduce = layers[self.config.reducer["name"]]\
                (dotdict(self.config.reducer["config"]))


        # ----- aggregator ----- 
        self.aggregate = layers[self.config.aggregator["name"]](dotdict(self.config.aggregator["config"]))


    def update(self, vectors, layer, layer_info):
        """ gets the inputs, run them through the layer, update the vectors dict
        """

        # ----- get inputs -----
        layer_inputs = [vectors[inp] for inp in layer_info["inputs"]]

        # ----- get output of layer -----
        layer_outputs = layer(*layer_inputs)

        for output, output_info in zip(layer_outputs, layer_info["outputs"]):
            name = output_info["name"]
            res_connection_add = output_info.get("res_connection_add", [])
            res_connection_concat = output_info.get("res_connection_concat", [])

            # if anything to add
            for res_add_vector in res_connection_add:
                output = output + vectors[res_add_vector]
            
            # if anything to concat
            for res_concat_vector in res_connection_concat:
                output = torch.cat([output, vectors[res_concat_vector]], -1)

            # updating
            vectors[name] = output
        
    def add_delimiter(self, prem, hypo):
        batch_size = prem.size(0)
        delimiter = prem.new_tensor(np.zeros(batch_size)).unsqueeze(-1)

        prem_delimiter_hypo = torch.cat([prem, delimiter, hypo], -1)

        return prem_delimiter_hypo


    def forward(self, batch):

        prem_delimiter_hypo = self.add_delimiter(batch.premise, batch.hypothesis)
        
        #######################
        # ----- embedding -----
        #######################
        prem_delimiter_hypo_embed = self.embed(prem_delimiter_hypo)


        ##########################
        # ----- pos encoding -----
        ##########################
        if self.pos_encoding:
            prem_delimiter_hypo_embed = self.pos_encoding(prem_delimiter_hypo_embed)

        vectors = {
            "pdh": prem_delimiter_hypo_embed,
        }


        #####################
        # ----- encoder -----
        #####################

        # ----- n times -----
        for layers in self.nx_layers:

            # ----- n times -----
            for layer_name, layer in zip(self.config.structure , layers):
                layer_info = self.config.layers[layer_name]
                self.update(vectors, layer, layer_info)
                



        #####################
        # ----- reducer -----
        #####################
        self.update(vectors, self.reduce, self.config.reducer)



        ########################
        # ----- aggregator -----
        ########################
        self.update(vectors, self.aggregate, self.config.aggregator)


        ########################
        # ----- scores -----
        ########################

        scores = vectors["scores"]

        del vectors

        return scores


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

class dotdict(dict):	
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def print_vectors(vectors):
    print("="* 79)
    for key,value in vectors.items():
        print(key + ": ", value.size())