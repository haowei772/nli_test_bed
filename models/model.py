import copy
import torch
import torch.nn as nn

from utils.model_utils import clones

from .embedding import Embedding, PositionalEncoding
from .layers import layers

class Model(nn.Module):

    def __init__(self, config, vocab):
        super(Model, self).__init__()

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
        nx_h_layers = []
        nx_p_layers = []
        nx_c_layers = []

        # TODO: using deepcopy gives an error
        # nx times
        for _ in range(self.config.nx):

            h_layers = []
            p_layers = []
            common_layers = []

            # creating layer
            for layer_n in self.config.structure:
                
                # when a layer has both h and p
                if all(x in self.config.layers[layer_n] for x in ["h", "p"]):

                    # if share layer
                    if self.config.layers[layer_n]["p"].get('share', False) or \
                        self.config.layers[layer_n]["h"].get('share', False):

                        h_layer_info = self.config.layers[layer_n]["h"]
                        h_layer = layers[h_layer_info['name']](dotdict(h_layer_info['config']))
                        h_layers.append(h_layer)
                        p_layers.append(h_layer)

                    else:

                        h_layer_info = self.config.layers[layer_n]["h"]
                        h_layer = layers[h_layer_info['name']](dotdict(h_layer_info['config']))
                        h_layers.append(h_layer)

                        p_layer_info = self.config.layers[layer_n]["p"]
                        p_layer = layers[p_layer_info['name']](dotdict(p_layer_info['config']))
                        p_layers.append(p_layer)

                    common_layers.append(Dummy())

                # when a layer is common
                else:
                    common_layer_info = self.config.layers[layer_n]
                    common_layer = layers[common_layer_info['name']](dotdict(common_layer_info['config']))
                    common_layers.append(common_layer)

                    h_layers.append(Dummy())
                    p_layers.append(Dummy())

            nx_h_layers.append(nn.ModuleList(h_layers))
            nx_p_layers.append(nn.ModuleList(p_layers))
            nx_c_layers.append(nn.ModuleList(common_layers))

        self.nx_h_layers = nn.ModuleList(nx_h_layers)
        self.nx_p_layers = nn.ModuleList(nx_p_layers)
        self.nx_c_layers = nn.ModuleList(nx_c_layers)

        # ----- reducer ----- 
        self.h_reduce = None
        self.p_reduce = None
        if self.config.reducer:
            # if share layer
            if self.config.reducer["p"].get('share', False) or \
                self.config.reducer["h"].get('share', False):

                self.h_reduce = self.p_reduce = layers[self.config.reducer["p"]["name"]]\
                    (dotdict(self.config.reducer["p"]["config"]))
            else:

                self.h_reduce = layers[self.config.reducer["h"]["name"]]\
                    (dotdict(self.config.reducer["h"]["config"]))
                self.p_reduce = layers[self.config.reducer["p"]["name"]]\
                    (dotdict(self.config.reducer["p"]["config"]))



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
    

    def forward(self, batch):
        
        #######################
        # ----- embedding -----
        #######################
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)


        ##########################
        # ----- pos encoding -----
        ##########################
        if self.pos_encoding:
            prem_embed = self.pos_encoding(prem_embed)
            hypo_embed = self.pos_encoding(hypo_embed)

        vectors = {
            "h": hypo_embed,
            "p": prem_embed
        }


        #####################
        # ----- encoder -----
        #####################

        # ----- n times -----
        for h_layers, p_layers, common_layers in \
            zip(self.nx_h_layers, self.nx_p_layers, self.nx_c_layers):

            # ----- n times -----
            for layer_name, layer_h, layer_p, layer_c in \
                zip(self.config.structure ,h_layers, p_layers, common_layers):

                layer_info = self.config.layers[layer_name]


                if not isinstance(layer_h, Dummy) and\
                    not isinstance(layer_p, Dummy):

                    self.update(vectors, layer_h, layer_info["h"])
                    self.update(vectors, layer_p, layer_info["p"])

                elif not isinstance(layer_c, Dummy):

                    self.update(vectors, layer_c, layer_info)




        #####################
        # ----- reducer -----
        #####################
        self.update(vectors, self.h_reduce, self.config.reducer["h"])
        self.update(vectors, self.p_reduce, self.config.reducer["p"])



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