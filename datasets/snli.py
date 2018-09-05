"""
Borrowed from https://github.com/galsang/BIMPM-pytorch
"""

from torchtext import data
from torchtext import datasets

from utils.text_utils import tokenize

class SNLI():
    def __init__(self, config):
        
        # dirty fix
        device = config.device
        if device == 'cpu':
            device = -1

        # if tokenize
        if not config.tokenize:
            tokenize = None

        # fields
        self.TEXT = data.ReversibleField(tokenize=tokenize, lower=True)
        self.LABEL = data.ReversibleField(sequential=False, unk_token=None)

        # data split
        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        # build vocabs
        self.TEXT.build_vocab(self.train, self.dev, self.test)
        self.LABEL.build_vocab(self.train)

        # create iterators
        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                        batch_sizes=(config.batch_size, 
                                                    config.batch_size, 
                                                    config.batch_size), 
                                        device=device
                                        )
        
        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        config.n_embed = len(self.TEXT.vocab)
        config.d_out = len(self.LABEL.vocab)


        