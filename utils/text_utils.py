import os
import spacy
import torch

spacy_en = spacy.load('en')

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def add_vocab_vectors(field, config):
    """ given torchtext field, get vector cache if available
    if not, load, set, and save the vectors
    """
    if os.path.isfile(config.vector_cache):
        field.vocab.vectors = torch.load(config.vector_cache)
    else:
        field.vocab.load_vectors(config.word_vectors)
        makedirs(os.path.dirname(config.vector_cache))
        torch.save(field.vocab.vectors, config.vector_cache)

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise