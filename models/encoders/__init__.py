from .lstm import LSTM
from .transformer import Transformer, TransformerInterAttention
from .transformer1 import Transformer1

encoders = {
    'lstm': LSTM,
    'transformer': Transformer,
    'transformer1': Transformer1,
    'transformer_inter_attention': TransformerInterAttention
}