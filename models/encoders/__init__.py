from .lstm import LSTM
from .transformer import Transformer, TransformerInterAttention

encoders = {
    'lstm': LSTM,
    'transformer': Transformer,
    'transformer_inter_attention': TransformerInterAttention
}