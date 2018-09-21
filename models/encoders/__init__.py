from .lstm import LSTM
from .transformer import Transformer, TransformerInterAttn
from .rnn import RNN
from .rnn_inter_attn import RNNInterAttn

encoders = {
    'rnn': RNN,
    'lstm': LSTM,
    'transformer': Transformer,
    'transformer_inter_attn': TransformerInterAttn,
    'rnn_inter_attn': RNNInterAttn
}