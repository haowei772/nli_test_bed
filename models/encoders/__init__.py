from .lstm import LSTM
from .transformer import Transformer, TransformerWInput
from .rnn import RNN
from .rnn_inter_attn import RNNInterAttn

encoders = {
    'rnn': RNN,
    'lstm': LSTM,
    'transformer': Transformer,
    'transformer_w_input': TransformerWInput,
    'rnn_inter_attn': RNNInterAttn
}