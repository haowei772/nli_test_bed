from .lstm import LSTM
from .transformer import Transformer

encoders = {
    'lstm': LSTM,
    'transformer': Transformer,
}