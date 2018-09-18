from .lstm import LSTM
from .transformer import Transformer, TransformerWInput

encoders = {
    'lstm': LSTM,
    'transformer': Transformer,
    'transformer_w_input': TransformerWInput,
}