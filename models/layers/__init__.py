from .attention_vanilla import AttentionVanilla
from .transformer import Transformer, AttentionMultiHead
from .rnn import RNN
from .mean import Mean
from .sum import Sum
from .last_layer import LastLayer
from .linear_aggregate import LinearAggregate
from .decomposable_attn import DecomposableAttn
from .decomposable_encoder import DecomposableEncoder
from .bimpm import BIMPM

layers = {
    "attention_vanilla": AttentionVanilla,
    "attention_multihead": AttentionMultiHead,
    'rnn': RNN,
    'transformer': Transformer,
    'decomposable_attn': DecomposableAttn,
    'decomposable_encoder': DecomposableEncoder,
    'mean': Mean,
    'sum': Sum,
    'last_layer': LastLayer,
    'linear_aggregate': LinearAggregate,
    'bimpm': BIMPM
}