from .attention_vanilla import AttentionVanilla
from .transformer import Transformer, AttentionMultiHead
from .rnn import RNN
from .mean import Mean
from .last_layer import LastLayer
from .linear_aggregate import LinearAggregate

layers = {
    "attention_vanilla": AttentionVanilla,
    "attention_multihead": AttentionMultiHead,
    'rnn': RNN,
    'transformer': Transformer,
    'mean': Mean,
    'last_layer': LastLayer,
    'linear_aggregate': LinearAggregate
}