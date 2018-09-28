from .attention_vanilla import AttentionVanilla
from .attention_multihead import AttentionMultihead
from .transformer import Transformer, AttentionMultiHead
from .rnn import RNN
from .mean import Mean
from .last_layer import LastLayer
from .linear_aggregate import LinearAggregate

layers = {
    "attention_vanilla": AttentionVanilla,
    "attention_multihead": AttentionMultihead,
    'rnn': RNN,
    'transformer': Transformer,
    'attention_multi_head': AttentionMultiHead,
    'mean': Mean,
    'last_layer': LastLayer,
    'linear_aggregate': LinearAggregate
}