import torch.nn as nn

from .attention_vanilla import AttentionVanilla
from .attention_multihead import AttentionMultihead

'''
Dict containing all the attention
Add new attention to this dict
'''
attentions = {
    "attention_vanilla": AttentionVanilla,
    "attention_multihead": AttentionMultihead,
}


'''
Main attention class
'''
class Attention(nn.Module):
    def __init__(self, attention_name, config):
        super(Attention, self).__init__()
        if attention_name not in attentions:
            raise NotImplementedError(f"reducer '{attention_name}' not implemented")
        
        self.attention = attentions[attention_name](config)
    
    def forward(self, x, context=None):
        """
        x=query
        context=key=value
        
        x (batch_size, _ ,d_hidden)
        context (batch_size, _ ,d_hidden)
        output (batch_size, _ ,d_hidden)
        """
        return self.attention(x, context)