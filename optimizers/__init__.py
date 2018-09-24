from .adam import Adam
from .adam_scheduled import AdamScheduled
from .openai_adam import OpenAIAdam


optimizers = {
    'adam': Adam,
    'adam_scheduled': AdamScheduled,
    'adam_openai': OpenAIAdam,
}