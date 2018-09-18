from .adam import Adam
from .openai_adam import OpenAIAdam


optimizers = {
    'adam': Adam,
    'adam_openai': OpenAIAdam,
}