from .nli import NLI
from .nli2 import NLI2
from .nli_siamese import NLISiamese

models = {
    'nli_siamese': NLISiamese,
    'nli': NLI,
}