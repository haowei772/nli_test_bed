from utils.utils import dotdict

config = dotdict({
    'model': 'snli_classifier',
    'dataset': 'snli',
    'tokenize': False,
    'seed': 42,
    'gpu': 0,
    'epochs': 10,
    'lr': 0.001,
    'batch_size': 16,
    'd_embed': 100,
    'd_proj': 300,
    'n_layers': 1,
    'n_cells': 1,
    'd_hidden': 300,
    'dp_ratio': 0.2,
    'print_every_n_batch': 100,
    'criterion': 'cross_entropy_loss',
    'optimizer': 'adam',
    'loss_compute': 'simple_loss_compute',
    'train_embed': True,
    'apply_weight_embed': True,
    'positional_encoding': True,
    'dropout_pe': 0.2,
    'max_len': 500, # TODO: not implemented yet
})