import os
import json
import time
from argparse import ArgumentParser
from pprint import pprint
from models.model import Model
from datasets import datasets
from criteria import criteria
from optimizers import optimizers
from loss_computers import loss_computers

class dotdict(dict):	
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def get_model(config, vocab):
	""" return model object
	"""
	return Model(config, vocab)
	# if not config.model:
	# 	raise NotImplementedError("'model' not defined in config")

	# if config.model not in models:
	# 	raise NotImplementedError("available models: ", [k for k,_ in models.items()])

	# return models[config.model](config, vocab)


def get_data(config):
	""" return dataset object
	"""
	if not config.dataset:
		raise NotImplementedError("'dataset' not defined in config")
	
	if config.dataset not in datasets:
		raise NotImplementedError("available datasets: ", [k for k,_ in datasets.items()])
	
	return datasets[config.dataset](config)


def get_criterion(config):
	if not config.criterion:
		raise NotImplementedError("'criterion' not defined in config")
	
	if config.criterion not in criteria:
		raise NotImplementedError("available criteria: ", [k for k,_ in criteria.items()])
	
	return criteria[config.criterion](config)


def get_optimizer(config, model):
	if not config.optimizer:
		raise NotImplementedError("'optimizer' not defined in config")

	if config.optimizer not in optimizers:
		raise NotImplementedError("available optimizers: ", [k for k,_ in optimizers.items()])
	
	return optimizers[config.optimizer](config, model)


def get_loss_compute(config, criterion, optimizer):
	if not config.loss_compute:
		raise NotImplementedError("'loss_compute' not defined in config")
	
	if config.loss_compute not in loss_computers:
		raise NotImplementedError("available loss_computers: ", [k for k,_ in loss_computers.items()])
	
	return loss_computers[config.loss_compute](config, criterion, optimizer)


def parse_args_get_config():
	parser = ArgumentParser()
	parser.add_argument('mode',
						choices=['train', 'test', 'interactive'],
						help="pipeline mode")
	parser.add_argument('config',
						help='model to be used')
	parser.add_argument('--gpu',
						default='0',
						help='index of GPU to be used (0 is CPU)')

	args = parser.parse_args()

	# ----- load config json file -----
	with open(args.config, 'r') as f:
		config = json.load(f)


	# ----- flattening the config structure -----
	config_flat = {}
	for k, v in config.items():
		for in_k, in_v in v.items():
    			config_flat[in_k] = in_v
	config = config_flat

	# ----- generate this run name -----
	run_name = config['name'] + "__" + time.strftime("%Y%m%d-%H%M%S")

	# ----- add info from argparser -----
	config.update({
		'mode': args.mode,
		'gpu': args.gpu,
		'run_name': run_name
	})

	return dotdict(config)


def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f