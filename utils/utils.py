from models import models
from datasets import datasets
from criteria import criteria
from optimizers import optimizers
from loss_computers import loss_computers

class dotdict(dict):	
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


def get_model(config):
	""" return model object
	"""
	if not config.model:
		raise NotImplementedError("'model' not defined in config")

	if config.model not in models:
		raise NotImplementedError(f"model {config.model} not implemented")

	return models[config.model](config)


def get_data(config):
	""" return dataset object
	"""
	if not config.dataset:
		raise NotImplementedError("'dataset' not defined in config")
	
	if config.dataset not in datasets:
		raise NotImplementedError(f"dataset {config.dataset} not implemented")
	
	return datasets[config.dataset](config)


def get_criterion(config):
	if not config.criterion:
		raise NotImplementedError("'criterion' not defined in config")
	
	if config.criterion not in criteria:
		raise NotImplementedError(f"criterion {config.criterion} not implemented")
	
	return criteria[config.criterion](config)


def get_optimizer(config, model):
	if not config.optimizer:
		raise NotImplementedError("'optimizer' not defined in config")

	if config.optimizer not in optimizers:
		raise NotImplementedError(f"optimizer '{config.optimizer}' not implemented")
	
	return optimizers[config.optimizer](config, model)


def get_loss_compute(config, criterion, optimizer):
	if not config.loss_compute:
		raise NotImplementedError("'loss_compute' not defined in config")
	
	if config.loss_compute not in loss_computers:
		raise NotImplementedError(f"loss_compute '{config.loss_compute}' not implemented")
	
	return loss_computers[config.loss_compute](config, criterion, optimizer)


	