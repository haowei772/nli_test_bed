import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as O

from config import config
from utils.train_utils import get_device_info, set_seed, run_epoch
from utils.utils import (get_data, get_model, get_criterion, 
                            get_optimizer, get_loss_compute)

def main():
    # ----- set seed -----
    set_seed()

    # ----- get device -----
    device, n_gpu = get_device_info(config)
    config.device = device

    # ----- load data object -----
    print("Loading data")
    data = get_data(config)

    # ----- create or load model -----
    print("Loading model")
    model = get_model(config)
    model.to(device)

    # ----- create criterion -----
    print("Loading criterion")
    criterion = get_criterion(config)

    # ----- create optimizer -----
    print("Loading optimizer")
    optimizer = get_optimizer(config, model)

    # ----- get loss compute function -----
    print("Loading loss compute function")
    loss_compute = get_loss_compute(config, criterion, optimizer)
    loss_compute_dev = get_loss_compute(config, criterion, None)

    # ----- train -----
    print("Training")
    for i in range(config.epochs):
        model.train()
        t_loss = run_epoch(i, data.train_iter, model, loss_compute, device)

        model.eval()

if __name__ == "__main__":
    main()