import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as O

from config import config
from utils.train_utils import get_device_info, set_seed, run_epoch, save_model
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
    model = get_model(config, data.vocab)
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
    best_dev_acc = -1
    for i in range(config.epochs):
        model.train()
        run_epoch(i, data.train_iter, model, loss_compute, device, mode='train')

        # ----- dev -----
        model.eval()
        with torch.no_grad():
            dev_acc = run_epoch(i, data.dev_iter, model, loss_compute_dev, device, mode='eval')
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, config, dev_acc, i)

if __name__ == "__main__":
    main()