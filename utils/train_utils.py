import random
import time
import torch
import numpy as np
from config import config


def get_device_info(config):
    """ get device info
    """
    device = torch.device("cuda") if torch.cuda.is_available() and config.gpu else 'cpu'
    n_gpu = torch.cuda.device_count()
    print("device:", device)
    print("number of gpus:", n_gpu)
    return device, n_gpu


def set_seed():
    """ Set random seeds
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def run_epoch(epoch, data_iter, model, loss_compute, device, mode='train'):
    """ Training function
    """
    start = time.time()
    total_loss, n_correct, n_total = 0, 0, 0
    data_iter.init_epoch()

    for i, batch in enumerate(data_iter):

        # ----- forward pass ----- 
        out = model(batch)

        # ----- loss compute and backprop ----- 
        loss = loss_compute(out, batch)

        # ----- total loss ----- 
        total_loss += loss

        # ----- accuracy ----- 
        n_correct += (torch.max(out, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        acc = 100. * n_correct/n_total

        # ----- log ----- 
        if i % config.print_every_n_batch == 1:
            elapsed = time.time() - start
            print(f"Epoch: {epoch} Step: {i} Loss: {loss} Accuracy: {acc} Elapsed Time: {elapsed}")
            start = time.time()
    
    return total_loss