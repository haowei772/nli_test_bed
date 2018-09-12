import os
import random
import time
import torch
import glob
import numpy as np
from config import config

def save_model(model, config, acc, iterations):
    snapshot_prefix = os.path.join(config.save_path, 'best_snapshot')
    snapshot_path = snapshot_prefix + '_devacc_{}__iter_{}_model.pt'.format(dev_acc, iterations)

    # save model, delete previous 'best_snapshot' files
    torch.save(model, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)

def log_temporary(file, line):
    with open(file, "a") as myfile:
        myfile.write(line)


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
    acc_total = 0.0
    data_iter.init_epoch()

    for i, batch in enumerate(data_iter):

        # ----- forward pass ----- 
        out = model(batch)

        # ----- loss compute and backprop ----- 
        loss = loss_compute(out, batch)

        # ----- accuracy ----- 
        n_correct += (torch.max(out, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        acc = 100. * n_correct/n_total
        acc_total += acc

        # ----- log ----- 
        if i % config.print_every_n_batch == 1:
            elapsed = time.time() - start
            line = f"Mode: {mode} Epoch: {epoch} Step: {i} Loss: {loss.item()} Accuracy: {acc} Elapsed Time: {elapsed}\n"
            print(line)
            log_temporary(config.log_file, line)
            start = time.time()

    print("acc_total before: ", acc_total)
    acc_total = acc_total/len(data_iter)
    print("acc_total: ", acc_total)
    return acc_total