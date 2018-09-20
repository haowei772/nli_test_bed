import os
import random
import time
import torch
import glob
import numpy as np
from tqdm import tqdm

def save_model(model, config, acc, iterations):
    snapshot_prefix = os.path.join(config.save_path, 'best_snapshot')
    snapshot_path = snapshot_prefix + '_acc_{}__iter_{}_model.pt'.format(acc, iterations)
    make_path(config.save_path)

    # save model, delete previous 'best_snapshot' files
    torch.save(model.state_dict(), snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)


def restore_model(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError("model restore path not found")
    model.load_state_dict(torch.load(path))


def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

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


def set_seed(config):
    """ Set random seeds
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def run_epoch(logger, config, epoch, data_iter, model, loss_compute, device, mode='train'):
    """ Training function
    """
    start = time.time()
    total_loss, n_correct, n_total = 0, 0, 0
    data_iter.init_epoch()
    logger.log(mode="train")

    for batch in tqdm(data_iter):

        # ----- forward pass ----- 
        out = model(batch)

        # ----- loss compute and backprop ----- 
        loss = loss_compute(out, batch)
        total_loss += loss.data[0]

        # ----- accuracy ----- 
        n_correct += (torch.max(out, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        acc = 100. * n_correct/n_total


    elapsed = time.time() - start
    acc_total = n_correct/n_total*100
    loss = total_loss/len(data_iter)

    logger.log(mode=mode, epoch=epoch, acc_total=acc_total, loss=loss, elapsed=elapsed )
    
    return acc_total