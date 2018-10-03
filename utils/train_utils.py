import os
import random
import time
import torch
import glob
import numpy as np
from tqdm import tqdm
from itertools import compress

def save_model(model, config, acc, iterations):
    snapshot_prefix = os.path.join(config.save_path, 'best_snapshot')
    snapshot_path = snapshot_prefix + "_" + config.run_name + '_acc_{:.2f}__iter_{}_model.pt'.format(acc, iterations)
    make_path(config.save_path)

    # save model, delete previous 'best_snapshot' files
    torch.save(model.state_dict(), snapshot_path)
    for f in glob.glob(snapshot_prefix + "_" + config.run_name + '*'):
        if f != snapshot_path:
            os.remove(f)


def restore_model(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("model restore path not found")
    model.load_state_dict(torch.load(path, map_location=device))


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


def run_epoch(logger, config, epoch, data, data_iter, model, loss_compute, device, mode='train', save_misclassified=False):
    """ Training function
    """
    start = time.time()
    loss, acc, size = 0, 0, 0
    data_iter.init_epoch()

    premise_wrong = []
    hypothesis_wrong = []
    correct_labels = []
    
    for i, batch in enumerate(tqdm(data_iter)):

        # ----- forward pass ----- 
        out = model(batch)

        # ----- loss compute and backprop ----- 
        batch_loss = loss_compute(out, batch)
        loss += batch_loss.item()

        # ----- accuracy ----- 
        _, pred = out.max(dim=1)
        res = (pred == batch.label)
        acc += (res).sum().float()
        size += len(pred)

        # ----- get misclassified samples -----
        # TODO: save it!
        if save_misclassified:
            res_not = (pred != batch.label)
            res_data = res_not.cpu().data.numpy().tolist()
            p_text = data.TEXT.reverse(batch.premise.data)
            h_text = data.TEXT.reverse(batch.hypothesis.data)

            premise_wrong.extend(list(compress(p_text, res_data)))
            hypothesis_wrong.extend(list(compress(h_text, res_data)))
            correct_labels.extend(batch.label.cpu().data.numpy().tolist())
        
        del batch_loss, out, pred


    elapsed = time.time() - start

    acc /= size
    acc = acc.cpu().data[0]

    logger.add_scalar(f"loss/{mode}", total_loss, epoch)
    logger.add_scalar(f"acc/{mode}", acc, epoch)

    
    return acc_total, loss