import os
import random
import time
import torch
import glob
import numpy as np
from tqdm import tqdm

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


def run_epoch(logger, config, epoch, data_iter, model, loss_compute, device, 
        dev_iter=None, dev_loss_compute=None, mode='train', log=True):
    """ Training function
    """
    start = time.time()
    total_loss, n_correct, n_total = 0, 0, 0
    data_iter.init_epoch()

    for i, batch in enumerate(tqdm(data_iter)):

        # ----- forward pass ----- 
        out = model(batch)

        # ----- loss compute and backprop ----- 
        loss = loss_compute(out, batch)
        total_loss += loss.item()

        # ----- accuracy ----- 
        n_correct += (torch.max(out, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        acc = 100. * n_correct/n_total

        # ----- log ----- 
        if log and i % config.print_every_n_batch == 1:
            logger.add_scalar(f"loss/{mode}", total_loss, epoch * len(data_iter) + i)
            logger.add_scalar(f"acc/{mode}", acc, epoch * len(data_iter) + i)

            # ----- dev ----- 
            if dev_iter:
                with torch.no_grad():
                    model.eval()
                    dev_acc, dev_loss = run_epoch(logger, config, i, dev_iter, 
                        model, dev_loss_compute, device, dev_iter=None, 
                        dev_loss_compute=None, mode='dev', log=False)
                        
                    model.train()
                    logger.add_scalar(f"loss/dev", dev_loss, epoch * len(data_iter) + i)
                    logger.add_scalar(f"acc/dev", dev_acc, epoch * len(data_iter) + i)
        
        del loss, out


    elapsed = time.time() - start
    acc_total = n_correct/n_total*100
    loss = total_loss/len(data_iter)

    logger.add_scalar(f"loss/{mode}", loss, epoch * len(data_iter) + i)
    logger.add_scalar(f"acc/{mode}", acc_total, epoch * len(data_iter) + i)
    # logger.log(mode=mode, epoch=epoch, acc_total=acc_total, loss=loss, elapsed=elapsed )
    
    return acc_total, loss