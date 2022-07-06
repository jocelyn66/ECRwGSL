import datetime
import os
import torch
import dgl
import torch.nn.functional as F

LOG_DIR = './logs/'


def get_savedir(dataset, model, init_g, gslearner, grid_search):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    gs = 'gs_' if grid_search else ''
    save_dir = os.path.join(LOG_DIR, date, dataset,
                            gs + model + '_' + init_g + '+' + '+' + gslearner + '_' + dt.strftime('_%H_%M_%S'))
    os.makedirs(save_dir)
    return save_dir


def count_params(model):
    """Count total number of trainable parameters in model"""
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l


operations = {
    'add': torch.add,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y.clamp_max(-1e-15) if y < 0 else x / y.clamp_min(1e-15),
    'max': torch.maximum,
    'min': torch.minimum,
}


activations = {
    'exp': torch.exp,
    'sig': torch.sigmoid,
    'soft': F.softplus,
    'tanh': torch.tanh,
    '': None
}
