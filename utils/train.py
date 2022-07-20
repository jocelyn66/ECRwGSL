import datetime
import os
import torch
import torch.nn.functional as F
import numpy as np 
import scipy.sparse as sp

LOG_DIR = '../logs/'


def get_savedir(dataset, model, encoder, decoder, grid_search):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    gs = 'gs_' if grid_search else ''
    save_dir = os.path.join(LOG_DIR, date, dataset,
                            gs + model + '_' + encoder + '+' + decoder + dt.strftime('_%H_%M_%S'))
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


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# 处理adj
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_new_item(dict, item, idx):
    if not item in dict.keys():
        dict[item] = []
    dict[item].append(idx)


def preprocess_adjacency(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5))  # ??
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized
