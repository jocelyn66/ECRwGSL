import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class GATLayer(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, use_bias=False, act=None, dropout=0.0):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        if abs(dropout) > 1e-5:
            ValueError("dropout has not yet completed in layers.Linear")

    def forward(self, x):
        pass
