from pkgutil import get_data
import torch.nn as nn
import gsl
from utils.name2object import name2gsl, name2init
from initializer import *


class ECRModel(nn.Module):

    def __init__(self, args, g_data):
        super(ECRModel, self).__init__(args)
        name2init(g_data)
        self.gsl = getattr(gsl, name2gsl[args.gsl])(args.rank, args.rank, args.rank, args.dropout)

    def forward(self, batch):
        adj = self.gsl.forward(batch)
        return adj
    