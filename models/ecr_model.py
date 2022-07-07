from pkgutil import get_data
import torch.nn as nn
import gsl
from utils.name2object import name2gsl, name2init
from initializer import *


class ECRModel(nn.Module):

    def __init__(self, args):
        super(ECRModel, self).__init__(args, g_data)
        name2init(g_data)
        self.gsl = getattr(gsl, name2gsl[args.gsl])(args.rank, args.rank, args.rank)

    def forward(self, batch):
        self.gsl.forward(batch)
    
