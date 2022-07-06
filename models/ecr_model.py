import torch.nn as nn
import gsl
from utils.name2object import name2gsl, name2init
import initializer


class ECRModel(nn.Module):

    def __init__(self, args):
        super(ECRModel, self).__init__()
        pass
        self.initializer = getattr(initializer, name2init)(args) if args.init_g else None
        self.gsl = getattr(gsl, name2gsl[args.gsl])(args)

    def forward(self):
        pass
