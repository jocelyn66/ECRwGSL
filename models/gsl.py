import torch.nn as nn
from layers.gsl_layer import *


class GAT(nn.Module):

    def __init__(self, args):
        super(GAT, self).__init__()
        pass

        assert args.n_layers > 0
        self.layers = nn.ModuleList()
        self.build_layers(args)
        self.name = args.gsl

    def build_layers(self, args):
        pass

        for i in range(self.n_layers):
            self.layers.append(
                GATLayer())

    def forward(self, examples):
        pass

        for _, layer in enumerate(self.layers):
            layer(examples)
