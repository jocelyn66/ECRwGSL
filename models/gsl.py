import torch.nn as nn
from layers.gsl_layer import *
from torch_geometric.nn import GATConv


class GAT_Net(nn.Module):
    def __init__(self, in_features, hidden, out_features, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(in_features, hidden, heads=heads)
        self.gat2 = GATConv(hidden*heads, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        pass    # adj

        return F.log_softmax(x, dim=1)
