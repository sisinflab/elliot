from abc import ABC

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch


class EdgeEdgeLayer(MessagePassing, ABC):
    def __init__(self, feature_dim, embed_dim):
        super(EdgeEdgeLayer, self).__init__(aggr='add')
        self.lin1 = torch.nn.Linear(feature_dim, embed_dim)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index):
        row, col = edge_index
        deg_row = degree(row, x.size(0), dtype=x.dtype)
        deg_col = degree(col, x.size(0), dtype=x.dtype)
        deg = deg_row + deg_col
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.leaky_relu(self.propagate(edge_index, x=x, norm=norm))

    def message(self, x_j, norm):
        return norm.view(-1, 1) * self.lin1(x_j)
