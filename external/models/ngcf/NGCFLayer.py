from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

torch.manual_seed(42)


class NGCFLayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels):
        super(NGCFLayer, self).__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, adj):
        # row, col = edge_index
        # deg_row = degree(row, x.size(0), dtype=x.dtype)
        # deg_col = degree(col, x.size(0), dtype=x.dtype)
        # deg = deg_row + deg_col
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        t1 = adj.matmul(x)
        t2 = t1.mul(x)

        # return self.leaky_relu(self.lin1(x) + self.propagate(edge_index, x=x, norm=norm))
        return self.leaky_relu(self.lin1(t1) + self.propagate(adj, x=t2))

    def message(self, x_i, x_j, norm):
        return norm.view(-1, 1) * (self.lin1(x_j) + self.lin2(x_i * x_j))

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
