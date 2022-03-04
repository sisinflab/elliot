from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul, mul_nnz
from torch_geometric.utils import softmax


class GraphRefiningLayer(MessagePassing, ABC):
    def __init__(self, rows, size_rows):
        super(GraphRefiningLayer, self).__init__(aggr='add')
        self.leaky_relu = torch.nn.LeakyReLU()
        self.rows = rows
        self.size_rows = size_rows
        self.alpha = torch.repeat_interleave(torch.tensor([1]), self.rows.shape[0])

    def forward(self, x, rows_attr, cols_attr, edge_index):
        self.alpha = torch.mul(rows_attr, cols_attr).sum(dim=-1)
        # self.alpha = softmax(self.alpha, self.rows, self.size_rows).view(-1, 1)
        edge_index = mul_nnz(edge_index, self.alpha, layout='coo')
        return self.leaky_relu(self.propagate(edge_index, x=x))

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
