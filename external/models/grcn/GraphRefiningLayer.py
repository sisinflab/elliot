from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul, mul_nnz
from torch_geometric.utils import softmax


class GraphRefiningLayer(MessagePassing, ABC):
    def __init__(self, rows, has_act, ptr, ptr_full=None):
        super(GraphRefiningLayer, self).__init__(aggr='add')
        self.rows = rows
        self.alpha = torch.repeat_interleave(torch.tensor([1]), self.rows.shape[0])
        self.has_act = has_act
        self.ptr = ptr
        self.ptr_full = ptr_full
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, rows_attr, cols_attr, edge_index, full=False):
        self.alpha = torch.mul(rows_attr, cols_attr).sum(dim=-1)
        if not full:
            self.alpha = softmax(src=self.alpha, ptr=self.ptr).view(-1)
        else:
            self.alpha = softmax(src=self.alpha, ptr=self.ptr_full).view(-1)
        edge_index = mul_nnz(edge_index, self.alpha, layout='coo')
        return self.leaky_relu(self.propagate(edge_index, x=x)) if self.has_act else self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
