from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul, mul_nnz


class GraphConvolutionalLayer(MessagePassing, ABC):
    def __init__(self):
        super(GraphConvolutionalLayer, self).__init__(aggr='add')
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, weights):
        edge_index = mul_nnz(edge_index, weights, layout='coo')
        return self.leaky_relu(self.propagate(edge_index, x=x))

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
