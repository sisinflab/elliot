from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from .MyLinear import MyLinear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul


class NGCFLayer(MessagePassing, ABC):
    def __init__(self, weights, bias, normalize=True):
        super(NGCFLayer, self).__init__(aggr='add')
        self.lin1 = MyLinear(weights[0], bias[0])
        self.lin2 = MyLinear(weights[1], bias[1])
        self.leaky_relu = torch.nn.LeakyReLU()
        self.normalize = normalize

    def forward(self, x, edge_index):
        if self.normalize:
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim), add_self_loops=True, dtype=x.dtype)
        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t, x):
        side_embeddings = matmul(adj_t, x, reduce=self.aggr)
        return self.leaky_relu(self.lin1(side_embeddings)) + self.leaky_relu(self.lin2(torch.mul(side_embeddings, x)))
