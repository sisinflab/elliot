from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul


class NGCFLayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(NGCFLayer, self).__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.normalize = normalize

    def forward(self, x, edge_index):
        if self.normalize:
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim), add_self_loops=True, dtype=x.dtype)
        return self.leaky_relu(self.lin1(x) + self.propagate(edge_index, x=x))

    def message_and_aggregate(self, adj_t, x):
        return self.lin1(matmul(adj_t, x, reduce=self.aggr)) + \
               self.lin2(torch.mul(matmul(adj_t, x, reduce=self.aggr), x))
