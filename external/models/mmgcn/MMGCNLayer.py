from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
from torch_geometric.nn.inits import uniform


class MMGCNLayer(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim, aggregation):
        super(MMGCNLayer, self).__init__(aggr=aggregation)
        self.aggregation = aggregation
        self.weight = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        uniform(in_dim, self.weight)

    def forward(self, x, edge_index):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size()[0], x.size()[0]), x=x)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
