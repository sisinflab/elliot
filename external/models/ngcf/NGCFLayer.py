from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul


class NGCFLayer(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim, normalize=True):
        super(NGCFLayer, self).__init__(aggr='add')
        self.w1 = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.w2 = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.leaky_relu = torch.nn.LeakyReLU()
        self.normalize = normalize

    def forward(self, x, edge_index):
        if self.normalize:
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim), add_self_loops=True, dtype=x.dtype)
        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t, x):
        side_embeddings = matmul(adj_t, x, reduce=self.aggr)
        return self.leaky_relu(torch.matmul(side_embeddings, self.w1)) + \
               self.leaky_relu(torch.matmul(torch.mul(side_embeddings, x), self.w2))
