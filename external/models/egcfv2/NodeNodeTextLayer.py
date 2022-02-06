from abc import ABC

from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul, mul_nnz


class NodeNodeTextLayer(MessagePassing, ABC):
    def __init__(self, feature_dim, embed_dim, normalize=True):
        super(NodeNodeTextLayer, self).__init__(aggr='add')
        self.normalize = normalize
        self.lin1 = torch.nn.Linear(feature_dim, embed_dim)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, node_attr, edge_attr):
        edge_attr = self.leaky_relu(self.lin1(edge_attr))
        weights = torch.nn.functional.cosine_similarity(torch.mul(node_attr, edge_attr), node_attr, dim=1)
        edge_index = mul_nnz(edge_index, weights)

        print(edge_index)
        exit()

        if self.normalize:
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim), add_self_loops=True, dtype=x.dtype)

        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
