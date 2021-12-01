from abc import ABC

import torch
from torch_geometric.nn import MessagePassing


class DGCFLayer(MessagePassing, ABC):
    def __init__(self):
        super(DGCFLayer, self).__init__(aggr='add', node_dim=-3)

    @staticmethod
    def weighted_degree(index, weights, num_nodes, dtype):
        out = torch.zeros((weights.shape[0], num_nodes,), dtype=dtype, device=weights.device)
        return out.scatter_add_(1, index.repeat(weights.shape[0], 1), weights)

    def forward(self, x, edge_index, edge_index_intents):
        normalized_edge_index_intents = torch.softmax(edge_index_intents, dim=0)
        row, col = edge_index
        deg_row = self.weighted_degree(index=row, weights=normalized_edge_index_intents, num_nodes=x.size(0),
                                       dtype=x.dtype)
        deg_col = self.weighted_degree(index=col, weights=normalized_edge_index_intents, num_nodes=x.size(0),
                                       dtype=x.dtype)
        deg = deg_row + deg_col
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[:, row] * deg_inv_sqrt[:, col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return torch.unsqueeze(norm.permute(1, 0), -1) * x_j
