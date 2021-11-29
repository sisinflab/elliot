from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class DGCFLayer(MessagePassing, ABC):
    def __init__(self):
        super(DGCFLayer, self).__init__(aggr='add', node_dim=-3)

    def forward(self, x, edge_index, edge_index_intents):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        normalized_edge_index_intents = torch.softmax(edge_index_intents, dim=0)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_i, x_j):
        p = torch.softmax(torch.sum(x_i * x_j, dim=2), dim=0)
        return torch.unsqueeze(p, 2) * x_j
