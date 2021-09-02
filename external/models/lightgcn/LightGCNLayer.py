from abc import ABC

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightGCNLayer(MessagePassing, ABC):
    def __init__(self):
        super(LightGCNLayer, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
