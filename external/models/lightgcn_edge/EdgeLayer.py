from abc import ABC

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class EdgeLayer(MessagePassing, ABC):
    def __init__(self, normalize=True):
        super(EdgeLayer, self).__init__(aggr='add')
        self.normalize = normalize

    def forward(self, x, edge_index, edge_attr):
        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)
        else:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return edge_attr * x_j
