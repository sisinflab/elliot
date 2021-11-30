from abc import ABC

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch


class LGACNLayer(MessagePassing, ABC):
    def __init__(self, embed_k):
        super(LGACNLayer, self).__init__(aggr='add')
        self.tanh = torch.nn.Tanh()
        self.f = torch.nn.Linear(in_features=embed_k * 2, out_features=1, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg_row = degree(row, x.size(0), dtype=x.dtype)
        deg_col = degree(col, x.size(0), dtype=x.dtype)
        deg = deg_row + deg_col
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_i, x_j, norm):
        alpha_i_j = self.tanh(self.f(torch.cat((x_i, x_j), dim=1)))
        return norm.view(-1, 1) * alpha_i_j * x_j
