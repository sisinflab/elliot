from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GraphSAGELayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGELayer, self).__init__(aggr="mean")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.relu(self.lin(self.propagate(edge_index, x=x)))
