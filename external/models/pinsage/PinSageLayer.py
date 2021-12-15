from abc import ABC

import torch
from torch_geometric.nn import MessagePassing


class PinSageLayer(MessagePassing, ABC):
    def __init__(self, message_in_channels, message_out_channels, convolution_out_channels):
        super(PinSageLayer, self).__init__(aggr="mean")
        self.lin1 = torch.nn.Linear(message_in_channels, message_out_channels)
        self.lin2 = torch.nn.Linear(message_in_channels + message_out_channels, convolution_out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        out = self.relu(self.lin2(torch.cat((x, self.propagate(edge_index, x=x)), dim=1)))
        out = out / torch.unsqueeze(torch.norm(out, 2, dim=1), dim=1)
        return out

    def message(self, x_j):
        return self.relu(self.lin1(x_j))
