from abc import ABC

import torch
from torch_geometric.nn import MessagePassing


class MMGCNLayer(MessagePassing, ABC):
    def __init__(self, d, dm, dmp, aggregation, combination):
        super(MMGCNLayer, self).__init__(aggr=aggregation)
        self.aggregation = aggregation
        self.combination = combination
        self.lin1 = torch.nn.Linear(dm, dmp, bias=False)
        self.lin2 = torch.nn.Linear(dm, d, bias=False)
        if self.combination == 'co':
            self.lin3 = torch.nn.Linear(dmp + d, dmp, bias=False)
        elif self.combination == 'ele':
            self.lin3 = torch.nn.Linear(dmp, d, bias=False)
        else:
            raise NotImplementedError('This aggregation function has not been implemented yet!')
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x_m, x_id, edge_index):
        h_m = self.propagate(edge_index, x=x_m)
        x_hat_m = self.leaky_relu(self.lin2(x_m)) + x_id
        if self.combination == 'co':
            return self.leaky_relu(self.lin3(torch.cat((h_m, x_hat_m), dim=1)))
        elif self.combination == 'ele':
            return self.leaky_relu(self.lin3(h_m) + x_hat_m)
        else:
            raise NotImplementedError('This aggregation function has not been implemented yet!')

    def message(self, x_j):
        return self.lin1(x_j)
