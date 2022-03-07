from abc import ABC

from torch_geometric.nn import MessagePassing
from torch_sparse import matmul


class LATTICELayer(MessagePassing, ABC):
    def __init__(self):
        super(LATTICELayer, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)