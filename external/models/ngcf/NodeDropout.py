from abc import ABC

import torch
import numpy as np


class NodeDropout(torch.nn.Module, ABC):
    # as presented in the original paper
    def __init__(self, node_dropout, num_users, num_items):
        super(NodeDropout, self).__init__()

        self.node_dropout = node_dropout
        self.num_users = num_users
        self.num_items = num_items
        self.all_nodes_indices = torch.arange(0, self.num_users + self.num_items)

    def forward(self, edge_index):
        if self.node_dropout:
            n_nodes_to_drop = round((self.num_users + self.num_items) * self.node_dropout)
            nodes_to_drop = self.all_nodes_indices[torch.randperm(self.num_users + self.num_items)][:n_nodes_to_drop]
            users_to_drop = nodes_to_drop[nodes_to_drop < self.num_users]
            items_to_drop = nodes_to_drop[nodes_to_drop >= self.num_users]
            mask_users = np.invert(np.in1d(edge_index[0].detach().cpu(), users_to_drop.numpy()))
            mask_items = np.invert(np.in1d(edge_index[1].detach().cpu(), items_to_drop.numpy()))
            mask_users_items = mask_users + mask_items
            dropout_users = torch.unsqueeze(edge_index[0, mask_users_items], 0)
            dropout_items = torch.unsqueeze(edge_index[1, mask_users_items], 0)
            return torch.cat([dropout_users, dropout_items])
        else:
            return edge_index
