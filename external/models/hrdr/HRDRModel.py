"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import torch
import numpy as np
from collections import OrderedDict


class HRDRModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 weight_size,
                 n_layers,
                 node_dropout,
                 message_dropout,
                 edge_index,
                 random_seed,
                 name="HRDR",
                 **kwargs
                 ):
        super().__init__()
        torch.manual_seed(random_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.weight_size = weight_size
        self.n_layers = n_layers
        self.node_dropout = node_dropout if node_dropout else [0.0] * self.n_layers
        self.message_dropout = message_dropout if message_dropout else [0.0] * self.n_layers
        self.weight_size_list = [self.embed_k] + self.weight_size
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        # mlp for user and item ratings
        user_projection_rating_network_list = [('linear_0', torch.nn.Linear(
                in_features=self.num_items, out_features=self.user_projection_rating[0]
            )), ('relu_0', torch.nn.ReLU())]
        for layer in range(1, len(self.user_projection_rating)):
            user_projection_rating_network_list.append(('linear_{0}'.format(layer), torch.nn.Linear(
                in_features=self.user_projection_rating[layer], out_features=self.user_projection_rating[layer + 1]
            )))
            user_projection_rating_network_list.append(('relu_{0}'.format(layer), torch.nn.ReLU()))
        user_projection_rating_network = torch.nn.Sequential(OrderedDict(user_projection_rating_network_list))
        user_projection_rating_network.to(self.device)

        item_projection_rating_network_list = [('linear_0', torch.nn.Linear(
            in_features=self.num_users, out_features=self.item_projection_rating[0]
        )), ('relu_0', torch.nn.ReLU())]
        for layer in range(1, len(self.item_projection_rating)):
            item_projection_rating_network_list.append(('linear_{0}'.format(layer), torch.nn.Linear(
                in_features=self.item_projection_rating[layer], out_features=self.item_projection_rating[layer + 1]
            )))
            item_projection_rating_network_list.append(('relu_{0}'.format(layer), torch.nn.ReLU()))
        item_projection_rating_network = torch.nn.Sequential(OrderedDict(item_projection_rating_network_list))
        item_projection_rating_network.to(self.device)

        # cnn for user and item reviews
        user_review_cnn_network_list = [('conv_0', torch.nn.Conv2d(
                in_channels=1,
                out_channels=self.user_review_cnn[0],
                kernel_size=[3, 3]
        )), ('relu_0', torch.nn.ReLU())]
        for layer in range(1, len(self.user_review_cnn)):
            user_review_cnn_network_list.append(('conv_{0}'.format(layer), torch.nn.Conv2d(
                in_channels=self.user_review_cnn[layer],
                out_channels=self.user_review_cnn[layer + 1],
                kernel_size=[3, 3]
            )))
            user_review_cnn_network_list.append(('relu_{0}'.format(layer), torch.nn.ReLU()))
        user_review_cnn_network_list.append(torch.nn.MaxPool2d(kernel_size=[]))

        self.sigmoid = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, inputs, **kwargs):
        user, item, user_pos, item_pos = inputs
        xui = 0.0

        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        user, item, user_pos, item_pos = batch
        xui = self.forward(inputs=(user, item, user_pos, item_pos))

        loss = torch.square(xui - 1.0)
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network.parameters()],
                                           dim=0).sum(dim=0)) * 2
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
