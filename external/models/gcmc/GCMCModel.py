"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC
from torch_geometric.nn import GCNConv
from .NodeDropout import NodeDropout
from collections import OrderedDict

import torch
import torch_geometric
import numpy as np


class GCMCModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 convolutional_layer_size,
                 dense_layer_size,
                 n_convolutional_layers,
                 n_dense_layers,
                 node_dropout,
                 dense_layer_dropout,
                 edge_index,
                 random_seed,
                 name="GCMC",
                 **kwargs
                 ):
        super().__init__()
        torch.manual_seed(random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.convolutional_layer_size = [self.embed_k] + convolutional_layer_size
        self.dense_layer_size = [self.convolutional_layer_size[-1]] + dense_layer_size
        self.n_convolutional_layers = n_convolutional_layers
        self.n_dense_layers = n_dense_layers
        self.node_dropout = node_dropout if node_dropout else [0.0] * (
                self.n_convolutional_layers + self.n_dense_layers)
        self.dense_layer_dropout = dense_layer_dropout if dense_layer_dropout else [0.0] * self.n_dense_layers
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k))))
        self.Gi = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_items, self.embed_k))))
        self.Q = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.dense_layer_size[-1], self.dense_layer_size[-1]))))

        # Convolutional part
        convolutional_network_list = []
        for layer in range(self.n_convolutional_layers):
            convolutional_network_list.append((NodeDropout(self.node_dropout[layer], self.num_users, self.num_items),
                                               'edge_index -> edge_index'))
            convolutional_network_list.append((GCNConv(in_channels=self.convolutional_layer_size[layer],
                                                       out_channels=self.convolutional_layer_size[layer + 1],
                                                       add_self_loops=False,
                                                       bias=False), 'x, edge_index -> x'))
            convolutional_network_list.append((torch.nn.ReLU(), 'x -> x'))
        self.convolutional_network = torch_geometric.nn.Sequential('x, edge_index', convolutional_network_list)

        # Dense part
        dense_network_list = []
        for layer in range(self.n_dense_layers):
            dense_network_list.append(('dense_' + str(layer), torch.nn.Linear(in_features=self.dense_layer_size[layer],
                                                                              out_features=self.dense_layer_size[
                                                                                  layer + 1],
                                                                              bias=False)))
            dense_network_list.append(('relu_' + str(layer), torch.nn.ReLU()))
        self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _propagate_embeddings(self):
        current_embeddings = torch.cat((self.Gu, self.Gi), 0)

        for layer in range(0, self.n_convolutional_layers * 3, 3):
            dropout_edge_index = list(
                self.convolutional_network.children()
            )[0][layer](self.edge_index)
            current_embeddings = list(
                self.convolutional_network.children()
            )[0][layer + 1](current_embeddings, dropout_edge_index)
            current_embeddings = list(
                self.convolutional_network.children()
            )[0][layer + 2](current_embeddings)

        for layer in range(0, self.n_dense_layers * 2, 2):
            current_embeddings = list(
                self.dense_network.children()
            )[layer](current_embeddings)
            current_embeddings = list(
                self.dense_network.children()
            )[layer + 1](current_embeddings)

        Zu, Zi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
        return Zu, Zi

    def forward(self, inputs, **kwargs):
        Zu, Zi = self._propagate_embeddings()
        user, item = inputs
        zeta_u = torch.squeeze(Zu[user])
        zeta_i = torch.squeeze(Zi[item])

        xui = torch.sigmoid_(torch.matmul(torch.transpose(zeta_u, 0, 1), torch.matmul(self.Q, zeta_i)))

        return xui, zeta_u, zeta_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu[start:stop], torch.transpose(self.Gi, 0, 1))

    def train_step(self, batch):
        user, item = batch
        xui, zeta_u, zeta_i = self.forward(inputs=(user, item))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(gamma_u, 2) +
                               torch.norm(gamma_pos, 2) +
                               torch.norm(gamma_neg, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network.parameters()],
                                           dim=0).sum(dim=0)) * 2
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    @staticmethod
    def get_top_k(preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask), preds, torch.tensor(-np.inf)), k=k, sorted=True)
