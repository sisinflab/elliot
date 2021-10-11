"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC

from .FeatureProjection import FeatureProjection
from .DisenGCNLayer import DisenGCNLayer
from collections import OrderedDict

import torch
import torch_geometric
import numpy as np


class DisenGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 weight_size,
                 n_layers,
                 disen_k,
                 temperature,
                 routing_iterations,
                 message_dropout,
                 edge_index,
                 random_seed,
                 name="DisenGCN",
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
        self.disen_k = disen_k
        self.temperature = temperature
        self.routing_iterations = routing_iterations
        self.message_dropout = message_dropout if message_dropout else [0.0] * self.n_layers
        self.weight_size_list = [self.embed_k] + self.weight_size
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        disengcn_network_list = []
        for layer in range(self.n_layers):
            projection_layer = torch.nn.Sequential(OrderedDict([('feat_proj_' + str(layer), (FeatureProjection(
                self.weight_size_list[layer],
                self.weight_size_list[layer + 1],
                self.disen_k[layer])))]))
            disentangle_layer = torch_geometric.nn.Sequential('x, edge_index', [
                (DisenGCNLayer(self.temperature), 'x, edge_index -> x')])
            disengcn_network_list.append(('disen_gcn_' + str(layer), torch.nn.Sequential(projection_layer,
                                                                                         disentangle_layer)))
            disengcn_network_list.append(('dropout_' + str(layer), torch.nn.Dropout(self.message_dropout[layer])))

        self.disengcn_network = torch.nn.Sequential(OrderedDict(disengcn_network_list))
        self.disengcn_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, inputs, **kwargs):
        user, item, neigh_user, neigh_item = inputs
        current_edge_index_u = torch.tensor([[0] * len(neigh_user), list(range(1, len(neigh_user) + 1))])
        current_edge_index_i = torch.tensor([[0] * len(neigh_item), list(range(1, len(neigh_item) + 1))])
        users_embeddings = self.Gu[[user] + neigh_item]
        items_embeddings = self.Gi[[item] + neigh_user]
        embeddings_zeta_u = torch.cat((torch.unsqueeze(users_embeddings[0], 0), items_embeddings[1:]), 0)
        embeddings_zeta_i = torch.cat((torch.unsqueeze(items_embeddings[0], 0), users_embeddings[1:]), 0)
        for layer in range(0, self.n_layers * 2, 2):
            zeta_u = list(self.disengcn_network.children())[layer][0](embeddings_zeta_u.to(self.device))
            zeta_i = list(self.disengcn_network.children())[layer][0](embeddings_zeta_i.to(self.device))
            for t in range(self.routing_iterations):
                c_u = list(self.disengcn_network.children())[layer][1](zeta_u.to(self.device),
                                                                       current_edge_index_u.to(self.device))[0]
                c_i = list(self.disengcn_network.children())[layer][1](zeta_i.to(self.device),
                                                                       current_edge_index_i.to(self.device))[0]
                zeta_u[0] = c_u
                zeta_i[0] = c_i
            embeddings_zeta_u = zeta_u.reshape(zeta_u.shape[0], zeta_u.shape[1] * zeta_u.shape[2])
            embeddings_zeta_i = zeta_i.reshape(zeta_i.shape[0], zeta_i.shape[1] * zeta_i.shape[2])
            embeddings_zeta_u = list(self.disengcn_network.children())[layer + 1](embeddings_zeta_u.to(self.device))
            embeddings_zeta_i = list(self.disengcn_network.children())[layer + 1](embeddings_zeta_i.to(self.device))

        xui = torch.sum(torch.unsqueeze(embeddings_zeta_u[0], 0) * torch.unsqueeze(embeddings_zeta_i[0], 0), 1)

        return xui, embeddings_zeta_u[0], embeddings_zeta_i[0]

    def predict(self, start, stop, **kwargs):
        zeta_u = self.projection_network(self.Gu)
        zeta_i = self.projection_network(self.Gi)

        all_zeta = torch.cat((zeta_u, zeta_i), 0)
        self.disentangle_network.eval()
        all_zeta = self.disentangle_network(all_zeta, self.edge_index)
        self.disentangle_network.train()
        zeta_u, zeta_i = torch.split(all_zeta, [self.num_users, self.num_items], 0)
        c_u = zeta_u.reshape(zeta_u.shape[0], zeta_u.shape[1] * zeta_u.shape[2])
        c_i = zeta_i.reshape(zeta_i.shape[0], zeta_i.shape[1] * zeta_i.shape[2])

        return torch.matmul(c_u[start:stop], torch.transpose(c_i, 0, 1))

    def train_step(self, batch):
        user, pos, neg, neigh_user, neigh_pos_items, neigh_neg_items = batch
        xu_pos, zeta_u, zeta_i_pos = self.forward(inputs=(user, pos, neigh_user, neigh_pos_items))
        xu_neg, _, zeta_i_neg = self.forward(inputs=(user, neg, neigh_user, neigh_neg_items))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(zeta_u, 2) +
                               torch.norm(zeta_i_pos, 2) +
                               torch.norm(zeta_i_neg, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.disengcn_network.parameters()],
                                           dim=0).sum(dim=0)) * 2
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    @staticmethod
    def get_top_k(preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask), preds, torch.tensor(-np.inf)), k=k, sorted=True)
