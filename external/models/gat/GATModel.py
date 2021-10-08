"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC
from torch_geometric.nn import GATConv

import torch
import torch_geometric
import numpy as np


class GATModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 weight_size,
                 n_layers,
                 heads,
                 message_dropout,
                 edge_index,
                 random_seed,
                 name="GAT",
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
        self.heads = heads
        self.message_dropout = message_dropout if message_dropout else [0.0] * self.n_layers
        self.weight_size_list = [self.embed_k] + self.weight_size
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        propagation_network_list = []

        for layer in range(self.n_layers - 1):
            propagation_network_list.append((GATConv(in_channels=self.weight_size_list[layer],
                                                     out_channels=self.weight_size_list[layer + 1],
                                                     heads=self.heads[layer],
                                                     dropout=self.message_dropout[layer],
                                                     add_self_loops=False,
                                                     concat=True), 'x, edge_index -> x'))
            propagation_network_list.append((torch.nn.ELU(), 'x -> x'))

        propagation_network_list.append((GATConv(in_channels=self.weight_size_list[self.n_layers - 1],
                                                 out_channels=self.weight_size_list[self.n_layers],
                                                 heads=self.heads[self.n_layers - 1],
                                                 dropout=self.message_dropout[self.n_layers - 1],
                                                 add_self_loops=False,
                                                 concat=False), 'x, edge_index -> x'))
        propagation_network_list.append((torch.nn.Identity(), 'x -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        current_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)

        for layer in range(0, self.n_layers * 2, 2):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    current_embeddings = list(
                        self.propagation_network.children()
                    )[0][layer](current_embeddings.to(self.device), self.edge_index.to(self.device))
                    current_embeddings = list(
                        self.propagation_network.children()
                    )[0][layer + 1](current_embeddings.to(self.device))
            else:
                current_embeddings = list(
                    self.propagation_network.children()
                )[0][layer](current_embeddings.to(self.device), self.edge_index.to(self.device))
                current_embeddings = list(
                    self.propagation_network.children()
                )[0][layer + 1](current_embeddings.to(self.device))

        if evaluate:
            self.propagation_network.train()

        gu, gi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user], gi[pos]))
        xu_neg = self.forward(inputs=(gu[user], gi[neg]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
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
