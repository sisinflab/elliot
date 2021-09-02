"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC

from .LightGCNLayer import LightGCNLayer
import torch
import torch_geometric
import numpy as np


class LightGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 edge_index,
                 random_seed,
                 name="LightGCN",
                 **kwargs
                 ):
        super().__init__()
        torch.manual_seed(random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.n_layers = n_layers
        self.weight_size_list = [self.embed_k] * (self.n_layers + 1)
        self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))

        propagation_network_list = []

        for layer in range(self.n_layers):
            propagation_network_list.append((LightGCNLayer(), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _propagate_embeddings(self):
        # Extract gu_0 and gi_0 to begin embedding updating for L layers
        gu_0 = self.Gu
        gi_0 = self.Gi

        ego_embeddings = torch.cat((gu_0, gi_0), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_layers):
            all_embeddings += [list(
                self.propagation_network.children()
            )[0][layer](all_embeddings[layer], self.edge_index)]

        all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.Gu = torch.nn.Parameter(gu)
        self.Gi = torch.nn.Parameter(gi)

    def forward(self, inputs, **kwargs):
        user, item = inputs
        gamma_u = torch.squeeze(self.Gu[user])
        gamma_i = torch.squeeze(self.Gi[item])

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu[start:stop], torch.transpose(self.Gi, 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        self._propagate_embeddings()
        xu_pos, gamma_u, gamma_pos = self.forward(inputs=(user, pos))
        xu_neg, _, gamma_neg = self.forward(inputs=(user, neg))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(gamma_u, 2) +
                               torch.norm(gamma_pos, 2) +
                               torch.norm(gamma_neg, 2)) * 2
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    @staticmethod
    def get_top_k(preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask), preds, torch.tensor(-np.inf)), k=k, sorted=True)
