"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC

from .NGCFLayer import NGCFLayer
from .NodeDropout import NodeDropout
import torch
import torch_geometric
import numpy as np
import random

from torch_sparse import SparseTensor


class NGCFModel(torch.nn.Module, ABC):
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
                 name="NGFC",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

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

        self.adj = SparseTensor(row=torch.cat([self.edge_index[0], self.edge_index[1]], dim=0),
                                col=torch.cat([self.edge_index[1], self.edge_index[0]], dim=0),
                                sparse_sizes=(self.num_users + self.num_items,
                                              self.num_users + self.num_items))

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        propagation_network_list = []
        self.dropout_layers = []

        for layer in range(self.n_layers):
            propagation_network_list.append((NodeDropout(self.node_dropout[layer],
                                                         self.num_users,
                                                         self.num_items),
                                             'edge_index -> edge_index'))
            propagation_network_list.append((NGCFLayer(self.weight_size_list[layer],
                                                       self.weight_size_list[layer + 1]), 'x, edge_index -> x'))
            self.dropout_layers.append(torch.nn.Dropout(p=self.message_dropout[layer]))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]
        embedding_idx = 0

        for layer in range(0, self.n_layers * 2, 2):
            if not evaluate:
                dropout_edge_index = list(
                    self.propagation_network.children()
                )[layer](self.edge_index.to(self.device))
                adj = SparseTensor(row=torch.cat([dropout_edge_index[0], dropout_edge_index[1]], dim=0),
                                   col=torch.cat([dropout_edge_index[1], dropout_edge_index[0]], dim=0),
                                   sparse_sizes=(self.num_users + self.num_items,
                                                 self.num_users + self.num_items))
                all_embeddings += [self.dropout_layers[embedding_idx](list(
                    self.propagation_network.children()
                )[layer + 1](all_embeddings[embedding_idx].to(self.device), adj.to(self.device)))]
            else:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer + 1](all_embeddings[embedding_idx].to(self.device), self.adj.to(self.device))]

            embedding_idx += 1

        if evaluate:
            self.propagation_network.train()

        all_embeddings = torch.cat(all_embeddings, 1)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
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
        xu_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]]))
        xu_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network.parameters()],
                                           dim=0).sum(dim=0))
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
