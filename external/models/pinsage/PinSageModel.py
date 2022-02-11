"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import networkx as nx

from .PinSageLayer import PinSageLayer
import torch
import torch_geometric
import numpy as np
import random
from collections import OrderedDict


class PinSageModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 message_weight_size,
                 convolution_weight_size,
                 out_weight_size,
                 t_top_nodes,
                 n_layers,
                 delta,
                 adj,
                 random_seed,
                 name="PinSage",
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
        self.message_weight_size = message_weight_size
        self.convolution_weight_size = convolution_weight_size
        self.out_weight_size = out_weight_size
        self.t_top_nodes = t_top_nodes
        self.n_layers = n_layers
        self.delta = delta
        self.message_weight_size_list = [self.embed_k] + self.message_weight_size
        self.convolution_weight_size_list = list(self.convolution_weight_size)
        self.out_weight_size_list = list(self.out_weight_size)
        self.adj = adj

        self.Gu = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        # self._get_personalized_page_rank()

        propagation_network_list = [(PinSageLayer(self.message_weight_size_list[0],
                                                  self.message_weight_size_list[1],
                                                  self.convolution_weight_size_list[0]), 'x, edge_index -> x')]

        for layer in range(1, self.n_layers):
            propagation_network_list.append((PinSageLayer(self.convolution_weight_size_list[layer - 1],
                                                          self.message_weight_size_list[layer + 1],
                                                          self.convolution_weight_size_list[layer]),
                                             'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)

        out_network_list = [('out_0', torch.nn.Linear(in_features=self.convolution_weight_size_list[-1],
                                                      out_features=self.out_weight_size_list[0]))]
        out_network_list += [('relu_0', torch.nn.ReLU())]
        out_network_list += [('out_1', torch.nn.Linear(in_features=self.out_weight_size_list[0],
                                                       out_features=self.out_weight_size_list[1], bias=False))]

        self.out_network = torch.nn.Sequential(OrderedDict(out_network_list))

        self.propagation_network.to(self.device)
        self.out_network.to(self.device)

        self.loss = torch.nn.MarginRankingLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # def _get_personalized_page_rank(self):
    #
    #     def pagerank(node):
    #         return nx.pagerank(self.graph, personalization={node: 1})
    #
    #     results = []
    #     for n in list(self.graph.nodes):
    #         results.append(pagerank(n))

    def propagate_embeddings(self, evaluate=False):
        current_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)

        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    current_embeddings = list(
                        self.propagation_network.children()
                    )[layer](current_embeddings.to(self.device), self.adj.to(self.device))
            else:
                current_embeddings = list(
                    self.propagation_network.children()
                )[layer](current_embeddings.to(self.device), self.adj.to(self.device))

        if evaluate:
            self.out_network.eval()
            with torch.no_grad():
                current_embeddings = self.out_network(current_embeddings.to(self.device))
            self.propagation_network.train()
            self.out_network.train()
        else:
            current_embeddings = self.out_network(current_embeddings.to(self.device))

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

        loss = self.loss(xu_pos, xu_neg, torch.ones_like(xu_pos))
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
