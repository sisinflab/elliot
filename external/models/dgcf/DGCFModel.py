"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC

from .DGCFLayer import DGCFLayer

import torch
import torch_geometric
import numpy as np


class DGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 intents,
                 routing_iterations,
                 edge_index,
                 random_seed,
                 name="DGCF",
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
        self.n_layers = n_layers
        self.intents = intents
        self.routing_iterations = routing_iterations
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.edge_index_intents = torch.ones((self.intents, self.edge_index.shape[1]), dtype=torch.float32)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        dgcf_network_list = []
        for layer in range(self.n_layers):
            dgcf_network_list.append((DGCFLayer(), 'x, edge_index -> x'))

        self.dgcf_network = torch_geometric.nn.Sequential('x, edge_index', dgcf_network_list)
        self.dgcf_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        ego_embeddings = torch.reshape(torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0),
                                       (self.num_users + self.num_items, self.intents, self.embed_k // self.intents))
        all_embeddings = [ego_embeddings]
        row, col = self.edge_index
        col -= self.num_users

        for layer in range(self.n_layers):
            current_edge_index_intents = self.edge_index_intents.to(self.device)
            current_embeddings = all_embeddings[layer]
            _, current_0_gi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
            for routing in range(self.routing_iterations):
                if not evaluate:
                    current_embeddings = list(
                        self.dgcf_network.children()
                    )[layer](all_embeddings[layer].to(self.device),
                             self.edge_index.to(self.device),
                             current_edge_index_intents.to(self.device))
                    current_t_gu, _ = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
                    current_edge_index_intents += torch.sum(
                        current_t_gu[row].to(self.device) * torch.tanh(current_0_gi[col].to(self.device)).to(
                            self.device),
                        dim=-1).permute(1, 0)
                else:
                    self.dgcf_network.eval()
                    with torch.no_grad():
                        current_embeddings = list(
                            self.dgcf_network.children()
                        )[layer](all_embeddings[layer].to(self.device),
                                 self.edge_index.to(self.device),
                                 current_edge_index_intents.to(self.device))
                        current_t_gu, _ = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
                        current_edge_index_intents += torch.sum(
                            current_t_gu[row].to(self.device) * torch.tanh(current_0_gi[col].to(self.device)).to(
                                self.device),
                            dim=-1).permute(1, 0)
            self.edge_index_intents = current_edge_index_intents
            all_embeddings += [current_embeddings]

        if evaluate:
            self.dgcf_network.train()

        all_embeddings = sum(all_embeddings)
        return all_embeddings

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        all_embeddings = self.propagate_embeddings()

        # independence loss
        loss_ind = 0.0
        for intent in range(self.intents):
            for intent_p in range(self.intents):
                if intent != intent_p:
                    loss_ind += (torch.cov(
                        (all_embeddings[:, intent].to(self.device), all_embeddings[:, intent_p].to(self.device))) / (
                                     torch.sqrt(torch.var(all_embeddings[:, intent].to(self.device)) * torch.var(
                                         all_embeddings[:, intent_p].to(self.device)))))

        # bpr loss
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        gu, gi = torch.reshape(gu, (gu.shape[0], gu.shape[1] * gu.shape[2])), torch.reshape(gi, (
            gi.shape[0], gi.shape[1] * gi.shape[2]))
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user], gi[pos]))
        xu_neg = self.forward(inputs=(gu[user], gi[neg]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss_bpr = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2)) * 2
        loss_bpr += reg_loss

        # sum and optimize according to the overall loss
        loss = loss_bpr + loss_ind
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
