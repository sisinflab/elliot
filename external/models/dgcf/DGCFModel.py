"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

from .DGCFLayer import DGCFLayer

import torch
import torch_geometric
import numpy as np
import random


class DGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w_bpr,
                 l_w_ind,
                 ind_batch_size,
                 n_layers,
                 intents,
                 routing_iterations,
                 edge_index,
                 random_seed,
                 name="DGCF",
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
        self.l_w_bpr = l_w_bpr
        self.l_w_ind = l_w_ind
        self.ind_batch_size = ind_batch_size
        self.n_layers = n_layers
        self.intents = intents
        self.routing_iterations = routing_iterations
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.edge_index_intents = torch.ones((self.intents, self.edge_index.shape[1]), dtype=torch.float32)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
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
        current_edge_index = self.edge_index.clone()
        row, col = current_edge_index[:, :current_edge_index.shape[1] // 2]
        col -= self.num_users

        for layer in range(self.n_layers):
            current_edge_index_intents = self.edge_index_intents.to(self.device)
            current_embeddings = all_embeddings[layer]
            current_0_gu, current_0_gi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
            for _ in range(self.routing_iterations):
                if not evaluate:
                    current_embeddings = list(
                        self.dgcf_network.children()
                    )[layer](all_embeddings[layer].to(self.device),
                             self.edge_index.to(self.device),
                             current_edge_index_intents.to(self.device))
                    current_t_gu, current_t_gi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
                    with torch.no_grad():  # the update is done manually, the tensor is not learned during the training
                        users_items = torch.sum(
                            current_t_gu[row].to(self.device) * torch.tanh(current_0_gi[col].to(self.device)).to(
                                self.device), dim=-1)
                        items_users = torch.sum(
                            current_t_gi[col].to(self.device) * torch.tanh(current_0_gu[row].to(self.device)).to(
                                self.device), dim=-1)
                        all_interactions = torch.cat([users_items, items_users], dim=0)
                        current_edge_index_intents = torch.softmax(current_edge_index_intents.clone(),
                                                                   dim=0) + all_interactions.permute(1, 0)
                else:
                    self.dgcf_network.eval()
                    with torch.no_grad():
                        current_embeddings = list(
                            self.dgcf_network.children()
                        )[layer](all_embeddings[layer].to(self.device),
                                 self.edge_index.to(self.device),
                                 current_edge_index_intents.to(self.device))
                        current_t_gu, current_t_gi = torch.split(current_embeddings, [self.num_users, self.num_items],
                                                                 0)
                        users_items = torch.sum(
                            current_t_gu[row].to(self.device) * torch.tanh(current_0_gi[col].to(self.device)).to(
                                self.device), dim=-1)
                        items_users = torch.sum(
                            current_t_gi[col].to(self.device) * torch.tanh(current_0_gu[row].to(self.device)).to(
                                self.device), dim=-1)
                        all_interactions = torch.cat([users_items, items_users], dim=0)
                        current_edge_index_intents = torch.softmax(current_edge_index_intents.clone(),
                                                                   dim=0) + all_interactions.permute(1, 0)
            self.edge_index_intents = current_edge_index_intents
            all_embeddings += [current_embeddings]

        if evaluate:
            self.dgcf_network.train()

        all_embeddings = sum(all_embeddings)
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

    def sample_users_items_for_loss_ind(self):
        # we sample ind_batch_size users and items because of memory issues as underlined in:
        # https://github.com/xiangwang1223/disentangled_graph_collaborative_filtering/blob/56dbc30ad82519d2d0655ca01eec935674923b7b/DGCF_v1/DGCF.py#311

        # sample users
        perm = torch.randperm(self.Gu.shape[0])
        sampled_users = perm[:self.ind_batch_size]

        # sample items
        perm = torch.randperm(self.Gi.shape[0])
        sampled_items = perm[:self.ind_batch_size]

        return sampled_users, sampled_items

    @staticmethod
    def get_loss_ind(x1, x2):
        # reference: https://recbole.io/docs/_modules/recbole/model/general_recommender/dgcf.html
        def _create_centered_distance(x):
            r = torch.sum(x * x, dim=1, keepdim=True)
            v = r - 2 * torch.mm(x, x.T + r.T)
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            D = torch.sqrt(v + 1e-8)
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(d1, d2):
            v = torch.sum(d1 * d2) / (d1.shape[0] * d1.shape[0])
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            dcov = torch.sqrt(v + 1e-8)
            return dcov

        D1 = _create_centered_distance(x1)
        D2 = _create_centered_distance(x2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        loss_ind = dcov_12 / (torch.sqrt(value) + 1e-10)
        return loss_ind

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()

        # independence loss
        loss_ind = torch.tensor(0.0, device=self.device)
        if self.intents > 1 and self.l_w_ind > 1e-9:
            sampled_users, sampled_items = self.sample_users_items_for_loss_ind()
            sampled_users.to(self.device)
            sampled_items.to(self.device)
            gu_sampled = gu[sampled_users]
            gi_sampled = gi[sampled_items]
            sampled_embeddings = torch.cat((gu_sampled.to(self.device), gi_sampled.to(self.device)), dim=0)
            for intent in range(self.intents - 1):
                loss_ind += self.get_loss_ind(sampled_embeddings[:, intent].to(self.device),
                                              sampled_embeddings[:, intent + 1].to(self.device))
            loss_ind /= ((self.intents + 1.0) * self.intents / 2)
            loss_ind *= self.l_w_ind

        # bpr loss
        gu, gi = torch.reshape(gu, (gu.shape[0], gu.shape[1] * gu.shape[2])), torch.reshape(gi, (
            gi.shape[0], gi.shape[1] * gi.shape[2]))
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user], gi[pos]))
        xu_neg = self.forward(inputs=(gu[user], gi[neg]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss_bpr = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w_bpr * (torch.norm(self.Gu, 2) +
                                   torch.norm(self.Gi, 2))
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
