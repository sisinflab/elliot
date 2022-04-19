from abc import ABC

from .UUIILayer import UUIILayer

import torch
import torch_geometric
import numpy as np
import random
from torch_sparse import SparseTensor, mul, sum


class UUIIModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_uu_layers,
                 num_ii_layers,
                 learning_rate,
                 embed_k,
                 l_w,
                 top_k_uu,
                 top_k_ii,
                 sim_uu,
                 sim_ii,
                 random_seed,
                 name="UUII",
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
        self.top_k_uu = top_k_uu
        self.top_k_ii = top_k_ii
        self.n_uu_layers = num_uu_layers
        self.n_ii_layers = num_ii_layers

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # user-user graph
        ur = torch.tensor(list(range(self.num_users)), dtype=torch.int64, device=self.device)
        users_rows = torch.repeat_interleave(ur, self.top_k_uu).to(self.device)
        weighted_adj_uu = self.build_knn_neighbourhood(sim_uu.to_dense().detach(), self.top_k_uu, users_rows, self.num_users)
        self.sim_uu = self.compute_normalized_laplacian(weighted_adj_uu)

        # item-item graph
        ir = torch.tensor(list(range(self.num_items)), dtype=torch.int64, device=self.device)
        items_rows = torch.repeat_interleave(ir, self.top_k_ii).to(self.device)
        weighted_adj_ii = self.build_knn_neighbourhood(sim_ii.to_dense().detach(), self.top_k_ii, items_rows, self.num_items)
        self.sim_ii = self.compute_normalized_laplacian(weighted_adj_ii)

        # graph convolutional network for user-user graph
        propagation_network_uu_list = []
        for layer in range(self.n_uu_layers):
            propagation_network_uu_list.append((UUIILayer(), 'x, edge_index -> x'))
        self.propagation_network_uu = torch_geometric.nn.Sequential('x, edge_index', propagation_network_uu_list)
        self.propagation_network_uu.to(self.device)

        # graph convolutional network for item-item graph
        propagation_network_ii_list = []
        for layer in range(self.n_ii_layers):
            propagation_network_ii_list.append((UUIILayer(), 'x, edge_index -> x'))
        self.propagation_network_ii = torch_geometric.nn.Sequential('x, edge_index', propagation_network_ii_list)
        self.propagation_network_ii.to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def build_knn_neighbourhood(self, adj, topk, rows, n_nodes):
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        cols = torch.flatten(knn_ind).to(self.device)
        values = torch.flatten(knn_val).to(self.device)
        weighted_adj = SparseTensor(row=rows,
                                    col=cols,
                                    value=values,
                                    sparse_sizes=(n_nodes, n_nodes))
        return weighted_adj

    @staticmethod
    def compute_normalized_laplacian(adj):
        deg = sum(adj, dim=-1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    def propagate_embeddings(self):
        user_embeddings = [self.Gu.weight.to(self.device)]
        for layer in range(self.n_uu_layers):
            user_embeddings += [list(self.propagation_network_uu.children())[layer](
                user_embeddings[layer].to(self.device),
                self.sim_uu.to(self.device))]

        item_embeddings = [self.Gi.weight.to(self.device)]
        for layer in range(self.n_ii_layers):
            item_embeddings += [list(self.propagation_network_ii.children())[layer](
                item_embeddings[layer].to(self.device),
                self.sim_ii.to(self.device))]

        user_embeddings = torch.stack(user_embeddings, dim=1)
        user_embeddings = user_embeddings.mean(dim=1, keepdim=False)

        item_embeddings = torch.stack(item_embeddings, dim=1)
        item_embeddings = item_embeddings.mean(dim=1, keepdim=False)
        return user_embeddings, item_embeddings

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(gu[user], gi[pos]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(gu[user], gi[neg]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.mean(torch.nn.functional.softplus(-difference))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
