"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

# from torch_geometric.nn import LGConv

import torch
import torch_geometric
import numpy as np
import random


class LATTICEModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_layers,
                 num_ui_layers,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 l_w,
                 modalities,
                 l_m,
                 top_k,
                 multimodal_features,
                 adj,
                 random_seed,
                 name="LATTICE",
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
        self.embed_k_multimod = embed_k_multimod
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.modalities = modalities
        self.l_m = l_m
        self.top_k = top_k
        self.n_layers = num_layers
        self.n_ui_layers = num_ui_layers
        self.adj = adj

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # multimodal features
        self.Gim = torch.nn.ParameterDict()
        self.Sim = dict()
        self.Si = None
        self.projection_m = torch.nn.ModuleDict()
        self.importance_weights_m = torch.nn.Parameter(
            torch.tensor(len(self.modalities) * [float(1 / len(self.modalities))]))
        self.importance_weights_m.to(self.device)
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        ir = torch.tensor(list(range(self.num_items)), dtype=torch.int64, device=self.device)
        self.items_rows = torch.repeat_interleave(ir, self.top_k).to(self.device)
        for m_id, m in enumerate(modalities):
            self.Gim[m] = torch.nn.Embedding.from_pretrained(
                torch.tensor(multimodal_features[m_id], dtype=torch.float32, device=self.device),
                freeze=False).weight
            self.Gim[m].to(self.device)
            current_sim = self.build_sim(self.Gim[m].detach())
            weighted_adj = self.build_knn_neighbourhood(current_sim)
            self.Sim[m] = self.compute_normalized_laplacian(weighted_adj)
            self.Sim[m].to(self.device)
            self.projection_m[m] = torch.nn.Linear(in_features=self.multimodal_features_shapes[m_id],
                                                   out_features=self.embed_k_multimod)
            self.projection_m[m].to(self.device)

        # lightgcn as user-item graph model for recommendation
        propagation_network_list = []
        for layer in range(self.n_ui_layers):
            propagation_network_list.append((LGConv(normalize=False), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    @staticmethod
    def build_sim(context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_knn_neighbourhood(self, adj):
        knn_val, knn_ind = torch.topk(adj, self.top_k, dim=-1)
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return weighted_adjacency_matrix

    @staticmethod
    def compute_normalized_laplacian(adj):
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return L_norm

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.96 ** (epoch / 50))
        return scheduler

    def propagate_embeddings(self, build_item_graph=False):
        if build_item_graph:
            weights = torch.cat([torch.unsqueeze(w, 0) for w in self.importance_weights_m], dim=0)
            softmax_weights = torch.softmax(weights, dim=0)
            learned_adj_addendum = []
            original_adj_addendum = []
            for m_id, m in enumerate(self.modalities):
                projected_m = self.projection_m[m](self.Gim[m].to(self.device))
                current_sim = self.build_sim(projected_m)
                weighted_adj = self.build_knn_neighbourhood(current_sim)
                learned_adj_addendum += [softmax_weights[m_id] * weighted_adj]
                original_adj_addendum += [softmax_weights[m_id] * self.Sim[m]]
            learned_adj = torch.stack(learned_adj_addendum, dim=1)
            learned_adj = learned_adj.sum(dim=1)
            learned_adj = self.compute_normalized_laplacian(learned_adj)
            original_adj = torch.stack(original_adj_addendum, dim=1)
            original_adj = original_adj.sum(dim=1)
            self.Si = (1 - self.l_m) * learned_adj + self.l_m * original_adj
        else:
            self.Si = self.Si.detach()

        item_embedding = self.Gi.weight
        for layer in range(self.n_layers):
            item_embedding = torch.mm(self.Si, item_embedding)

        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(self.n_ui_layers):
            all_embeddings += [torch.nn.functional.normalize(list(
                self.propagation_network.children()
            )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device)), p=2, dim=1)]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi + torch.nn.functional.normalize(item_embedding.to(self.device), p=2, dim=1)

    def forward(self, inputs, **kwargs):
        gum, gim = inputs
        gamma_u_m = torch.squeeze(gum).to(self.device)
        gamma_i_m = torch.squeeze(gim).to(self.device)

        xui = torch.sum(gamma_u_m * gamma_i_m, 1)

        return xui, gamma_u_m, gamma_i_m

    def predict(self, gum, gim, **kwargs):
        return torch.matmul(gum.to(self.device), torch.transpose(gim.to(self.device), 0, 1))

    def train_step(self, batch, build_item_graph):
        gum, gim = self.propagate_embeddings(build_item_graph)
        user, pos, neg = batch
        xu_pos, gamma_u_m, gamma_i_pos_m = self.forward(inputs=(gum[user], gim[pos]))
        xu_neg, _, gamma_i_neg_m = self.forward(inputs=(gum[user], gim[neg]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.mean(torch.nn.functional.softplus(-difference))
        reg_loss = self.l_w * (1 / 2) * (gamma_u_m.norm(2).pow(2) +
                                         gamma_i_pos_m.norm(2).pow(2) +
                                         gamma_i_neg_m.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
