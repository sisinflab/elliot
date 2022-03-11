"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

from .LATTICELayer import LATTICELayer
from torch_geometric.nn import LGConv

import torch
import torch_geometric
import numpy as np
import random
from torch_sparse import SparseTensor, mul, mul_nnz, fill_diag, sum, add


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
        self.importance_weights_m = torch.nn.ParameterList()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        ir = torch.tensor(list(range(self.num_items)), dtype=torch.int64, device=self.device)
        self.items_rows = torch.repeat_interleave(ir, self.top_k).to(self.device)
        for m_id, m in enumerate(modalities):
            self.Gim[m] = torch.nn.Embedding.from_pretrained(torch.nn.functional.normalize(
                torch.tensor(multimodal_features[m_id], dtype=torch.float32, device=self.device), p=2, dim=1),
                freeze=False).weight
            self.Gim[m].to(self.device)
            current_feature = torch.tensor(multimodal_features[m_id], dtype=torch.float32)
            current_feature = current_feature / torch.norm(current_feature, p=2, dim=-1, keepdim=True)
            current_sim = torch.mm(current_feature, current_feature.transpose(1, 0))
            knn_val, knn_ind = torch.topk(current_sim, self.top_k, dim=-1)
            items_cols = torch.flatten(knn_ind).to(self.device)
            values = torch.flatten(knn_val).to(self.device)
            weighted_adj = SparseTensor(row=self.items_rows,
                                        col=items_cols,
                                        value=values,
                                        sparse_sizes=(self.num_items, self.num_items))
            self.Sim[m] = self.apply_norm(weighted_adj, self.items_rows, items_cols)
            self.Sim[m].to(self.device)
            self.projection_m[m] = torch.nn.Linear(in_features=self.multimodal_features_shapes[m_id],
                                                   out_features=self.embed_k_multimod)
            self.projection_m[m].to(self.device)
            self.importance_weights_m.append(torch.nn.Parameter(
                torch.tensor(data=[float(1 / len(self.modalities))])))
            self.importance_weights_m[m_id].to(self.device)

        # graph convolutional network for item-item multimodal graphs
        propagation_network_list = []
        for layer in range(self.n_layers):
            propagation_network_list.append((LATTICELayer(), 'x, edge_index -> x'))
        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        # lightgcn as user-item graph model for recommendation
        propagation_network_list = []
        for layer in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=False), 'x, edge_index -> x'))

        self.propagation_network_recommend = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network_recommend.to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    def apply_norm(self, adj_t, items_rows, items_cols, add_self_loops=True):
        unweighted_adj = SparseTensor(row=items_rows,
                                      col=items_cols,
                                      sparse_sizes=(self.num_items, self.num_items))

        if add_self_loops:
            adj_t = fill_diag(adj_t, 1.)
        deg = sum(unweighted_adj, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.96 ** (epoch / 50))
        return scheduler

    def propagate_embeddings(self, build_item_graph=False, evaluate=False):
        projected_m = dict()
        for m_id, m in enumerate(self.modalities):
            projected_m[m] = self.projection_m[m](self.Gim[m].to(self.device))
        if build_item_graph:
            weights = torch.cat([torch.unsqueeze(w, 0) for w in self.importance_weights_m], dim=0)
            softmax_weights = torch.softmax(weights, dim=0)
            weighted_adjs = dict()
            learned_adj_addendum = []
            original_adj_addendum = []
            for m_id, m in enumerate(self.modalities):
                projected_m[m] = projected_m[m] / torch.norm(projected_m[m], p=2, dim=-1, keepdim=True)
                current_sim = torch.mm(projected_m[m], projected_m[m].transpose(1, 0))
                knn_val, knn_ind = torch.topk(current_sim, self.top_k, dim=-1)
                items_cols = torch.flatten(knn_ind).to(self.device)
                values = torch.flatten(knn_val).to(self.device)
                weighted_adj = SparseTensor(row=self.items_rows,
                                            col=items_cols,
                                            value=values,
                                            sparse_sizes=(self.num_items, self.num_items))
                weighted_adjs[m] = self.apply_norm(weighted_adj, self.items_rows, items_cols)
                learned_adj_addendum.append(mul_nnz(weighted_adjs[m],
                                                    softmax_weights[m_id].repeat((weighted_adjs[m].nnz(),)).to(
                                                        self.device),
                                                    layout='coo'))
                original_adj_addendum.append(mul_nnz(self.Sim[m],
                                                     softmax_weights[m_id].repeat((self.Sim[m].nnz(),)).to(
                                                         self.device),
                                                     layout='coo'))
            learned_adj = learned_adj_addendum[0]
            for i in range(1, len(learned_adj_addendum)):
                learned_adj = add(learned_adj, learned_adj_addendum[i])
            learned_adj = self.apply_norm(learned_adj, learned_adj.storage.row(), learned_adj.storage.col())
            original_adj = original_adj_addendum[0]
            for i in range(1, len(original_adj_addendum)):
                original_adj = add(original_adj, original_adj_addendum[i])
            original_adj = self.apply_norm(original_adj, original_adj.storage.row(), original_adj.storage.col())
            first = mul_nnz(learned_adj, torch.tensor([1 - self.l_m]).repeat((learned_adj.nnz(),)).to(self.device),
                            layout='coo')
            second = mul_nnz(original_adj, torch.tensor([self.l_m]).repeat((original_adj.nnz(),)).to(self.device),
                             layout='coo')
            self.Si = add(first, second)
        else:
            self.Si = self.Si.detach()

        item_embedding = self.Gi.weight
        for layer in range(self.n_layers):
            item_embedding = list(self.propagation_network.children())[layer](item_embedding.to(self.device),
                                                                                 self.Si)

        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_ui_layers):
            if evaluate:
                self.propagation_network_recommend.eval()
                with torch.no_grad():
                    all_embeddings += [torch.nn.functional.normalize(list(
                        self.propagation_network_recommend.children()
                    )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device)), p=2, dim=1)]
            else:
                all_embeddings += [torch.nn.functional.normalize(list(
                    self.propagation_network_recommend.children()
                )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device)), p=2, dim=1)]

        if evaluate:
            self.propagation_network_recommend.train()

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
