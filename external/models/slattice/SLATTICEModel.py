from abc import ABC

from .SLATTICELayer import SLATTICELayer

import torch
import torch_geometric
import numpy as np
import random
from torch_sparse import SparseTensor, mul, sum


class SLATTICEModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_layers,
                 num_ui_layers,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 l_w,
                 interaction_modalities,
                 top_k,
                 sim_multimodal,
                 adj,
                 random_seed,
                 name="SLATTICE",
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
        self.interaction_modalities = interaction_modalities
        self.top_k = top_k
        self.n_layers = num_layers
        self.n_ui_layers = num_ui_layers
        self.sim_multimodal = sim_multimodal
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
        self.projection_m = torch.nn.ModuleDict()
        self.Sim = dict()
        ir = torch.tensor(list(range(self.num_items)), dtype=torch.int64, device=self.device)
        self.items_rows = torch.repeat_interleave(ir, self.top_k).to(self.device)
        for m_id, m in enumerate(self.interaction_modalities):
            self.Gim[m] = torch.nn.Embedding(self.num_items, self.embed_k_multimod).weight
            self.Gim[m].to(self.device)
            current_sim = self.sim_multimodal[m_id].to_dense().detach()
            weighted_adj = self.build_knn_neighbourhood(current_sim, self.top_k)
            self.Sim[m] = self.compute_normalized_laplacian(weighted_adj)
            self.Sim[m].to(self.device)
            self.projection_m[m] = torch.nn.Linear(in_features=self.embed_k_multimod,
                                                   out_features=self.embed_k)
            self.projection_m[m].to(self.device)

        # graph convolutional network for item-item multimodal graphs
        propagation_network_list = []
        for layer in range(self.n_layers):
            propagation_network_list.append((SLATTICELayer(), 'x, edge_index -> x'))
        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        # lightgcn as user-item graph model for recommendation
        propagation_network_list = []
        for layer in range(self.n_ui_layers):
            propagation_network_list.append((SLATTICELayer(), 'x, edge_index -> x'))

        self.propagation_network_recommend = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network_recommend.to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    def build_knn_neighbourhood(self, adj, topk):
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        items_cols = torch.flatten(knn_ind).to(self.device)
        values = torch.flatten(knn_val).to(self.device)
        weighted_adj = SparseTensor(row=self.items_rows,
                                    col=items_cols,
                                    value=values,
                                    sparse_sizes=(self.num_items, self.num_items))
        return weighted_adj

    @staticmethod
    def compute_normalized_laplacian(adj):
        deg = sum(adj, dim=-1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.96 ** (epoch / 50))
        return scheduler

    def propagate_embeddings(self):
        item_embeddings_m = []
        for m_id, m in enumerate(self.interaction_modalities):
            item_embedding = self.Gim[m]
            for layer in range(self.n_layers):
                item_embedding = list(self.propagation_network.children())[layer](item_embedding.to(self.device),
                                                                                  self.Sim[m].to(self.device))
            item_embeddings_m += [self.projection_m[m](item_embedding.to(self.device))]

        item_embeddings_m = torch.stack(item_embeddings_m)
        item_embeddings_m = torch.sum(item_embeddings_m, dim=0)
        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(self.n_ui_layers):
            all_embeddings += [torch.nn.functional.normalize(list(
                self.propagation_network_recommend.children()
            )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device)), p=2, dim=1)]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi + torch.nn.functional.normalize(item_embeddings_m.to(self.device), p=2, dim=1)

    def forward(self, inputs, **kwargs):
        gum, gim = inputs
        gamma_u_m = torch.squeeze(gum).to(self.device)
        gamma_i_m = torch.squeeze(gim).to(self.device)

        xui = torch.sum(gamma_u_m * gamma_i_m, 1)

        return xui, gamma_u_m, gamma_i_m

    def predict(self, gum, gim, **kwargs):
        return torch.matmul(gum.to(self.device), torch.transpose(gim.to(self.device), 0, 1))

    def train_step(self, batch):
        gum, gim = self.propagate_embeddings()
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
