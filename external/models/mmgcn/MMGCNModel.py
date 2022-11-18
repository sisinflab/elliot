"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

from .MMGCNLayer import MMGCNLayer
import torch
import torch_geometric
from collections import OrderedDict
import numpy as np
import random


class MMGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 l_w,
                 num_layers,
                 modalities,
                 aggregation,
                 concatenation,
                 has_id,
                 multimodal_features,
                 adj,
                 random_seed,
                 name="MMGCN",
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
        self.aggregation = aggregation
        self.concatenation = concatenation
        self.n_layers = num_layers
        self.has_id = has_id
        self.adj = adj

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # multimodal collaborative embeddings
        self.Gum = torch.nn.ParameterDict()
        self.proj_multimodal = torch.nn.ModuleDict()
        self.propagation_network_multimodal = torch.nn.ModuleDict()
        self.linear_network_multimodal = torch.nn.ModuleDict()
        self.g_linear_network_multimodal = torch.nn.ModuleDict()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        for m_id, m in enumerate(modalities):
            if self.embed_k_multimod[m_id]:
                self.Gum[m] = torch.nn.Embedding(self.num_users, self.embed_k_multimod[m_id]).weight
                torch.nn.init.xavier_uniform_(self.Gum[m])
                self.Gum[m].to(self.device)
                self.proj_multimodal[m] = torch.nn.Linear(self.multimodal_features_shapes[m_id],
                                                          self.embed_k_multimod[m_id])
                self.proj_multimodal[m].to(self.device)
                propagation_network_list = [(MMGCNLayer(self.embed_k_multimod[m_id],
                                                        self.embed_k_multimod[m_id],
                                                        self.aggregation), 'x, edge_index -> x')]
                torch.nn.init.xavier_uniform_(propagation_network_list[0][0].weight)
                linear_network_list = [('linear_0', torch.nn.Linear(self.embed_k_multimod[m_id], self.embed_k))]
                torch.nn.init.xavier_uniform_(linear_network_list[0][1].weight)
                g_linear_network_list = [('g_linear_0', torch.nn.Linear(self.embed_k_multimod[m_id] + self.embed_k,
                                                                        self.embed_k)) if self.concatenation else
                                         (('g_linear_0', torch.nn.Linear(self.embed_k_multimod[m_id],
                                                                         self.embed_k)))]
                torch.nn.init.xavier_uniform_(g_linear_network_list[0][1].weight)
            else:
                self.Gum[m] = torch.nn.Embedding(self.num_users, self.multimodal_features_shapes[m_id]).weight
                torch.nn.init.xavier_uniform_(self.Gum[m])
                self.Gum[m].to(self.device)
                propagation_network_list = [(MMGCNLayer(self.multimodal_features_shapes[m_id],
                                                        self.multimodal_features_shapes[m_id],
                                                        self.aggregation), 'x, edge_index -> x')]
                torch.nn.init.xavier_uniform_(propagation_network_list[0][0].weight)
                linear_network_list = [('linear_0', torch.nn.Linear(self.multimodal_features_shapes[m_id],
                                                                    self.embed_k))]
                torch.nn.init.xavier_uniform_(linear_network_list[0][1].weight)
                g_linear_network_list = [
                    ('g_linear_0', torch.nn.Linear(self.multimodal_features_shapes[m_id] + self.embed_k,
                                                   self.embed_k)) if self.concatenation else
                    (('g_linear_0', torch.nn.Linear(self.multimodal_features_shapes[m_id],
                                                    self.embed_k)))]
                torch.nn.init.xavier_uniform_(g_linear_network_list[0][1].weight)
            for layer in range(1, self.n_layers):
                propagation_network_list.append((MMGCNLayer(self.embed_k,
                                                            self.embed_k,
                                                            self.aggregation), 'x, edge_index -> x'))
                torch.nn.init.xavier_uniform_(propagation_network_list[layer][0].weight)
                linear_network_list.append(
                    (f'linear_{layer}', torch.nn.Linear(self.embed_k, self.embed_k)))
                torch.nn.init.xavier_uniform_(linear_network_list[layer][1].weight)
                g_linear_network_list.append((f'g_linear_{layer}', torch.nn.Linear(self.embed_k + self.embed_k,
                                                                                   self.embed_k))
                                             if self.concatenation else
                                             (f'g_linear_{layer}', torch.nn.Linear(self.embed_k,
                                                                                   self.embed_k)))
                torch.nn.init.xavier_uniform_(g_linear_network_list[layer][1].weight)
            self.propagation_network_multimodal[m] = torch_geometric.nn.Sequential('x, edge_index',
                                                                                   propagation_network_list)
            self.propagation_network_multimodal[m].to(self.device)
            self.linear_network_multimodal[m] = torch.nn.Sequential(OrderedDict(linear_network_list))
            self.linear_network_multimodal[m].to(self.device)
            self.g_linear_network_multimodal[m] = torch.nn.Sequential(OrderedDict(g_linear_network_list))
            self.g_linear_network_multimodal[m].to(self.device)

        # multimodal features
        self.Fm = []
        for m_id, m in enumerate(modalities):
            self.Fm += [torch.tensor(multimodal_features[m_id], dtype=torch.float32, device=self.device)]

        # placeholder for calculated user and item embeddings
        self.user_embeddings = torch.nn.init.xavier_uniform_(torch.rand((self.num_users, self.embed_k)))
        self.user_embeddings.to(self.device)
        self.item_embeddings = torch.nn.init.xavier_uniform_(torch.rand((self.num_items, self.embed_k)))
        self.item_embeddings.to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self):
        x_all_m = []
        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)

        for m_id, m in enumerate(self.modalities):
            temp_features = self.proj_multimodal[m](self.Fm[m_id].to(self.device)) if self.embed_k_multimod[m_id] else \
                self.Fm[m_id].to(self.device)
            x_all_m += [torch.nn.functional.normalize(
                torch.cat((self.Gum[m].to(self.device), temp_features.to(self.device)), 0))]
            for layer in range(self.n_layers):
                h = torch.nn.functional.leaky_relu(list(
                    self.propagation_network_multimodal[m].children()
                )[layer](x_all_m[m_id].to(self.device), self.adj.to(self.device)))
                x_hat = torch.nn.functional.leaky_relu(
                    list(self.linear_network_multimodal[m].children())[layer](
                        x_all_m[m_id].to(self.device))) + ego_embeddings if \
                    self.has_id else torch.nn.functional.leaky_relu(
                    list(self.linear_network_multimodal[m].children())[layer](x_all_m[m_id].to(self.device)))
                x_all_m[m_id] = torch.nn.functional.leaky_relu(
                    list(self.g_linear_network_multimodal[m].children())[layer](
                        torch.cat((h.to(self.device), x_hat.to(self.device)), dim=1))) if \
                    self.concatenation else torch.nn.functional.leaky_relu(
                    list(self.g_linear_network_multimodal[m].children())[layer](h) + x_hat.to(self.device))

        x_all = torch.stack(x_all_m)
        x_all = torch.mean(x_all, dim=0)
        gum, gim = torch.split(x_all, [self.num_users, self.num_items], 0)
        self.user_embeddings = gum
        self.item_embeddings = gim
        return gum, gim

    def forward(self, inputs, **kwargs):
        gum, gim = inputs
        gamma_u_m = torch.squeeze(gum).to(self.device)
        gamma_i_m = torch.squeeze(gim).to(self.device)

        xui = torch.sum(gamma_u_m * gamma_i_m, 1)

        return xui

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.user_embeddings[start: stop].to(self.device),
                            torch.transpose(self.item_embeddings.to(self.device), 0, 1))

    def train_step(self, batch):
        gum, gim = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gum[user[:, 0]], gim[pos[:, 0]]))
        xu_neg = self.forward(inputs=(gum[user[:, 0]], gim[neg[:, 0]]))

        loss = -torch.mean(torch.log(torch.sigmoid(xu_pos - xu_neg)))
        reg_loss = self.l_w * ((self.Gu.weight[np.concatenate([user[:, 0], user[:, 0]])].pow(2) +
                                self.Gi.weight[np.concatenate([pos[:, 0], neg[:, 0]])].pow(2)).mean() + self.Gum[
                                   self.modalities[0]].pow(2).mean())
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
