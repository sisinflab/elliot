"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

from .GraphRefiningLayer import GraphRefiningLayer
from .GraphConvolutionalLayer import GraphConvolutionalLayer
import torch
import torch_geometric
import numpy as np
import random


class GRCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 l_w,
                 num_layers,
                 num_routings,
                 modalities,
                 aggregation,
                 weight_mode,
                 pruning,
                 has_act,
                 fusion_mode,
                 multimodal_features,
                 adj,
                 adj_user,
                 rows,
                 cols,
                 ptr,
                 ptr_full,
                 random_seed,
                 name="GRCN",
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
        self.weight_mode = weight_mode
        self.pruning = pruning
        self.has_act = has_act
        self.fusion_mode = fusion_mode
        self.n_layers = num_layers
        self.n_routings = num_routings
        self.adj = adj
        self.adj_user = adj_user
        self.rows = torch.tensor(rows, dtype=torch.int64)
        self.cols = torch.tensor(cols, dtype=torch.int64)
        self.ptr = ptr
        self.ptr_full = ptr_full

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # multimodal collaborative embeddings
        self.Gum = torch.nn.ParameterDict()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        for m_id, m in enumerate(modalities):
            self.Gum[m] = torch.nn.Embedding(self.num_users, self.embed_k_multimod).weight
            torch.nn.init.xavier_uniform_(self.Gum[m])
            self.Gum[m].to(self.device)

        # multimodal features
        self.Fm = []
        for m_id, m in enumerate(modalities):
            self.Fm += [torch.tensor(multimodal_features[m_id], dtype=torch.float32, device=self.device)]

        # graph convolutional network
        propagation_graph_convolutional_network_list = []
        for layer in range(self.n_layers):
            propagation_graph_convolutional_network_list.append((
                GraphConvolutionalLayer(self.has_act), 'x, edge_index -> x'
            ))
        self.propagation_graph_convolutional_network = torch_geometric.nn.Sequential(
            'x, edge_index', propagation_graph_convolutional_network_list)
        self.propagation_graph_convolutional_network.to(self.device)

        # graph refining network for each modality
        self.projection_multimodal = torch.nn.ModuleDict()
        self.propagation_graph_refining_network = torch.nn.ModuleDict()
        for m_id, m in enumerate(self.modalities):
            self.projection_multimodal[m] = torch.nn.Linear(in_features=self.multimodal_features_shapes[m_id],
                                                            out_features=self.embed_k_multimod)
            self.projection_multimodal[m].to(self.device)
            propagation_graph_refining_network_list = []
            for layer in range(self.n_routings):
                propagation_graph_refining_network_list.append(
                    (GraphRefiningLayer(self.rows, self.has_act, self.ptr.to(self.device)), 'x, edge_index -> x'))
            propagation_graph_refining_network_list.append(
                (GraphRefiningLayer(self.rows, self.has_act, self.ptr.to(self.device),
                                    self.ptr_full.to(self.device)), 'x, edge_index -> x'))
            self.propagation_graph_refining_network[m] = torch_geometric.nn.Sequential(
                'x, edge_index', propagation_graph_refining_network_list)
            self.propagation_graph_refining_network[m].to(self.device)

        # model specific parameters
        self.model_specific_conf = torch.nn.Embedding(self.num_users + self.num_items, len(self.modalities))
        torch.nn.init.xavier_uniform_(self.model_specific_conf.weight)
        self.model_specific_conf.to(self.device)

        # placeholder for calculated user and item embeddings
        self.user_embeddings = torch.nn.init.xavier_uniform_(torch.rand((self.num_users, self.embed_k)))
        self.user_embeddings.to(self.device)
        self.item_embeddings = torch.nn.init.xavier_uniform_(torch.rand((self.num_items, self.embed_k)))
        self.item_embeddings.to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self):
        # Graph Refining Layers
        x_all_m = []
        alphas_m = []
        for m_id, m in enumerate(self.modalities):
            gum = torch.nn.functional.normalize(self.Gum[m].to(self.device))
            gim = torch.nn.functional.normalize(
                torch.nn.functional.leaky_relu(self.projection_multimodal[m](self.Fm[m_id].to(self.device))))
            x_all_m += [torch.cat((gum, gim), dim=0)]

            for t in range(self.n_routings):
                x_all_hat_m = list(
                    self.propagation_graph_refining_network[m].children()
                )[t](x_all_m[m_id].to(self.device),
                     gum[self.rows].to(self.device),
                     gim[self.cols - self.num_users].to(self.device),
                     self.adj_user.to(self.device))
                gum += x_all_hat_m[:self.num_users]
                gum = torch.nn.functional.normalize(gum)
                x_all_m[m_id] = torch.cat((gum, gim), dim=0)

            x_all_m[m_id] = torch.cat((gum, gim), dim=0)
            x_all_hat_m = list(
                self.propagation_graph_refining_network[m].children()
            )[-1](x_all_m[m_id].to(self.device),
                  torch.cat((gum[self.rows], gim[self.cols - self.num_users]), dim=0).to(self.device),
                  torch.cat((gim[self.cols - self.num_users], gum[self.rows]), dim=0).to(self.device),
                  self.adj.to(self.device),
                  True)
            x_all_m[m_id] = x_all_m[m_id] + x_all_hat_m
            alphas_m += [list(self.propagation_graph_refining_network[m].children())[-1].alpha]

        # Graph Convolutional Layers
        x_all = torch.cat(x_all_m, dim=1)
        alphas = torch.stack(alphas_m, dim=1)
        if self.weight_mode == 'mean':
            alphas = torch.sum(alphas, dim=1) / len(self.modalities)
        elif self.weight_mode == 'max':
            alphas, _ = torch.max(alphas, dim=1)
        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf.weight[self.rows],
                                    self.model_specific_conf.weight[self.cols]), dim=0)
            alphas = alphas * confidence
            alphas, _ = torch.max(alphas, dim=1)
        else:
            raise NotImplementedError('This weight mode has not been implemented yet!')

        if self.pruning:
            alphas = torch.relu(alphas)

        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [torch.nn.functional.normalize(ego_embeddings)]
        for layer in range(self.n_layers):
            all_embeddings += [torch.nn.functional.normalize(
                list(self.propagation_graph_convolutional_network.children())[layer](
                    all_embeddings[layer].to(self.device),
                    self.adj.to(self.device),
                    alphas.to(self.device)))]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1)

        if self.fusion_mode == 'concat':
            x = torch.cat([all_embeddings, x_all], dim=1)
        elif self.fusion_mode == 'id':
            x = all_embeddings
        elif self.fusion_mode == 'mean':
            x = torch.mean(torch.stack(x_all_m + [all_embeddings]), dim=0)
        else:
            raise NotImplementedError('This fusion mode has not been implemented yet!')

        gu, gi = torch.split(x, [self.num_users, self.num_items], 0)
        self.user_embeddings = gu
        self.item_embeddings = gi
        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.user_embeddings[start: stop].to(self.device),
                            torch.transpose(self.item_embeddings.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(gu[user], gi[pos]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(gu[user], gi[neg]))

        loss = -torch.mean(torch.log(torch.sigmoid(xu_pos - xu_neg)))
        reg_content_loss = torch.sum(torch.stack(
            [self.Gum[m][np.concatenate([user, user])].pow(2).mean() for m in self.modalities]))
        reg_loss = self.l_w * ((self.Gu.weight[np.concatenate([user, user])].pow(2) +
                                self.Gi.weight[np.concatenate([pos, neg])].pow(2)).mean())
        loss += (reg_loss + reg_content_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
