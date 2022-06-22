from abc import ABC
from torch_geometric.nn import GCNConv
from collections import OrderedDict

import torch
import torch_geometric
import numpy as np
import random


class GCMCModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 convolutional_layer_size,
                 dense_layer_size,
                 n_convolutional_layers,
                 n_dense_layers,
                 num_relations,
                 adj_ratings,
                 accumulation,
                 random_seed,
                 name="GCMC",
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
        self.n_convolutional_layers = n_convolutional_layers
        self.n_dense_layers = n_dense_layers
        self.convolutional_layer_size = [self.embed_k] + ([convolutional_layer_size] * self.n_convolutional_layers)
        self.num_relations = num_relations

        if accumulation not in ['stack', 'sum']:
            raise NotImplementedError('This accumulation method has not been implemented yet!')

        self.accumulation = accumulation
        self.dense_layer_size = [self.convolutional_layer_size[-1] if self.accumulation == 'sum' else
                                 self.convolutional_layer_size[-1] * self.num_relations] + (
                                            [dense_layer_size] * self.n_dense_layers)

        self.adj_ratings = adj_ratings

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)
        self.Q = torch.nn.ParameterList()
        for _ in range(self.num_relations):
            q_r = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty((self.dense_layer_size[-1], self.dense_layer_size[-1]))))
            q_r.to(self.device)
            self.Q.append(q_r)

        # Convolutional part
        self.convolutions = torch.nn.ModuleList()
        for _ in range(self.num_relations):
            convolutional_network_list = []
            for layer in range(self.n_convolutional_layers):
                convolutional_network_list.append((GCNConv(in_channels=self.convolutional_layer_size[layer],
                                                           out_channels=self.convolutional_layer_size[layer + 1],
                                                           add_self_loops=False,
                                                           bias=False),
                                                   'x, edge_index -> x'))
            convolutional_network = torch_geometric.nn.Sequential('x, edge_index', convolutional_network_list)
            convolutional_network.to(self.device)
            self.convolutions.append(convolutional_network)

        # Dense part
        dense_network_list = []
        for layer in range(self.n_dense_layers):
            dense_network_list.append(('dense_' + str(layer), torch.nn.Linear(in_features=self.dense_layer_size[layer],
                                                                              out_features=self.dense_layer_size[
                                                                                  layer + 1],
                                                                              bias=False)))
            dense_network_list.append(('relu_' + str(layer), torch.nn.ReLU()))
        self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
        self.dense_network.to(self.device)

        self.loss = torch.nn.NLLLoss(reduction='sum')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        current_embeddings = []
        for _ in range(self.num_relations):
            current_embeddings += [torch.cat((self.Gu, self.Gi), 0)]

        for r in range(self.num_relations):
            for layer in range(self.n_convolutional_layers):
                if evaluate:
                    self.convolutions.eval()
                    with torch.no_grad():
                        current_embeddings[r] = torch.relu(list(
                            self.convolutions[r].children()
                        )[layer](current_embeddings[r].to(self.device), self.adj_ratings[r].to(self.device)))
                else:
                    current_embeddings[r] = torch.relu(list(
                        self.convolutions[r].children()
                    )[layer](current_embeddings[r].to(self.device), self.adj_ratings[r].to(self.device)))

        if self.accumulation == 'stack':
            current_embeddings = torch.cat(current_embeddings, 1)
        else:
            current_embeddings = [torch.unsqueeze(ce, 0) for ce in current_embeddings]
            current_embeddings = torch.cat(current_embeddings, 0)
            current_embeddings = torch.sum(current_embeddings, 0)

        if evaluate:
            self.dense_network.eval()
            with torch.no_grad():
                current_embeddings = self.dense_network(current_embeddings.to(self.device))
        else:
            current_embeddings = self.dense_network(current_embeddings.to(self.device))

        if evaluate:
            self.convolutions.train()
            self.dense_network.train()

        zu, zi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
        return zu, zi

    def forward(self, inputs, **kwargs):
        zu, zi = inputs
        zeta_u = torch.squeeze(zu).to(self.device)
        zeta_i = torch.squeeze(zi).to(self.device)

        xui_r = []
        for r in range(self.num_relations):
            xui_r.append(torch.unsqueeze(
                torch.sum(zeta_u.to(self.device) * torch.matmul(zeta_i.to(self.device), self.Q[r].to(self.device)), 1),
                1))
        pui = torch.cat(xui_r, 1)
        xui = torch.sum(torch.arange(0, 5).to(self.device) * torch.softmax(pui.to(self.device), 1), 1)
        return xui, pui

    def predict(self, zu, zi, batch_user, batch_item, **kwargs):
        xui, _ = self.forward((zu, zi))
        return torch.reshape(xui, [batch_user, batch_item])

    def train_step(self, batch):
        zu, zi = self.propagate_embeddings()
        user, item, r = batch
        xui, pui = self.forward(inputs=(zu[user], zi[item]))

        loss = self.loss(torch.nn.functional.log_softmax(pui, 1), torch.tensor(r, device=self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device),
                                      torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf, dtype=torch.double).to(self.device)), k=k, sorted=True)
