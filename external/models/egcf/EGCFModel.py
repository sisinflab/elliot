"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC

from torch_geometric.nn import GCNConv
from collections import OrderedDict

import torch
import torch_geometric
import numpy as np


class EGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 weight_size_projection_node_edge,
                 weight_size_nodes,
                 weight_size_edges,
                 weight_size_nodes_edges,
                 l_w,
                 n_layers,
                 edge_features,
                 edge_index,
                 node_edge_index,
                 edge_edge_index,
                 trainable_edges,
                 random_seed,
                 name="EGCF",
                 **kwargs
                 ):
        super().__init__()
        torch.manual_seed(random_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.weight_size_projection_node_edge_list = weight_size_projection_node_edge
        self.weight_size_nodes_list = weight_size_nodes
        self.weight_size_edges_list = weight_size_edges
        self.weight_size_nodes_edges_list = weight_size_nodes_edges
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.n_layers = n_layers
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.node_edge_index = torch.tensor(node_edge_index, dtype=torch.int64)
        self.edge_edge_index = torch.tensor(edge_edge_index, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)
        self.trainable_edges = trainable_edges

        if self.trainable_edges:
            self.Ge = torch.nn.Parameter(
                torch.tensor(edge_features, dtype=torch.float32)
            )
        else:
            self.Ge = torch.tensor(edge_features, dtype=torch.float32, device=self.device)

        propagation_network_nn_list = [(GCNConv(in_channels=self.embed_k,
                                                out_channels=self.weight_size_nodes_list[0],
                                                add_self_loops=True), 'x, edge_index -> x')]
        propagation_network_ee_list = [(GCNConv(in_channels=self.Ge.shape[1],
                                                out_channels=self.weight_size_edges_list[0],
                                                add_self_loops=True), 'x, edge_index -> x')]
        propagation_network_ne_list = [(GCNConv(in_channels=self.weight_size_projection_node_edge_list[-1],
                                                out_channels=self.weight_size_nodes_edges_list[0],
                                                add_self_loops=True), 'x, edge_index -> x')]

        for layer in range(1, self.n_layers):
            propagation_network_nn_list.append(
                (GCNConv(in_channels=self.weight_size_nodes_list[layer - 1] + self.weight_size_edges_list[layer - 1],
                         out_channels=self.weight_size_nodes_list[layer],
                         add_self_loops=True), 'x, edge_index -> x'))
            propagation_network_ee_list.append(
                (GCNConv(in_channels=self.weight_size_edges_list[layer - 1] + self.weight_size_edges_list[layer - 1],
                         out_channels=self.weight_size_edges_list[layer],
                         add_self_loops=True), 'x, edge_index -> x'))
            propagation_network_ne_list.append(
                (GCNConv(in_channels=self.weight_size_nodes_list[layer - 1] + self.weight_size_edges_list[layer - 1],
                         out_channels=self.weight_size_nodes_edges_list[layer],
                         add_self_loops=True), 'x, edge_index -> x'))

        self.propagation_network_nn = torch_geometric.nn.Sequential('x, edge_index', propagation_network_nn_list)
        self.propagation_network_nn.to(self.device)
        self.propagation_network_ee = torch_geometric.nn.Sequential('x, edge_index', propagation_network_ee_list)
        self.propagation_network_ee.to(self.device)
        self.propagation_network_ne = torch_geometric.nn.Sequential('x, edge_index', propagation_network_ne_list)
        self.propagation_network_ne.to(self.device)

        projection_layer_nodes_list = [('feat_proj_nodes_0',
                                        torch.nn.Linear(in_features=self.embed_k,
                                                        out_features=self.weight_size_projection_node_edge_list[0])),
                                       ('relu_0', torch.nn.ReLU())]
        projection_layer_edges_list = [('feat_proj_edges_0',
                                        torch.nn.Linear(in_features=self.Ge.shape[1],
                                                        out_features=self.weight_size_projection_node_edge_list[0])),
                                       ('relu_0', torch.nn.ReLU())]
        for layer in range(1, len(self.weight_size_projection_node_edge_list)):
            projection_layer_nodes_list.append(('feat_proj_nodes_' + str(layer),
                                                torch.nn.Linear(
                                                    in_features=self.weight_size_projection_node_edge_list[layer - 1],
                                                    out_features=self.weight_size_projection_node_edge_list[layer])))
            projection_layer_nodes_list.append(('relu_' + str(layer), torch.nn.ReLU()))
            projection_layer_edges_list.append(('feat_proj_edges_' + str(layer),
                                                torch.nn.Linear(
                                                    in_features=self.weight_size_projection_node_edge_list[layer - 1],
                                                    out_features=self.weight_size_projection_node_edge_list[layer])))
            projection_layer_edges_list.append(('relu_' + str(layer), torch.nn.ReLU()))

        self.projection_network_nodes = torch.nn.Sequential(OrderedDict(projection_layer_nodes_list))
        self.projection_network_nodes.to(self.device)
        self.projection_network_edges = torch.nn.Sequential(OrderedDict(projection_layer_edges_list))
        self.projection_network_edges.to(self.device)

        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        node_node_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        edge_edge_embeddings = self.Ge

        # we project node and edge embeddings into the same latent space for the node-edge propagation network
        node_node_embeddings_projected = self.projection_network_nodes(node_node_embeddings.to(self.device))
        edge_edge_embeddings_projected = self.projection_network_edges(edge_edge_embeddings.to(self.device))
        node_edge_embeddings = torch.cat((node_node_embeddings_projected.to(self.device),
                                          edge_edge_embeddings_projected.to(self.device)), 0)

        for layer in range(self.n_layers):
            if not evaluate:
                # first, we propagate node-node embeddings
                node_node_embeddings = list(
                    self.propagation_network_nn.children()
                )[layer](node_node_embeddings.to(self.device), self.edge_index.to(self.device))
                # then, we propagate edge-edge embedding
                edge_edge_embeddings = list(
                    self.propagation_network_ee.children()
                )[layer](edge_edge_embeddings.to(self.device), self.edge_edge_index.to(self.device))
                # finally, we propagate node-edge embeddings
                node_edge_embeddings = list(
                    self.propagation_network_ne.children()
                )[layer](node_edge_embeddings.to(self.device), self.node_edge_index.to(self.device))
            else:
                self.propagation_network_nn.eval()
                self.propagation_network_ee.eval()
                self.propagation_network_ne.eval()
                with torch.no_grad():
                    # first, we propagate node-node embeddings
                    node_node_embeddings = list(
                        self.propagation_network_nn.children()
                    )[layer](node_node_embeddings.to(self.device), self.edge_index.to(self.device))
                    # then, we propagate edge-edge embedding
                    edge_edge_embeddings = list(
                        self.propagation_network_ee.children()
                    )[layer](edge_edge_embeddings.to(self.device), self.edge_edge_index.to(self.device))
                    # finally, we propagate node-edge embeddings
                    node_edge_embeddings = list(
                        self.propagation_network_ne.children()
                    )[layer](node_edge_embeddings.to(self.device), self.node_edge_index.to(self.device))
            node_edge_node_embeddings, node_edge_edge_embeddings = \
                torch.split(node_edge_embeddings, [self.num_users + self.num_items,
                                                   node_edge_embeddings.shape[0] - (self.num_users + self.num_items)],
                            0)
            node_node_embeddings = torch.cat((node_node_embeddings, node_edge_node_embeddings), dim=1)
            edge_edge_embeddings = torch.cat((edge_edge_embeddings, node_edge_edge_embeddings), dim=1)
            node_edge_embeddings = torch.cat((node_node_embeddings.to(self.device),
                                              edge_edge_embeddings.to(self.device)), 0)
        if evaluate:
            self.propagation_network_nn.train()
            self.propagation_network_ee.train()
            self.propagation_network_ne.train()

        gu, gi = torch.split(node_node_embeddings, [self.num_users, self.num_items], 0)
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

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2) +
                               torch.norm(self.Ge, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network_nn.parameters()],
                                           dim=0).sum(dim=0) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network_ee.parameters()],
                                           dim=0).sum(dim=0) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network_ne.parameters()],
                                           dim=0).sum(dim=0) +
                               torch.stack([torch.norm(value, 2) for value in self.projection_network_edges.parameters()],
                                           dim=0).sum(dim=0) +
                               torch.stack([torch.norm(value, 2) for value in self.projection_network_edges.parameters()],
                                           dim=0).sum(dim=0)) * 2
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
