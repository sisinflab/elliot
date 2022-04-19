from abc import ABC

from torch_geometric.nn import LGConv
from .NodeNodeTextLayer import NodeNodeTextLayer

import torch
import torch_geometric
import numpy as np
import random


class EGCFv2Model(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 edge_features,
                 node_node_adj,
                 rows,
                 cols,
                 random_seed,
                 name="EGCFv2",
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
        self.n_layers = n_layers
        self.alpha = torch.tensor([1 / (k + 1) for k in range(self.n_layers + 1)])

        self.node_node_adj = node_node_adj
        self.rows, self.cols = torch.tensor(rows, dtype=torch.int64), torch.tensor(cols, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        self.Gut = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gut.to(self.device)
        self.Git = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Git.to(self.device)

        self.edge_embeddings_interactions = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        self.edge_embeddings_interactions = torch.cat([self.edge_embeddings_interactions,
                                                       self.edge_embeddings_interactions], dim=0)
        self.feature_dim = edge_features.shape[1]

        # create node-node collaborative
        propagation_node_node_collab_list = []
        for _ in range(self.n_layers):
            propagation_node_node_collab_list.append((LGConv(), 'x, edge_index -> x'))

        self.node_node_collab_network = torch_geometric.nn.Sequential('x, edge_index',
                                                                      propagation_node_node_collab_list)
        self.node_node_collab_network.to(self.device)

        # create node-node textual
        propagation_node_node_textual_list = []
        for _ in range(self.n_layers):
            propagation_node_node_textual_list.append(
                (NodeNodeTextLayer(), 'x, edge_index -> x'))

        self.node_node_textual_network = torch_geometric.nn.Sequential('x, edge_index',
                                                                       propagation_node_node_textual_list)
        self.node_node_textual_network.to(self.device)

        # projection layer
        self.projection = torch.nn.Linear(self.edge_embeddings_interactions.shape[-1], self.embed_k)
        self.projection.to(self.device)

        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        node_node_collab_emb = [torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)]
        node_node_textual_emb = [torch.cat((self.Gut.to(self.device), self.Git.to(self.device)), 0)]
        edge_embeddings_interactions_projected = self.projection(self.edge_embeddings_interactions)

        for layer in range(self.n_layers):
            user_item_embeddings_interactions = torch.cat([
                node_node_textual_emb[layer][:self.num_users][self.rows],
                node_node_textual_emb[layer][self.num_users:][self.cols - self.num_users]], dim=0)
            item_user_embeddings_interactions = torch.cat([
                node_node_textual_emb[layer][self.num_users:][self.cols - self.num_users],
                node_node_textual_emb[layer][:self.num_users][self.rows]], dim=0)

            if evaluate:
                self.node_node_collab_network.eval()
                self.node_node_textual_network.eval()
                with torch.no_grad():
                    # node-node collaborative graph
                    node_node_collab_emb += [list(
                        self.node_node_collab_network.children()
                    )[layer](node_node_collab_emb[layer].to(self.device),
                             self.node_node_adj.to(self.device))]

                    # node-node textual graph
                    node_node_textual_emb += [list(
                        self.node_node_textual_network.children()
                    )[layer](node_node_textual_emb[layer].to(self.device),
                             self.node_node_adj.to(self.device),
                             user_item_embeddings_interactions.to(self.device),
                             item_user_embeddings_interactions.to(self.device),
                             edge_embeddings_interactions_projected.to(self.device))]
                self.node_node_collab_network.train()
                self.node_node_textual_network.train()
            else:
                # node-node collaborative graph
                node_node_collab_emb += [list(
                    self.node_node_collab_network.children()
                )[layer](node_node_collab_emb[layer].to(self.device),
                         self.node_node_adj.to(self.device))]

                # node-node textual graph
                node_node_textual_emb += [list(
                    self.node_node_textual_network.children()
                )[layer](node_node_textual_emb[layer].to(self.device),
                         self.node_node_adj.to(self.device),
                         user_item_embeddings_interactions.to(self.device),
                         item_user_embeddings_interactions.to(self.device),
                         edge_embeddings_interactions_projected.to(self.device))]

        node_node_collab_emb = sum([node_node_collab_emb[k] * self.alpha[k] for k in range(len(node_node_collab_emb))])
        node_node_textual_emb = sum(
            [node_node_textual_emb[k] * self.alpha[k] for k in range(len(node_node_textual_emb))])
        gu, gi = torch.split(node_node_collab_emb, [self.num_users, self.num_items], 0)
        gut, git = torch.split(node_node_textual_emb, [self.num_users, self.num_items], 0)
        return gu, gi, gut, git

    def forward(self, inputs, **kwargs):
        gu, gi, gut, git = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)
        gamma_u_t = torch.squeeze(gut).to(self.device)
        gamma_i_t = torch.squeeze(git).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1) + torch.sum(gamma_u_t * gamma_i_t, 1)

        return xui

    def predict(self, gu, gi, gut, git, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1)) + \
               torch.matmul(gut.to(self.device), torch.transpose(git.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi, gut, git = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]], gut[user[:, 0]], git[pos[:, 0]]))
        xu_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]], gut[user[:, 0]], git[neg[:, 0]]))
        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2) +
                               torch.norm(self.Gut, 2) +
                               torch.norm(self.Git, 2)).sum(dim=0)
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
