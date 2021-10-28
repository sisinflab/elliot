"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC

from .MMGCNLayer import MMGCNLayer
import torch
import torch_geometric
import numpy as np


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
                 combination,
                 multimodal_features,
                 edge_index,
                 random_seed,
                 name="MMGCN",
                 **kwargs
                 ):
        super().__init__()
        torch.manual_seed(random_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.embed_k_multimod = embed_k_multimod
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.modalities = modalities
        self.aggregation = aggregation
        self.combination = combination
        self.n_layers = num_layers
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        # collaborative embeddings
        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        # multimodal collaborative embeddings
        self.Gum = dict()
        self.Gim = dict()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        for m_id, m in enumerate(modalities):
            self.Gum[m] = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.multimodal_features_shapes[m_id])))
            )
            self.Gum[m].to(self.device)
            self.Gim[m] = torch.nn.Parameter(
                torch.tensor(multimodal_features[m_id], dtype=torch.float32)
            )
            self.Gim[m].to(self.device)

        self.propagation_network_multimodal = dict()

        for m_id, m in enumerate(self.modalities):
            propagation_network_list = [(MMGCNLayer(self.embed_k,
                                                    self.multimodal_features_shapes[m_id],
                                                    self.embed_k_multimod[m_id],
                                                    self.aggregation,
                                                    self.combination), 'x, edge_index -> x')]
            for layer in range(1, self.n_layers):
                propagation_network_list.append((MMGCNLayer(self.embed_k,
                                                            propagation_network_list[-1][0].lin1.out_features,
                                                            self.embed_k_multimod[m_id],
                                                            self.aggregation,
                                                            self.combination), 'x, edge_index -> x'))

            self.propagation_network_multimodal[m] = torch_geometric.nn.Sequential('x, edge_index',
                                                                                   propagation_network_list)
            self.propagation_network_multimodal[m].to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        x_id = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        x_all_m = dict()

        for m_id, m in enumerate(self.modalities):
            x_all_m[m] = torch.cat((self.Gum[m].to(self.device), self.Gim[m].to(self.device)), 0)

        for m_id, m in enumerate(self.modalities):
            for layer in range(self.n_layers):
                if not evaluate:
                    x_all_m[m] = list(
                        self.propagation_network_multimodal[m].children()
                    )[layer](x_all_m[m].to(self.device), x_id.to(self.device), self.edge_index.to(self.device))
                else:
                    self.propagation_network_multimodal[m].eval()
                    with torch.no_grad():
                        x_all_m[m] = list(
                            self.propagation_network_multimodal[m].children()
                        )[layer](x_all_m[m].to(self.device), x_id.to(self.device), self.edge_index.to(self.device))
                if evaluate:
                    self.propagation_network_multimodal[m].train()

        all_embeddings = torch.cat(all_embeddings, 1)
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

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user], gi[pos]))
        xu_neg = self.forward(inputs=(gu[user], gi[neg]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network.parameters()],
                                           dim=0).sum(dim=0)) * 2
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
