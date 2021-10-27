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
                 weight_size,
                 modalities,
                 aggregation,
                 combination,
                 multimodal_features,
                 n_layers,
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
        self.weight_size = weight_size
        self.modalities = modalities
        self.aggregation = aggregation
        self.combination = combination
        self.multimodal_features = [torch.tensor(mf) for mf in multimodal_features]
        self.multimodal_features_shapes = [mf.shape[1] for mf in self.multimodal_features]
        self.n_layers = n_layers
        self.weight_size_list = [self.embed_k] + self.weight_size
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        # users collaborative embeddings
        self.Gu = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)

        # users multimodal collaborative embeddings
        self.Gum = dict()
        for m_id, m in enumerate(modalities):
            self.Gum[m] = torch.nn.Parameter(
                torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k_multimod[m_id])))
            )
            self.Gum[m].to(self.device)

        self.propagation_network_multimodal = dict()

        for m_id, m in enumerate(self.modalities):
            propagation_network_list = []
            for layer in range(self.n_layers):
                propagation_network_list.append((MMGCNLayer(self.embed_k,
                                                            self.multimodal_features_shapes[m_id],
                                                            self.weight_size[layer],
                                                            self.aggregation,
                                                            self.combination), 'x, edge_index -> x'))

            self.propagation_network_multimodal[m] = torch_geometric.nn.Sequential('x, edge_index',
                                                                                   propagation_network_list)
            self.propagation_network_multimodal[m].to(self.device)

        out = list(self.propagation_network_multimodal['visual'].children())[0](self.Gum['visual'].to(self.device),
                                                                                self.multimodal_features[0].to(self.device),
                                                                                self.Gu.to(self.device),
                                                                                self.edge_index)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]
        embedding_idx = 0

        for layer in range(self.n_layers):
            if not evaluate:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer + 1](all_embeddings[embedding_idx].to(self.device), self.edge_index.to(self.device))]
            else:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer + 1](all_embeddings[embedding_idx].to(self.device), self.edge_index.to(self.device))]

            embedding_idx += 1

        if evaluate:
            self.propagation_network.train()

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
