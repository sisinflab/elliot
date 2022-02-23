"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import torch
import numpy as np
import random


class BPRMFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 random_seed,
                 name="BPRMF",
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

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Bu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(self.num_users)))
        self.Bu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)
        self.Bi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(self.num_items)))
        self.Bi.to(self.device)
        self.B = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(1)))
        self.B.to(self.device)

        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l_w)

    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu[users[:, 0]]).to(self.device)
        beta_u = torch.squeeze(self.Bu[users[:, 0]]).to(self.device)
        gamma_i = torch.squeeze(self.Gi[items[:, 0]]).to(self.device)
        beta_i = torch.squeeze(self.Bi[items[:, 0]]).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1) + beta_u + beta_i + self.B

        return xui

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu[start:stop].to(self.device),
                            torch.transpose(self.Gi.to(self.device), 0, 1)) + self.Bu + self.Bi + self.B

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(user, pos))
        xu_neg = self.forward(inputs=(user, neg))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
