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


class VBPRModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 multimodal_features,
                 random_seed,
                 name="VBPR",
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

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # multimodal
        self.F = torch.nn.Embedding.from_pretrained(torch.nn.functional.normalize(
            torch.tensor(multimodal_features, dtype=torch.float32, device=self.device), p=2, dim=1))
        self.F.to(self.device)
        self.feature_shape = self.F.shape[1]
        self.proj = torch.nn.Linear(in_features=self.feature_shape, out_features=self.embed_k)
        self.proj.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.96 ** (epoch / 50))
        return scheduler

    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)
        effe_i = torch.squeeze(self.F.weight[items]).to(self.device)
        proj_i = self.proj(effe_i).to(self.device)
        gamma_i += torch.nn.functional.normalize(proj_i, p=2, dim=1)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start_user, stop_user, **kwargs):
        proj_i = self.proj(self.F.weight).to(self.device)
        gamma_i = self.Gi.weight.to(self.device) + torch.nn.functional.normalize(proj_i, p=2, dim=1)
        return torch.matmul(self.Gu.weight[start_user:stop_user].to(self.device), torch.transpose(gamma_i, 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user, pos))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(user, neg))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.mean(torch.nn.functional.softplus(-difference))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
