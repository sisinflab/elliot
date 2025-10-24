"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import json
import torch
from torch import nn

from elliot.dataset.samplers.mf_samplers import BPRMFSampler
from elliot.recommender.base_recommender import GeneralRecommender


class BPRMF(GeneralRecommender):
    """
    Batch Bayesian Personalized Ranking with Matrix Factorization

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        l_w: Regularization coefficient for latent factors

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        BPRMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.1
    """

    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_factors", "factors", "factors", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_l_w", "l_w", "l_w", 0.1, float, None)
        ]
        self.sampler = BPRMFSampler(data.i_train_dict, seed)
        super(BPRMF, self).__init__(data, params, seed, logger)

        self.Gu = torch.nn.Embedding(self._num_users, self._factors, device=self.device)
        self.Gi = torch.nn.Embedding(self._num_items, self._factors, device=self.device)

        nn.init.xavier_uniform_(self.Gu.weight)
        nn.init.xavier_uniform_(self.Gi.weight)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def forward(self, inputs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu(users))
        gamma_i = torch.squeeze(self.Gi(items))
        xui = torch.sum(gamma_u * gamma_i, 1)
        return xui, gamma_u, gamma_i

    def predict(self, start, stop):
        return torch.matmul(self.Gu.weight[start:stop], self.Gi.weight.T)

    def train_step(self, batch, *args):
        user, pos, neg = batch
        xu_pos, gu, gi_pos = self.forward(inputs=(user, pos))
        xu_neg, _, gi_neg = self.forward(inputs=(user, neg))
        reg = 0.5 * (gu.square().sum() + gi_pos.square().sum() + gi_neg.square().sum()) / user.size(0)
        loss = -torch.mean(nn.functional.logsigmoid(xu_pos - xu_neg)) + self._l_w * reg
        return loss

    # def end_training(self, dataset_name):
    #     with open(f'./results/{dataset_name}/performance/' + 'freq_users.json', 'w') as f:
    #         json.dump(self.sampler.freq_users, f)
    #     with open(f'./results/{dataset_name}/performance/' + 'freq_items.json', 'w') as f:
    #         json.dump(self.sampler.freq_items, f)
