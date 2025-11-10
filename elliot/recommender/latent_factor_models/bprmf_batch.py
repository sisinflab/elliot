"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import json
import torch
from torch import nn

from elliot.dataset.samplers import BPRMFSampler
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_uniform_init


class BPRMFBatch(GeneralRecommender):
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
          learning_rate: 0.001
          lambda_weights: 0.1
    """
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1

    def __init__(self, data, params, seed, logger):
        self.sampler = BPRMFSampler(data.i_train_dict, seed)
        super(BPRMFBatch, self).__init__(data, params, seed, logger)

        # Embeddings
        self.Gu = nn.Embedding(self._num_users, self.factors)
        self.Gi = nn.Embedding(self._num_items, self.factors)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
        gamma_u = torch.squeeze(self.Gu(user))
        gamma_i = torch.squeeze(self.Gi(item))

        xui = torch.sum(gamma_u * gamma_i, 1)
        return xui, gamma_u, gamma_i

    def train_step(self, batch, *args):
        user, pos, neg = [x.to(self._device) for x in batch]

        xu_pos, gu, gi_pos = self.forward(user, pos)
        xu_neg, _, gi_neg = self.forward(user, neg)
        reg = 0.5 * (gu.square().sum() + gi_pos.square().sum() + gi_neg.square().sum()) / user.size(0)
        loss = -torch.mean(nn.functional.logsigmoid(xu_pos - xu_neg)) + self.lambda_weights * reg

        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)

        # Retrieve embeddings
        user_e_all = self.Gu.weight
        item_e_all = self.Gi.weight

        # Select only the embeddings in the current batch
        u_embedding_batch = user_e_all[user_indices]

        predictions = torch.matmul(u_embedding_batch, item_e_all.T)
        return predictions.to(self._device)

    # def end_training(self, dataset_name):
    #     with open(f'./results/{dataset_name}/performance/' + 'freq_users.json', 'w') as f:
    #         json.dump(self.sampler.freq_users, f)
    #     with open(f'./results/{dataset_name}/performance/' + 'freq_items.json', 'w') as f:
    #         json.dump(self.sampler.freq_items, f)
