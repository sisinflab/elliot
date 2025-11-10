"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,daniele.malitesta@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import PWPosNegSampler
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_uniform_init


class LogisticMF(GeneralRecommender):
    """
    Logistic Matrix Factorization

    For further details, please refer to the `paper <https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf>`_

    Args:
        factors: Number of factors of feature embeddings
        lr: Learning rate
        reg: Regularization coefficient
        alpha: Parameter for confidence estimation

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        LogisticMatrixFactorization:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          learning_rate: 0.001
          lambda_weights: 0.1
          alpha: 0.5
    """
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1
    alpha: float = 0.5

    def __init__(self, data, params, seed, logger):
        self.sampler = PWPosNegSampler(data.i_train_dict)
        super(LogisticMF, self).__init__(data, params, seed, logger)

        # Embeddings
        self.Gu = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.Gi = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.Bu = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.Bi = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Optimizer
        # NOTE: Removed Adagrad optimizer due to its poor performance
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.sampler.events = self._data.transactions * 2
        self.transactions = self._data.transactions * 2

        # Init embedding weights
        self.bias = [self.Bu, self.Bi]
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
        gamma_u = self.Gu(user)
        gamma_i = self.Gi(item)
        beta_u = self.Bu(user)
        beta_i = self.Bi(item)

        xui = torch.sum(gamma_u * gamma_i, dim=-1) + beta_u + beta_i
        return xui, gamma_u, gamma_i

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]
        label = label.float()
        output, g_u, g_i = self.forward(user, pos)

        loss = torch.sum(-(self.alpha * label * output -
                           (1 + self.alpha * label) * torch.log1p(torch.exp(output))))

        reg_loss = self.lambda_weights * (torch.sum(g_u.pow(2)) + torch.sum(g_i.pow(2)))
        loss = loss + reg_loss

        steps = args[0]
        inputs = ([self.Gu.weight, self.Bu.weight] if steps > self._data.transactions
                  else [self.Gi.weight, self.Bi.weight])

        return loss, inputs

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)

        # Retrieve embeddings
        user_e_all = self.Gu.weight
        item_e_all = self.Gi.weight
        user_b_all = self.Bu.weight
        item_b_all = self.Bi.weight

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]
        u_bias_batch = user_b_all[user_indices]

        predictions = torch.matmul(u_embeddings_batch, item_e_all.T) + u_bias_batch + item_b_all.T
        return predictions.to(self._device)
