"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,daniele.malitesta@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.base_recommender import GeneralRecommender


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
        self.sampler = pws.Sampler(data.i_train_dict)
        super(LogisticMF, self).__init__(data, params, seed, logger)

        # Embeddings
        self.Gu = nn.Parameter(torch.empty(self._num_users, self.factors))
        self.Gi = nn.Parameter(torch.empty(self._num_items, self.factors))
        self.Bu = nn.Parameter(torch.zeros(self._num_users))
        self.Bi = nn.Parameter(torch.zeros(self._num_items))

        # Optimizer
        # NOTE: Removed Adagrad optimizer due to its poor performance
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.sampler.events = self._data.transactions * 2
        self.transactions = self._data.transactions * 2

        # Init embedding weights
        self._init_weights('xavier_uniform', [self.Gu, self.Gi])

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
        gamma_u = self.Gu[user]
        gamma_i = self.Gi[item]
        beta_u = self.Bu[user]
        beta_i = self.Bi[item]

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
        inputs = [self.Gu, self.Bu] if steps > self._data.transactions else [self.Gi, self.Bi]

        return loss, inputs

    def predict(self, start, stop):
        Bu_batch = self.Bu[start:stop].unsqueeze(-1)
        Bi_all = self.Bi.unsqueeze(0)

        predictions = torch.matmul(self.Gu[start:stop], self.Gi.T) + Bu_batch + Bi_all
        return predictions.to(self._device)
