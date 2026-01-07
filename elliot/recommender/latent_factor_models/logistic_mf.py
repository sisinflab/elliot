"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,daniele.malitesta@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import PointWisePosNegSampler
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

    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1
    alpha: float = 0.5

    def __init__(self, data, params, seed, logger):
        super(LogisticMF, self).__init__(data, params, seed, logger)

        # Embeddings
        self.Gu = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.Gi = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.Bu = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.Bi = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Optimizer
        # NOTE: Removed Adagrad optimizer due to its poor performance
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.transactions = self._data.transactions * 2

        # Init embedding weights
        self.bias = [self.Bu, self.Bi]
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def get_training_dataloader(self):
        dataloader = self._data.training_dataloader(
            PointWisePosNegSampler, self._seed, transactions=self.transactions
        )
        return dataloader

    def forward(self, user, item):
        user_e = self.Gu(user)
        item_e = self.Gi(item)
        user_b = self.Bu(user)
        item_b = self.Bi(item)

        xui = torch.mul(user_e, item_e).sum(dim=1) + user_b + item_b
        return xui

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]
        label = label.float()

        output = self.forward(user, pos)

        reg = self.Gu.weight[user].pow(2).sum() + self.Gi.weight[pos].pow(2).sum()
        loss = torch.sum(
            - (self.alpha * label * output - (1 + self.alpha * label) * torch.log1p(torch.exp(output)))
        ) + self.lambda_weights * reg

        steps = args[0]
        inputs = ([self.Gu.weight, self.Bu.weight] if steps > self._data.transactions
                  else [self.Gi.weight, self.Bi.weight])

        return loss, inputs

    def predict_full(self, user_indices):
        # Retrieve embeddings
        user_e_all = self.Gu.weight
        item_e_all = self.Gi.weight
        user_b_all = self.Bu.weight
        item_b_all = self.Bi.weight

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]
        u_bias_batch = user_b_all[user_indices]

        # Compute predictions
        predictions = (
            torch.matmul(u_embeddings_batch, item_e_all.T) +
            u_bias_batch +
            item_b_all.T
        )

        return predictions.to(self._device)

    def predict_sampled(self, user_indices, item_indices):
        # Retrieve embeddings
        u_embeddings_batch = self.Gu(user_indices)
        i_embeddings_candidate = self.Gi(item_indices.clamp(min=0))
        u_bias_batch = self.Bu(user_indices)
        i_bias_candidate = self.Bi(item_indices.clamp(min=0))

        # Compute predictions
        predictions = (
            torch.einsum(
                "bi,bji->bj", u_embeddings_batch, i_embeddings_candidate
            ) +
            u_bias_batch +
            i_bias_candidate.squeeze(-1)
        )

        return predictions.to(self._device)
