"""
Module description:

Mnih, Andriy, and Russ R. Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems 20 (2007)

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.utils import GaussianNoise


class PMF(GeneralRecommender):
    """
    Probabilistic Matrix Factorization

    For further details, please refer to the `paper <https://papers.nips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        reg: Regularization coefficient
        gaussian_variance: Variance of the Gaussian distribution

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        PMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 50
          learning_rate: 0.001
          lambda_weights: 0.0025
          gaussian_variance: 0.1
    """
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.0025
    gaussian_variance: float = 0.1

    def __init__(self, data, params, seed, logger):
        self.sampler = pws.Sampler(data.i_train_dict)
        super(PMF, self).__init__(data, params, seed, logger)

        # Embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)

        # Gaussian noise
        self.noise = GaussianNoise(self.gaussian_variance)

        # Activation function, loss and optimizer
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self._init_weights('xavier_normal')

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        mf_output = torch.mul(user_mf_e, item_mf_e).sum(dim=1)
        output = self.sigmoid(mf_output)
        return output

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]

        output = self.noise(self.forward(user, pos))
        reg = self.user_mf_embedding.weight.pow(2).sum() + self.item_mf_embedding.weight.pow(2).sum()
        loss = self.loss(label.float(), output) + self.lambda_weights * reg

        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)

        # Retrieve embeddings
        user_e_all = self.user_mf_embedding.weight
        item_e_all = self.item_mf_embedding.weight

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]

        predictions = torch.matmul(u_embeddings_batch, item_e_all.T)
        predictions = self.sigmoid(predictions)
        return predictions.to(self._device)
