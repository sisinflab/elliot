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
          lr: 0.001
          reg: 0.0025
          gaussian_variance: 0.1
    """

    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_factors", "factors", "factors", 50, None, None),
            ("_l_w", "reg", "reg", 0.0025, None, None),
            ("_gvar", "gaussian_variance", "gvar", 0.1, None, None),
        ]
        self.sampler = pws.Sampler(data.i_train_dict)
        super(PMF, self).__init__(data, params, seed, logger)

        self.user_mf_embedding = nn.Embedding(self._num_users, self._factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self._factors, dtype=torch.float32)

        nn.init.normal_(self.user_mf_embedding.weight, std=0.01)
        nn.init.normal_(self.item_mf_embedding.weight, std=0.01)

        self.noise = GaussianNoise(self._gvar)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def forward(self, inputs):
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        if self.training:
            mf_output = torch.mul(user_mf_e, item_mf_e).sum(dim=1)
        else:
            mf_output = torch.matmul(user_mf_e, item_mf_e.T)
        output = self.sigmoid(mf_output)
        return output

    def train_step(self, batch, *args):
        user, pos, label = batch
        output = self.noise(self.forward(inputs=(user, pos)))
        reg = self.user_mf_embedding.weight.pow(2).sum() + self.item_mf_embedding.weight.pow(2).sum()
        loss = self.loss(label.float(), output) + self._l_w * reg
        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)
        item_indices = torch.arange(self._num_items)
        output = self.forward(inputs=(user_indices, item_indices))
        return output
