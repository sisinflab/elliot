"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.base_recommender import GeneralRecommender

# NOTE: Model with poor performance. Consider to use FunkSVD, instead.
class MF(GeneralRecommender):
    """
    Matrix Factorization

    For further details, please refer to the `paper <https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        reg: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        MF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          reg: 0.1
    """

    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "reg", "reg", 0.1, None, None)
        ]
        self.sampler = pws.Sampler(data.i_train_dict)
        super(MF, self).__init__(data, params, seed, logger)

        self.user_mf_embedding = nn.Embedding(self._num_users, self._factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self._factors, dtype=torch.float32)

        nn.init.xavier_uniform_(self.user_mf_embedding.weight)
        nn.init.xavier_uniform_(self.item_mf_embedding.weight)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def forward(self, inputs):
        user, item = inputs
        u = self.user_mf_embedding(user)
        i = self.item_mf_embedding(item)
        if self.training:
            output = torch.mul(u, i).sum(dim=1)
        else:
            output = torch.matmul(u, i.T)
        return output

    def train_step(self, batch, *args):
        user, pos, label = batch
        output = self.forward(inputs=(user, pos))
        reg = self.user_mf_embedding.weight.pow(2).sum() + self.item_mf_embedding.weight.pow(2).sum()
        loss = self.loss(label.float(), output) + self._l_w * reg
        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)
        item_indices = torch.arange(self._num_items)
        output = self.forward(inputs=(user_indices, item_indices))
        return output
