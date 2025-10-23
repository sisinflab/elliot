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


class FunkSVD(GeneralRecommender):
    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_lambda_weights", "reg_w", "reg_w", 0.1, None, None),
            ("_lambda_bias", "reg_b", "reg_b", 0.001, None, None),
        ]
        super(FunkSVD, self).__init__(data, params, seed, logger)

        self.sampler = pws.Sampler(self._data.i_train_dict)

        self.user_mf_embedding = nn.Embedding(self._num_users, self._factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self._factors, dtype=torch.float32)
        self.user_bias_embedding = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.item_bias_embedding = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        nn.init.xavier_uniform_(self.user_mf_embedding.weight)
        nn.init.xavier_uniform_(self.item_mf_embedding.weight)
        nn.init.zeros_(self.user_bias_embedding.weight)
        nn.init.zeros_(self.item_bias_embedding.weight)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def forward(self, inputs):
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_bias_e = self.user_bias_embedding(user)
        item_bias_e = self.item_bias_embedding(item)
        return user_mf_e, item_mf_e, user_bias_e, item_bias_e

    def l2_reg(self):
        return (
            self._lambda_weights * (
                self.user_mf_embedding.weight.pow(2).sum() +
                self.item_mf_embedding.weight.pow(2).sum()
            ) +
            self._lambda_bias * (
                self.user_bias_embedding.weight.pow(2).sum() +
                self.item_bias_embedding.weight.pow(2).sum()
            )
        )

    def train_step(self, batch):
        user, pos, label = batch
        u, i, ub, ib = self.forward(inputs=(user, pos))
        output = torch.mul(u, i).sum(dim=1) + ub.squeeze() + ib.squeeze()
        loss = self.loss(label.float(), output) + self.l2_reg()
        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)
        item_indices = torch.arange(self._num_items)
        u, i, ub, ib = self.forward(inputs=(user_indices, item_indices))
        output = torch.matmul(u, i.T) + ub + ib.T
        return output
