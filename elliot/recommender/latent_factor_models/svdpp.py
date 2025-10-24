"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import custom_pointwise_sparse_sampler as cpss
from elliot.recommender.base_recommender import GeneralRecommender


class SVDpp(GeneralRecommender):
    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_lambda_weights", "reg_w", "reg_w", 0.1, None, None),
            ("_lambda_bias", "reg_b", "reg_b", 0.001, None, None),
        ]
        self.sampler = cpss.Sampler(data.i_train_dict, data.sp_i_train)
        super(SVDpp, self).__init__(data, params, seed, logger)

        self.user_mf_embedding = nn.Embedding(self._num_users, self._factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self._factors, dtype=torch.float32)
        self.item_y_embedding = nn.Embedding(self._num_items, self._factors, dtype=torch.float32)
        self.user_bias_embedding = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.item_bias_embedding = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        self.bias_ = nn.Parameter(torch.Tensor([0]))

        nn.init.xavier_uniform_(self.user_mf_embedding.weight)
        nn.init.xavier_uniform_(self.item_mf_embedding.weight)
        nn.init.xavier_uniform_(self.item_y_embedding.weight)
        nn.init.zeros_(self.user_bias_embedding.weight)
        nn.init.zeros_(self.item_bias_embedding.weight)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def forward(self, inputs):
        user, item = inputs
        u = self.user_mf_embedding(user)
        i = self.item_mf_embedding(item)
        ub = self.user_bias_embedding(user)
        ib = self.item_bias_embedding(item)

        offsets, indices, _ = self._sp_i_train[user].csr()
        puyj = nn.functional.embedding_bag(
            input=indices,
            weight=self.item_y_embedding.weight,
            offsets=offsets[:-1],
            mode='mean'
        )

        if self.training:
            output = torch.mul((puyj + u), i).sum(dim=-1) + ub.squeeze() + ib.squeeze() + self.bias_
        else:
            output = torch.matmul((puyj + u), i.T) + ub + ib.T + self.bias_

        return output

    def l2_reg(self):
        return (
            self._lambda_weights * (
                self.user_mf_embedding.weight.pow(2).sum() +
                self.item_mf_embedding.weight.pow(2).sum() +
                self.item_y_embedding.weight.pow(2).sum()
            ) +
            self._lambda_bias * (
                self.user_bias_embedding.weight.pow(2).sum() +
                self.item_bias_embedding.weight.pow(2).sum()
            )
        )

    def train_step(self, batch, *args):
        user, item, label = batch
        output = self.forward(inputs=(user, item))
        loss = self.loss(label.float(), output) + self.l2_reg()
        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)
        item_indices = torch.arange(self._num_items)
        output = self.forward(inputs=(user_indices, item_indices))
        return output
