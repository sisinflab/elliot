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
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1
    lambda_bias: float = 0.001

    def __init__(self, data, params, seed, logger):
        self.sampler = cpss.Sampler(data.i_train_dict, data.sp_i_train)
        super(SVDpp, self).__init__(data, params, seed, logger)

        # Embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.item_y_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.user_bias_embedding = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.item_bias_embedding = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Global bias
        self.bias_ = nn.Parameter(torch.Tensor([0]))

        # Loss and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self._init_weights('xavier_uniform', [self.user_mf_embedding, self.item_mf_embedding, self.item_y_embedding])
        self._init_weights('zeros', [self.user_bias_embedding, self.item_bias_embedding])

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
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

        output = torch.mul((puyj + u), i).sum(dim=-1) + ub.squeeze() + ib.squeeze() + self.bias_
        return output

    def train_step(self, batch, *args):
        user, item, label = [x.to(self._device) for x in batch]

        output = self.forward(user, item)
        loss = self.loss(label.float(), output) + self._l2_reg()

        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)

        # Retrieve embeddings
        user_e_all = self.user_mf_embedding.weight
        item_e_all = self.item_mf_embedding.weight
        user_b_all = self.user_bias_embedding.weight
        item_b_all = self.item_bias_embedding.weight

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]
        u_bias_batch = user_b_all[user_indices]

        offsets, indices, _ = self._sp_i_train[user_indices].csr()
        puyj = nn.functional.embedding_bag(
            input=indices,
            weight=self.item_y_embedding.weight,
            offsets=offsets[:-1],
            mode='mean'
        )
        output = torch.matmul(
            (puyj + u_embeddings_batch), item_e_all.T
        ) + u_bias_batch + item_b_all.T + self.bias_
        return output

    def _l2_reg(self):
        return (
            self.lambda_weights * (
                self.user_mf_embedding.weight.pow(2).sum() +
                self.item_mf_embedding.weight.pow(2).sum() +
                self.item_y_embedding.weight.pow(2).sum()
            ) +
            self.lambda_bias * (
                self.user_bias_embedding.weight.pow(2).sum() +
                self.item_bias_embedding.weight.pow(2).sum()
            )
        )
