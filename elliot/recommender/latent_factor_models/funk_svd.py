"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import PWPosNegSampler
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_uniform_init


class FunkSVD(GeneralRecommender):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1
    lambda_bias: float = 0.001

    def __init__(self, data, params, seed, logger):
        self.sampler = PWPosNegSampler(data.i_train_dict)
        super(FunkSVD, self).__init__(data, params, seed, logger)

        # Embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.user_bias_embedding = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.item_bias_embedding = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Loss and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self.bias = [self.user_bias_embedding, self.item_bias_embedding]
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
        u = self.user_mf_embedding(user)
        i = self.item_mf_embedding(item)
        ub = self.user_bias_embedding(user)
        ib = self.item_bias_embedding(item)

        return torch.mul(u, i).sum(dim=1) + ub.squeeze() + ib.squeeze()

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]

        output = self.forward(user, pos)

        reg = (
            self.lambda_weights * (
                self.user_mf_embedding.weight.pow(2).sum() +
                self.item_mf_embedding.weight.pow(2).sum()
            ) +
            self.lambda_bias * (
                self.user_bias_embedding.weight.pow(2).sum() +
                self.item_bias_embedding.weight.pow(2).sum()
            )
        )
        loss = self.loss(label.float(), output) + reg

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

        predictions = torch.matmul(u_embeddings_batch, item_e_all.T) + u_bias_batch + item_b_all.T
        return predictions.to(self._device)
