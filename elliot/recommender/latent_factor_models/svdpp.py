"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import torch
from torch import nn

from elliot.dataset.samplers import CustomPointWiseSparseSampler
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_uniform_init


class SVDpp(GeneralRecommender):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1
    lambda_bias: float = 0.001

    def __init__(self, data, params, seed, logger):
        super(SVDpp, self).__init__(data, params, seed, logger)

        # Embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.item_y_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.user_bias_embedding = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.item_bias_embedding = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Global bias
        self.bias_ = nn.Parameter(torch.zeros(1))

        # Loss and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self.bias = [self.user_bias_embedding, self.item_bias_embedding]
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def get_training_dataloader(self):
        dataloader = self._data.training_dataloader(CustomPointWiseSparseSampler, self._seed)
        return dataloader

    def forward(self, user, item):
        u = self.user_mf_embedding(user)
        i = self.item_mf_embedding(item)
        ub = self.user_bias_embedding(user)
        ib = self.item_bias_embedding(item)

        puyj = self._compute_user_representation(user)

        output = torch.mul((puyj + u), i).sum(dim=-1) + ub.squeeze() + ib.squeeze() + self.bias_
        return output

    def train_step(self, batch, *args):
        user, item, label = [x.to(self._device) for x in batch]

        output = self.forward(user, item)

        reg = (
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
        loss = self.loss(label.float(), output) + reg

        return loss

    def predict_full(self, user_indices):
        # Retrieve embeddings
        user_e_all = self.user_mf_embedding.weight
        item_e_all = self.item_mf_embedding.weight
        user_b_all = self.user_bias_embedding.weight
        item_b_all = self.item_bias_embedding.weight

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]
        u_bias_batch = user_b_all[user_indices]

        # Compute predictions
        puyj = self._compute_user_representation(user_indices)

        predictions = torch.matmul(
            (puyj + u_embeddings_batch), item_e_all.T
        ) + u_bias_batch + item_b_all.T + self.bias_

        return predictions.to(self._device)

    def predict_sampled(self, user_indices, item_indices):
        # Retrieve embeddings
        u_embeddings_batch = self.user_mf_embedding(user_indices)
        i_embeddings_candidate = self.item_mf_embedding(item_indices.clamp(min=0))
        u_bias_batch = self.user_bias_embedding(user_indices)
        i_bias_candidate = self.item_bias_embedding(item_indices.clamp(min=0))

        # Compute predictions
        puyj = self._compute_user_representation(user_indices)

        predictions = (
            torch.einsum(
                "bi,bji->bj", (puyj + u_embeddings_batch), i_embeddings_candidate
            ) +
            u_bias_batch +
            i_bias_candidate.squeeze(-1)
        )

        return predictions.to(self._device)

    def _compute_user_representation(self, users):
        item_y_all = self.item_y_embedding.weight
        offsets, indices, _ = self._data.sp_i_train_tensor[users].csr()

        puyj = nn.functional.embedding_bag(
            input=indices,
            weight=item_y_all,
            offsets=offsets[:-1],
            mode='mean'
        )

        return puyj
