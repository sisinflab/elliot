"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import torch

from elliot.dataset.samplers import PairWiseSampler
from elliot.recommender.base_recommender import Recommender
from elliot.recommender.init import normal_init


class BPRMF(Recommender):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.05
    lambda_bias: float = 0.0
    lambda_user: float = 0.0025
    lambda_pos_i: float = 0.0025
    lambda_neg_i: float = 0.00025

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

        # Embeddings
        self._user_factors = np.empty((self._num_users, self.factors), dtype=np.float32)
        self._item_factors = np.empty((self._num_items, self.factors), dtype=np.float32)
        self._user_bias = np.empty(self._num_users, dtype=np.float32)
        self._item_bias = np.empty(self._num_items, dtype=np.float32)

        # Global bias
        self._global_bias = 0

        # Init embedding weights
        self._modules = [self._user_factors, self._item_factors, self._user_bias, self._item_bias]
        self._bias = [self._user_bias, self._item_bias]
        self.apply(normal_init)

        self.params_to_save = ['_user_bias', '_item_bias', '_user_factors', '_item_factors']

    def get_training_dataloader(self):
        dataloader = self._data.training_dataloader(PairWiseSampler, self._seed)
        return dataloader

    def train_step(self, batch, *args):
        users, pos_items, neg_items = [b.numpy() for b in batch]

        for u, i, j in zip(users, pos_items, neg_items):
            # Extract embeddings
            u_factors = self._user_factors[u]
            i_factors = self._item_factors[i]
            j_factors = self._item_factors[j]
            b_i = self._item_bias[i]
            b_j = self._item_bias[j]

            # Compute scores difference
            x_ui = self._user_factors[u] @ self._item_factors[i] + self._item_bias[i]
            x_uj = self._user_factors[u] @ self._item_factors[j] + self._item_bias[j]
            x_uij = x_ui - x_uj

            # BPR loss
            z = 1 / (1 + np.exp(x_uij))

            lr = self.learning_rate

            # Bias updates
            self._item_bias[i] += lr * (z - self.lambda_bias * b_i)
            self._item_bias[j] += lr * (-z - self.lambda_bias * b_j)

            # User factors update
            delta_u = (i_factors - j_factors) * z - self.lambda_user * u_factors
            self._user_factors[u] += lr * delta_u

            # Positive item update
            delta_i = u_factors * z - self.lambda_pos_i * i_factors
            self._item_factors[i] += lr * delta_i

            # Negative item update
            delta_j = -u_factors * z - self.lambda_neg_i * j_factors
            self._item_factors[j] += lr * delta_j

        return 0

    def predict_full(self, user_indices):
        user_indices = user_indices.numpy()

        u_embeddings_batch = self._user_factors[user_indices]
        i_embeddings_all = self._item_factors
        u_bias_batch = self._user_bias[user_indices]
        i_bias_all = self._item_bias

        predictions = (
            u_embeddings_batch @ i_embeddings_all.T +
            u_bias_batch[:, None] +
            i_bias_all[None, :] +
            self._global_bias
        )

        predictions = torch.from_numpy(predictions)
        return predictions

    def predict_sampled(self, user_indices, item_indices):
        user_indices = user_indices.numpy()
        item_indices = item_indices.clamp(min=0).numpy()

        u_embeddings_batch = self._user_factors[user_indices]
        i_embeddings_candidate = self._item_factors[item_indices]
        u_bias_batch = self._user_bias[user_indices]
        i_bias_candidate = self._item_bias[item_indices]

        predictions = (
            np.einsum(
                "bi,bji->bj", u_embeddings_batch, i_embeddings_candidate
            ) +
            u_bias_batch[:, None] +
            i_bias_candidate +
            self._global_bias
        )

        predictions = torch.from_numpy(predictions)
        return predictions
