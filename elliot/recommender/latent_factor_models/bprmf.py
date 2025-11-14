"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import torch
from torch import nn

from elliot.dataset.samplers import CustomSampler
from elliot.recommender.base_recommender import Recommender, GeneralRecommender
from elliot.recommender.init import normal_init, xavier_normal_init
from elliot.recommender.utils import device, classproperty


class BPRMF_numpy(Recommender):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.05
    lambda_bias: float = 0.0
    lambda_user: float = 0.0025
    lambda_pos_i: float = 0.0025
    lambda_neg_i: float = 0.00025

    def __init__(self, data, params, seed, logger):
        self.sampler = CustomSampler(data.i_train_dict)
        super().__init__(data, params, seed, logger)

        # Embeddings
        self._user_factors = np.empty((self._num_users, self.factors), dtype=np.float32)
        self._item_factors = np.empty((self._num_items, self.factors), dtype=np.float32)
        self._user_bias = np.empty(self._num_users, dtype=np.float32)
        self._item_bias = np.empty(self._num_items, dtype=np.float32)

        # Global bias
        self._global_bias = 0

        # Init embedding weights
        self.modules = [self._user_factors, self._item_factors, self._user_bias, self._item_bias]
        self.bias = [self._user_bias, self._item_bias]
        self.apply(normal_init)

        self.params_to_save = ['_user_bias', '_item_bias', '_user_factors', '_item_factors']

    def train_step(self, batch, *args):
        users, pos_items, neg_items = batch

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

    def predict(self, start, stop):
        return (
            self._global_bias
            + self._user_bias[start:stop, None]
            + self._item_bias[None, :]
            + self._user_factors[start:stop] @ self._item_factors.T
        )


class BPRMF_pytorch(GeneralRecommender):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.05
    lambda_bias: float = 0.0
    lambda_user: float = 0.0025
    lambda_pos_i: float = 0.0025
    lambda_neg_i: float = 0.00025

    def __init__(self, data, params, seed, logger):
        self.sampler = CustomSampler(data.i_train_dict)
        super().__init__(data, params, seed, logger)

        # Embeddings
        self._user_factors = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self._item_factors = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self._user_bias = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self._item_bias = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Global bias
        self._global_bias = nn.Parameter(torch.zeros(1))

        # Optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self.bias = [self._user_bias, self._item_bias]
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def train_step(self, batch, *args):
        users, pos_items, neg_items = [x.to(self._device) for x in batch]
        loss = torch.zeros(1)

        for u, i, j in zip(users, pos_items, neg_items):
            # Extract embeddings
            user_vec = self._user_factors(u)
            pos_vec = self._item_factors(i)
            neg_vec = self._item_factors(j)
            b_u = self._user_bias(u)
            b_i = self._item_bias(i)
            b_j = self._item_bias(j)

            # Compute scores difference
            x_ui = torch.matmul(user_vec, pos_vec) + b_u.squeeze() + b_i.squeeze()
            x_uj = torch.matmul(user_vec, neg_vec) + b_u.squeeze() + b_j.squeeze()
            x_uij = x_ui - x_uj

            # BPR loss
            loss = -nn.functional.logsigmoid(x_uij)

            # Regularization
            reg = (
                self.lambda_user * user_vec.pow(2).sum()
                + self.lambda_pos_i * pos_vec.pow(2).sum()
                + self.lambda_neg_i * neg_vec.pow(2).sum()
                + self.lambda_bias * (b_i.pow(2).sum() + b_j.pow(2).sum())
            )

            loss = loss + reg

        return loss

    def predict(self, start, stop):
        user_indices = torch.arange(start, stop)

        # Retrieve embeddings
        user_e_all = self._user_factors.weight
        item_e_all = self._item_factors.weight
        user_b_all = self._user_bias.weight
        item_b_all = self._item_bias.weight

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]
        u_bias_batch = user_b_all[user_indices]

        predictions = torch.matmul(
            u_embeddings_batch, item_e_all.T
        ) + u_bias_batch + item_b_all.T + self._global_bias
        return predictions.to(self._device)


class BPRMF:
    implementations = {
        "numpy": BPRMF_numpy,
        "pytorch": BPRMF_pytorch
    }

    @classproperty
    def type(cls):
        if device.type == "cuda":
            return cls.implementations['pytorch'].type
        else:
            return cls.implementations['numpy'].type

    def __new__(cls, *args):
        if device.type == "cuda":
            return cls.implementations['pytorch'](*args)
        else:
            return cls.implementations['numpy'](*args)
