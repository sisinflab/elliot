"""
Module description:

"""


import numpy as np
import torch

from elliot.recommender.base_recommender import BaseRecommender
from elliot.recommender.init import normal_init
from elliot.utils.registry import model_registry


class AbstractMF2020(BaseRecommender):
    """
    Matrix Factorization (implementation from "Neural Collaborative Filtering vs. Matrix Factorization Revisited")

    For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/3383313.3412488>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        bias_regularization: Regularization coefficient for the bias
        user_regularization: Regularization coefficient for user latent factors
        positive_item_regularization: Regularization coefficient for positive item latent factors
        negative_item_regularization: Regularization coefficient for negative item latent factors
        update_negative_item_factors:
        update_users:
        update_items:
        update_bias:

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        MF2020 (or MF2020Batch):
          meta:
            save_recs: True
          epochs: 10
          factors: 10
          learning_rate: 0.001
          lambda_weights: 0.0025
    """

    # Model hyperparameters
    factors: int
    learning_rate: float
    lambda_weights: float
    m: int

    def __init__(self, params, interactions, seed, *args, **kwargs):
        super().__init__(params, interactions, seed, *args, **kwargs)

        # Embeddings
        self._user_factors = np.empty((self._num_users, self.factors), dtype=np.float32)
        self._item_factors = np.empty((self._num_items, self.factors), dtype=np.float32)
        self._user_bias = np.empty(self._num_users, dtype=np.float32)
        self._item_bias = np.empty(self._num_items, dtype=np.float32)

        # Global bias
        self._global_bias = 0

        self.transactions = self._interactions.transactions * (self.m + 1)

        # Init embedding weights
        self.modules = [self._user_factors, self._item_factors, self._user_bias, self._item_bias]
        self.bias = [self._user_bias, self._item_bias]
        self.apply(normal_init)

        self.params_to_save = ['_global_bias', '_user_bias', '_item_bias', '_user_factor', '_item_factor']

    def get_training_dataloader(self, batch_size):
        dataloader = self._interactions.get_dataloader(
            "MFPointWisePosNegSampler", batch_size, self._seed, m=self.m
        )
        return dataloader

    def predict(self, user_indices, item_indices=None):
        user_indices = user_indices.numpy()

        # Select only the embeddings in the current batch
        user_embeddings = self._user_factors[user_indices]
        user_bias = self._user_bias[user_indices]

        # Compute predictions
        if item_indices is None:
            item_embeddings = self._item_factors
            item_bias = self._item_bias[None, :]
            einsum_string = "be,ie->bi"
        else:
            item_indices = item_indices.clamp(min=0).numpy()
            item_embeddings = self._item_factors[item_indices]
            item_bias = self._item_bias[item_indices]
            einsum_string = "be,bse->bs"

        predictions = (
            np.einsum(
                einsum_string, user_embeddings, item_embeddings
            )
            + user_bias[:, None]
            + item_bias
            + self._global_bias
        )

        predictions = torch.from_numpy(predictions)
        return predictions


@model_registry.register()
class MF2020(AbstractMF2020):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.05
    lambda_weights: float = 0.0
    m: int = 0

    def __init__(self, params, interactions, seed, *args, **kwargs):
        super().__init__(params, interactions, seed, *args, **kwargs)

    def train_step(self, batch, *args):
        batch = [x.numpy() for x in batch]
        sum_of_loss = 0
        lr = self.learning_rate
        reg = self.lambda_weights
        batch = np.column_stack(batch)

        for user, item, rating in batch:
            gb_ = self._global_bias
            uf_ = self._user_factors[user]
            if_ = self._item_factors[item]
            ub_ = self._user_bias[user]
            ib_ = self._item_bias[item]

            prediction = gb_ + ub_ + ib_ + np.dot(uf_, if_)

            sigmoid = 1 / (1 + np.exp(-prediction))
            this_loss = np.logaddexp(0, prediction) - rating * prediction

            grad = rating - sigmoid

            new_uf = uf_ + lr * (grad * if_ - reg * uf_)
            new_if = if_ + lr * (grad * uf_ - reg * if_)

            self._user_factors[user, :] = new_uf
            self._item_factors[item, :] = new_if
            self._user_bias[user] += lr * (grad - reg * ub_)
            self._item_bias[item] += lr * (grad - reg * ib_)
            self._global_bias += lr * (grad - reg * gb_)

            sum_of_loss += this_loss

        return sum_of_loss


@model_registry.register()
class MF2020Batch(AbstractMF2020):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.05
    lambda_weights: float = 0.0
    m: int = 0

    def __init__(self, params, interactions, seed, *args, **kwargs):
        super().__init__(params, interactions, seed, *args, **kwargs)

    def train_step(self, batch, *args):
        u_idx, i_idx, rating = tuple(x.numpy() for x in batch)
        lr = self.learning_rate
        reg = self.lambda_weights

        gb_ = self._global_bias
        uf_ = self._user_factors[u_idx]
        if_ = self._item_factors[i_idx]
        ub_ = self._user_bias[u_idx]
        ib_ = self._item_bias[i_idx]

        prediction = gb_ + ub_ + ib_ + (uf_ * if_).sum(axis=-1)

        sigmoid = 1 / (1 + np.exp(-prediction))
        this_loss = np.logaddexp(0, prediction) - rating * prediction

        grad = rating - sigmoid

        np.add.at(self._user_factors, u_idx, lr * (np.expand_dims(grad, axis=-1) * if_ - reg * uf_))
        np.add.at(self._item_factors, i_idx, lr * (np.expand_dims(grad, axis=-1) * uf_ - reg * if_))
        np.add.at(self._user_bias, u_idx, lr * (grad - reg * ub_))
        np.add.at(self._item_bias, i_idx, lr * (grad - reg * ib_))
        self._global_bias += lr * (np.sum(grad) - reg * gb_)

        return np.sum(this_loss)
