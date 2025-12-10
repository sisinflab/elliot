"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import torch
from tqdm import tqdm

from elliot.recommender.base_recommender import Recommender
from elliot.recommender.init import normal_init


class NonNegMF(Recommender):
    """
    Non-Negative Matrix Factorization

    For further details, please refer to the `paper <https://ieeexplore.ieee.org/document/6748996>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        reg: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NonNegMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          learning_rate: 0.001
          lambda_weights: 0.1
    """

    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

        self._i_train = self._data.get_train_dict(private=True)
        self._global_mean = np.mean(self._data.sp_i_train_ratings)

        # Embeddings
        self._user_factors = np.empty((self._num_users, self.factors), dtype=np.float32)
        self._item_factors = np.empty((self._num_items, self.factors), dtype=np.float32)
        self._user_bias = np.empty(self._num_users, dtype=np.float32)
        self._item_bias = np.empty(self._num_items, dtype=np.float32)

        # Init embedding weights
        self.modules = [self._user_factors, self._item_factors, self._user_bias, self._item_bias]
        self.bias = [self._user_bias, self._item_bias]
        self.apply(normal_init)

        self.params_to_save = ['_user_bias', '_item_bias', '_user_embeddings', '_item_embeddings']

    def train_step(self, *args):
        # (re)initialize nums and denominators to zero
        user_num = np.zeros_like(self._user_factors)
        user_denom = np.zeros_like(self._user_factors)
        item_num = np.zeros_like(self._item_factors)
        item_denom = np.zeros_like(self._item_factors)

        user_iter = tqdm(
            self._i_train.items(),
            desc="Computing",
            total=len(self._i_train)
        )

        # Compute numerators and denominators for users and items factors
        for u, u_ratings in user_iter:
            items = np.array(list(u_ratings.keys()))
            r_ui = np.array(list(u_ratings.values()))

            # compute current estimation and error
            est = (
                self._global_mean +
                self._user_bias[u] +
                self._item_bias[items] +
                np.dot(self._user_factors[u], self._item_factors[items].T)
            )
            err = r_ui - est

            q_i = self._item_factors[items]
            p_u = self._user_factors[u]

            # update user bias
            for e in err:
                self._user_bias[u] += self.learning_rate * (
                    e - self.lambda_weights * self._user_bias[u]
                )

            # update items biases
            self._item_bias[items] += self.learning_rate * (
                err - self.lambda_weights * self._item_bias[items]
            )

            # compute numerators and denominators
            user_num[u] += np.sum(q_i * r_ui[:, None], axis=0)
            user_denom[u] += np.sum(q_i * est[:, None], axis=0)
            item_num[items] += p_u[None, :] * r_ui[:, None]
            item_denom[items] += p_u[None, :] * est[:, None]

        # Update user factors
        n_ratings = np.array([len(v) for v in self._i_train.values()])
        self._user_factors *= user_num / (
            user_denom + n_ratings[:, None] * self.lambda_weights * self._user_factors
        )

        # Update item factors
        I_train_T = self._data.sp_i_train.tocsc()
        n_ratings_item = np.diff(I_train_T.indptr)
        self._item_factors *= item_num / (
            item_denom + n_ratings_item[:, None] * self.lambda_weights * self._item_factors
        )

        return 0

    def predict_full(self, user_indices):
        user_indices = user_indices.numpy()

        # Retrieve embeddings
        u_embeddings_batch = self._user_factors[user_indices]
        i_embeddings_all = self._item_factors
        u_bias_batch = self._user_bias[user_indices]
        i_bias_all = self._item_bias

        # Compute predictions
        predictions = (
            u_embeddings_batch @ i_embeddings_all.T +
            u_bias_batch[:, None] +
            i_bias_all[None, :] +
            self._global_mean
        )

        predictions = torch.from_numpy(predictions)
        return predictions

    def predict_sampled(self, user_indices, item_indices):
        user_indices = user_indices.numpy()
        item_indices = item_indices.clamp(min=0).numpy()

        # Retrieve embeddings
        u_embeddings_batch = self._user_factors[user_indices]
        i_embeddings_candidate = self._item_factors[item_indices]
        u_bias_batch = self._user_bias[user_indices]
        i_bias_candidate = self._item_bias[item_indices]

        # Compute predictions
        predictions = (
            np.einsum(
                "bi,bji->bj", u_embeddings_batch, i_embeddings_candidate
            ) +
            u_bias_batch[:, None] +
            i_bias_candidate +
            self._global_mean
        )

        predictions = torch.from_numpy(predictions)
        return predictions
