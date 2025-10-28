"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from abc import abstractmethod

import numpy as np

from elliot.dataset.samplers.mf_samplers import MFSampler, MFSamplerRendle
from elliot.recommender.base_recommender import Recommender


class AbstractMF2020(Recommender):
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
          lr: 0.001
          reg: 0.0025
    """

    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_factors", "factors", "f", 10, int, None),
            ("_lr", "lr", "lr", 0.05, None, None),
            ("_reg", "reg", "reg", 0, None, None),
            ("_m", "m", "m", 0, int, None),
            ("_loc", "loc", "loc", 0, float, None),
            ("_scale", "scale", "scale", 0.1, float, None),
        ]
        self.sampler = MFSamplerRendle(data.i_train_dict, data.sp_i_train, seed)
        super().__init__(data, params, seed, logger)

        self._global_bias = 0
        self._user_bias = np.zeros(len(self._users))
        self._item_bias = np.zeros(len(self._items))
        self._user_factors = \
            np.random.normal(loc=self._loc, scale=self._scale, size=(len(self._users), self._factors))
        self._item_factors = \
            np.random.normal(loc=self._loc, scale=self._scale, size=(len(self._items), self._factors))

        self.transactions = data.transactions * (self._m + 1)
        self.sampler.m = self._m

        self.params_to_save = ['_global_bias', '_user_bias', '_item_bias', '_user_factor', '_item_factor']

    @abstractmethod
    def train_step(self, batch, *args):
        raise NotImplementedError()

    def predict(self, start, stop):
        return (
            self._global_bias
            + self._user_bias[start:stop, None]
            + self._item_bias[None, :]
            + self._user_factors[start:stop] @ self._item_factors.T
        )


class MF2020(AbstractMF2020):
    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

    def train_step(self, batch, *args):
        sum_of_loss = 0
        lr = self._lr
        reg = self._reg
        batch = np.column_stack(batch)

        for user, item, rating in batch:
            gb_ = self._global_bias
            uf_ = self._user_factors[user]
            if_ = self._item_factors[item]
            ub_ = self._user_bias[user]
            ib_ = self._item_bias[item]

            prediction = gb_ + ub_ + ib_ + np.dot(uf_, if_)

            if prediction > 0:
                one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
                sigmoid = 1.0 / one_plus_exp_minus_pred
                this_loss = (np.log(one_plus_exp_minus_pred) +
                             (1.0 - rating) * prediction)
            else:
                exp_pred = np.exp(prediction)
                sigmoid = exp_pred / (1.0 + exp_pred)
                this_loss = -rating * prediction + np.log(1.0 + exp_pred)

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


class MF2020Batch(AbstractMF2020):
    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

    def train_step(self, batch, *args):
        u_idx, i_idx, rating = batch
        lr = self._lr
        reg = self._reg

        gb_ = self._global_bias
        uf_ = self._user_factors[u_idx]
        if_ = self._item_factors[i_idx]
        ub_ = self._user_bias[u_idx]
        ib_ = self._item_bias[i_idx]

        prediction = gb_ + ub_ + ib_ + (uf_ * if_).sum(axis=-1)

        exp_pred = np.where(prediction > 0, 1.0 + np.exp(-prediction), np.exp(prediction))
        sigmoid = np.where(prediction > 0, 1.0 / exp_pred, exp_pred / (1.0 + exp_pred))

        this_loss = np.where(prediction > 0, np.log(exp_pred) + (1 - rating) * prediction,
                             -rating * prediction + np.log(1.0 + exp_pred))

        grad = rating - sigmoid

        np.add.at(self._user_factors, u_idx, lr * (np.expand_dims(grad, axis=-1) * if_ - reg * uf_))
        np.add.at(self._item_factors, i_idx, lr * (np.expand_dims(grad, axis=-1) * uf_ - reg * if_))
        np.add.at(self._user_bias, u_idx, lr * (grad - reg * ub_))
        np.add.at(self._item_bias, i_idx, lr * (grad - reg * ib_))
        self._global_bias += lr * (np.sum(grad) - reg * gb_)

        return np.sum(this_loss)

    # def update_factors(self, user: int, item: int, rating: float):
    #     uf_ = self._user_factors[user]
    #     if_ = self._item_factors[item]
    #     ub_ = self._user_bias[user]
    #     ib_ = self._item_bias[item]
    #     gb_ = self._global_bias
    #     lr = self._lr
    #     reg = self._reg
    #
    #
    #     prediction = gb_ + ub_ + ib_ + np.dot(uf_,if_)
    #     # prediction = gb_ + ub_ + ib_ + uf_ @ if_
    #
    #     if prediction > 0:
    #         one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
    #         sigmoid = 1.0 / one_plus_exp_minus_pred
    #         this_loss = (np.log(one_plus_exp_minus_pred) +
    #                      (1.0 - rating) * prediction)
    #     else:
    #         exp_pred = np.exp(prediction)
    #         sigmoid = exp_pred / (1.0 + exp_pred)
    #         this_loss = -rating * prediction + np.log(1.0 + exp_pred)
    #
    #     grad = rating - sigmoid
    #
    #     self._user_factors[user] += lr * (grad * if_ - reg * uf_)
    #     self._item_factors[item] += lr * (grad * uf_ - reg * if_)
    #     self._user_bias[user] += lr * (grad - reg * ub_)
    #     self._item_bias[item] += lr * (grad - reg * ib_)
    #     self._global_bias += lr * (grad - reg * gb_)
    #
    #     return this_loss
